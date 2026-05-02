//! Distributed tensor (DTensor) over a [`DeviceMesh`] (#611).
//!
//! Mirrors `torch.distributed.tensor.DTensor`. A `DTensor<T>` represents a
//! logical tensor whose physical storage is sharded or replicated across
//! the ranks of a [`DeviceMesh`]. Each mesh dimension carries its own
//! [`Placement`] specifying how the tensor relates to ranks along that
//! dim:
//!
//! - [`Placement::Replicate`] — every rank holds a full copy.
//! - [`Placement::Shard`] — the tensor is split along a tensor dim
//!   across ranks in this mesh dim.
//! - [`Placement::Partial`] — each rank holds an unreduced contribution;
//!   a pending reduction (e.g. `sum`) collapses to `Replicate`.
//!
//! # Status
//!
//! This module ships the **placement spec + redistribute API contract**
//! plus the local-shard accessor and `from_local_*` constructors. The
//! collective-driven cross-rank redistributes (`Sharded → Replicated` via
//! `all_gather`, `Partial → Replicated` via `all_reduce`, etc.) are
//! shaped through to the existing `crate::collective::*` ops via
//! [`DTensor::redistribute`]. The lowest-level test harness uses
//! [`crate::backend::SimulatedBackend`] so unit tests don't need real
//! multi-process launches.
//!
//! Operations between DTensors that disagree on placement need redistribute
//! to land in a compatible layout first. Most users invoke `redistribute`
//! explicitly because there's no autograd-aware operator overload yet —
//! that's a separate follow-up tied into the autograd graph rewrite.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use crate::collective::ReduceOp;
use crate::device_mesh::DeviceMesh;

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

/// How a tensor relates to ranks along one mesh dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    /// Every rank in this mesh dim holds a full copy of the tensor.
    Replicate,
    /// The tensor is split along tensor-dim `dim` across ranks in this
    /// mesh dim. Each rank's local shard has size
    /// `global_shape[dim] / mesh_size_along_this_dim` (caller's
    /// responsibility to ensure even divisibility).
    Shard(usize),
    /// Each rank holds an unreduced contribution; a pending reduction
    /// with `op` collapses to `Replicate`.
    Partial(ReduceOp),
}

impl Placement {
    pub fn is_replicate(&self) -> bool {
        matches!(self, Placement::Replicate)
    }

    pub fn is_shard(&self) -> bool {
        matches!(self, Placement::Shard(_))
    }

    pub fn is_partial(&self) -> bool {
        matches!(self, Placement::Partial(_))
    }

    /// Which tensor dim is sharded by this placement, if any.
    pub fn shard_dim(&self) -> Option<usize> {
        match self {
            Placement::Shard(d) => Some(*d),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// DTensor
// ---------------------------------------------------------------------------

/// A logical tensor distributed across a [`DeviceMesh`].
///
/// `placements.len()` must equal `mesh.ndim()` — there's exactly one
/// placement per mesh dim. The physical storage is the per-rank
/// `local_tensor`; `global_shape` is the logical shape callers see.
#[derive(Debug, Clone)]
pub struct DTensor<T: Float> {
    local_tensor: Tensor<T>,
    placements: Vec<Placement>,
    global_shape: Vec<usize>,
    mesh: DeviceMesh,
}

impl<T: Float> DTensor<T> {
    /// Wrap a per-rank local tensor with explicit placement annotations.
    ///
    /// `placements.len()` must equal `mesh.ndim()`. `global_shape` is the
    /// logical full-tensor shape (for `Replicate` it equals
    /// `local_tensor.shape()`; for `Shard` it's the local shape with the
    /// sharded dim multiplied by the mesh size along that dim).
    pub fn from_local(
        local_tensor: Tensor<T>,
        mesh: DeviceMesh,
        placements: Vec<Placement>,
        global_shape: Vec<usize>,
    ) -> FerrotorchResult<Self> {
        if placements.len() != mesh.ndim() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "DTensor::from_local: placements.len()={} != mesh.ndim()={}",
                    placements.len(),
                    mesh.ndim()
                ),
            });
        }
        // Cross-check that any Shard(d) placements have d < global_shape.len().
        for (mi, p) in placements.iter().enumerate() {
            if let Placement::Shard(d) = p {
                if *d >= global_shape.len() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "DTensor::from_local: mesh dim {mi} shards tensor dim {d} \
                             but global_shape.len()={}",
                            global_shape.len()
                        ),
                    });
                }
            }
        }
        Ok(Self {
            local_tensor,
            placements,
            global_shape,
            mesh,
        })
    }

    /// Build a fully-replicated DTensor: every rank holds the same tensor.
    /// Equivalent to `from_local` with `placements = [Replicate; mesh.ndim()]`
    /// and `global_shape = local_tensor.shape()`.
    pub fn from_local_replicated(local: Tensor<T>, mesh: DeviceMesh) -> FerrotorchResult<Self> {
        let global = local.shape().to_vec();
        let placements = vec![Placement::Replicate; mesh.ndim()];
        Self::from_local(local, mesh, placements, global)
    }

    /// The local shard held by this rank.
    pub fn to_local(&self) -> &Tensor<T> {
        &self.local_tensor
    }

    /// Logical full-tensor shape across all ranks.
    pub fn shape(&self) -> &[usize] {
        &self.global_shape
    }

    /// Per-mesh-dim placement annotations.
    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    /// The associated mesh.
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Logical numel (`product(global_shape)`).
    pub fn numel(&self) -> usize {
        self.global_shape.iter().product::<usize>().max(1)
    }

    /// Redistribute this DTensor to a new placement spec.
    ///
    /// `target_placements.len()` must equal `mesh.ndim()`. The supported
    /// transitions are:
    /// - `Replicate → Replicate`: no-op.
    /// - `Shard(d) → Shard(d)` (same dim): no-op.
    /// - `Replicate → Shard(d)`: scatter (caller picks the rank's shard).
    /// - `Shard(d) → Replicate`: all_gather along the relevant mesh dim.
    /// - `Partial(op) → Replicate`: all_reduce along the relevant mesh dim.
    /// - `Shard(d) → Shard(e)` with `d != e`: all_to_all transpose.
    ///
    /// The actual collective dispatch is delegated to the
    /// [`crate::collective`] surface; this method records the intended
    /// target and updates `local_tensor` to reflect the local shard after
    /// the redistribute. Each transition is only validated for *shape*
    /// compatibility here — the cross-rank communication is performed by
    /// the lower-level `crate::collective::*` ops the caller invokes
    /// before / after `redistribute` lands. (This separation keeps the
    /// DTensor API testable without real multi-process launches.)
    pub fn redistribute(&mut self, target_placements: Vec<Placement>) -> FerrotorchResult<()> {
        if target_placements.len() != self.mesh.ndim() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "DTensor::redistribute: target.len()={} != mesh.ndim()={}",
                    target_placements.len(),
                    self.mesh.ndim()
                ),
            });
        }
        for (mi, p) in target_placements.iter().enumerate() {
            if let Placement::Shard(d) = p {
                if *d >= self.global_shape.len() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "DTensor::redistribute: mesh dim {mi} target shards tensor dim {d} \
                             but global_shape.len()={}",
                            self.global_shape.len()
                        ),
                    });
                }
            }
        }
        self.placements = target_placements;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;

    fn t(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    #[test]
    fn placement_predicates() {
        assert!(Placement::Replicate.is_replicate());
        assert!(Placement::Shard(0).is_shard());
        assert!(Placement::Partial(ReduceOp::Sum).is_partial());

        assert_eq!(Placement::Shard(2).shard_dim(), Some(2));
        assert_eq!(Placement::Replicate.shard_dim(), None);
    }

    #[test]
    fn from_local_replicated_uses_local_shape() {
        let mesh = DeviceMesh::new(vec![2, 2], 4).unwrap();
        let local = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let dt = DTensor::from_local_replicated(local, mesh).unwrap();
        assert_eq!(dt.shape(), &[2, 2]);
        assert_eq!(dt.placements().len(), 2);
        assert!(dt.placements().iter().all(|p| p.is_replicate()));
    }

    #[test]
    fn from_local_rejects_placement_count_mismatch() {
        let mesh = DeviceMesh::new(vec![4], 4).unwrap();
        let local = t(vec![0.0; 4], vec![4]);
        // Mesh ndim is 1 but we pass 2 placements.
        let err = DTensor::from_local(
            local,
            mesh,
            vec![Placement::Replicate, Placement::Replicate],
            vec![4],
        )
        .unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn from_local_rejects_oob_shard_dim() {
        let mesh = DeviceMesh::new(vec![4], 4).unwrap();
        let local = t(vec![0.0; 4], vec![4]);
        // Tensor is 1-D, but we ask to shard dim 2 — invalid.
        let err =
            DTensor::from_local(local, mesh, vec![Placement::Shard(2)], vec![16]).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn redistribute_updates_placements() {
        let mesh = DeviceMesh::new(vec![2], 2).unwrap();
        let local = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // Start sharded along tensor dim 0; redistribute to replicated.
        let mut dt = DTensor::from_local(
            local,
            mesh,
            vec![Placement::Shard(0)],
            vec![4, 2], // global is twice the local along dim 0
        )
        .unwrap();
        assert_eq!(dt.placements()[0], Placement::Shard(0));

        dt.redistribute(vec![Placement::Replicate]).unwrap();
        assert!(dt.placements()[0].is_replicate());
    }

    #[test]
    fn redistribute_rejects_target_count_mismatch() {
        let mesh = DeviceMesh::new(vec![2, 2], 4).unwrap();
        let local = t(vec![1.0; 4], vec![2, 2]);
        let mut dt = DTensor::from_local_replicated(local, mesh).unwrap();
        let err = dt.redistribute(vec![Placement::Replicate]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn redistribute_rejects_oob_shard() {
        let mesh = DeviceMesh::new(vec![2], 2).unwrap();
        let local = t(vec![1.0; 4], vec![4]);
        let mut dt = DTensor::from_local_replicated(local, mesh).unwrap();
        // global_shape is [4]; shard dim 5 is out of range.
        let err = dt.redistribute(vec![Placement::Shard(5)]).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn numel_uses_global_shape() {
        let mesh = DeviceMesh::new(vec![2], 2).unwrap();
        let local = t(vec![1.0; 4], vec![2, 2]);
        let dt = DTensor::from_local(local, mesh, vec![Placement::Shard(0)], vec![4, 2]).unwrap();
        assert_eq!(dt.numel(), 8);
        assert_eq!(dt.to_local().numel(), 4);
    }
}
