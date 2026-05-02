//! `DeviceMesh` — multi-dimensional rank layout. (#591)
//!
//! Mirrors `torch.distributed.DeviceMesh`. A mesh is an n-dimensional
//! arrangement of ranks: e.g. for 2-D parallelism with 8 GPUs split into
//! 2-way data parallel × 4-way tensor parallel, the mesh has shape `[2, 4]`
//! with ranks `[[0, 1, 2, 3], [4, 5, 6, 7]]`.
//!
//! The mesh exposes:
//! - The rank's coordinate within the mesh (`coords()`)
//! - The list of ranks along each dimension that share the same coords on
//!   every other dim (`ranks_along_dim()`) — used to construct sub-groups
//!   for collective ops scoped to one parallelism axis.
//!
//! Sub-group / sub-backend creation is a separate concern handled by
//! [`SubBackend`](crate::backend::SubBackend); this module is
//! infrastructure-agnostic and just maintains the index math.

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// An n-D arrangement of ranks. The product of `shape` must equal the
/// world size; ranks are laid out in row-major order (last dim varies
/// fastest).
#[derive(Debug, Clone)]
pub struct DeviceMesh {
    shape: Vec<usize>,
    /// Names for each mesh dimension, e.g. `["dp", "tp"]`. Optional —
    /// callers can pass `None` to skip naming.
    dim_names: Option<Vec<String>>,
}

impl DeviceMesh {
    /// Create a mesh with the given shape. `world_size` must equal the
    /// product of `shape` (verified eagerly so misconfigured launches
    /// fail loudly instead of silently splitting wrong).
    pub fn new(shape: Vec<usize>, world_size: usize) -> FerrotorchResult<Self> {
        if shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "DeviceMesh: shape must be non-empty".into(),
            });
        }
        let prod: usize = shape.iter().product::<usize>().max(1);
        if prod != world_size {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("DeviceMesh: shape product {prod} != world_size {world_size}"),
            });
        }
        for (i, &d) in shape.iter().enumerate() {
            if d == 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("DeviceMesh: dim {i} is 0"),
                });
            }
        }
        Ok(Self {
            shape,
            dim_names: None,
        })
    }

    /// Variant of [`new`] that also attaches names to each dim.
    /// `dim_names.len()` must match `shape.len()`.
    pub fn new_with_names(
        shape: Vec<usize>,
        dim_names: Vec<String>,
        world_size: usize,
    ) -> FerrotorchResult<Self> {
        if dim_names.len() != shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DeviceMesh: dim_names.len()={} != shape.len()={}",
                    dim_names.len(),
                    shape.len()
                ),
            });
        }
        let mut mesh = Self::new(shape, world_size)?;
        mesh.dim_names = Some(dim_names);
        Ok(mesh)
    }

    /// Mesh shape (`[dp, tp, ...]`).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Optional dim names.
    pub fn dim_names(&self) -> Option<&[String]> {
        self.dim_names.as_deref()
    }

    /// Dimensionality of the mesh.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of ranks in the mesh (= world_size).
    pub fn size(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Resolve a dim name to its index.
    pub fn dim_index(&self, name: &str) -> FerrotorchResult<usize> {
        let names = self
            .dim_names
            .as_ref()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "DeviceMesh: no dim names registered".into(),
            })?;
        names
            .iter()
            .position(|n| n == name)
            .ok_or(FerrotorchError::InvalidArgument {
                message: format!("DeviceMesh: dim name '{name}' not found"),
            })
    }

    /// Convert a rank to its multi-dim coordinate within the mesh.
    /// Row-major: the last dim varies fastest.
    pub fn coords(&self, rank: usize) -> FerrotorchResult<Vec<usize>> {
        if rank >= self.size() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DeviceMesh: rank {rank} out of range for mesh size {}",
                    self.size()
                ),
            });
        }
        let mut out = vec![0usize; self.shape.len()];
        let mut r = rank;
        for i in (0..self.shape.len()).rev() {
            out[i] = r % self.shape[i];
            r /= self.shape[i];
        }
        Ok(out)
    }

    /// Inverse of [`coords`]: convert a coordinate to its rank.
    pub fn rank_of(&self, coords: &[usize]) -> FerrotorchResult<usize> {
        if coords.len() != self.shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DeviceMesh: coords len {} != ndim {}",
                    coords.len(),
                    self.shape.len()
                ),
            });
        }
        let mut rank = 0usize;
        for (i, &c) in coords.iter().enumerate() {
            if c >= self.shape[i] {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "DeviceMesh: coord[{i}] = {c} out of range for dim size {}",
                        self.shape[i]
                    ),
                });
            }
            rank = rank * self.shape[i] + c;
        }
        Ok(rank)
    }

    /// All ranks along `dim` that share `rank`'s coordinates on every
    /// other dim. Useful for constructing per-axis collective groups
    /// (e.g. one TP group per data-parallel slice).
    ///
    /// Returns the ranks in increasing-coord order on `dim`.
    pub fn ranks_along_dim(&self, dim: usize, rank: usize) -> FerrotorchResult<Vec<usize>> {
        if dim >= self.shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DeviceMesh: dim {dim} out of range for ndim {}",
                    self.shape.len()
                ),
            });
        }
        let mut coords = self.coords(rank)?;
        let mut ranks = Vec::with_capacity(self.shape[dim]);
        for d in 0..self.shape[dim] {
            coords[dim] = d;
            ranks.push(self.rank_of(&coords)?);
        }
        Ok(ranks)
    }

    /// All sub-groups along `dim`: a partitioning of the world into
    /// disjoint groups of `shape[dim]` ranks each, such that every
    /// group consists of ranks differing only on `dim`. Useful for
    /// building sub-backends in bulk.
    pub fn groups_along_dim(&self, dim: usize) -> FerrotorchResult<Vec<Vec<usize>>> {
        if dim >= self.shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DeviceMesh: dim {dim} out of range for ndim {}",
                    self.shape.len()
                ),
            });
        }
        let world = self.size();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut seen = vec![false; world];
        for rank in 0..world {
            if seen[rank] {
                continue;
            }
            let g = self.ranks_along_dim(dim, rank)?;
            for &r in &g {
                seen[r] = true;
            }
            groups.push(g);
        }
        Ok(groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_shape_must_match_world_size() {
        let err = DeviceMesh::new(vec![2, 3], 5).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn mesh_zero_dim_rejected() {
        let err = DeviceMesh::new(vec![2, 0], 0).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn mesh_coords_roundtrip_2d() {
        let m = DeviceMesh::new(vec![2, 4], 8).unwrap();
        // 2x4 layout in row-major:
        //   [[0, 1, 2, 3],
        //    [4, 5, 6, 7]]
        for rank in 0..8 {
            let coords = m.coords(rank).unwrap();
            assert_eq!(coords[0], rank / 4);
            assert_eq!(coords[1], rank % 4);
            assert_eq!(m.rank_of(&coords).unwrap(), rank);
        }
    }

    #[test]
    fn mesh_ranks_along_dim_returns_correct_axis() {
        let m = DeviceMesh::new(vec![2, 4], 8).unwrap();
        // rank 5 has coords (1, 1).
        // along dim 0 (data-parallel axis): same col → ranks (0, 1) and (1, 1) = [1, 5]
        // along dim 1 (tensor-parallel axis): same row → ranks 4..=7
        assert_eq!(m.ranks_along_dim(0, 5).unwrap(), vec![1, 5]);
        assert_eq!(m.ranks_along_dim(1, 5).unwrap(), vec![4, 5, 6, 7]);
    }

    #[test]
    fn mesh_groups_along_dim_partition_world() {
        let m = DeviceMesh::new(vec![2, 4], 8).unwrap();
        // Along dim 0: 4 groups of 2 (one per col).
        let g0 = m.groups_along_dim(0).unwrap();
        assert_eq!(g0.len(), 4);
        for g in &g0 {
            assert_eq!(g.len(), 2);
        }
        // Sorted union covers every rank exactly once.
        let mut all: Vec<usize> = g0.iter().flatten().copied().collect();
        all.sort();
        assert_eq!(all, (0..8).collect::<Vec<_>>());

        // Along dim 1: 2 groups of 4 (one per row).
        let g1 = m.groups_along_dim(1).unwrap();
        assert_eq!(g1.len(), 2);
        assert_eq!(g1[0], vec![0, 1, 2, 3]);
        assert_eq!(g1[1], vec![4, 5, 6, 7]);
    }

    #[test]
    fn mesh_with_dim_names_resolve_index() {
        let m = DeviceMesh::new_with_names(vec![2, 4], vec!["dp".to_string(), "tp".to_string()], 8)
            .unwrap();
        assert_eq!(m.dim_index("dp").unwrap(), 0);
        assert_eq!(m.dim_index("tp").unwrap(), 1);
        assert!(m.dim_index("missing").is_err());
    }

    #[test]
    fn mesh_new_with_names_rejects_mismatched_lengths() {
        let err = DeviceMesh::new_with_names(vec![2, 4], vec!["only_one_name".to_string()], 8)
            .unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn mesh_oob_rank_errors() {
        let m = DeviceMesh::new(vec![2, 2], 4).unwrap();
        assert!(m.coords(4).is_err());
    }

    #[test]
    fn mesh_oob_coord_errors() {
        let m = DeviceMesh::new(vec![2, 2], 4).unwrap();
        let err = m.rank_of(&[0, 5]).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn mesh_3d_correctness() {
        // 2x2x3 = 12 ranks, row-major.
        let m = DeviceMesh::new(vec![2, 2, 3], 12).unwrap();
        // rank 7: 7 = 1*6 + 0*3 + 1 → coords (1, 0, 1)
        assert_eq!(m.coords(7).unwrap(), vec![1, 0, 1]);
        assert_eq!(m.rank_of(&[1, 0, 1]).unwrap(), 7);
        // along innermost dim from rank 7: same (1, 0, *) → 6, 7, 8
        assert_eq!(m.ranks_along_dim(2, 7).unwrap(), vec![6, 7, 8]);
    }
}
