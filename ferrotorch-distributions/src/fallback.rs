//! Opt-in CPU fallback gate for distribution methods.
//!
//! ferrotorch-distributions' compute bodies are entirely on host. When a
//! distribution method is invoked with CUDA tensors, the existing code path
//! does host readback (`data_vec()`), runs CPU math, then re-uploads results
//! via `TensorStorage::cpu(...)` and `Tensor::to(device)`. That is the silent
//! CPU-fallback anti-pattern (`rust-gpu-discipline` §3): the user named a
//! GPU tensor and the work secretly ran on CPU.
//!
//! Until per-distribution GPU kernels exist (Category A scope, tracked
//! separately), the policy here mirrors PyTorch's MPS fallback: error by
//! default on CUDA inputs, allow opt-in CPU fallback gated by the
//! `FERROTORCH_ENABLE_GPU_FALLBACK` environment variable, log a
//! `tracing::warn!` on every fallback so the slow path is loud.
//!
//! Each distribution method that performs the readback/recompute pattern
//! invokes [`check_gpu_fallback_opt_in`] at entry, passing all tensor
//! operands plus a static op label. The CPU compute body following the
//! guard is unchanged.
//!
//! # PyTorch parity
//!
//! PyTorch raises `NotImplementedError` when a CUDA tensor calls an op with
//! no CUDA kernel. The single documented escape hatch is
//! `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple MPS, which logs a
//! `UserWarning` per call. We mirror that shape.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

/// Environment variable name read on every guarded call. Setting it to any
/// value (`1`, `true`, anything non-empty as long as the var is *present*)
/// enables the CPU fallback path; unsetting it makes CUDA inputs an error.
pub(crate) const FALLBACK_ENV_VAR: &str = "FERROTORCH_ENABLE_GPU_FALLBACK";

/// Guard the entry of a distribution method against silent CPU fallback on
/// CUDA inputs.
///
/// Behaviour:
///
/// - All inputs are CPU tensors → `Ok(())`. The CPU compute body that
///   follows is correct as-is.
/// - At least one input is a CUDA tensor and `FERROTORCH_ENABLE_GPU_FALLBACK`
///   is set in the process environment → `Ok(())`, with a `tracing::warn!`
///   logged per call. The CPU compute body runs and the user is told once
///   per call that they hit a slow path.
/// - At least one input is a CUDA tensor and the env var is unset →
///   `Err(FerrotorchError::NotImplementedOnCuda { op })`. This is the
///   PyTorch-parity default.
///
/// `op` should be a `"<DistributionName>::<method>"` literal (e.g.
/// `"Normal::log_prob"`) so the error and warning name the call site.
pub(crate) fn check_gpu_fallback_opt_in<T>(
    inputs: &[&Tensor<T>],
    op: &'static str,
) -> FerrotorchResult<()>
where
    T: Float,
{
    let any_cuda = inputs.iter().any(|t| t.is_cuda());
    if !any_cuda {
        return Ok(());
    }
    if std::env::var(FALLBACK_ENV_VAR).is_ok() {
        tracing::warn!(
            target: "ferrotorch::gpu_fallback",
            op = op,
            env_var = FALLBACK_ENV_VAR,
            "ferrotorch-distributions: {op} has no GPU implementation; \
             CPU fallback enabled because {FALLBACK_ENV_VAR} is set. \
             Unset {FALLBACK_ENV_VAR} to make this an error.",
        );
        return Ok(());
    }
    Err(FerrotorchError::NotImplementedOnCuda { op })
}

#[cfg(test)]
mod tests {
    //! Behaviour tests for the fallback gate.
    //!
    //! Tests in this module mutate process-global environment state
    //! (`FERROTORCH_ENABLE_GPU_FALLBACK`) and so must run serially. They
    //! share a single `Mutex` to enforce that, regardless of cargo's test
    //! threading. Adding `serial_test` was considered but it isn't a
    //! workspace dependency, and a local `Mutex` is one line of overhead.
    //!
    //! The CUDA-input arms require an initialised GPU backend. They run
    //! only under `--features cuda`; without the feature they compile out
    //! entirely so the CPU-input arm still exercises the gate's `Ok(())`
    //! path on machines without GPUs.
    use super::*;
    use ferrotorch_core::creation;
    use std::sync::Mutex;

    /// Serializes env-var-mutating tests within this module. All tests in
    /// this `mod tests` lock this mutex on entry.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Helper: run `f` with the fallback env var unset.
    fn with_env_unset<R>(f: impl FnOnce() -> R) -> R {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: tests in this module are serialised through ENV_LOCK,
        // so no other test thread is reading or writing this env var
        // concurrently. The mutation is local to the test binary's
        // process; no FFI or signal handler observes the change.
        unsafe {
            std::env::remove_var(FALLBACK_ENV_VAR);
        }
        f()
    }

    /// Helper: run `f` with the fallback env var set to `"1"`.
    #[cfg(feature = "cuda")]
    fn with_env_set<R>(f: impl FnOnce() -> R) -> R {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: tests in this module are serialised through ENV_LOCK.
        unsafe {
            std::env::set_var(FALLBACK_ENV_VAR, "1");
        }
        let out = f();
        // SAFETY: same justification as above.
        unsafe {
            std::env::remove_var(FALLBACK_ENV_VAR);
        }
        out
    }

    #[test]
    fn cpu_inputs_ok_regardless_of_env_var() {
        with_env_unset(|| {
            let t = creation::scalar(1.0f32).unwrap();
            assert!(check_gpu_fallback_opt_in(&[&t], "TestDist::op").is_ok());
        });
    }

    #[test]
    fn empty_inputs_ok() {
        with_env_unset(|| {
            let inputs: &[&Tensor<f32>] = &[];
            assert!(check_gpu_fallback_opt_in(inputs, "TestDist::op").is_ok());
        });
    }

    // ------------------------------------------------------------------
    // CUDA-input behaviour. These require a working GPU and the `cuda`
    // feature, which compiles ferrotorch-gpu and lets us actually
    // construct CUDA tensors.
    // ------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_input_errors_when_env_var_unset() {
        with_env_unset(|| {
            ferrotorch_gpu::init_cuda_backend().expect("init GPU backend for fallback tests");
            let cpu = creation::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
            let device = ferrotorch_core::device::Device::Cuda(0);
            let cuda = cpu.to(device).expect("upload to CUDA");
            let result = check_gpu_fallback_opt_in(&[&cuda], "TestDist::op");
            assert!(matches!(
                result,
                Err(FerrotorchError::NotImplementedOnCuda { op: "TestDist::op" })
            ));
        });
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_input_ok_when_env_var_set_with_cpu_correct_result() {
        with_env_set(|| {
            ferrotorch_gpu::init_cuda_backend().expect("init GPU backend for fallback tests");
            let cpu = creation::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
            let device = ferrotorch_core::device::Device::Cuda(0);
            let cuda = cpu.to(device).expect("upload to CUDA");
            let result = check_gpu_fallback_opt_in(&[&cuda], "TestDist::op");
            assert!(result.is_ok());
            // After the guard returns Ok, the existing CPU compute body in
            // each distribution method runs unchanged. Sanity-check that
            // the readback path (`data_vec()`) still produces the original
            // values — that is the whole behavioural contract: the CPU
            // body is correct.
            let host = cuda.data_vec().expect("readback CUDA tensor");
            assert_eq!(host, vec![1.0f32, 2.0, 3.0]);
        });
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mixed_cpu_and_cuda_inputs_treated_as_cuda() {
        with_env_unset(|| {
            ferrotorch_gpu::init_cuda_backend().expect("init GPU backend for fallback tests");
            let cpu = creation::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
            let device = ferrotorch_core::device::Device::Cuda(0);
            let cuda = cpu.clone().to(device).expect("upload to CUDA");
            let result = check_gpu_fallback_opt_in(&[&cpu, &cuda], "TestDist::op");
            assert!(matches!(
                result,
                Err(FerrotorchError::NotImplementedOnCuda { op: "TestDist::op" })
            ));
        });
    }
}
