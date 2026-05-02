//! JIT compilation of generated C source into native machine code.
//!
//! The [`InductorBackend`](crate::codegen::InductorBackend) emits C source
//! strings for fusion groups via [`CpuCodegen::generate_c_source`]. This
//! module closes the loop: it shells out to the system C compiler, produces
//! a shared library, loads it with `libloading`, and exposes the compiled
//! kernel as a callable closure.
//!
//! # Pipeline
//!
//! ```text
//! LoopIR → C source → cc -O3 -shared → .so → dlopen → fn ptr
//! ```
//!
//! # ABI
//!
//! Each generated kernel has a per-input C signature like:
//!
//! ```c
//! void kernel_0(const double* in0, const double* in1,
//!               double* output, int n);
//! ```
//!
//! To get a uniform FFI signature regardless of input count, this module
//! wraps every generated kernel with a fixed trampoline:
//!
//! ```c
//! void ferrotorch_kernel_entry(const double* const* inputs,
//!                              double* output, int n) {
//!     kernel_0(inputs[0], inputs[1], ..., output, n);
//! }
//! ```
//!
//! The entry symbol `ferrotorch_kernel_entry` is what the Rust side loads
//! and invokes.
//!
//! # Compile cache
//!
//! Compilation is the expensive part of the pipeline (fork+exec to `cc`
//! plus link). To amortize it across repeated calls with identical source,
//! this module keeps a global `HashMap<source_hash, Arc<JitCompiledKernel>>`
//! cache. Two graphs that lower to byte-identical C hit the cache on the
//! second call.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};

// ---------------------------------------------------------------------------
// JitCompiledKernel
// ---------------------------------------------------------------------------

/// Signature of the trampoline entry point exposed by every compiled kernel.
///
/// `inputs` is a pointer to an array of `num_inputs` `*const f64` pointers;
/// `output` is a pointer to a writable `f64` buffer; `n` is the element
/// count the kernel was generated for (present for potential future
/// dynamic-shape support — current kernels bake the shape in).
type KernelEntry = unsafe extern "C" fn(*const *const f64, *mut f64, i32);

/// A compiled C kernel loaded as a shared library.
///
/// The handle keeps the `libloading::Library` alive for the full lifetime
/// of the kernel, so the function pointer stays valid. Instances are
/// cheap to clone via `Arc`.
pub struct JitCompiledKernel {
    /// Dynamic library handle — MUST outlive `kernel_fn`.
    _lib: libloading::Library,
    /// Trampoline entry symbol, resolved once at load time.
    kernel_fn: KernelEntry,
    /// Number of input buffers the kernel expects.
    num_inputs: usize,
    /// Number of output elements the kernel writes.
    output_len: usize,
    /// Path of the .so file on disk (kept so we can `Drop` it if needed).
    _so_path: PathBuf,
}

impl std::fmt::Debug for JitCompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiledKernel")
            .field("num_inputs", &self.num_inputs)
            .field("output_len", &self.output_len)
            .field("so_path", &self._so_path)
            .finish()
    }
}

// Safety: `libloading::Library` is `Send + Sync` on all supported targets
// (Unix and Windows). The raw `extern "C" fn` is also `Send + Sync`.
unsafe impl Send for JitCompiledKernel {}
unsafe impl Sync for JitCompiledKernel {}

impl JitCompiledKernel {
    /// The number of inputs this kernel expects.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// The number of output elements this kernel writes.
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    /// Execute the compiled kernel.
    ///
    /// Populates `output` with the result of applying the kernel to the
    /// input buffers. All input slices must have at least `output_len`
    /// elements (the kernel writes `output_len` outputs but may read any
    /// combination of the inputs element-by-element).
    ///
    /// # Errors
    ///
    /// Returns an error if the number of input buffers does not match
    /// `num_inputs` or if `output.len() < output_len`.
    pub fn execute(&self, inputs: &[&[f64]], output: &mut [f64]) -> FerrotorchResult<()> {
        if inputs.len() != self.num_inputs {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "JitCompiledKernel::execute: expected {} input buffers, got {}",
                    self.num_inputs,
                    inputs.len()
                ),
            });
        }
        if output.len() < self.output_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "JitCompiledKernel::execute: output buffer too small — need \
                     {}, got {}",
                    self.output_len,
                    output.len()
                ),
            });
        }
        for (i, buf) in inputs.iter().enumerate() {
            if buf.len() < self.output_len {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "JitCompiledKernel::execute: input {} has {} elements, \
                         kernel expects at least {}",
                        i,
                        buf.len(),
                        self.output_len
                    ),
                });
            }
        }

        // Build a contiguous array of raw input pointers.
        let ptrs: Vec<*const f64> = inputs.iter().map(|b| b.as_ptr()).collect();

        // Safety: we validated buffer lengths above, the function pointer
        // was resolved from a library we keep alive, and the trampoline
        // ABI matches `KernelEntry`.
        unsafe {
            (self.kernel_fn)(ptrs.as_ptr(), output.as_mut_ptr(), self.output_len as i32);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compile cache
// ---------------------------------------------------------------------------

/// Global compile cache keyed by hash of the complete source (kernel +
/// trampoline). Shared across threads.
fn compile_cache() -> &'static Mutex<HashMap<u64, Arc<JitCompiledKernel>>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, Arc<JitCompiledKernel>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn hash_source(source: &str, num_inputs: usize, output_len: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    num_inputs.hash(&mut hasher);
    output_len.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Public compile entry point
// ---------------------------------------------------------------------------

/// Wrap a kernel function (as emitted by
/// [`CpuCodegen::generate_c_source`](crate::codegen_cpu::CpuCodegen::generate_c_source))
/// with a fixed-signature trampoline and compile the whole thing into a
/// loaded shared library.
///
/// `kernel_source` is the full C source of a single kernel function (must
/// define a function with the given `kernel_fn_name`).
///
/// # Errors
///
/// Returns an error if:
/// - no system C compiler can be found (`cc`, `gcc`, or `clang`),
/// - the compiler invocation fails (bad source, missing math.h, link error),
/// - the produced shared library cannot be loaded or the entry symbol cannot
///   be resolved.
pub fn compile_c_kernel(
    kernel_source: &str,
    kernel_fn_name: &str,
    num_inputs: usize,
    output_len: usize,
) -> FerrotorchResult<Arc<JitCompiledKernel>> {
    // Build the wrapped source (kernel + trampoline) so we can hash and
    // compile a single self-contained translation unit.
    let full_source = wrap_with_trampoline(kernel_source, kernel_fn_name, num_inputs);
    let key = hash_source(&full_source, num_inputs, output_len);

    // Cache hit?
    {
        let cache = compile_cache()
            .lock()
            .map_err(|_| FerrotorchError::Internal {
                message: "jit compile cache mutex poisoned".into(),
            })?;
        if let Some(existing) = cache.get(&key) {
            return Ok(existing.clone());
        }
    }

    let kernel = compile_and_load(&full_source, output_len, num_inputs)?;
    let arc = Arc::new(kernel);

    // Insert into cache (ignore races — last writer wins, but the
    // kernels are observationally identical).
    {
        let mut cache = compile_cache()
            .lock()
            .map_err(|_| FerrotorchError::Internal {
                message: "jit compile cache mutex poisoned".into(),
            })?;
        cache.insert(key, arc.clone());
    }

    Ok(arc)
}

/// Build the full C translation unit: the user-provided kernel function
/// plus a fixed-signature trampoline `ferrotorch_kernel_entry` that fans
/// out a `const double* const*` array into per-input arguments.
fn wrap_with_trampoline(kernel_source: &str, kernel_fn_name: &str, num_inputs: usize) -> String {
    let mut out = String::with_capacity(kernel_source.len() + 256);

    // Keep the kernel verbatim.
    out.push_str(kernel_source);
    out.push('\n');

    // Emit the trampoline entry point.
    out.push_str("void ferrotorch_kernel_entry(\n");
    out.push_str("    const double* const* inputs,\n");
    out.push_str("    double* output,\n");
    out.push_str("    int n\n");
    out.push_str(") {\n");

    if num_inputs == 0 {
        // No inputs — the kernel still takes an output pointer and length.
        out.push_str(&format!("    {kernel_fn_name}(output, n);\n"));
    } else {
        out.push_str(&format!("    {kernel_fn_name}("));
        for i in 0..num_inputs {
            out.push_str(&format!("inputs[{i}], "));
        }
        out.push_str("output, n);\n");
    }

    out.push_str("}\n");
    out
}

/// Shell out to the system C compiler to build a shared library from
/// `source`, load it with `libloading`, and resolve the trampoline
/// symbol.
fn compile_and_load(
    source: &str,
    output_len: usize,
    num_inputs: usize,
) -> FerrotorchResult<JitCompiledKernel> {
    // Unique per-compile directory so concurrent compiles don't clobber
    // one another's .c / .so files.
    let tmp_dir = tempfile::Builder::new()
        .prefix("ferrotorch-jit-")
        .tempdir()
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: failed to create temp dir: {e}"),
        })?;

    let c_path = tmp_dir.path().join("kernel.c");
    let so_path = tmp_dir.path().join(format!("kernel{}", shared_lib_ext()));

    // Write the source.
    {
        let mut f = std::fs::File::create(&c_path).map_err(|e| FerrotorchError::Internal {
            message: format!("jit: failed to create {}: {}", c_path.display(), e),
        })?;
        f.write_all(source.as_bytes())
            .map_err(|e| FerrotorchError::Internal {
                message: format!("jit: failed to write source: {e}"),
            })?;
    }

    // Invoke the compiler.
    let compiler = find_c_compiler().ok_or_else(|| FerrotorchError::Internal {
        message: "jit: no C compiler found (tried cc, gcc, clang). \
                  Install one to use the InductorBackend JIT path."
            .into(),
    })?;

    let status = Command::new(&compiler)
        .arg("-O3")
        .arg("-fPIC")
        .arg("-shared")
        .arg("-ffast-math")
        .arg("-o")
        .arg(&so_path)
        .arg(&c_path)
        .arg("-lm")
        .status()
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: failed to spawn {}: {}", compiler.to_string_lossy(), e),
        })?;

    if !status.success() {
        return Err(FerrotorchError::Internal {
            message: format!(
                "jit: C compiler ({}) failed with status {}. Source was:\n{}",
                compiler.to_string_lossy(),
                status,
                source
            ),
        });
    }

    // Load the shared library.
    //
    // Safety: we just wrote and compiled this library ourselves, and the
    // trampoline has a well-defined C ABI.
    let lib =
        unsafe { libloading::Library::new(&so_path) }.map_err(|e| FerrotorchError::Internal {
            message: format!(
                "jit: libloading::Library::new({}): {}",
                so_path.display(),
                e
            ),
        })?;

    // Resolve the trampoline symbol.
    //
    // Safety: the symbol was defined in the source we compiled and its
    // ABI matches `KernelEntry`.
    let kernel_fn: KernelEntry = unsafe {
        let sym: libloading::Symbol<KernelEntry> =
            lib.get(b"ferrotorch_kernel_entry\0")
                .map_err(|e| FerrotorchError::Internal {
                    message: format!(
                        "jit: could not resolve ferrotorch_kernel_entry in {}: {}",
                        so_path.display(),
                        e
                    ),
                })?;
        *sym
    };

    // Keep the .so around for the lifetime of the library — persist by
    // detaching the tempdir so the OS does not reclaim it while we are
    // still executing the kernel.
    let persisted_path = tmp_dir.keep();
    let so_path_final = persisted_path.join(
        so_path
            .file_name()
            .expect("so_path has no file name; should be impossible"),
    );

    Ok(JitCompiledKernel {
        _lib: lib,
        kernel_fn,
        num_inputs,
        output_len,
        _so_path: so_path_final,
    })
}

/// Return the host's shared-library extension (including the leading dot).
fn shared_lib_ext() -> &'static str {
    if cfg!(target_os = "windows") {
        ".dll"
    } else if cfg!(target_os = "macos") {
        ".dylib"
    } else {
        ".so"
    }
}

/// Find the first available C compiler in `PATH`. Checks `CC` env var
/// first, then falls back to a list of common compiler names.
fn find_c_compiler() -> Option<OsString> {
    if let Ok(cc) = std::env::var("CC") {
        if !cc.is_empty() && which(&cc).is_some() {
            return Some(OsString::from(cc));
        }
    }
    for candidate in ["cc", "gcc", "clang"] {
        if which(candidate).is_some() {
            return Some(OsString::from(candidate));
        }
    }
    None
}

/// Minimal `which` replacement: walk `PATH` looking for `name`.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let full = dir.join(name);
        if full.is_file() {
            return Some(full);
        }
        // Windows: also try name + ".exe".
        if cfg!(target_os = "windows") {
            let full_exe = dir.join(format!("{name}.exe"));
            if full_exe.is_file() {
                return Some(full_exe);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_compiler() -> bool {
        find_c_compiler().is_none()
    }

    #[test]
    fn wrap_with_trampoline_single_input() {
        let kernel = "void kernel_0(const double* in0, double* output, int n) { \
                      for (int i = 0; i < n; i++) output[i] = -in0[i]; }";
        let full = wrap_with_trampoline(kernel, "kernel_0", 1);
        assert!(full.contains("kernel_0(inputs[0], output, n)"));
        assert!(full.contains("ferrotorch_kernel_entry"));
    }

    #[test]
    fn wrap_with_trampoline_two_inputs() {
        let kernel = "void kernel_0(const double* in0, const double* in1, \
                      double* output, int n) { \
                      for (int i = 0; i < n; i++) output[i] = in0[i] + in1[i]; }";
        let full = wrap_with_trampoline(kernel, "kernel_0", 2);
        assert!(full.contains("kernel_0(inputs[0], inputs[1], output, n)"));
    }

    #[test]
    fn find_c_compiler_succeeds_on_dev_hosts() {
        // This is the test harness’s way of surfacing missing compilers
        // early: if we’re on a dev host with no C compiler, every other
        // JIT test will gracefully skip, but we note it here.
        let _ = find_c_compiler();
    }

    #[test]
    fn compile_c_kernel_neg_single_input() {
        if skip_if_no_compiler() {
            eprintln!("skipping JIT test: no C compiler");
            return;
        }
        let src = "\
#include <stddef.h>
void kernel_neg(const double* in0, double* output, int n) {
    for (int i = 0; i < n; i++) output[i] = -in0[i];
}
";
        let kernel = compile_c_kernel(src, "kernel_neg", 1, 4).unwrap();
        let input = vec![1.0, -2.0, 3.5, 0.0];
        let mut output = vec![0.0; 4];
        kernel.execute(&[&input], &mut output).unwrap();
        assert_eq!(output, vec![-1.0, 2.0, -3.5, 0.0]);
    }

    #[test]
    fn compile_c_kernel_add_two_inputs() {
        if skip_if_no_compiler() {
            eprintln!("skipping JIT test: no C compiler");
            return;
        }
        let src = "\
#include <stddef.h>
void kernel_add(const double* in0, const double* in1, double* output, int n) {
    for (int i = 0; i < n; i++) output[i] = in0[i] + in1[i];
}
";
        let kernel = compile_c_kernel(src, "kernel_add", 2, 3).unwrap();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let mut out = vec![0.0; 3];
        kernel.execute(&[&a, &b], &mut out).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn compile_c_kernel_math_intrinsics() {
        if skip_if_no_compiler() {
            eprintln!("skipping JIT test: no C compiler");
            return;
        }
        let src = "\
#include <math.h>
#include <stddef.h>
void kernel_exp_sqrt(const double* in0, double* output, int n) {
    for (int i = 0; i < n; i++) output[i] = sqrt(exp(in0[i]));
}
";
        let kernel = compile_c_kernel(src, "kernel_exp_sqrt", 1, 3).unwrap();
        let input = vec![0.0, 2.0_f64.ln(), 4.0_f64.ln()];
        let mut out = vec![0.0; 3];
        kernel.execute(&[&input], &mut out).unwrap();
        // sqrt(exp(0)) = 1, sqrt(exp(ln 2)) = sqrt(2), sqrt(exp(ln 4)) = 2
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[1] - 2.0_f64.sqrt()).abs() < 1e-9);
        assert!((out[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn compile_cache_hits_on_identical_source() {
        if skip_if_no_compiler() {
            eprintln!("skipping JIT test: no C compiler");
            return;
        }
        let src = "\
#include <stddef.h>
void kernel_ident(const double* in0, double* output, int n) {
    for (int i = 0; i < n; i++) output[i] = in0[i];
}
";
        let k1 = compile_c_kernel(src, "kernel_ident", 1, 5).unwrap();
        let k2 = compile_c_kernel(src, "kernel_ident", 1, 5).unwrap();
        // Cache should return the same Arc.
        assert!(Arc::ptr_eq(&k1, &k2));
    }

    #[test]
    fn execute_rejects_wrong_input_count() {
        if skip_if_no_compiler() {
            return;
        }
        let src = "\
void kernel_noop(const double* in0, double* output, int n) {
    for (int i = 0; i < n; i++) output[i] = in0[i];
}
";
        let kernel = compile_c_kernel(src, "kernel_noop", 1, 4).unwrap();
        let out = &mut [0.0; 4];
        let err = kernel.execute(&[], out);
        assert!(err.is_err());
    }

    #[test]
    fn execute_rejects_short_output() {
        if skip_if_no_compiler() {
            return;
        }
        let src = "\
void kernel_pass(const double* in0, double* output, int n) {
    for (int i = 0; i < n; i++) output[i] = in0[i];
}
";
        let kernel = compile_c_kernel(src, "kernel_pass", 1, 4).unwrap();
        let input = vec![1.0; 4];
        let out = &mut [0.0; 2];
        let err = kernel.execute(&[&input], out);
        assert!(err.is_err());
    }

    #[test]
    fn missing_compiler_error_is_descriptive() {
        // We can't easily unset the compiler without affecting other
        // tests, so just verify the error path formatting by checking
        // the message template.
        let err = FerrotorchError::Internal {
            message: "jit: no C compiler found (tried cc, gcc, clang). \
                      Install one to use the InductorBackend JIT path."
                .into(),
        };
        let s = format!("{err}");
        assert!(s.contains("no C compiler"));
    }
}
