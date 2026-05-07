//! Build script for `ferrotorch-gpu`.
//!
//! Currently has a single responsibility: when the opt-in `cusparselt`
//! feature is enabled, locate `cusparseLt.h` on the host, run `bindgen`
//! to emit `cusparselt_sys.rs` into `OUT_DIR`, and instruct cargo to
//! link against `libcusparseLt.so`.
//!
//! When the feature is **off**, this script is a no-op — the default
//! workspace build does not require libclang or the cuSPARSELt SDK.
//!
//! Probe order for the cuSPARSELt SDK header:
//!   1. `$CUSPARSELT_INCLUDE_DIR/cusparseLt.h`
//!   2. `$CUDA_PATH/include/cusparseLt.h`
//!   3. `/usr/local/cuda/include/cusparseLt.h`
//!   4. `/usr/local/cuda-12.9/include/cusparseLt.h`
//!   5. `/usr/local/cuda-12.8/include/cusparseLt.h`
//!   6. `/usr/include/cusparseLt.h`
//!   7. `/opt/nvidia/cusparselt/include/cusparseLt.h`
//!
//! NVIDIA distributes cuSPARSELt as a separate SDK from the CUDA
//! toolkit (it ships in its own tarball / RPM); on systems without it
//! installed the build script emits a `cargo::warning=` and aborts so
//! the user sees a clear path to fix.

fn main() {
    // The script runs unconditionally — but every action below is gated
    // on `CARGO_FEATURE_CUSPARSELT`, which cargo sets only when the
    // `cusparselt` feature is active. Re-run if that gate flips or any
    // probed env var changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUSPARSELT");
    println!("cargo:rerun-if-env-changed=CUSPARSELT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUSPARSELT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if std::env::var_os("CARGO_FEATURE_CUSPARSELT").is_some() {
        #[cfg(feature = "cusparselt")]
        cusparselt::generate();
    }
}

#[cfg(feature = "cusparselt")]
mod cusparselt {
    use std::path::{Path, PathBuf};

    /// Header probe + bindgen run + link directives.
    pub fn generate() {
        let header = match locate_header() {
            Some(p) => p,
            None => {
                println!(
                    "cargo:warning=cusparselt feature is enabled but `cusparseLt.h` was not found on this host. Set CUSPARSELT_INCLUDE_DIR to the directory containing cusparseLt.h, or install the NVIDIA cuSPARSELt SDK (https://docs.nvidia.com/cuda/cusparselt/getting_started.html). Searched: $CUSPARSELT_INCLUDE_DIR, $CUDA_PATH/include, /usr/local/cuda/include, /usr/local/cuda-12.*/include, /usr/include, /opt/nvidia/cusparselt/include."
                );
                panic!(
                    "ferrotorch-gpu: cusparselt feature requires cusparseLt.h but none of the probed locations contained it. See cargo:warning above for resolution."
                );
            }
        };

        // Tell rustc to link against `libcusparseLt.so`. The library
        // search path defaults to the system loader path; the user can
        // extend it via CUSPARSELT_LIB_DIR for non-default install
        // prefixes (e.g. /opt/nvidia/cusparselt/lib64).
        if let Ok(dir) = std::env::var("CUSPARSELT_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
        // Common implicit search paths so `LD_LIBRARY_PATH` is not the
        // only way to find the lib at runtime.
        for candidate in [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12.9/lib64",
            "/usr/local/cuda-12.8/lib64",
            "/usr/lib64",
            "/opt/nvidia/cusparselt/lib64",
        ] {
            if Path::new(candidate).exists() {
                println!("cargo:rustc-link-search=native={candidate}");
            }
        }
        println!("cargo:rustc-link-lib=cusparseLt");

        // Re-run if the located header changes.
        println!("cargo:rerun-if-changed={}", header.display());

        let header_str = header.to_string_lossy().to_string();
        let mut builder = bindgen::Builder::default()
            .header(header_str.clone())
            .allowlist_function("cusparseLt.*")
            .allowlist_type("cusparseLt.*")
            .allowlist_var("CUSPARSELT_.*")
            .allowlist_var("CUSPARSE_.*")
            .allowlist_type("cudaDataType.*")
            .allowlist_type("cudaStream_t")
            .allowlist_type("cusparseStatus_t")
            .allowlist_type("cusparseOperation_t")
            .allowlist_type("cusparseComputeType.*")
            .allowlist_type("cusparseOrder_t")
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .derive_default(true)
            .derive_debug(true)
            .layout_tests(false)
            .generate_comments(false);

        // Add include path containing the header so bindgen finds the
        // CUDA toolkit headers it transitively depends on.
        if let Some(parent) = header.parent() {
            builder = builder.clang_arg(format!("-I{}", parent.display()));
        }
        for path in cuda_include_dirs() {
            builder = builder.clang_arg(format!("-I{}", path.display()));
        }

        let bindings = builder
            .generate()
            .expect("bindgen failed to generate cusparseLt bindings");

        let out_path = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"))
            .join("cusparselt_sys.rs");
        bindings
            .write_to_file(&out_path)
            .expect("failed to write cusparselt_sys.rs");
    }

    fn locate_header() -> Option<PathBuf> {
        let candidates: Vec<PathBuf> = [
            std::env::var_os("CUSPARSELT_INCLUDE_DIR").map(PathBuf::from),
            std::env::var_os("CUDA_PATH").map(|p| PathBuf::from(p).join("include")),
            Some(PathBuf::from("/usr/local/cuda/include")),
            Some(PathBuf::from("/usr/local/cuda-12.9/include")),
            Some(PathBuf::from("/usr/local/cuda-12.8/include")),
            Some(PathBuf::from("/usr/include")),
            Some(PathBuf::from("/opt/nvidia/cusparselt/include")),
        ]
        .into_iter()
        .flatten()
        .map(|d| d.join("cusparseLt.h"))
        .collect();
        candidates.into_iter().find(|p| p.exists())
    }

    fn cuda_include_dirs() -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Some(p) = std::env::var_os("CUDA_PATH") {
            out.push(PathBuf::from(p).join("include"));
        }
        for c in [
            "/usr/local/cuda/include",
            "/usr/local/cuda-12.9/include",
            "/usr/local/cuda-12.8/include",
            "/usr/include",
        ] {
            let p = PathBuf::from(c);
            if p.exists() {
                out.push(p);
            }
        }
        out
    }
}
