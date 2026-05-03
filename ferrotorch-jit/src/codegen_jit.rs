// `unsafe` is needed exactly twice in this module: once to transmute the
// cranelift-produced code pointer into a typed `extern "C" fn`, and once to
// invoke that pointer from `JitCompiledKernel::execute`. Each `unsafe { … }`
// block carries a SAFETY comment naming the invariant it upholds. The
// `unsafe impl Send/Sync` blocks have their invariants documented immediately
// preceding the impl.
#![allow(unsafe_code)]

//! In-process JIT compilation of [`LoopIR`] via cranelift.
//!
//! The [`InductorBackend`](crate::codegen::InductorBackend) lowers fusion
//! groups to [`LoopIR`]; this module translates that IR into cranelift's
//! SSA form, JIT-compiles it into mapped executable memory pages, and
//! exposes the resulting kernel as a callable closure.
//!
//! No subprocess is forked, no shared library is written to disk, no
//! `dlopen` / `libloading` is involved. The pipeline is:
//!
//! ```text
//! LoopIR → cranelift IR → cranelift JIT → fn ptr to executable pages
//! ```
//!
//! This is the pure-Rust replacement for the previous `rustc --crate-type=cdylib`
//! shell-out + `libloading` path. It eliminates the runtime toolchain
//! dependency, the on-disk `.so` artifact, and the FFI attack surface that
//! came with loading a separately compiled library.
//!
//! # ABI
//!
//! Each kernel is compiled with the C ABI:
//!
//! ```ignore
//! extern "C" fn ferrotorch_kernel_entry(
//!     inputs: *const *const f64,
//!     output: *mut f64,
//!     n: i32,
//! )
//! ```
//!
//! The C ABI is a cranelift codegen target choice, not a foreign-language
//! dependency: the function never leaves this process and the caller is
//! Rust on both sides of the call.
//!
//! # Math intrinsics
//!
//! `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `pow` are not native cranelift
//! ops. We bind them at JIT time to the corresponding pure-Rust [`libm`]
//! implementations via cranelift's symbol resolution. `f64::exp()` and
//! friends in `std` dispatch to the same code; we just expose it under a
//! stable symbol name the JIT can call.
//!
//! # Compile cache
//!
//! Cranelift compiles in milliseconds (vs. seconds for the previous
//! rustc-shellout path), but identical fusion groups still reuse a cached
//! kernel. The cache key is the FNV-1a hash of the `LoopIR`'s `Debug`
//! representation plus the `(num_inputs, output_len)` shape pair, so it is
//! deterministic across processes and Rust releases.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, OnceLock};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, FuncRef, InstBuilder, MemFlags, Type, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};

use crate::codegen_ir::{BinOpKind, Expr, LoopIR, UnaryOpKind};

// ---------------------------------------------------------------------------
// JitCompiledKernel
// ---------------------------------------------------------------------------

/// Signature of the kernel's entry point.
///
/// `inputs` points to an array of `num_inputs` `*const f64` pointers (one
/// per kernel input buffer); `output` is the writable result buffer; `n`
/// is the element count the kernel was lowered for.
type KernelEntry = unsafe extern "C" fn(*const *const f64, *mut f64, i32);

/// A `LoopIR` kernel JIT-compiled to native code via cranelift.
///
/// Owns the [`JITModule`] that holds the executable memory pages backing
/// `kernel_fn`; dropping the kernel deallocates those pages. Cheap to clone
/// via `Arc`.
pub struct JitCompiledKernel {
    /// Cranelift JIT module holding the executable pages. Must outlive
    /// `kernel_fn`. Wrapped in `Mutex` so we can free it on drop without
    /// requiring `&mut self`.
    _module: Mutex<JITModule>,
    /// The compiled trampoline entry point, resolved once at build time.
    kernel_fn: KernelEntry,
    /// Number of input buffers the kernel expects.
    num_inputs: usize,
    /// Number of output elements the kernel writes.
    output_len: usize,
}

// `_module` is intentionally omitted from Debug — its internal state is
// implementation-defined and adds no diagnostic value beyond confirming
// the kernel was built.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for JitCompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiledKernel")
            .field("num_inputs", &self.num_inputs)
            .field("output_len", &self.output_len)
            .finish()
    }
}

// SAFETY: `JITModule` is `Send` (its internal state is owned heap-allocated
// data and a memory mapper). The `kernel_fn` is a raw function pointer to
// `Mutex`-protected executable memory; the pointer can move between threads
// because `JITModule` ensures the pages stay mapped for the kernel's
// lifetime, and the kernel itself reads only from caller-provided buffers
// (no shared mutable state).
unsafe impl Send for JitCompiledKernel {}
// SAFETY: `kernel_fn` is a raw function pointer to compiled code that reads
// from caller-provided immutable input buffers and writes to caller-provided
// disjoint output buffers. There is no shared mutable state inside the
// kernel; concurrent calls from multiple threads are race-free.
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
    /// Populates `output` with the kernel's result. Each input slice must
    /// have at least `output_len` elements (the kernel writes `output_len`
    /// outputs but may read any combination of the inputs element-by-
    /// element).
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the number of input
    /// buffers does not match `num_inputs`, if `output.len() < output_len`,
    /// or if any input buffer is shorter than `output_len`.
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
                    "JitCompiledKernel::execute: output buffer too small — \
                     need {}, got {}",
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

        // Build a contiguous array of raw input pointers — the trampoline
        // expects `*const *const f64`.
        let ptrs: Vec<*const f64> = inputs.iter().map(|b| b.as_ptr()).collect();

        // SAFETY: input/output buffer lengths were validated above; the
        // function pointer was produced by cranelift in `build_kernel`
        // with the matching `KernelEntry` ABI; the JITModule that owns the
        // executable pages is held alive by `_module` for the lifetime of
        // `self`, so the pages cannot be unmapped while we're calling.
        unsafe {
            (self.kernel_fn)(ptrs.as_ptr(), output.as_mut_ptr(), self.output_len as i32);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compile cache
// ---------------------------------------------------------------------------

fn compile_cache() -> &'static Mutex<HashMap<u64, Arc<JitCompiledKernel>>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, Arc<JitCompiledKernel>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Deterministic FNV-1a 64-bit hash of the kernel's `LoopIR` plus its shape.
///
/// We hash the `Debug` representation of the `LoopIR` slice; this is stable
/// across processes and Rust releases (the `LoopIR` types derive `Debug`
/// deterministically), so the cache key is portable.
fn hash_loops(loops: &[LoopIR], num_inputs: usize, output_len: usize) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h: u64 = FNV_OFFSET_BASIS;
    let mix = |h: &mut u64, byte: u8| {
        *h ^= u64::from(byte);
        *h = h.wrapping_mul(FNV_PRIME);
    };

    let mut buf = String::new();
    let _ = write!(buf, "{loops:?}");
    for &b in buf.as_bytes() {
        mix(&mut h, b);
    }
    for &b in &num_inputs.to_le_bytes() {
        mix(&mut h, b);
    }
    for &b in &output_len.to_le_bytes() {
        mix(&mut h, b);
    }
    h
}

// ---------------------------------------------------------------------------
// Public compile entry point
// ---------------------------------------------------------------------------

/// JIT-compile a [`LoopIR`] kernel into in-process executable memory via
/// cranelift.
///
/// `loops` is the body of the kernel; the trampoline that fans out the raw
/// `*const *const f64` input array into per-buffer slots is generated
/// automatically based on `num_inputs`. `output_len` is the element count
/// of the result buffer (used for input-shape validation in `execute`).
///
/// # Errors
///
/// Returns [`FerrotorchError::Internal`] if cranelift initialisation,
/// codegen, or finalisation fails. Returns [`FerrotorchError::InvalidArgument`]
/// if the `LoopIR` contains constructs the JIT path doesn't support — see
/// [`jit_supports`] for the predicate. Callers (notably
/// [`crate::codegen::InductorBackend`]) should fall back to the interpreter
/// when this returns the unsupported-IR variant.
pub fn compile_loop_ir_kernel(
    loops: &[LoopIR],
    num_inputs: usize,
    output_len: usize,
) -> FerrotorchResult<Arc<JitCompiledKernel>> {
    if !jit_supports(loops) {
        return Err(FerrotorchError::InvalidArgument {
            message: "ferrotorch-jit: LoopIR contains constructs not yet supported by the cranelift JIT (If, Mod, or unknown FnCall); fall back to the interpreter"
                .into(),
        });
    }

    let key = hash_loops(loops, num_inputs, output_len);

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

    let kernel = build_kernel(loops, num_inputs, output_len)?;
    let arc = Arc::new(kernel);

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

// ---------------------------------------------------------------------------
// JIT-supported predicate
// ---------------------------------------------------------------------------

/// Whether the cranelift JIT codegen handles every construct in `loops`.
///
/// The current codegen covers what `find_fusion_groups` actually emits for
/// elementwise + reduction fusion: loops, scalar arithmetic, indexed
/// loads/stores, and the documented unary intrinsic set. Anything outside
/// that subset (`If`, modulus, unknown function calls) is rejected here so
/// the orchestrator can fall back to the interpreter without the codegen
/// ever encountering an unhandled variant.
pub fn jit_supports(loops: &[LoopIR]) -> bool {
    loops.iter().all(jit_supports_stmt)
}

fn jit_supports_stmt(stmt: &LoopIR) -> bool {
    match stmt {
        LoopIR::Loop {
            start, end, body, ..
        } => jit_supports_expr(start) && jit_supports_expr(end) && jit_supports(body),
        LoopIR::Store { index, value, .. } => jit_supports_expr(index) && jit_supports_expr(value),
        LoopIR::Let { value, .. }
        | LoopIR::Assign { value, .. }
        | LoopIR::Accumulate { value, .. } => jit_supports_expr(value),
        // Conditionals require multi-block IR with bool plumbing the LoopIR
        // doesn't expose (no Cmp variant); not yet wired.
        LoopIR::If { .. } => false,
        LoopIR::Comment(_) => true,
    }
}

fn jit_supports_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) | Expr::Const(_) | Expr::IntConst(_) => true,
        Expr::BinOp { op, lhs, rhs } => {
            !matches!(op, BinOpKind::Mod) && jit_supports_expr(lhs) && jit_supports_expr(rhs)
        }
        // Single-child recursive cases share the same body — collapse via
        // an or-pattern that re-binds each child field to a common name.
        Expr::UnaryOp { operand: child, .. }
        | Expr::Index { index: child, .. }
        | Expr::Cast { operand: child, .. } => jit_supports_expr(child),
        // Only `powf` is currently routed to a libm intrinsic.
        Expr::FnCall { name, args } => name == "powf" && args.iter().all(jit_supports_expr),
    }
}

// ---------------------------------------------------------------------------
// Cranelift codegen
// ---------------------------------------------------------------------------

/// Stable symbol names that the JIT module resolves to `libm` functions.
mod sym {
    pub const EXP: &str = "ferrotorch_libm_exp";
    pub const LOG: &str = "ferrotorch_libm_log";
    pub const SQRT: &str = "ferrotorch_libm_sqrt";
    pub const SIN: &str = "ferrotorch_libm_sin";
    pub const COS: &str = "ferrotorch_libm_cos";
    pub const TANH: &str = "ferrotorch_libm_tanh";
    pub const POW: &str = "ferrotorch_libm_pow";
}

fn register_libm_symbols(builder: &mut JITBuilder) {
    builder.symbol(sym::EXP, libm::exp as *const u8);
    builder.symbol(sym::LOG, libm::log as *const u8);
    builder.symbol(sym::SQRT, libm::sqrt as *const u8);
    builder.symbol(sym::SIN, libm::sin as *const u8);
    builder.symbol(sym::COS, libm::cos as *const u8);
    builder.symbol(sym::TANH, libm::tanh as *const u8);
    builder.symbol(sym::POW, libm::pow as *const u8);
}

#[derive(Clone, Copy)]
struct Intrinsics {
    exp: FuncRef,
    log: FuncRef,
    sqrt: FuncRef,
    tanh: FuncRef,
    pow: FuncRef,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ValueKind {
    Float,
    Int,
}

fn build_kernel(
    loops: &[LoopIR],
    num_inputs: usize,
    output_len: usize,
) -> FerrotorchResult<JitCompiledKernel> {
    // Default ISA flags. cranelift_native picks the host triple/CPU; we use
    // the defaults from `cranelift_codegen::settings::builder()` because
    // they're enough for an in-process JIT and don't require pulling in
    // cranelift_native.
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: cranelift settings: {e}"),
        })?;
    // cranelift-jit requires non-PIC: the JITed code lives in mapped pages
    // we own and patches relocations directly, so position-independent
    // codegen is both unnecessary and rejected by `cranelift-jit`'s backend
    // assertions.
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: cranelift settings: {e}"),
        })?;
    let isa_builder = cranelift_native::builder().map_err(|e| FerrotorchError::Internal {
        message: format!("jit: cranelift_native::builder: {e}"),
    })?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: ISA finish: {e}"),
        })?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_libm_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);

    let ptr_ty = module.target_config().pointer_type();

    // Kernel signature: extern "C" fn(*const *const f64, *mut f64, i32).
    let mut kernel_sig = module.make_signature();
    kernel_sig.params.push(AbiParam::new(ptr_ty));
    kernel_sig.params.push(AbiParam::new(ptr_ty));
    kernel_sig.params.push(AbiParam::new(types::I32));

    // libm intrinsic signatures.
    let mut sig1 = module.make_signature();
    sig1.params.push(AbiParam::new(types::F64));
    sig1.returns.push(AbiParam::new(types::F64));
    let mut sig2 = module.make_signature();
    sig2.params.push(AbiParam::new(types::F64));
    sig2.params.push(AbiParam::new(types::F64));
    sig2.returns.push(AbiParam::new(types::F64));

    let kernel_id = module
        .declare_function("ferrotorch_kernel_entry", Linkage::Export, &kernel_sig)
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: declare kernel: {e}"),
        })?;

    let exp_id = module
        .declare_function(sym::EXP, Linkage::Import, &sig1)
        .map_err(internal)?;
    let log_id = module
        .declare_function(sym::LOG, Linkage::Import, &sig1)
        .map_err(internal)?;
    let sqrt_id = module
        .declare_function(sym::SQRT, Linkage::Import, &sig1)
        .map_err(internal)?;
    let _sin_id = module
        .declare_function(sym::SIN, Linkage::Import, &sig1)
        .map_err(internal)?;
    let _cos_id = module
        .declare_function(sym::COS, Linkage::Import, &sig1)
        .map_err(internal)?;
    let tanh_id = module
        .declare_function(sym::TANH, Linkage::Import, &sig1)
        .map_err(internal)?;
    let pow_id = module
        .declare_function(sym::POW, Linkage::Import, &sig2)
        .map_err(internal)?;

    let mut ctx = module.make_context();
    ctx.func.signature = kernel_sig;

    let intrinsics = Intrinsics {
        exp: module.declare_func_in_func(exp_id, &mut ctx.func),
        log: module.declare_func_in_func(log_id, &mut ctx.func),
        sqrt: module.declare_func_in_func(sqrt_id, &mut ctx.func),
        tanh: module.declare_func_in_func(tanh_id, &mut ctx.func),
        pow: module.declare_func_in_func(pow_id, &mut ctx.func),
    };

    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let inputs_ptr = builder.block_params(entry)[0];
        let output_ptr = builder.block_params(entry)[1];
        // n is i32 from the C ABI; widen to i64 for index arithmetic.
        let n_i32 = builder.block_params(entry)[2];
        let _n_i64 = builder.ins().sextend(types::I64, n_i32);

        let mut emitter = Emitter {
            builder: &mut builder,
            inputs_ptr,
            output_ptr,
            ptr_ty,
            var_map: HashMap::new(),
            intrinsics,
        };
        emitter.emit_stmts(loops);

        builder.ins().return_(&[]);
        builder.seal_block(entry);
        builder.finalize();
    }

    module
        .define_function(kernel_id, &mut ctx)
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: define kernel: {e}"),
        })?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| FerrotorchError::Internal {
            message: format!("jit: finalize: {e}"),
        })?;

    let code_ptr = module.get_finalized_function(kernel_id);
    // SAFETY: cranelift just emitted a function with the C ABI signature
    // `extern "C" fn(*const *const f64, *mut f64, i32)`. The KernelEntry
    // type matches this signature exactly. The pointer is valid for the
    // lifetime of `module`, which JitCompiledKernel owns below.
    let kernel_fn: KernelEntry = unsafe { std::mem::transmute::<*const u8, KernelEntry>(code_ptr) };

    Ok(JitCompiledKernel {
        _module: Mutex::new(module),
        kernel_fn,
        num_inputs,
        output_len,
    })
}

// Owned by-value to be usable as `.map_err(internal)`. The cost of owning a
// `ModuleError` per call is trivial (it's a small enum); a `&Error` signature
// would force every call site into `|e| internal(&e)`.
#[allow(clippy::needless_pass_by_value)]
fn internal(e: cranelift_module::ModuleError) -> FerrotorchError {
    FerrotorchError::Internal {
        message: format!("jit: cranelift module: {e}"),
    }
}

// ---------------------------------------------------------------------------
// LoopIR → cranelift IR emitter
// ---------------------------------------------------------------------------

struct Emitter<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    inputs_ptr: Value,
    output_ptr: Value,
    ptr_ty: Type,
    var_map: HashMap<String, (Variable, ValueKind)>,
    intrinsics: Intrinsics,
}

impl Emitter<'_, '_> {
    fn declare_var(&mut self, name: &str, kind: ValueKind) -> Variable {
        let cl_ty = match kind {
            ValueKind::Float => types::F64,
            ValueKind::Int => types::I64,
        };
        let var = self.builder.declare_var(cl_ty);
        self.var_map.insert(name.to_string(), (var, kind));
        var
    }

    fn lookup_var(&self, name: &str) -> (Variable, ValueKind) {
        *self
            .var_map
            .get(name)
            .unwrap_or_else(|| panic!("ferrotorch-jit: undefined variable `{name}` in LoopIR"))
    }

    fn emit_stmts(&mut self, stmts: &[LoopIR]) {
        for s in stmts {
            self.emit_stmt(s);
        }
    }

    fn emit_stmt(&mut self, stmt: &LoopIR) {
        match stmt {
            LoopIR::Loop {
                var,
                start,
                end,
                body,
            } => self.emit_loop(var, start, end, body),
            LoopIR::Store {
                buffer,
                index,
                value,
            } => {
                let idx = self.emit_int_expr(index);
                let val = self.emit_float_expr(value);
                let addr = self.buffer_addr(buffer, idx);
                self.builder.ins().store(MemFlags::trusted(), val, addr, 0);
            }
            LoopIR::Let { var, value } => {
                // Float by default — `Let` in elementwise fusion always
                // initialises a float accumulator (`acc = 0.0`) or temporary.
                let v = self.emit_float_expr(value);
                let cl_var = self.declare_var(var, ValueKind::Float);
                self.builder.def_var(cl_var, v);
            }
            LoopIR::Assign { var, value } => {
                let (cl_var, kind) = self.lookup_var(var);
                let v = match kind {
                    ValueKind::Float => self.emit_float_expr(value),
                    ValueKind::Int => self.emit_int_expr(value),
                };
                self.builder.def_var(cl_var, v);
            }
            LoopIR::Accumulate { var, value } => {
                let (cl_var, kind) = self.lookup_var(var);
                let new_v = match kind {
                    ValueKind::Float => self.emit_float_expr(value),
                    ValueKind::Int => self.emit_int_expr(value),
                };
                let old = self.builder.use_var(cl_var);
                let sum = match kind {
                    ValueKind::Float => self.builder.ins().fadd(old, new_v),
                    ValueKind::Int => self.builder.ins().iadd(old, new_v),
                };
                self.builder.def_var(cl_var, sum);
            }
            LoopIR::If { .. } => {
                // Filtered out by `jit_supports`; reaching here is an
                // invariant violation rather than a user error.
                unreachable!(
                    "If reached cranelift codegen — `jit_supports` should have rejected it"
                );
            }
            LoopIR::Comment(_) => {}
        }
    }

    fn emit_loop(&mut self, var: &str, start: &Expr, end: &Expr, body: &[LoopIR]) {
        let header = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit = self.builder.create_block();

        let loop_var = self.declare_var(var, ValueKind::Int);
        let start_v = self.emit_int_expr(start);
        self.builder.def_var(loop_var, start_v);
        self.builder.ins().jump(header, &[]);

        self.builder.switch_to_block(header);
        let i = self.builder.use_var(loop_var);
        let end_v = self.emit_int_expr(end);
        let cont = self.builder.ins().icmp(IntCC::SignedLessThan, i, end_v);
        self.builder.ins().brif(cont, body_block, &[], exit, &[]);

        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);
        self.emit_stmts(body);
        let i = self.builder.use_var(loop_var);
        let one = self.builder.ins().iconst(types::I64, 1);
        let next = self.builder.ins().iadd(i, one);
        self.builder.def_var(loop_var, next);
        self.builder.ins().jump(header, &[]);

        self.builder.seal_block(header);
        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
    }

    fn emit_float_expr(&mut self, expr: &Expr) -> Value {
        match expr {
            Expr::Const(v) => self.builder.ins().f64const(*v),
            Expr::IntConst(v) => {
                let i = self.builder.ins().iconst(types::I64, *v);
                self.builder.ins().fcvt_from_sint(types::F64, i)
            }
            Expr::Var(name) => {
                let (cl_var, kind) = self.lookup_var(name);
                let v = self.builder.use_var(cl_var);
                match kind {
                    ValueKind::Float => v,
                    ValueKind::Int => self.builder.ins().fcvt_from_sint(types::F64, v),
                }
            }
            Expr::Index { buffer, index } => {
                let idx = self.emit_int_expr(index);
                let addr = self.buffer_addr(buffer, idx);
                self.builder
                    .ins()
                    .load(types::F64, MemFlags::trusted(), addr, 0)
            }
            Expr::BinOp { op, lhs, rhs } => {
                let l = self.emit_float_expr(lhs);
                let r = self.emit_float_expr(rhs);
                match op {
                    BinOpKind::Add => self.builder.ins().fadd(l, r),
                    BinOpKind::Sub => self.builder.ins().fsub(l, r),
                    BinOpKind::Mul => self.builder.ins().fmul(l, r),
                    BinOpKind::Div => self.builder.ins().fdiv(l, r),
                    BinOpKind::Mod => unreachable!("filtered by jit_supports"),
                }
            }
            Expr::UnaryOp { op, operand } => self.emit_unary(*op, operand),
            Expr::FnCall { name, args } => match name.as_str() {
                "powf" => {
                    debug_assert_eq!(args.len(), 2);
                    let a = self.emit_float_expr(&args[0]);
                    let b = self.emit_float_expr(&args[1]);
                    let pow_ref = self.intrinsics.pow;
                    self.call_fn(pow_ref, &[a, b])
                }
                other => unreachable!("filtered by jit_supports: {other}"),
            },
            Expr::Cast {
                target_type,
                operand,
            } => match target_type.as_str() {
                "f64" | "double" | "float" => self.emit_float_expr(operand),
                _ => {
                    let v = self.emit_int_expr(operand);
                    self.builder.ins().fcvt_from_sint(types::F64, v)
                }
            },
        }
    }

    fn emit_unary(&mut self, op: UnaryOpKind, operand: &Expr) -> Value {
        let v = self.emit_float_expr(operand);
        match op {
            UnaryOpKind::Neg => self.builder.ins().fneg(v),
            UnaryOpKind::Abs => self.builder.ins().fabs(v),
            UnaryOpKind::Sqrt => {
                let r = self.intrinsics.sqrt;
                self.call_fn(r, &[v])
            }
            UnaryOpKind::Exp => {
                let r = self.intrinsics.exp;
                self.call_fn(r, &[v])
            }
            UnaryOpKind::Log => {
                let r = self.intrinsics.log;
                self.call_fn(r, &[v])
            }
            UnaryOpKind::Tanh => {
                let r = self.intrinsics.tanh;
                self.call_fn(r, &[v])
            }
            UnaryOpKind::Sigmoid => {
                // 1 / (1 + exp(-v))
                let neg = self.builder.ins().fneg(v);
                let exp_neg = {
                    let r = self.intrinsics.exp;
                    self.call_fn(r, &[neg])
                };
                let one = self.builder.ins().f64const(1.0);
                let denom = self.builder.ins().fadd(one, exp_neg);
                self.builder.ins().fdiv(one, denom)
            }
            UnaryOpKind::Relu => {
                // max(v, 0)
                let zero = self.builder.ins().f64const(0.0);
                self.builder.ins().fmax(v, zero)
            }
            UnaryOpKind::Gelu => {
                // x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
                let x = v;
                let half = self.builder.ins().f64const(0.5);
                let one = self.builder.ins().f64const(1.0);
                let c1 = self.builder.ins().f64const(0.7978845608_f64);
                let c2 = self.builder.ins().f64const(0.044715_f64);
                let x2 = self.builder.ins().fmul(x, x);
                let x3 = self.builder.ins().fmul(x2, x);
                let c2x3 = self.builder.ins().fmul(c2, x3);
                let inner_sum = self.builder.ins().fadd(x, c2x3);
                let inner = self.builder.ins().fmul(c1, inner_sum);
                let t = {
                    let r = self.intrinsics.tanh;
                    self.call_fn(r, &[inner])
                };
                let one_p_t = self.builder.ins().fadd(one, t);
                let half_x = self.builder.ins().fmul(x, half);
                self.builder.ins().fmul(half_x, one_p_t)
            }
            UnaryOpKind::Silu => {
                // v / (1 + exp(-v))
                let neg = self.builder.ins().fneg(v);
                let exp_neg = {
                    let r = self.intrinsics.exp;
                    self.call_fn(r, &[neg])
                };
                let one = self.builder.ins().f64const(1.0);
                let denom = self.builder.ins().fadd(one, exp_neg);
                self.builder.ins().fdiv(v, denom)
            }
        }
    }

    fn emit_int_expr(&mut self, expr: &Expr) -> Value {
        match expr {
            Expr::IntConst(v) => self.builder.ins().iconst(types::I64, *v),
            Expr::Const(v) => self.builder.ins().iconst(types::I64, *v as i64),
            Expr::Var(name) => {
                let (cl_var, kind) = self.lookup_var(name);
                let v = self.builder.use_var(cl_var);
                match kind {
                    ValueKind::Int => v,
                    ValueKind::Float => self.builder.ins().fcvt_to_sint(types::I64, v),
                }
            }
            Expr::BinOp { op, lhs, rhs } => {
                let l = self.emit_int_expr(lhs);
                let r = self.emit_int_expr(rhs);
                match op {
                    BinOpKind::Add => self.builder.ins().iadd(l, r),
                    BinOpKind::Sub => self.builder.ins().isub(l, r),
                    BinOpKind::Mul => self.builder.ins().imul(l, r),
                    BinOpKind::Div => self.builder.ins().sdiv(l, r),
                    BinOpKind::Mod => unreachable!("filtered by jit_supports"),
                }
            }
            Expr::Cast { operand, .. } => self.emit_int_expr(operand),
            Expr::Index { .. } | Expr::UnaryOp { .. } | Expr::FnCall { .. } => {
                unreachable!("non-integer expression in integer context")
            }
        }
    }

    /// Resolve the base pointer of a buffer name to a cranelift Value.
    ///
    /// `out` / `output` → the kernel's output pointer.
    /// `inN` (for some integer N) → `inputs[N]`, loaded from the
    /// `*const *const f64` array.
    fn buffer_base(&mut self, name: &str) -> Value {
        if name == "out" || name == "output" {
            return self.output_ptr;
        }
        if let Some(suffix) = name.strip_prefix("in") {
            if let Ok(idx) = suffix.parse::<usize>() {
                let elem_size = std::mem::size_of::<*const f64>() as i32;
                let off = (idx as i32)
                    .checked_mul(elem_size)
                    .expect("input index byte offset overflow");
                return self.builder.ins().load(
                    self.ptr_ty,
                    MemFlags::trusted(),
                    self.inputs_ptr,
                    off,
                );
            }
        }
        panic!("ferrotorch-jit: unknown buffer name `{name}`");
    }

    fn buffer_addr(&mut self, buffer: &str, index: Value) -> Value {
        let base = self.buffer_base(buffer);
        let elem_size = self.builder.ins().iconst(types::I64, 8);
        let byte_off = self.builder.ins().imul(index, elem_size);
        self.builder.ins().iadd(base, byte_off)
    }

    fn call_fn(&mut self, fr: FuncRef, args: &[Value]) -> Value {
        let call = self.builder.ins().call(fr, args);
        self.builder.inst_results(call)[0]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_ir::{self, BinOpKind, Expr, LoopIR};
    use crate::graph::IrOpKind;

    fn elementwise_loops(ops: &[IrOpKind], inputs: &[&str], n: usize) -> Vec<LoopIR> {
        codegen_ir::lower_to_loops(ops, inputs, "out", n)
    }

    #[test]
    fn jit_supports_elementwise_loops() {
        let loops = elementwise_loops(&[IrOpKind::Neg], &["in0"], 4);
        assert!(jit_supports(&loops));
    }

    #[test]
    fn jit_supports_rejects_if_statement() {
        let loops = vec![LoopIR::If {
            condition: Expr::var("c"),
            then_body: vec![],
            else_body: vec![],
        }];
        assert!(!jit_supports(&loops));
    }

    #[test]
    fn jit_supports_rejects_modulus() {
        let loops = vec![LoopIR::Let {
            var: "x".into(),
            value: Expr::bin(BinOpKind::Mod, Expr::int(10), Expr::int(3)),
        }];
        assert!(!jit_supports(&loops));
    }

    #[test]
    fn compile_loop_ir_kernel_neg_single_input() {
        let loops = elementwise_loops(&[IrOpKind::Neg], &["in0"], 4);
        let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
        let input = vec![1.0_f64, -2.0, 3.5, 0.0];
        let mut output = vec![0.0; 4];
        kernel.execute(&[&input], &mut output).unwrap();
        assert_eq!(output, vec![-1.0, 2.0, -3.5, 0.0]);
    }

    #[test]
    fn compile_loop_ir_kernel_add_two_inputs() {
        let loops = elementwise_loops(&[IrOpKind::Add], &["in0", "in1"], 3);
        let kernel = compile_loop_ir_kernel(&loops, 2, 3).unwrap();
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![10.0_f64, 20.0, 30.0];
        let mut out = vec![0.0; 3];
        kernel.execute(&[&a, &b], &mut out).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn compile_loop_ir_kernel_relu() {
        let loops = elementwise_loops(&[IrOpKind::Relu], &["in0"], 4);
        let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
        let input = vec![-1.0_f64, 0.0, 1.0, 2.5];
        let mut out = vec![0.0; 4];
        kernel.execute(&[&input], &mut out).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.5]);
    }

    #[test]
    fn compile_loop_ir_kernel_sqrt_exp_chain() {
        let loops = elementwise_loops(&[IrOpKind::Exp, IrOpKind::Sqrt], &["in0"], 3);
        let kernel = compile_loop_ir_kernel(&loops, 1, 3).unwrap();
        let input = vec![0.0_f64, 2.0_f64.ln(), 4.0_f64.ln()];
        let mut out = vec![0.0; 3];
        kernel.execute(&[&input], &mut out).unwrap();
        // sqrt(exp(0)) = 1; sqrt(exp(ln 2)) = sqrt(2); sqrt(exp(ln 4)) = 2.
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[1] - 2.0_f64.sqrt()).abs() < 1e-9);
        assert!((out[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn compile_cache_hits_on_identical_loops() {
        let loops = elementwise_loops(&[IrOpKind::Neg], &["in0"], 5);
        let k1 = compile_loop_ir_kernel(&loops, 1, 5).unwrap();
        let k2 = compile_loop_ir_kernel(&loops, 1, 5).unwrap();
        assert!(Arc::ptr_eq(&k1, &k2));
    }

    #[test]
    fn execute_rejects_wrong_input_count() {
        let loops = elementwise_loops(&[IrOpKind::Neg], &["in0"], 4);
        let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
        let mut out = [0.0; 4];
        let err = kernel.execute(&[], &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn execute_rejects_short_output() {
        let loops = elementwise_loops(&[IrOpKind::Neg], &["in0"], 4);
        let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        let mut tiny = [0.0; 2];
        let err = kernel.execute(&[&input], &mut tiny);
        assert!(err.is_err());
    }

    #[test]
    fn unsupported_loop_ir_returns_err() {
        let loops = vec![LoopIR::If {
            condition: Expr::var("c"),
            then_body: vec![],
            else_body: vec![],
        }];
        let err = compile_loop_ir_kernel(&loops, 1, 1).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }
}
