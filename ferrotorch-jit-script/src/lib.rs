//! `#[ferrotorch_jit_script::script]` — declarative graph construction. (#625)
//!
//! Annotate a Rust function with `#[script]` and the macro rewrites the body
//! to build an `IrGraph` instead of running eagerly. The wrapped function
//! returns a closure with the same signature that, when called, executes
//! the captured graph via `ferrotorch_jit::trace`.
//!
//! # What's supported in the body
//!
//! The script body is restricted to a small recognized subset:
//!
//! - `let x = …;` bindings (no shadowing across statements is required)
//! - Function calls on tensors: `add(&a, &b)`, `mul(&a, &b)`, `sum(&t)`,
//!   `mean(&t)`, `relu(&t)`, `sigmoid(&t)`, `tanh(&t)`, `mm(&a, &b)` ...
//!   (anything in `ferrotorch_core::grad_fns::{arithmetic, reduction,
//!   activation, linalg}` that takes `&Tensor` args and returns `Tensor`)
//! - A trailing expression as the function's return value
//!
//! Anything else (control flow, struct fields, non-tensor values) is left
//! untouched — the macro just emits the rewritten body as-is and `trace`
//! captures it at the autograd level.
//!
//! # Why this is a shim over `trace`
//!
//! The macro doesn't reimplement the IR builder. Instead, it ensures the
//! function's inputs are leaf tensors with `requires_grad=true` (so the
//! autograd graph is built), then calls the existing `trace` to capture
//! the IR. That keeps op coverage in lockstep with trace and avoids a
//! second source of truth for op-name → IR mapping.

#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
#![allow(missing_docs)] // tracked workspace-wide in the rustdoc pass

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{FnArg, ItemFn};

/// Apply `#[script]` to a function `fn(args...) -> Tensor<T>` to compile
/// it into a `ferrotorch_jit::TracedModule<T>`-returning function.
///
/// The example below is marked `ignore` because this is a `proc-macro`
/// crate: it cannot itself import `ferrotorch_jit::TracedModule` or
/// `ferrotorch_core::Tensor` at doctest-compile time (proc-macro crates
/// can only export proc-macro items and pull procedural-macro deps; they
/// cannot depend on consumer crates). The example is exercised
/// end-to-end by the integration tests in
/// `ferrotorch-jit-script/tests/script_macro.rs`.
///
/// ```ignore
/// use ferrotorch_jit_script::script;
/// use ferrotorch_core::Tensor;
/// use ferrotorch_core::grad_fns::arithmetic::{mul, add};
/// use ferrotorch_core::grad_fns::reduction::sum;
///
/// #[script]
/// fn weighted_sum(a: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32> {
///     let prod = mul(&a, &w)?;
///     sum(&prod)
/// }
///
/// // `weighted_sum(...)` now returns `FerrotorchResult<TracedModule<f32>>`
/// // built by tracing the body once with the supplied tensors.
/// ```
///
/// # Errors
///
/// Emits a `compile_error!` (via `syn::Error`) if the annotated function's
/// return type isn't one of the recognized shapes:
///
/// - `Tensor<T>`
/// - `FerrotorchResult<Tensor<T>>`
/// - `Result<Tensor<T>, _>`
///
/// Previously, an unrecognized return type silently fell back to
/// `TracedModule<f32>`, producing a wrong-dtype wrapper for e.g.
/// `Tensor<f64>` callers. The macro now refuses the input instead.
#[proc_macro_attribute]
pub fn script(attr: TokenStream, item: TokenStream) -> TokenStream {
    match script_impl(attr, item) {
        Ok(ts) => ts.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn script_impl(_attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream2> {
    let input: ItemFn = syn::parse(item)?;
    let vis = &input.vis;
    let sig = &input.sig;
    let ident = &sig.ident;
    let block = &input.block;
    let inputs = &sig.inputs;
    let output = &sig.output;

    // Walk the function's typed args once. `arg_clones` is the
    // example-input slice handed to `trace`; `arg_unpacks` are the
    // index-based bindings inside the closure body. Both projections
    // share the same filter (skip `Self` receivers) and ordering, so a
    // single pass keeps them in lockstep.
    let typed_args: Vec<&syn::PatType> = inputs
        .iter()
        .filter_map(|a| match a {
            FnArg::Typed(pt) => Some(pt),
            FnArg::Receiver(_) => None,
        })
        .collect();
    let arg_clones: Vec<TokenStream2> = typed_args
        .iter()
        .map(|pt| {
            let pat = &pt.pat;
            quote! { #pat .clone() }
        })
        .collect();
    let arg_unpacks: Vec<TokenStream2> = typed_args
        .iter()
        .enumerate()
        .map(|(i, pt)| {
            let pat = &pt.pat;
            quote! { let #pat = inputs[#i].clone(); }
        })
        .collect();

    // Determine T from the return type. Unrecognized return shapes are
    // a hard error: silently defaulting to f32 (the historic behaviour)
    // produced a `TracedModule<f32>` wrapper for callers that returned
    // e.g. `Tensor<f64>`, with no diagnostic. Failing here surfaces the
    // mistake at macro-expansion time as a clean `compile_error!`.
    let scalar_ty = match output {
        syn::ReturnType::Type(_, ty) => extract_tensor_param(ty.as_ref()).ok_or_else(|| {
            syn::Error::new_spanned(
                ty,
                "ferrotorch-jit-script: function must return Tensor<T>, \
                 FerrotorchResult<Tensor<T>>, or Result<Tensor<T>, _>",
            )
        })?,
        syn::ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                sig,
                "ferrotorch-jit-script: function must declare a return type \
                 of Tensor<T>, FerrotorchResult<Tensor<T>>, or Result<Tensor<T>, _>",
            ));
        }
    };

    // Capture the user's return type tokens so the expansion mentions
    // any names they imported (e.g. `FerrotorchResult`) — otherwise
    // those imports look unused after macro expansion.
    let user_return_ty: TokenStream2 = match output {
        syn::ReturnType::Type(_, ty) => quote! { #ty },
        syn::ReturnType::Default => {
            quote! { ::ferrotorch_core::FerrotorchResult<::ferrotorch_core::Tensor<#scalar_ty>> }
        }
    };

    // Generated wrapper:
    // - Builds the example-input slice from the caller's args.
    // - Defines an inner closure with the user's body (so all the let
    //   bindings and ops just compile as normal Rust).
    // - Calls trace(closure, &[args...]) to capture the IR graph.
    // - Wraps the graph in a TracedModule and returns it.
    let expanded = quote! {
        #vis fn #ident ( #inputs ) -> ::ferrotorch_core::FerrotorchResult<
            ::ferrotorch_jit::TracedModule<#scalar_ty>
        > {
            let __script_inputs: ::std::vec::Vec<::ferrotorch_core::Tensor<#scalar_ty>> =
                vec![ #( #arg_clones ),* ];
            let __script_inputs_for_trace: ::std::vec::Vec<::ferrotorch_core::Tensor<#scalar_ty>> =
                __script_inputs
                    .iter()
                    .map(|t| t.clone().requires_grad_(true))
                    .collect();
            let __graph = ::ferrotorch_jit::trace(
                |inputs: &[::ferrotorch_core::Tensor<#scalar_ty>]|
                    -> ::ferrotorch_core::FerrotorchResult<::ferrotorch_core::Tensor<#scalar_ty>>
                {
                    #( #arg_unpacks )*
                    let __script_result: #user_return_ty = (|| #block)();
                    __script_result
                },
                &__script_inputs_for_trace,
            )?;
            Ok(::ferrotorch_jit::TracedModule::<#scalar_ty>::new(__graph))
        }
    };
    Ok(expanded)
}

/// Recognized return-type entry point for the `#[script]` macro.
///
/// Returns the dtype `TokenStream` extracted from the annotated function's
/// return type. The recognized shapes are:
///
/// - `Tensor<T>` — yields `T`
/// - `FerrotorchResult<Tensor<T>>` — yields `T` (recurses through the
///   `FerrotorchResult` wrapper)
/// - `Result<Tensor<T>, _>` — yields `T` (recurses through the `Result`
///   wrapper, ignoring the error type)
///
/// Returns `None` when the type doesn't match any of those shapes; the
/// caller turns that into a `compile_error!` diagnostic.
///
/// Recursion through `Result` / `FerrotorchResult` wrappers is bounded
/// (see `extract_tensor_param_inner`'s depth cap) so a malformed input
/// like `Result<Result<Result<...>, _>, _>` cannot blow the macro's
/// stack — it returns `None` once the cap is reached, falling through
/// to the same `compile_error!` path as any other unrecognized shape.
fn extract_tensor_param(ty: &syn::Type) -> Option<TokenStream2> {
    extract_tensor_param_inner(ty, 0)
}

/// Maximum nesting depth for the recognized `Result<...>` /
/// `FerrotorchResult<...>` wrappers around `Tensor<T>`.
///
/// `Result<Result<FerrotorchResult<Tensor<T>>, _>, _>` is depth 3, which
/// is already pathological; anything deeper is malformed. The cap exists
/// to make the recursion total — a hand-crafted, syntactically valid but
/// nonsense input (an arbitrarily nested `Result<...>`) cannot blow the
/// macro's stack during expansion.
const MAX_RETURN_TYPE_DEPTH: u8 = 4;

fn extract_tensor_param_inner(ty: &syn::Type, depth: u8) -> Option<TokenStream2> {
    if depth > MAX_RETURN_TYPE_DEPTH {
        return None;
    }
    let syn::Type::Path(p) = ty else {
        return None;
    };
    let path = &p.path;
    let last = path.segments.last()?;
    let ident_str = last.ident.to_string();
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    if ident_str == "Tensor" {
        // First generic arg is the scalar type.
        if let Some(syn::GenericArgument::Type(t)) = args.args.first() {
            let ts = quote! { #t };
            return Some(ts);
        }
    }
    if ident_str == "FerrotorchResult" || ident_str == "Result" {
        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
            return extract_tensor_param_inner(inner, depth + 1);
        }
    }
    None
}
