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

use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, parse_macro_input};

/// Apply `#[script]` to a function `fn(args...) -> Tensor<T>` to compile
/// it into a `ferrotorch_jit::TracedModule<T>`-returning function.
///
/// Example:
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
#[proc_macro_attribute]
pub fn script(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let vis = &input.vis;
    let sig = &input.sig;
    let ident = &sig.ident;
    let block = &input.block;
    let inputs = &sig.inputs;
    let output = &sig.output;

    // Collect simple `name: Tensor<T>` arg idents so we can pass them
    // to `trace` as the example-input slice.
    let mut arg_idents: Vec<proc_macro2::TokenStream> = Vec::new();
    for arg in inputs {
        if let FnArg::Typed(pat_ty) = arg {
            let pat = &pat_ty.pat;
            arg_idents.push(quote! { #pat .clone() });
        }
    }

    let body_fn_inputs = inputs.clone();

    // Generated wrapper:
    // - Defines an inner closure with the user's body (so all the let
    //   bindings and ops just compile as normal Rust).
    // - Calls trace(closure, &[args...]) to capture the IR graph.
    // - Wraps the graph in a TracedModule and returns it.
    //
    // The closure takes `&[Tensor<T>]` (matching trace's signature) and
    // unpacks them by index into the same arg names the user wrote.
    let arg_names: Vec<&syn::Pat> = inputs
        .iter()
        .filter_map(|a| match a {
            FnArg::Typed(pt) => Some(&*pt.pat),
            _ => None,
        })
        .collect();
    let arg_unpacks: Vec<proc_macro2::TokenStream> = arg_names
        .iter()
        .enumerate()
        .map(|(i, name)| quote! { let #name = inputs[#i].clone(); })
        .collect();

    // Determine T from the return type Tensor<T>: walk the syn Type.
    // For simplicity we pin T = f32 in the generated code unless the
    // return is Tensor<f64>. Most callers use f32; f64 falls back to
    // explicit annotation.
    let scalar_ty = match output {
        syn::ReturnType::Type(_, ty) => extract_tensor_param(ty).unwrap_or_else(|| quote! { f32 }),
        _ => quote! { f32 },
    };

    // Capture the user's return type tokens so the expansion mentions
    // any names they imported (e.g. `FerrotorchResult`) — otherwise
    // those imports look unused after macro expansion.
    let user_return_ty: proc_macro2::TokenStream = match output {
        syn::ReturnType::Type(_, ty) => quote! { #ty },
        _ => quote! { ::ferrotorch_core::FerrotorchResult<::ferrotorch_core::Tensor<#scalar_ty>> },
    };

    let expanded = quote! {
        #vis fn #ident ( #body_fn_inputs ) -> ::ferrotorch_core::FerrotorchResult<
            ::ferrotorch_jit::TracedModule<#scalar_ty>
        > {
            let __script_inputs: ::std::vec::Vec<::ferrotorch_core::Tensor<#scalar_ty>> =
                vec![ #( #arg_idents ),* ];
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
    expanded.into()
}

/// Extract the `T` from `Tensor<T>` (or `FerrotorchResult<Tensor<T>>`,
/// or `Result<Tensor<T>, _>`). Returns `None` if the shape doesn't match.
fn extract_tensor_param(ty: &syn::Type) -> Option<proc_macro2::TokenStream> {
    let path = if let syn::Type::Path(p) = ty {
        &p.path
    } else {
        return None;
    };
    let last = path.segments.last()?;
    let ident_str = last.ident.to_string();
    let args = match &last.arguments {
        syn::PathArguments::AngleBracketed(a) => a,
        _ => return None,
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
            return extract_tensor_param(inner);
        }
    }
    None
}
