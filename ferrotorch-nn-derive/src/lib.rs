//! Derive macro for the `Module<T>` trait in `ferrotorch-nn`.
//!
//! Generates the boilerplate methods (`parameters`, `parameters_mut`,
//! `named_parameters`, `train`, `eval`, `is_training`) so the user only
//! needs to write `forward()`.
//!
//! # Field attributes
//!
//! | Attribute       | Meaning                                                |
//! |-----------------|--------------------------------------------------------|
//! | `#[param]`      | This field is a `Parameter<T>` — registered directly.  |
//! | `#[submodule]`  | This field implements `Module<T>` — recurse into it.   |
//! | `#[skip]`       | Ignore this field entirely.                            |
//! | *(none)*        | Ignored (same as `#[skip]`), except for `training: bool` which is managed automatically. |
//!
//! The struct **must** contain a `training: bool` field. The derive will
//! generate `train()`, `eval()`, and `is_training()` using it, and will
//! propagate train/eval to all `#[submodule]` fields.
//!
//! # Example
//!
//! The example below is marked `ignore` because this is a `proc-macro` crate:
//! it cannot itself import `ferrotorch_nn::Module` or `ferrotorch_core::Tensor`
//! at doctest-compile time (proc-macro crates can only export proc-macro items
//! and pull procedural-macro deps; they cannot depend on consumer crates).
//! The example is exercised end-to-end by the integration tests in
//! `ferrotorch-nn/tests/derive_module.rs`.
//!
//! ```ignore
//! use ferrotorch_nn::{Module, Parameter, Linear};
//! use ferrotorch_nn_derive::Module;
//!
//! #[derive(Module)]
//! struct MyModel<T: Float> {
//!     #[param]     weight: Parameter<T>,
//!     #[param]     bias: Parameter<T>,
//!     #[submodule] layer1: Linear<T>,
//!     #[submodule] layer2: Linear<T>,
//!     #[skip]      hidden_size: usize,
//!     training: bool,
//! }
//! ```

#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
#![allow(missing_docs)] // tracked workspace-wide in the rustdoc pass

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{Data, DeriveInput, Fields, GenericParam, Generics, TypeParam, parse_macro_input};

/// Derive the `Module<T>` trait for a struct.
///
/// See the [crate-level documentation](crate) for attribute usage.
#[proc_macro_derive(Module, attributes(param, submodule, skip))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_module_impl(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

/// Classification of a struct field for code generation.
#[derive(Debug)]
enum FieldKind {
    /// `#[param]` — a `Parameter<T>` field.
    Param,
    /// `#[submodule]` — a field that implements `Module<T>`.
    Submodule,
    /// The `training: bool` field managed by the derive.
    Training,
    /// `#[skip]` or unannotated — ignored.
    Skip,
}

#[derive(Debug)]
struct ClassifiedField {
    ident: Ident,
    kind: FieldKind,
}

// `derive_module_impl` exceeds clippy's 100-line threshold because it is the
// single code-generation entry point: classify fields, validate, find float
// param, build six method bodies, then assemble one `quote!` block.
// Splitting purely to satisfy the lint would scatter the `quote!` template
// across helpers that share a dozen captured locals — net readability loss.
#[allow(clippy::too_many_lines)]
// `parse_macro_input!` (in `derive_module`) yields an owned `DeriveInput`,
// which is the standard proc-macro shape; taking by reference here would
// force callers to bind a temporary first. This is a proc-macro convention,
// not a hot-path concern.
#[allow(clippy::needless_pass_by_value)]
fn derive_module_impl(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let generics = &input.generics;

    // --- Extract fields (named structs only) --------------------------------

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "#[derive(Module)] only supports structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "#[derive(Module)] only supports structs",
            ));
        }
    };

    // --- Classify each field ------------------------------------------------

    let mut classified: Vec<ClassifiedField> = Vec::new();
    let mut has_training = false;

    for field in fields {
        // We've already matched `Fields::Named(...)` above, so every field
        // here has an ident. Defending in depth against a future refactor
        // that broadens the match: surface a `compile_error!` rather than
        // an `unwrap`-ICE if this invariant is ever violated.
        let ident = field
            .ident
            .as_ref()
            .ok_or_else(|| {
                syn::Error::new_spanned(
                    field,
                    "ferrotorch-nn-derive: expected named field (this is a bug — \
                     please report at https://github.com/ferrotorch/ferrotorch/issues)",
                )
            })?
            .clone();

        let has_param = field.attrs.iter().any(|a| a.path().is_ident("param"));
        let has_submodule = field.attrs.iter().any(|a| a.path().is_ident("submodule"));
        let has_skip = field.attrs.iter().any(|a| a.path().is_ident("skip"));

        // Validate: at most one of #[param], #[submodule], #[skip].
        let attr_count = u8::from(has_param) + u8::from(has_submodule) + u8::from(has_skip);
        if attr_count > 1 {
            return Err(syn::Error::new_spanned(
                field,
                "field cannot have more than one of #[param], #[submodule], #[skip]",
            ));
        }

        let kind = if has_param {
            FieldKind::Param
        } else if has_submodule {
            FieldKind::Submodule
        } else if has_skip {
            FieldKind::Skip
        } else if ident == "training" {
            has_training = true;
            FieldKind::Training
        } else {
            // Unannotated and not `training` — skip by default.
            FieldKind::Skip
        };

        classified.push(ClassifiedField { ident, kind });
    }

    if !has_training {
        return Err(syn::Error::new(
            Span::call_site(),
            "#[derive(Module)] requires a `training: bool` field",
        ));
    }

    // --- Find the Float type parameter --------------------------------------
    // We look for a type parameter that has a `Float` bound.
    // If none is found, we fall back to the first type parameter.

    let float_param = find_float_param(generics)?;

    // --- Generate method bodies ---------------------------------------------

    let params: Vec<&ClassifiedField> = classified
        .iter()
        .filter(|f| matches!(f.kind, FieldKind::Param))
        .collect();
    let submodules: Vec<&ClassifiedField> = classified
        .iter()
        .filter(|f| matches!(f.kind, FieldKind::Submodule))
        .collect();

    // parameters(&self) -> Vec<&Parameter<T>>
    let parameters_body = {
        let param_pushes = params.iter().map(|f| {
            let id = &f.ident;
            quote! { params.push(&self.#id); }
        });
        let submod_extends = submodules.iter().map(|f| {
            let id = &f.ident;
            quote! { params.extend(self.#id.parameters()); }
        });
        quote! {
            let mut params = ::std::vec::Vec::new();
            #(#param_pushes)*
            #(#submod_extends)*
            params
        }
    };

    // parameters_mut(&mut self) -> Vec<&mut Parameter<T>>
    let parameters_mut_body = {
        let param_pushes = params.iter().map(|f| {
            let id = &f.ident;
            quote! { params.push(&mut self.#id); }
        });
        let submod_extends = submodules.iter().map(|f| {
            let id = &f.ident;
            quote! { params.extend(self.#id.parameters_mut()); }
        });
        quote! {
            let mut params = ::std::vec::Vec::new();
            #(#param_pushes)*
            #(#submod_extends)*
            params
        }
    };

    // named_parameters(&self) -> Vec<(String, &Parameter<T>)>
    let named_parameters_body = {
        let param_pushes = params.iter().map(|f| {
            let id = &f.ident;
            let name_str = id.to_string();
            quote! { params.push((#name_str.to_string(), &self.#id)); }
        });
        let submod_extends = submodules.iter().map(|f| {
            let id = &f.ident;
            let prefix = id.to_string();
            quote! {
                for (name, p) in self.#id.named_parameters() {
                    params.push((::std::format!("{}.{}", #prefix, name), p));
                }
            }
        });
        quote! {
            let mut params = ::std::vec::Vec::new();
            #(#param_pushes)*
            #(#submod_extends)*
            params
        }
    };

    // train(&mut self)
    let train_body = {
        let submod_trains = submodules.iter().map(|f| {
            let id = &f.ident;
            quote! { self.#id.train(); }
        });
        quote! {
            self.training = true;
            #(#submod_trains)*
        }
    };

    // eval(&mut self)
    let eval_body = {
        let submod_evals = submodules.iter().map(|f| {
            let id = &f.ident;
            quote! { self.#id.eval(); }
        });
        quote! {
            self.training = false;
            #(#submod_evals)*
        }
    };

    // --- Assemble the impl block --------------------------------------------

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::ferrotorch_nn::Module<#float_param> for #name #ty_generics #where_clause {
            /// Delegates to the inherent `forward()` method that the user must
            /// define on this struct. Forgetting to define it produces a
            /// compile-time error instead of a runtime panic.
            fn forward(&self, input: &::ferrotorch_core::Tensor<#float_param>) -> ::ferrotorch_core::FerrotorchResult<::ferrotorch_core::Tensor<#float_param>> {
                self.forward(input)
            }

            fn parameters(&self) -> ::std::vec::Vec<&::ferrotorch_nn::Parameter<#float_param>> {
                #parameters_body
            }

            fn parameters_mut(&mut self) -> ::std::vec::Vec<&mut ::ferrotorch_nn::Parameter<#float_param>> {
                #parameters_mut_body
            }

            fn named_parameters(&self) -> ::std::vec::Vec<(::std::string::String, &::ferrotorch_nn::Parameter<#float_param>)> {
                #named_parameters_body
            }

            fn train(&mut self) {
                #train_body
            }

            fn eval(&mut self) {
                #eval_body
            }

            fn is_training(&self) -> bool {
                self.training
            }
        }
    };

    Ok(expanded)
}

/// Collect the idents of every type parameter declared on `generics`.
fn type_param_idents(generics: &Generics) -> Vec<&Ident> {
    generics
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Type(TypeParam { ident, .. }) => Some(ident),
            _ => None,
        })
        .collect()
}

/// True if `path` is a single-segment path with no qualifier or generic args
/// — i.e. a plain type-parameter reference like `T`, not `Self::Item`,
/// `<Self as Foo>::T`, or `Vec<T>`.
fn path_is_plain_type_param(path: &syn::Path) -> bool {
    path.segments.len() == 1 && matches!(path.segments[0].arguments, syn::PathArguments::None)
}

/// Find the type parameter with a `Float` bound, or fall back to the first
/// type parameter. Returns an error if the struct has no type parameters.
fn find_float_param(generics: &Generics) -> syn::Result<Ident> {
    let declared = type_param_idents(generics);

    // First pass: look for a parameter declaration with an explicit `Float`
    // bound (e.g. `T: Float`).
    for param in &generics.params {
        if let GenericParam::Type(TypeParam { ident, bounds, .. }) = param {
            for bound in bounds {
                if let syn::TypeParamBound::Trait(tb) = bound {
                    if tb
                        .path
                        .segments
                        .last()
                        .is_some_and(|seg| seg.ident == "Float")
                    {
                        return Ok(ident.clone());
                    }
                }
            }
        }
    }

    // Second pass: where-clause predicates (e.g. `where T: Float`).
    //
    // We only accept predicates whose bounded type is a plain single-segment
    // path that names one of the struct's declared type parameters. Anything
    // else — `Self::Item: Float`, `<Self as Foo>::T: Float`, `Vec<T>: Float`,
    // or a path naming an undeclared identifier — is not a generic-parameter
    // bound and must not be picked as the float type. (Earlier versions of
    // this function used `path.segments.first()`, which silently returned
    // `Self` for `Self::Item: Float` — the wrong qualifier rather than the
    // intended type parameter.)
    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(pt) = predicate {
                let bounds_float = pt.bounds.iter().any(|bound| {
                    matches!(
                        bound,
                        syn::TypeParamBound::Trait(tb)
                            if tb.path.segments.last().is_some_and(|seg| seg.ident == "Float")
                    )
                });
                if !bounds_float {
                    continue;
                }
                let syn::Type::Path(tp) = &pt.bounded_ty else {
                    continue;
                };
                if tp.qself.is_some() || !path_is_plain_type_param(&tp.path) {
                    continue;
                }
                let candidate = &tp.path.segments[0].ident;
                if declared.contains(&candidate) {
                    return Ok(candidate.clone());
                }
            }
        }
    }

    // Fallback: use the first type parameter.
    if let Some(first) = declared.first() {
        return Ok((*first).clone());
    }

    Err(syn::Error::new(
        Span::call_site(),
        "#[derive(Module)] requires at least one type parameter (e.g., `T: Float`)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn float_ident_for(input: &syn::DeriveInput) -> String {
        find_float_param(&input.generics).unwrap().to_string()
    }

    #[test]
    fn picks_inline_bound_param() {
        // `struct S<T: Float> { ... }` — first pass.
        let di: syn::DeriveInput = parse_quote! {
            struct S<T: Float> { x: T }
        };
        assert_eq!(float_ident_for(&di), "T");
    }

    #[test]
    fn picks_where_clause_param() {
        // `where T: Float` — second pass.
        let di: syn::DeriveInput = parse_quote! {
            struct S<T> where T: Float { x: T }
        };
        assert_eq!(float_ident_for(&di), "T");
    }

    // Regression test for the audit finding: previously, a where-clause with
    // a multi-segment bounded type like `Self::Item: Float` would match and
    // return the *first* segment (`Self`) — which is not a generic parameter
    // at all. The current implementation skips such predicates entirely and
    // falls back to the first declared type parameter.
    #[test]
    fn ignores_associated_type_in_where_clause() {
        // `where Self::Item: Float` — must NOT be picked. The fallback (first
        // declared type param, `T`) is the correct answer.
        let di: syn::DeriveInput = parse_quote! {
            struct S<T> where Self::Item: Float { x: T }
        };
        assert_eq!(float_ident_for(&di), "T");
    }

    #[test]
    fn ignores_qself_path_in_where_clause() {
        // `where <Self as Foo>::T: Float` — must NOT be picked.
        let di: syn::DeriveInput = parse_quote! {
            struct S<U> where <Self as Foo>::T: Float { x: U }
        };
        assert_eq!(float_ident_for(&di), "U");
    }

    #[test]
    fn fallback_when_no_float_bound() {
        // No `Float` bound anywhere — return the first type parameter.
        let di: syn::DeriveInput = parse_quote! {
            struct S<T: Clone> { x: T }
        };
        assert_eq!(float_ident_for(&di), "T");
    }

    #[test]
    fn errors_when_no_type_params() {
        let di: syn::DeriveInput = parse_quote! {
            struct S { x: u32 }
        };
        assert!(find_float_param(&di.generics).is_err());
    }
}
