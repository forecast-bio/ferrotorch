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

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    parse_macro_input, Data, DeriveInput, Fields, GenericParam, Generics, TypeParam,
};

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

struct ClassifiedField {
    ident: Ident,
    kind: FieldKind,
}

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
        let ident = field
            .ident
            .clone()
            .expect("named fields always have idents");

        let has_param = field.attrs.iter().any(|a| a.path().is_ident("param"));
        let has_submodule = field.attrs.iter().any(|a| a.path().is_ident("submodule"));
        let has_skip = field.attrs.iter().any(|a| a.path().is_ident("skip"));

        // Validate: at most one of #[param], #[submodule], #[skip].
        let attr_count = has_param as u8 + has_submodule as u8 + has_skip as u8;
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

/// Find the type parameter with a `Float` bound, or fall back to the first
/// type parameter. Returns an error if the struct has no type parameters.
fn find_float_param(generics: &Generics) -> syn::Result<Ident> {
    // First pass: look for a parameter with an explicit `Float` bound.
    for param in &generics.params {
        if let GenericParam::Type(TypeParam { ident, bounds, .. }) = param {
            for bound in bounds {
                if let syn::TypeParamBound::Trait(tb) = bound {
                    if tb.path.segments.last().is_some_and(|seg| seg.ident == "Float") {
                        return Ok(ident.clone());
                    }
                }
            }
        }
    }

    // Second pass: also check where clause predicates.
    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(pt) = predicate {
                for bound in &pt.bounds {
                    if let syn::TypeParamBound::Trait(tb) = bound {
                        if tb.path.segments.last().is_some_and(|seg| seg.ident == "Float") {
                            // Extract the type parameter ident from the bounded type.
                            if let syn::Type::Path(tp) = &pt.bounded_ty {
                                if let Some(seg) = tp.path.segments.first() {
                                    return Ok(seg.ident.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: use the first type parameter.
    for param in &generics.params {
        if let GenericParam::Type(TypeParam { ident, .. }) = param {
            return Ok(ident.clone());
        }
    }

    Err(syn::Error::new(
        Span::call_site(),
        "#[derive(Module)] requires at least one type parameter (e.g., `T: Float`)",
    ))
}
