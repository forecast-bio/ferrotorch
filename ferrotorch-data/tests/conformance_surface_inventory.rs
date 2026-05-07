//! ferrotorch-data conformance — surface inventory generator (Layer 1).
//!
//! Walks `src/lib.rs` and every `mod` it transitively declares, parses each
//! file with `syn`, and emits a sorted JSON inventory of every `pub` item to
//! `tests/conformance/_surface_inventory.toml` (TOML) and the companion
//! `tests/conformance/_surface.json` used by the strict coverage gate.
//!
//! Run to regenerate:
//!
//!   cargo test -p ferrotorch-data --test conformance_surface_inventory
//!
//! The committed JSON file is the denominator for the strict gate in
//! `conformance_surface_coverage.rs`. PRs that change the public surface show
//! up as a clean JSON diff; the gate fails if any new item is not covered.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use syn::{Item, ItemFn, ItemImpl, ItemMod, ItemStruct, Visibility};

#[derive(Debug)]
struct SurfaceItem {
    path: String,
    kind: &'static str,
    signature: String,
}

const CRATE_NAME: &str = "ferrotorch_data";

fn crate_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn out_json_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("_surface.json")
}

fn out_toml_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("_surface_inventory.toml")
}

fn fmt_tokens<T: quote::ToTokens>(t: &T) -> String {
    let s = quote::quote!(#t).to_string();
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn is_pub(vis: &Visibility) -> bool {
    matches!(vis, Visibility::Public(_))
}

fn fn_signature(item: &ItemFn) -> String {
    fmt_tokens(&item.sig)
}

fn struct_signature(item: &ItemStruct) -> String {
    let attrs: String = item
        .attrs
        .iter()
        .filter(|a| {
            let p = a.path();
            p.is_ident("non_exhaustive") || p.is_ident("derive")
        })
        .map(fmt_tokens)
        .collect::<Vec<_>>()
        .join(" ");
    let vis = fmt_tokens(&item.vis);
    let ident = item.ident.to_string();
    let generics = fmt_tokens(&item.generics);
    if attrs.is_empty() {
        format!("{vis} struct {ident}{generics}")
    } else {
        format!("{attrs} {vis} struct {ident}{generics}")
    }
    .trim()
    .to_string()
}

fn strip_generic_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '<' => {
                depth += 1;
                out.push(ch);
            }
            '>' => {
                depth -= 1;
                out.push(ch);
            }
            ' ' if depth > 0 => {}
            c => out.push(c),
        }
    }
    out
}

fn collect_methods(impl_block: &ItemImpl, module_path: &str, out: &mut Vec<SurfaceItem>) {
    if impl_block.trait_.is_some() {
        return;
    }
    let ty = fmt_tokens(&impl_block.self_ty);
    let ty_clean = strip_generic_whitespace(&ty);
    for item in &impl_block.items {
        if let syn::ImplItem::Fn(method) = item
            && is_pub(&method.vis)
        {
            let sig = fmt_tokens(&method.sig);
            out.push(SurfaceItem {
                path: format!("{module_path}::{ty_clean}::{}", method.sig.ident),
                kind: "method",
                signature: sig,
            });
        }
    }
}

fn collect_use_leaves(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, Option<String>)>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            collect_use_leaves(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let mut segs = prefix.clone();
            segs.push(n.ident.to_string());
            out.push((segs, None));
        }
        syn::UseTree::Rename(r) => {
            let mut segs = prefix.clone();
            segs.push(r.ident.to_string());
            out.push((segs, Some(r.rename.to_string())));
        }
        syn::UseTree::Glob(_) => {
            let mut segs = prefix.clone();
            segs.push("*".to_string());
            out.push((segs, None));
        }
        syn::UseTree::Group(g) => {
            for t in &g.items {
                collect_use_leaves(t, prefix, out);
            }
        }
    }
}

fn walk_items(items: &[Item], module_path: &str, dir: &Path, out: &mut Vec<SurfaceItem>) {
    for item in items {
        match item {
            Item::Fn(f) if is_pub(&f.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", f.sig.ident),
                kind: "fn",
                signature: fn_signature(f),
            }),
            Item::Struct(s) if is_pub(&s.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", s.ident),
                kind: "struct",
                signature: struct_signature(s),
            }),
            Item::Enum(e) if is_pub(&e.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", e.ident),
                kind: "enum",
                signature: format!(
                    "{} enum {}{}",
                    fmt_tokens(&e.vis),
                    e.ident,
                    fmt_tokens(&e.generics)
                ),
            }),
            Item::Trait(t) if is_pub(&t.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", t.ident),
                kind: "trait",
                signature: format!(
                    "{} trait {}{}",
                    fmt_tokens(&t.vis),
                    t.ident,
                    fmt_tokens(&t.generics)
                ),
            }),
            Item::Type(ty) if is_pub(&ty.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", ty.ident),
                kind: "type",
                signature: format!(
                    "{} type {}{} = {}",
                    fmt_tokens(&ty.vis),
                    ty.ident,
                    fmt_tokens(&ty.generics),
                    fmt_tokens(&ty.ty)
                ),
            }),
            Item::Const(c) if is_pub(&c.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", c.ident),
                kind: "const",
                signature: format!(
                    "{} const {}: {}",
                    fmt_tokens(&c.vis),
                    c.ident,
                    fmt_tokens(&c.ty)
                ),
            }),
            Item::Static(s) if is_pub(&s.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", s.ident),
                kind: "static",
                signature: format!(
                    "{} static {}: {}",
                    fmt_tokens(&s.vis),
                    s.ident,
                    fmt_tokens(&s.ty)
                ),
            }),
            Item::Use(u) if is_pub(&u.vis) => {
                let mut leaves = Vec::new();
                collect_use_leaves(&u.tree, &mut Vec::new(), &mut leaves);
                for (segments, alias) in leaves {
                    let display_name =
                        alias.unwrap_or_else(|| segments.last().cloned().unwrap_or_default());
                    if display_name.is_empty() || display_name == "*" {
                        out.push(SurfaceItem {
                            path: format!("{module_path}::*"),
                            kind: "re-export",
                            signature: format!("pub use {};", segments.join("::") + "::*"),
                        });
                    } else {
                        out.push(SurfaceItem {
                            path: format!("{module_path}::{display_name}"),
                            kind: "re-export",
                            signature: format!("pub use {};", segments.join("::")),
                        });
                    }
                }
            }
            Item::Mod(m) => walk_module(m, module_path, dir, out),
            Item::Impl(i) => collect_methods(i, module_path, out),
            _ => {}
        }
    }
}

fn walk_module(m: &ItemMod, parent_path: &str, parent_dir: &Path, out: &mut Vec<SurfaceItem>) {
    if !is_pub(&m.vis) {
        return;
    }
    let new_path = format!("{parent_path}::{}", m.ident);
    if let Some((_, items)) = &m.content {
        walk_items(items, &new_path, parent_dir, out);
    } else {
        let ident = m.ident.to_string();
        let candidate_a = parent_dir.join(format!("{ident}.rs"));
        let candidate_b = parent_dir.join(&ident).join("mod.rs");
        let (path, new_dir): (PathBuf, PathBuf) = if candidate_a.exists() {
            (candidate_a, parent_dir.to_path_buf())
        } else if candidate_b.exists() {
            (candidate_b, parent_dir.join(&ident))
        } else {
            out.push(SurfaceItem {
                path: new_path.clone(),
                kind: "fn",
                signature: format!("/* UNRESOLVED MODULE: pub mod {ident}; */"),
            });
            return;
        };
        let src =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let file =
            syn::parse_file(&src).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        walk_items(&file.items, &new_path, &new_dir, out);
    }
}

fn render_json(items: &[SurfaceItem]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"crate\": \"");
    s.push_str(CRATE_NAME);
    s.push_str("\",\n");
    s.push_str("  \"description\": \"Auto-generated by tests/conformance_surface_inventory.rs. Do not edit by hand.\",\n");
    s.push_str("  \"items\": [\n");
    for (i, it) in items.iter().enumerate() {
        s.push_str("    { \"path\": ");
        s.push_str(&json_escape(&it.path));
        s.push_str(", \"kind\": ");
        s.push_str(&json_escape(it.kind));
        s.push_str(", \"signature\": ");
        s.push_str(&json_escape(&it.signature));
        s.push_str(" }");
        if i + 1 < items.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

fn render_toml(items: &[SurfaceItem]) -> String {
    let mut s = String::new();
    s.push_str("# Auto-generated by tests/conformance_surface_inventory.rs.\n");
    s.push_str("# Do not edit by hand.\n\n");
    s.push_str(&format!(
        "[meta]\ncrate = \"{CRATE_NAME}\"\nitem_count = {}\n\n",
        items.len()
    ));
    for it in items {
        s.push_str("[[item]]\n");
        s.push_str(&format!("path = {:?}\n", it.path));
        s.push_str(&format!("kind = {:?}\n", it.kind));
        s.push_str(&format!("signature = {:?}\n\n", it.signature));
    }
    s
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[test]
fn surface_inventory_writes_json() {
    let lib_rs = crate_src_dir().join("lib.rs");
    let src = fs::read_to_string(&lib_rs).expect("read src/lib.rs");
    let file = syn::parse_file(&src).expect("parse src/lib.rs");

    let mut items = Vec::new();
    walk_items(&file.items, CRATE_NAME, &crate_src_dir(), &mut items);

    items.sort_by(|a, b| a.path.cmp(&b.path).then(a.kind.cmp(b.kind)));

    // Deduplicate (a method defined in two impl blocks would appear twice).
    let mut seen = BTreeMap::new();
    let mut unique = Vec::new();
    for it in items {
        let key = format!("{}|{}|{}", it.path, it.kind, it.signature);
        if seen.insert(key, ()).is_none() {
            unique.push(it);
        }
    }

    let json = render_json(&unique);
    let toml_body = render_toml(&unique);

    let conf_dir = out_json_path().parent().expect("conformance dir").to_path_buf();
    fs::create_dir_all(&conf_dir).expect("mkdir conformance");
    fs::write(out_json_path(), &json).expect("write _surface.json");
    fs::write(out_toml_path(), &toml_body).expect("write _surface_inventory.toml");

    eprintln!(
        "ferrotorch-data surface inventory: {} items written to {}",
        unique.len(),
        out_json_path().display()
    );

    // Sanity: must contain the core public API items.
    let must_contain = [
        // dataset module
        "ferrotorch_data::dataset::Dataset",
        "ferrotorch_data::dataset::IterableDataset",
        "ferrotorch_data::dataset::VecDataset",
        "ferrotorch_data::dataset::TensorDataset",
        "ferrotorch_data::dataset::ConcatDataset",
        "ferrotorch_data::dataset::ChainDataset",
        "ferrotorch_data::dataset::WorkerInfo",
        // sampler module
        "ferrotorch_data::sampler::Sampler",
        "ferrotorch_data::sampler::SequentialSampler",
        "ferrotorch_data::sampler::RandomSampler",
        "ferrotorch_data::sampler::BatchSampler",
        "ferrotorch_data::sampler::WeightedRandomSampler",
        "ferrotorch_data::sampler::DistributedSampler",
        "ferrotorch_data::sampler::shuffle_with_seed",
        // dataloader module
        "ferrotorch_data::dataloader::DataLoader",
        "ferrotorch_data::dataloader::WorkerMode",
        "ferrotorch_data::dataloader::ToDevice",
        // collate module
        "ferrotorch_data::collate::default_collate",
        "ferrotorch_data::collate::default_collate_pair",
        // transforms module
        "ferrotorch_data::transforms::Transform",
        "ferrotorch_data::transforms::Compose",
        "ferrotorch_data::transforms::Normalize",
        "ferrotorch_data::transforms::ToTensor",
        "ferrotorch_data::transforms::RandomHorizontalFlip",
        "ferrotorch_data::transforms::RandomCrop",
        "ferrotorch_data::transforms::manual_seed",
    ];

    let paths: Vec<&str> = unique.iter().map(|i| i.path.as_str()).collect();
    let mut missing: Vec<&str> = Vec::new();
    for needle in must_contain {
        if !paths.contains(&needle) {
            missing.push(needle);
        }
    }
    assert!(
        missing.is_empty(),
        "surface inventory missing {} expected ferrotorch-data items: {missing:?}",
        missing.len()
    );

    // Sanity: ferrotorch-data has 5 source files with meaningful pub surfaces.
    // A walker that returns < 20 unique items has almost certainly failed to
    // descend into at least one module.
    assert!(
        unique.len() >= 20,
        "surface inventory unexpectedly small ({} items); expected >= 20. \
         The module walker likely failed to descend into a source file.",
        unique.len()
    );
}
