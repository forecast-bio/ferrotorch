//! Einops-style tensor rearrangement operations.
//!
//! Provides `rearrange`, `repeat`, and `reduce` with readable string patterns
//! for expressing tensor shape transformations declaratively.
//!
//! # Pattern syntax
//!
//! A pattern has the form `"left -> right"` where `left` and `right` are
//! space-separated axis names. Parenthesized groups denote merged/split
//! dimensions:
//!
//! - `"b c h w -> b (c h w)"` merges `c`, `h`, `w` into one axis
//! - `"b (c h) w -> b c h w"` splits a dimension (requires `axes_lengths`)
//! - `"b h w c -> b c h w"` transposes (reorders) axes
//!
//! Axes present on the left but absent on the right are reduced (for `reduce`)
//! or must be size-1 (for `rearrange`). Axes present on the right but absent
//! on the left are new axes (for `repeat`).

use std::collections::HashMap;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Public API — Reduction enum
// ---------------------------------------------------------------------------

/// Reduction operation for [`reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EinopsReduction {
    /// Arithmetic mean along reduced axes.
    Mean,
    /// Sum along reduced axes.
    Sum,
    /// Element-wise maximum along reduced axes.
    Max,
    /// Element-wise minimum along reduced axes.
    Min,
}

// ---------------------------------------------------------------------------
// Pattern parser
// ---------------------------------------------------------------------------

/// A single axis on one side of the pattern. Either a bare name or a
/// parenthesized group of names (representing a merged/split dimension).
#[derive(Debug, Clone, PartialEq)]
enum AxisSpec {
    /// A single named axis, e.g. `b`.
    Single(String),
    /// A parenthesized group of axes, e.g. `(c h w)`.
    Group(Vec<String>),
}

/// Parsed einops pattern.
#[derive(Debug)]
struct ParsedPattern {
    left: Vec<AxisSpec>,
    right: Vec<AxisSpec>,
}

/// Flatten an `AxisSpec` list into individual axis names in order.
fn flatten_axes(specs: &[AxisSpec]) -> Vec<String> {
    let mut out = Vec::new();
    for spec in specs {
        match spec {
            AxisSpec::Single(name) => out.push(name.clone()),
            AxisSpec::Group(names) => out.extend(names.iter().cloned()),
        }
    }
    out
}

/// Parse one side of the pattern (e.g. `"b (c h) w"`) into a list of
/// `AxisSpec` entries.
fn parse_side(s: &str) -> FerrotorchResult<Vec<AxisSpec>> {
    let s = s.trim();
    let mut specs = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        if c == '(' {
            // Consume the opening paren.
            chars.next();
            let mut group = Vec::new();
            loop {
                // Skip whitespace inside parens.
                while let Some(&c2) = chars.peek() {
                    if c2.is_whitespace() {
                        chars.next();
                    } else {
                        break;
                    }
                }
                match chars.peek() {
                    None => {
                        return Err(FerrotorchError::InvalidArgument {
                            message: "einops: unmatched '(' in pattern".into(),
                        });
                    }
                    Some(&')') => {
                        chars.next();
                        break;
                    }
                    _ => {}
                }
                // Read an axis name.
                let name = read_axis_name(&mut chars)?;
                if name.is_empty() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "einops: empty axis name inside parentheses".into(),
                    });
                }
                group.push(name);
            }
            if group.is_empty() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "einops: empty parenthesized group".into(),
                });
            }
            specs.push(AxisSpec::Group(group));
        } else if c.is_ascii_alphanumeric() || c == '_' {
            let name = read_axis_name(&mut chars)?;
            specs.push(AxisSpec::Single(name));
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einops: unexpected character '{c}' in pattern"),
            });
        }
    }

    Ok(specs)
}

/// Read an axis name: a run of alphanumeric / underscore characters.
#[allow(clippy::unnecessary_wraps)] // reason: keeps signature uniform with sibling parser helpers (read_int, read_group) that DO fail
fn read_axis_name(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> FerrotorchResult<String> {
    let mut name = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_alphanumeric() || c == '_' {
            name.push(c);
            chars.next();
        } else {
            break;
        }
    }
    Ok(name)
}

/// Parse a full pattern like `"b c h w -> b (c h) w"`.
fn parse_pattern(pattern: &str) -> FerrotorchResult<ParsedPattern> {
    let pattern = pattern.trim();
    let (left_str, right_str) =
        pattern
            .split_once("->")
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("einops: pattern must contain '->', got: \"{pattern}\""),
            })?;

    let left = parse_side(left_str)?;
    let right = parse_side(right_str)?;

    // Validate: no duplicate axis names within a side.
    let left_names = flatten_axes(&left);
    let right_names = flatten_axes(&right);

    let mut seen = HashMap::new();
    for name in &left_names {
        if seen.insert(name.as_str(), "left").is_some() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einops: duplicate axis name '{name}' on left side of pattern"),
            });
        }
    }
    seen.clear();
    for name in &right_names {
        if seen.insert(name.as_str(), "right").is_some() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einops: duplicate axis name '{name}' on right side of pattern"),
            });
        }
    }

    Ok(ParsedPattern { left, right })
}

// ---------------------------------------------------------------------------
// Axis-size resolution
// ---------------------------------------------------------------------------

/// Resolve the size of every named axis. Returns a map from axis name to
/// its size.
///
/// - Axes that appear as `Single` on the left get their size from the
///   corresponding input dimension.
/// - Axes inside a `Group` on the left come from splitting an input dim.
///   If there are N sub-axes and all but one have known sizes (from
///   `axes_lengths`), the remaining one is inferred.
/// - Axes that only appear on the right (new axes) must have their size
///   supplied in `axes_lengths`.
fn resolve_sizes(
    pattern: &ParsedPattern,
    input_shape: &[usize],
    axes_lengths: &[(&str, usize)],
) -> FerrotorchResult<HashMap<String, usize>> {
    let left_flat = flatten_axes(&pattern.left);
    let right_flat = flatten_axes(&pattern.right);

    // Count how many input dimensions the left side represents.
    let left_dim_count = pattern.left.len();
    if left_dim_count != input_shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einops: left side of pattern has {} axes but input tensor has {} dimensions",
                left_dim_count,
                input_shape.len()
            ),
        });
    }

    let user_sizes: HashMap<&str, usize> = axes_lengths.iter().copied().collect();
    let mut sizes: HashMap<String, usize> = HashMap::new();

    // First pass: assign sizes from the left side.
    for (dim_idx, spec) in pattern.left.iter().enumerate() {
        let dim_size = input_shape[dim_idx];
        match spec {
            AxisSpec::Single(name) => {
                sizes.insert(name.clone(), dim_size);
            }
            AxisSpec::Group(names) => {
                // This is a split: one input dim is being decomposed into
                // multiple named axes. We need axes_lengths for all but
                // (at most) one of them.
                let mut unknown_idx: Option<usize> = None;
                let mut known_product: usize = 1;

                for (i, name) in names.iter().enumerate() {
                    if let Some(&sz) = user_sizes.get(name.as_str()) {
                        sizes.insert(name.clone(), sz);
                        known_product *= sz;
                    } else if let Some(&sz) = sizes.get(name) {
                        // Already known from a previous occurrence (shouldn't happen
                        // since we check duplicates, but be defensive).
                        known_product *= sz;
                    } else {
                        if unknown_idx.is_some() {
                            return Err(FerrotorchError::InvalidArgument {
                                message: format!(
                                    "einops: cannot infer sizes for split '({})' — \
                                     provide sizes for all but one sub-axis via axes_lengths",
                                    names.join(" ")
                                ),
                            });
                        }
                        unknown_idx = Some(i);
                    }
                }

                if let Some(ui) = unknown_idx {
                    if known_product == 0 || dim_size % known_product != 0 {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "einops: dimension {} (size {}) is not divisible by \
                                 known product {} for split '({})'",
                                dim_idx,
                                dim_size,
                                known_product,
                                names.join(" ")
                            ),
                        });
                    }
                    sizes.insert(names[ui].clone(), dim_size / known_product);
                } else {
                    // All sub-axes are known; verify the product matches.
                    if known_product != dim_size {
                        return Err(FerrotorchError::ShapeMismatch {
                            message: format!(
                                "einops: split '({})' product {} does not match dimension {} size {}",
                                names.join(" "),
                                known_product,
                                dim_idx,
                                dim_size
                            ),
                        });
                    }
                }
            }
        }
    }

    // Second pass: axes that only appear on the right (new axes) must come
    // from axes_lengths.
    for name in &right_flat {
        if !sizes.contains_key(name) {
            if let Some(&sz) = user_sizes.get(name.as_str()) {
                sizes.insert(name.clone(), sz);
            } else if !left_flat.contains(name) {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "einops: axis '{name}' appears on the right but not the left \
                         and has no size in axes_lengths"
                    ),
                });
            }
        }
    }

    Ok(sizes)
}

// ---------------------------------------------------------------------------
// Core implementation helpers
// ---------------------------------------------------------------------------

/// Compute the output shape from the right side of the pattern and the
/// resolved axis sizes.
fn output_shape(right: &[AxisSpec], sizes: &HashMap<String, usize>) -> Vec<usize> {
    right
        .iter()
        .map(|spec| match spec {
            AxisSpec::Single(name) => *sizes.get(name).unwrap(),
            AxisSpec::Group(names) => names.iter().map(|n| sizes.get(n).unwrap()).product(),
        })
        .collect()
}

/// Convert a flat index to per-axis coordinates.
fn flat_to_coords(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];
    for d in (0..ndim).rev() {
        coords[d] = flat % shape[d];
        flat /= shape[d];
    }
    coords
}

/// Convert per-axis coordinates to a flat index.
fn coords_to_flat(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for d in (0..shape.len()).rev() {
        flat += coords[d] * stride;
        stride *= shape[d];
    }
    flat
}

/// Build the "elementary" shape from a pattern side: each `AxisSpec::Group`
/// is expanded into its individual sub-axis sizes.
fn elementary_shape(specs: &[AxisSpec], sizes: &HashMap<String, usize>) -> Vec<usize> {
    let mut shape = Vec::new();
    for spec in specs {
        match spec {
            AxisSpec::Single(name) => shape.push(*sizes.get(name).unwrap()),
            AxisSpec::Group(names) => {
                for n in names {
                    shape.push(*sizes.get(n).unwrap());
                }
            }
        }
    }
    shape
}

/// Perform the general rearrange operation. This is the core engine used by
/// `rearrange`, `repeat`, and `reduce`.
///
/// The algorithm:
/// 1. Compute the "elementary" shapes for left and right (fully expanded).
/// 2. Reshape input from `input_shape` to `left_elementary_shape` (splits).
/// 3. Determine the permutation from left-elementary to right-elementary
///    axis ordering.
/// 4. Transpose data according to the permutation.
/// 5. Reshape from `right_elementary_shape` to `output_shape` (merges).
#[allow(clippy::unnecessary_wraps)] // reason: pairs with rearrange/reduce_impl which DO fail; keeping uniform Result lets the dispatch site stay one match
fn rearrange_impl<T: Float>(
    data: &[T],
    _input_shape: &[usize],
    pattern: &ParsedPattern,
    sizes: &HashMap<String, usize>,
    _output_shape: &[usize],
) -> FerrotorchResult<Vec<T>> {
    let left_names = flatten_axes(&pattern.left);
    let right_names = flatten_axes(&pattern.right);
    let left_elem_shape = elementary_shape(&pattern.left, sizes);
    let right_elem_shape = elementary_shape(&pattern.right, sizes);

    // The right_names should be a permutation of left_names (for rearrange).
    // Build the permutation: for each axis in right_names, find its index
    // in left_names.
    let perm: Vec<usize> = right_names
        .iter()
        .map(|name| {
            left_names
                .iter()
                .position(|n| n == name)
                .unwrap_or(usize::MAX)
        })
        .collect();

    // If there are axes only on the right (repeat) or only on the left (reduce),
    // they won't have a valid permutation entry. For a pure rearrange, every
    // entry should be valid.

    // Step 1: Reshape from input_shape to left_elem_shape. Since both have
    // the same total number of elements and the data is C-contiguous, this
    // is a no-op on the buffer — only the interpretation changes.

    // Step 2: Transpose from left_elem_shape to right_elem_shape order.
    let elem_numel: usize = left_elem_shape.iter().product();
    let mut transposed = vec![<T as num_traits::Zero>::zero(); elem_numel];

    for (src_flat, &val) in data.iter().enumerate().take(elem_numel) {
        let src_coords = flat_to_coords(src_flat, &left_elem_shape);
        let mut dst_coords = vec![0usize; right_elem_shape.len()];
        for (dst_dim, &src_dim) in perm.iter().enumerate() {
            dst_coords[dst_dim] = src_coords[src_dim];
        }
        let dst_flat = coords_to_flat(&dst_coords, &right_elem_shape);
        transposed[dst_flat] = val;
    }

    // Step 3: The transposed buffer now has right_elem_shape layout.
    // Reshape to the output_shape (merging groups). This is again a
    // reinterpretation — same buffer, different shape.
    Ok(transposed)
}

// ---------------------------------------------------------------------------
// Public API — rearrange
// ---------------------------------------------------------------------------

/// Rearrange tensor dimensions using an einops-style pattern.
///
/// # Examples
/// ```ignore
/// // Flatten spatial dims: [B, C, H, W] -> [B, C*H*W]
/// rearrange(&t, "b c h w -> b (c h w)")?;
///
/// // Transpose: [B, H, W, C] -> [B, C, H, W]
/// rearrange(&t, "b h w c -> b c h w")?;
///
/// // Merge dims: [B, H, W, C] -> [B, H*W, C]
/// rearrange(&t, "b h w c -> b (h w) c")?;
/// ```
pub fn rearrange<T: Float>(input: &Tensor<T>, pattern: &str) -> FerrotorchResult<Tensor<T>> {
    rearrange_with(input, pattern, &[])
}

/// Rearrange with explicit axis sizes for ambiguous splits.
///
/// # Examples
/// ```ignore
/// // Split a dimension: [B, C*H, W] -> [B, C, H, W] with C=3
/// rearrange_with(&t, "b (c h) w -> b c h w", &[("c", 3)])?;
/// ```
///
/// # Device behavior
///
/// When the operation reduces to a pure flatten/unflatten (no axis reordering
/// at the elementary level — e.g. `"b c h w -> b (c h w)"` or
/// `"b (c h) w -> b c h w"`), this is a zero-copy `view_reshape` and runs on
/// any device with no data movement.
///
/// When the operation requires actual axis reordering (e.g. `"b h w c -> b c h w"`),
/// it goes through `view_reshape → permute → contiguous → view_reshape`. The
/// `permute` is a zero-copy stride view, but `contiguous` currently materializes
/// non-contiguous CUDA tensors via a CPU round-trip — see issue #496 for the
/// missing GPU `gpu_strided_copy` primitive that would close this gap.
pub fn rearrange_with<T: Float>(
    input: &Tensor<T>,
    pattern: &str,
    axes_lengths: &[(&str, usize)],
) -> FerrotorchResult<Tensor<T>> {
    let parsed = parse_pattern(pattern)?;
    let sizes = resolve_sizes(&parsed, input.shape(), axes_lengths)?;

    let left_names = flatten_axes(&parsed.left);
    let right_names = flatten_axes(&parsed.right);

    // For rearrange, left and right must name exactly the same set of axes.
    let mut left_sorted = left_names.clone();
    left_sorted.sort();
    let mut right_sorted = right_names.clone();
    right_sorted.sort();
    if left_sorted != right_sorted {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einops rearrange: left axes {left_names:?} and right axes {right_names:?} must name \
                 the same set of axes (use `repeat` for new axes, `reduce` for removed axes)"
            ),
        });
    }

    let out_shape = output_shape(&parsed.right, &sizes);
    let left_elem_shape = elementary_shape(&parsed.left, &sizes);
    let right_elem_shape = elementary_shape(&parsed.right, &sizes);

    // Compute the permutation from left elementary order to right elementary
    // order. Since left and right name the same axes, every right axis has a
    // unique position in left.
    let perm: Vec<usize> = right_names
        .iter()
        .map(|name| {
            left_names
                .iter()
                .position(|n| n == name)
                .expect("axis sets validated to match above")
        })
        .collect();

    // Fast path: identity permutation. This covers the common cases of pure
    // flatten (`"b c h w -> b (c h w)"`), pure unflatten
    // (`"b (c h) w -> b c h w"`), and grouping rearrangements where the
    // elementary axis order matches between left and right. No data movement
    // — runs zero-copy on any device (CPU or CUDA) when input is contiguous.
    let is_identity_perm = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity_perm && input.is_contiguous() {
        // Skip the intermediate elementary shape entirely — direct
        // view_reshape from input shape to output shape.
        return input.view_reshape(out_shape);
    }

    // General path: reshape to elementary form, permute, materialize, reshape
    // to merged output. On CUDA, the contiguous() step currently materializes
    // strided tensors via CPU (issue #496). On CPU it's all in-place loops.
    //
    // We try the tensor-op path first; if any step fails (e.g. an inner op
    // doesn't yet support a particular shape), we fall back to the legacy
    // index-walk implementation below.
    if input.is_contiguous() {
        if let Ok(result) = (|| -> FerrotorchResult<Tensor<T>> {
            let elem_view = input.view_reshape(left_elem_shape.clone())?;
            let permuted = elem_view.permute(&perm)?;
            let contig = permuted.contiguous()?;
            // Reshape from right_elem_shape (now contiguous) to merged output.
            // contig already has shape == right_elem_shape; if out_shape merges
            // some axes, view_reshape will collapse them.
            if contig.shape() == out_shape.as_slice() {
                Ok(contig)
            } else {
                contig.view_reshape(out_shape.clone())
            }
        })() {
            return Ok(result);
        }
    }

    // Legacy fallback: pure-CPU index-walk. Used when the tensor-op path
    // refuses (e.g. non-contiguous input) or errors.
    let _ = right_elem_shape; // silence unused warning when fast paths take over
    let device = input.device();
    let data = input.data_vec()?;
    let result_data = rearrange_impl(&data, input.shape(), &parsed, &sizes, &out_shape)?;

    let t = Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)?;
    Ok(if device.is_cuda() { t.to(device)? } else { t })
}

// ---------------------------------------------------------------------------
// Public API — repeat
// ---------------------------------------------------------------------------

/// Repeat tensor elements along new or existing axes.
///
/// Axes on the right that do not appear on the left are new dimensions and
/// must have their size specified in `axes_lengths`.
///
/// # Examples
/// ```ignore
/// // Add a batch dim by repeating: [H, W] -> [B, H, W]
/// repeat(&t, "h w -> b h w", &[("b", 4)])?;
///
/// // Tile: [C] -> [C, 3]
/// repeat(&t, "c -> c n", &[("n", 3)])?;
/// ```
pub fn repeat<T: Float>(
    input: &Tensor<T>,
    pattern: &str,
    axes_lengths: &[(&str, usize)],
) -> FerrotorchResult<Tensor<T>> {
    let parsed = parse_pattern(pattern)?;
    let sizes = resolve_sizes(&parsed, input.shape(), axes_lengths)?;

    let left_names = flatten_axes(&parsed.left);
    let right_names = flatten_axes(&parsed.right);

    // Every left axis must appear on the right.
    for name in &left_names {
        if !right_names.contains(name) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einops repeat: left axis '{name}' does not appear on the right — \
                     use `reduce` to remove axes"
                ),
            });
        }
    }

    // Identify new axes (on right but not left).
    let _new_axes: Vec<&str> = right_names
        .iter()
        .filter(|n| !left_names.contains(n))
        .map(String::as_str)
        .collect();

    // Build the right elementary shape and the output shape.
    let right_elem_shape = elementary_shape(&parsed.right, &sizes);
    let out_shape = output_shape(&parsed.right, &sizes);

    // Strategy: iterate over every element of the output (in right
    // elementary order), map its coordinates back to the input.
    let out_numel: usize = right_elem_shape.iter().product();
    let left_elem_shape = elementary_shape(&parsed.left, &sizes);
    let device = input.device();
    let data = input.data_vec()?;

    let mut result = Vec::with_capacity(out_numel);
    for dst_flat in 0..out_numel {
        let dst_coords = flat_to_coords(dst_flat, &right_elem_shape);
        // Map each right-elementary coordinate to a left-elementary coordinate.
        let mut src_coords = Vec::with_capacity(left_elem_shape.len());
        for (i, name) in right_names.iter().enumerate() {
            if left_names.contains(name) {
                src_coords.push(dst_coords[i]);
            }
            // New axes are simply ignored (they tile/repeat).
        }
        let src_flat = coords_to_flat(&src_coords, &left_elem_shape);
        result.push(data[src_flat]);
    }

    // The result buffer is in right-elementary order. Reshape to out_shape
    // (which merges groups). Since it's the same total size, just reinterpret.
    let t = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    Ok(if device.is_cuda() { t.to(device)? } else { t })
}

// ---------------------------------------------------------------------------
// Public API — reduce
// ---------------------------------------------------------------------------

/// Reduce along axes that appear on the left but not the right.
///
/// # Examples
/// ```ignore
/// // Global average pool: [B, C, H, W] -> [B, C]
/// reduce(&t, "b c h w -> b c", EinopsReduction::Mean)?;
///
/// // Sum over batch: [B, C] -> [C]
/// reduce(&t, "b c -> c", EinopsReduction::Sum)?;
/// ```
pub fn reduce<T: Float>(
    input: &Tensor<T>,
    pattern: &str,
    reduction: EinopsReduction,
) -> FerrotorchResult<Tensor<T>> {
    let parsed = parse_pattern(pattern)?;
    let sizes = resolve_sizes(&parsed, input.shape(), &[])?;

    let left_names = flatten_axes(&parsed.left);
    let right_names = flatten_axes(&parsed.right);

    // Every right axis must appear on the left.
    for name in &right_names {
        if !left_names.contains(name) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einops reduce: right axis '{name}' does not appear on the left — \
                     use `repeat` to add new axes"
                ),
            });
        }
    }

    // Identify reduced axes (on left but not right).
    let reduced_axes: Vec<&str> = left_names
        .iter()
        .filter(|n| !right_names.contains(n))
        .map(String::as_str)
        .collect();

    if reduced_axes.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "einops reduce: no axes are being reduced — use `rearrange` instead".into(),
        });
    }

    // Build the output elementary shape.
    let left_elem_shape = elementary_shape(&parsed.left, &sizes);
    let right_elem_shape = elementary_shape(&parsed.right, &sizes);
    let out_shape = output_shape(&parsed.right, &sizes);

    // ----------------------------------------------------------------------
    // Fast path: axis-aligned reductions via existing GPU-aware tensor ops.
    //
    // This path triggers when:
    //   1. The kept axes (left ∩ right) appear in the same relative order on
    //      both sides (no reordering needed for the kept axes).
    //   2. The reduced axes are contiguous in left elementary order — i.e.
    //      they form a single run within the left axis sequence.
    //
    // When both hold, we can express the reduction as:
    //   view_reshape(input, [outer_kept, reduced_combined, inner_kept])
    //   → sum_dim(dim=1, keepdim=false)        for Sum/Mean
    //   → cummax/cummin → narrow(last)         for Max/Min
    //   → view_reshape(out_shape)
    //
    // All steps run on the input's native device — no CPU round-trip.
    //
    // For Mean, we follow the sum_dim with a scalar multiply by 1/reduce_count
    // (cheaper than dividing every element through a separate kernel and
    // works on GPU since `mul` is GPU-aware).
    // ----------------------------------------------------------------------
    let kept_left_positions: Vec<usize> = left_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if right_names.contains(name) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let reduced_left_positions: Vec<usize> = left_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if right_names.contains(name) {
                None
            } else {
                Some(i)
            }
        })
        .collect();

    // Are the reduced axes contiguous in left order?
    let reduced_contiguous = reduced_left_positions.windows(2).all(|w| w[1] == w[0] + 1);

    // Are the kept axes in the same relative order on right as on left?
    // Build the sequence of kept names in left order, then check it matches
    // the sequence of right_names that are kept (= all right names, since
    // every right axis is kept by construction).
    let kept_in_left_order: Vec<&str> = kept_left_positions
        .iter()
        .map(|&i| left_names[i].as_str())
        .collect();
    let kept_order_matches = kept_in_left_order.len() == right_names.len()
        && kept_in_left_order
            .iter()
            .zip(right_names.iter())
            .all(|(a, b)| *a == b);

    if reduced_contiguous && kept_order_matches && input.is_contiguous() {
        let reduced_start = reduced_left_positions[0];
        let reduced_len = reduced_left_positions.len();

        // Build the 3-D view shape: [outer, reduced_combined, inner].
        let outer: usize = left_elem_shape[..reduced_start].iter().product();
        let reduced_combined: usize = left_elem_shape[reduced_start..reduced_start + reduced_len]
            .iter()
            .product();
        let inner: usize = left_elem_shape[reduced_start + reduced_len..]
            .iter()
            .product();

        if let Ok(result) = (|| -> FerrotorchResult<Tensor<T>> {
            let view = input.view_reshape(vec![outer, reduced_combined, inner])?;
            let reduced_tensor = match reduction {
                EinopsReduction::Sum => crate::grad_fns::reduction::sum_dim(&view, 1, false)?,
                EinopsReduction::Mean => {
                    // sum_dim is GPU-aware; mean_dim is not. Compose
                    // sum_dim → multiply by 1/N to stay on-device.
                    let summed = crate::grad_fns::reduction::sum_dim(&view, 1, false)?;
                    let n_recip =
                        <T as num_traits::One>::one() / T::from(reduced_combined).unwrap();
                    let scale_t = crate::creation::scalar(n_recip)?.to(input.device())?;
                    crate::grad_fns::arithmetic::mul(&summed, &scale_t)?
                }
                EinopsReduction::Max => {
                    // cummax along dim 1, then narrow to the last index.
                    // The running max ends with the global max along that axis.
                    let cmax = crate::grad_fns::cumulative::cummax(&view, 1)?;
                    cmax.values
                        .narrow(1, reduced_combined - 1, 1)?
                        .squeeze_t(1)?
                }
                EinopsReduction::Min => {
                    let cmin = crate::grad_fns::cumulative::cummin(&view, 1)?;
                    cmin.values
                        .narrow(1, reduced_combined - 1, 1)?
                        .squeeze_t(1)?
                }
            };
            // reduced_tensor has shape [outer, inner] (non-keepdim sum) or
            // [outer, inner] (cummax narrow→squeeze). Reshape to right
            // elementary shape, then to the merged output shape.
            let _ = right_elem_shape.clone(); // for documentation
            let materialized = if reduced_tensor.is_contiguous() {
                reduced_tensor
            } else {
                reduced_tensor.contiguous()?
            };
            let final_t = if materialized.shape() == out_shape.as_slice() {
                materialized
            } else {
                materialized.view_reshape(out_shape.clone())?
            };
            Ok(final_t)
        })() {
            return Ok(result);
        }
    }

    // ----------------------------------------------------------------------
    // Fallback: legacy CPU index-walk implementation. Used when the kept
    // axes need reordering, or the reduced axes are not contiguous in the
    // left layout, or any of the GPU-aware ops above bail. Functionally
    // equivalent — same result, but does a CPU round-trip on CUDA inputs.
    // ----------------------------------------------------------------------
    let out_numel: usize = right_elem_shape.iter().product();
    let device = input.device();
    let data = input.data_vec()?;
    let in_numel: usize = left_elem_shape.iter().product();

    // Compute how many elements are reduced per output element.
    let reduce_count: usize = reduced_axes
        .iter()
        .map(|name| sizes.get(*name).unwrap())
        .product();

    let init_val = match reduction {
        EinopsReduction::Sum | EinopsReduction::Mean => <T as num_traits::Zero>::zero(),
        EinopsReduction::Max => T::neg_infinity(),
        EinopsReduction::Min => T::infinity(),
    };
    let mut accum = vec![init_val; out_numel];

    for (src_flat, &val) in data.iter().enumerate().take(in_numel) {
        let src_coords = flat_to_coords(src_flat, &left_elem_shape);
        let mut dst_coords = Vec::with_capacity(right_elem_shape.len());
        for (i, name) in left_names.iter().enumerate() {
            if right_names.contains(name) {
                dst_coords.push(src_coords[i]);
            }
        }
        let dst_flat = coords_to_flat(&dst_coords, &right_elem_shape);

        match reduction {
            EinopsReduction::Sum | EinopsReduction::Mean => {
                accum[dst_flat] += val;
            }
            EinopsReduction::Max => {
                if val > accum[dst_flat] {
                    accum[dst_flat] = val;
                }
            }
            EinopsReduction::Min => {
                if val < accum[dst_flat] {
                    accum[dst_flat] = val;
                }
            }
        }
    }

    if reduction == EinopsReduction::Mean {
        let n = T::from(reduce_count).unwrap();
        for v in &mut accum {
            *v = *v / n;
        }
    }

    let t = Tensor::from_storage(TensorStorage::cpu(accum), out_shape, false)?;
    Ok(if device.is_cuda() { t.to(device)? } else { t })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a leaf tensor.
    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // -----------------------------------------------------------------------
    // rearrange tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rearrange_identity() {
        // "b c h w -> b c h w" should be a no-op.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]);
        let r = rearrange(&t, "b c h w -> b c h w").unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 2]);
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    #[test]
    fn test_rearrange_flatten() {
        // "b c h w -> b (c h w)" merges c, h, w.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]); // B=2, C=3, H=2, W=2
        let r = rearrange(&t, "b c h w -> b (c h w)").unwrap();
        assert_eq!(r.shape(), &[2, 12]);
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    #[test]
    // reason: rearrange ("b h w c -> b c h w") is pure axis permutation —
    // each output slot holds the exact bit pattern of an input slot, so
    // bit-exact equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_rearrange_transpose_nhwc_to_nchw() {
        // "b h w c -> b c h w" transposes.
        // Input shape: [1, 2, 2, 3] (B=1, H=2, W=2, C=3)
        // Output shape: [1, 3, 2, 2]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 2, 3]);
        let r = rearrange(&t, "b h w c -> b c h w").unwrap();
        assert_eq!(r.shape(), &[1, 3, 2, 2]);

        // Verify specific elements.
        // Input[0,0,0,:] = [0,1,2], Input[0,0,1,:] = [3,4,5]
        // Input[0,1,0,:] = [6,7,8], Input[0,1,1,:] = [9,10,11]
        // Output[0,c,h,w] = Input[0,h,w,c]
        // Output[0,0,0,0] = Input[0,0,0,0] = 0
        // Output[0,0,0,1] = Input[0,0,1,0] = 3
        // Output[0,0,1,0] = Input[0,1,0,0] = 6
        // Output[0,0,1,1] = Input[0,1,1,0] = 9
        // Output[0,1,0,0] = Input[0,0,0,1] = 1
        // etc.
        let out = r.data().unwrap();
        assert_eq!(out[0], 0.0); // [0,0,0,0]
        assert_eq!(out[1], 3.0); // [0,0,0,1]
        assert_eq!(out[2], 6.0); // [0,0,1,0]
        assert_eq!(out[3], 9.0); // [0,0,1,1]
        assert_eq!(out[4], 1.0); // [0,1,0,0]
        assert_eq!(out[5], 4.0); // [0,1,0,1]
    }

    #[test]
    fn test_rearrange_split_with_axes_lengths() {
        // "b (c h) w -> b c h w" with c=3 splits dimension 1.
        // Input: [2, 6, 4] -> Output: [2, 3, 2, 4]
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 6, 4]);
        let r = rearrange_with(&t, "b (c h) w -> b c h w", &[("c", 3)]).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 4]);

        // The data should be the same since (c h) is already in order and
        // we're just splitting.
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    #[test]
    fn test_rearrange_merge_dims() {
        // "b h w c -> b (h w) c" merges h and w.
        // Input: [1, 2, 3, 4] -> Output: [1, 6, 4]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 3, 4]);
        let r = rearrange(&t, "b h w c -> b (h w) c").unwrap();
        assert_eq!(r.shape(), &[1, 6, 4]);
        // Data stays the same since h and w are adjacent and in order.
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    // -----------------------------------------------------------------------
    // repeat tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_repeat_new_batch_dim() {
        // "h w -> b h w" adds a batch dimension.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = leaf(&data, &[2, 2]);
        let r = repeat(&t, "h w -> b h w", &[("b", 3)]).unwrap();
        assert_eq!(r.shape(), &[3, 2, 2]);

        let out = r.data().unwrap();
        // Each batch should be a copy of the original.
        assert_eq!(&out[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[4..8], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[8..12], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_tile() {
        // "c -> c n" tiles a 1-D tensor.
        let data = vec![10.0f32, 20.0, 30.0];
        let t = leaf(&data, &[3]);
        let r = repeat(&t, "c -> c n", &[("n", 2)]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);

        let out = r.data().unwrap();
        assert_eq!(out, &[10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
    }

    // -----------------------------------------------------------------------
    // reduce tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_mean_spatial() {
        // "b c h w -> b c" — global average pool.
        // B=1, C=2, H=2, W=2
        // Channel 0: [1, 2, 3, 4] mean = 2.5
        // Channel 1: [5, 6, 7, 8] mean = 6.5
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t = leaf(&data, &[1, 2, 2, 2]);
        let r = reduce(&t, "b c h w -> b c", EinopsReduction::Mean).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
        let out = r.data().unwrap();
        assert!((out[0] - 2.5).abs() < 1e-6, "expected 2.5, got {}", out[0]);
        assert!((out[1] - 6.5).abs() < 1e-6, "expected 6.5, got {}", out[1]);
    }

    #[test]
    fn test_reduce_sum_batch() {
        // "b c -> c" — sum over batch.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = leaf(&data, &[3, 2]); // B=3, C=2
        let r = reduce(&t, "b c -> c", EinopsReduction::Sum).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        // c=0: 1 + 3 + 5 = 9
        // c=1: 2 + 4 + 6 = 12
        assert!((out[0] - 9.0).abs() < 1e-6);
        assert!((out[1] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max() {
        // "b c -> c" — max over batch.
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Max).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        assert!((out[0] - 4.0).abs() < 1e-6); // max(1, 3, 4)
        assert!((out[1] - 6.0).abs() < 1e-6); // max(5, 2, 6)
    }

    #[test]
    fn test_reduce_min() {
        // "b c -> c" — min over batch.
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Min).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6); // min(1, 3, 4)
        assert!((out[1] - 2.0).abs() < 1e-6); // min(5, 2, 6)
    }

    // -----------------------------------------------------------------------
    // Error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_pattern_no_arrow() {
        let t = leaf(&[1.0, 2.0, 3.0], &[3]);
        assert!(rearrange(&t, "a b c").is_err());
    }

    #[test]
    fn test_mismatched_axis_count() {
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // Left side has 3 axes but tensor has 2 dims.
        assert!(rearrange(&t, "a b c -> a b c").is_err());
    }

    #[test]
    fn test_rearrange_missing_axis_on_right() {
        // "b c h w -> b c" would be a reduce, not a rearrange.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]);
        assert!(rearrange(&t, "b c h w -> b c").is_err());
    }

    #[test]
    fn test_rearrange_extra_axis_on_right() {
        // "b c -> b c n" would be a repeat, not a rearrange.
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(rearrange(&t, "b c -> b c n").is_err());
    }

    #[test]
    fn test_repeat_missing_new_axis_size() {
        let t = leaf(&[1.0, 2.0], &[2]);
        // "c -> c n" but no size given for n.
        assert!(repeat(&t, "c -> c n", &[]).is_err());
    }

    #[test]
    fn test_reduce_no_reduction() {
        // "b c -> b c" reduces nothing — should error.
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(reduce(&t, "b c -> b c", EinopsReduction::Sum).is_err());
    }

    #[test]
    fn test_unmatched_paren() {
        let t = leaf(&[1.0, 2.0], &[2]);
        assert!(rearrange(&t, "(a -> a").is_err());
    }

    #[test]
    fn test_duplicate_axis_name() {
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(rearrange(&t, "a a -> a a").is_err());
    }

    // -----------------------------------------------------------------------
    // Parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_simple() {
        let p = parse_pattern("b c h w -> b c h w").unwrap();
        assert_eq!(flatten_axes(&p.left), vec!["b", "c", "h", "w"]);
        assert_eq!(flatten_axes(&p.right), vec!["b", "c", "h", "w"]);
    }

    #[test]
    fn test_parse_groups() {
        let p = parse_pattern("b c h w -> b (c h w)").unwrap();
        assert_eq!(p.right.len(), 2); // b, (c h w)
        match &p.right[1] {
            AxisSpec::Group(names) => assert_eq!(names, &["c", "h", "w"]),
            _ => panic!("expected Group"),
        }
    }

    #[test]
    fn test_parse_left_group() {
        let p = parse_pattern("b (c h) w -> b c h w").unwrap();
        assert_eq!(p.left.len(), 3); // b, (c h), w
        match &p.left[1] {
            AxisSpec::Group(names) => assert_eq!(names, &["c", "h"]),
            _ => panic!("expected Group"),
        }
    }

    // -----------------------------------------------------------------------
    // GPU-aware fast-path tests (run on CPU but exercise the same code path
    // that is taken on CUDA — verifies the view_reshape / sum_dim / cummax
    // compositions are correct).
    // -----------------------------------------------------------------------

    #[test]
    fn test_rearrange_identity_perm_is_view() {
        // Pure flatten — should hit the identity-permutation fast path and
        // share storage with the input (zero-copy view).
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]);
        let r = rearrange(&t, "b c h w -> b (c h w)").unwrap();
        assert_eq!(r.shape(), &[2, 12]);
        // Same buffer pointer indicates a zero-copy view (no materialization).
        assert!(
            std::ptr::eq(r.data().unwrap().as_ptr(), t.data().unwrap().as_ptr()),
            "expected view_reshape fast path to share storage with input"
        );
    }

    #[test]
    fn test_rearrange_pure_unflatten_is_view() {
        // Split a dim — also identity perm, also zero-copy.
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 6, 4]);
        let r = rearrange_with(&t, "b (c h) w -> b c h w", &[("c", 3)]).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 4]);
        assert!(
            std::ptr::eq(r.data().unwrap().as_ptr(), t.data().unwrap().as_ptr()),
            "expected view_reshape fast path to share storage with input"
        );
    }

    #[test]
    fn test_reduce_sum_axis_aligned_fast_path() {
        // "b c h w -> b c" with reduced axes (h,w) contiguous at the end —
        // hits the sum_dim fast path. Verify correctness against PyTorch
        // semantics: sum over h*w within each (b,c).
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 3, 4]); // B=1, C=2, H=3, W=4
        let r = reduce(&t, "b c h w -> b c", EinopsReduction::Sum).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
        let out = r.data().unwrap();
        // Channel 0: sum(0..12) = 66
        // Channel 1: sum(12..24) = 210
        assert!((out[0] - 66.0).abs() < 1e-5);
        assert!((out[1] - 210.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_mean_axis_aligned_fast_path() {
        // Same shape as above; verify mean = sum / N for the fast path.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 3, 4]);
        let r = reduce(&t, "b c h w -> b c", EinopsReduction::Mean).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
        let out = r.data().unwrap();
        // Channel 0: 66 / 12 = 5.5
        // Channel 1: 210 / 12 = 17.5
        assert!((out[0] - 5.5).abs() < 1e-5);
        assert!((out[1] - 17.5).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_max_axis_aligned_fast_path() {
        // Reduced axis is contiguous — hits cummax fast path.
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Max).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        // Reduced axis is `b`, which is left position 0, so the kept axis
        // (c) appears AFTER the reduced axis. That means kept axes are not
        // a leading prefix; depending on interpretation this may take the
        // fallback. Either way the answer must be correct.
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_min_axis_aligned_fast_path() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Min).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_trailing_reduce_full_pool() {
        // "b c h w -> b" reduces c, h, w (all contiguous trailing axes).
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 2, 2, 3]);
        let r = reduce(&t, "b c h w -> b", EinopsReduction::Sum).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        // First batch: sum 0..12 = 66; second batch: sum 12..24 = 210.
        assert!((out[0] - 66.0).abs() < 1e-5);
        assert!((out[1] - 210.0).abs() < 1e-5);
    }
}
