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
fn read_axis_name(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> FerrotorchResult<String> {
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
    let (left_str, right_str) = pattern.split_once("->").ok_or_else(|| {
        FerrotorchError::InvalidArgument {
            message: format!("einops: pattern must contain '->', got: \"{pattern}\""),
        }
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
                                dim_idx, dim_size, known_product,
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
                                names.join(" "), known_product, dim_idx, dim_size
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

    for src_flat in 0..elem_numel {
        let src_coords = flat_to_coords(src_flat, &left_elem_shape);
        let mut dst_coords = vec![0usize; right_elem_shape.len()];
        for (dst_dim, &src_dim) in perm.iter().enumerate() {
            dst_coords[dst_dim] = src_coords[src_dim];
        }
        let dst_flat = coords_to_flat(&dst_coords, &right_elem_shape);
        transposed[dst_flat] = data[src_flat];
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
                "einops rearrange: left axes {:?} and right axes {:?} must name \
                 the same set of axes (use `repeat` for new axes, `reduce` for removed axes)",
                left_names, right_names
            ),
        });
    }

    let out_shape = output_shape(&parsed.right, &sizes);
    let data = input.data()?;
    let result_data = rearrange_impl(data, input.shape(), &parsed, &sizes, &out_shape)?;

    Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)
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
    let _new_axes: Vec<&String> = right_names
        .iter()
        .filter(|n| !left_names.contains(n))
        .collect();

    // Build the right elementary shape and the output shape.
    let right_elem_shape = elementary_shape(&parsed.right, &sizes);
    let out_shape = output_shape(&parsed.right, &sizes);

    // Strategy: iterate over every element of the output (in right
    // elementary order), map its coordinates back to the input.
    let out_numel: usize = right_elem_shape.iter().product();
    let left_elem_shape = elementary_shape(&parsed.left, &sizes);
    let data = input.data()?;

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
    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
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
    let reduced_axes: Vec<&String> = left_names
        .iter()
        .filter(|n| !right_names.contains(n))
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

    let out_numel: usize = right_elem_shape.iter().product();
    let data = input.data()?;
    let in_numel: usize = left_elem_shape.iter().product();

    // Compute how many elements are reduced per output element.
    let reduce_count: usize = reduced_axes
        .iter()
        .map(|name| sizes.get(name.as_str()).unwrap())
        .product();

    // Accumulate: for each input element, figure out which output element
    // it contributes to.
    // Initialize accumulators.
    let init_val = match reduction {
        EinopsReduction::Sum | EinopsReduction::Mean => <T as num_traits::Zero>::zero(),
        EinopsReduction::Max => T::neg_infinity(),
        EinopsReduction::Min => T::infinity(),
    };
    let mut accum = vec![init_val; out_numel];

    for src_flat in 0..in_numel {
        let src_coords = flat_to_coords(src_flat, &left_elem_shape);
        // Map to output coordinates: keep only the axes that survive.
        let mut dst_coords = Vec::with_capacity(right_elem_shape.len());
        for (i, name) in left_names.iter().enumerate() {
            if right_names.contains(name) {
                dst_coords.push(src_coords[i]);
            }
        }
        let dst_flat = coords_to_flat(&dst_coords, &right_elem_shape);

        let val = data[src_flat];
        match reduction {
            EinopsReduction::Sum | EinopsReduction::Mean => {
                accum[dst_flat] = accum[dst_flat] + val;
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

    // For mean, divide by the number of reduced elements.
    if reduction == EinopsReduction::Mean {
        let n = T::from(reduce_count).unwrap();
        for v in &mut accum {
            *v = *v / n;
        }
    }

    Tensor::from_storage(TensorStorage::cpu(accum), out_shape, false)
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
        assert_eq!(out[0], 0.0);  // [0,0,0,0]
        assert_eq!(out[1], 3.0);  // [0,0,0,1]
        assert_eq!(out[2], 6.0);  // [0,0,1,0]
        assert_eq!(out[3], 9.0);  // [0,0,1,1]
        assert_eq!(out[4], 1.0);  // [0,1,0,0]
        assert_eq!(out[5], 4.0);  // [0,1,0,1]
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
}
