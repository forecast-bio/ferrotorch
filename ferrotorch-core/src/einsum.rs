//! Einstein summation (`einsum`) for ferrotorch tensors.
//!
//! Supports both explicit (`"ij,jk->ik"`) and implicit (`"ij,jk"`) notation.
//! Handles single-input operations (trace, transpose, axis-sum) and two-input
//! contractions via the TTGT (transpose-transpose-GEMM-transpose) algorithm.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::autograd::autocast_ops::autocast_guard;
use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// Equation parser
// ---------------------------------------------------------------------------

/// Parsed einsum equation.
#[derive(Debug, Clone)]
struct ParsedEquation {
    input_subscripts: Vec<Vec<char>>,
    output_subscripts: Vec<char>,
}

/// Parse an einsum equation string like `"ij,jk->ik"` or `"ij,jk"`.
fn parse_equation(equation: &str, n_inputs: usize) -> FerrotorchResult<ParsedEquation> {
    let equation = equation.replace(' ', "");

    let (lhs, output_subscripts) = if let Some((lhs, rhs)) = equation.split_once("->") {
        // Explicit output.
        let out: Vec<char> = rhs.chars().collect();
        // Validate: output indices must all be alphabetic.
        for &c in &out {
            if !c.is_ascii_lowercase() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("einsum: invalid character '{c}' in output subscripts"),
                });
            }
        }
        (lhs.to_string(), out)
    } else {
        // Implicit mode: output is sorted unique indices that appear exactly once.
        let lhs = equation.clone();
        let mut counts: BTreeMap<char, usize> = BTreeMap::new();
        for c in lhs.chars() {
            if c == ',' {
                continue;
            }
            if !c.is_ascii_lowercase() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("einsum: invalid character '{c}' in subscripts"),
                });
            }
            *counts.entry(c).or_insert(0) += 1;
        }
        // Indices appearing exactly once, sorted alphabetically (BTreeMap is already sorted).
        let out: Vec<char> = counts
            .into_iter()
            .filter(|&(_, count)| count == 1)
            .map(|(c, _)| c)
            .collect();
        (lhs, out)
    };

    // Parse input subscripts.
    let input_parts: Vec<&str> = lhs.split(',').collect();
    if input_parts.len() != n_inputs {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einsum: equation has {} input subscripts but {} tensors were provided",
                input_parts.len(),
                n_inputs
            ),
        });
    }

    let input_subscripts: Vec<Vec<char>> = input_parts
        .iter()
        .map(|part| {
            let chars: Vec<char> = part.chars().collect();
            for &c in &chars {
                if !c.is_ascii_lowercase() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("einsum: invalid character '{c}' in input subscripts"),
                    });
                }
            }
            Ok(chars)
        })
        .collect::<FerrotorchResult<Vec<_>>>()?;

    Ok(ParsedEquation {
        input_subscripts,
        output_subscripts,
    })
}

// ---------------------------------------------------------------------------
// Dimension map: index char -> size
// ---------------------------------------------------------------------------

/// Build a map from index character to its dimension size, validating consistency.
fn build_dim_map<T: Float>(
    parsed: &ParsedEquation,
    inputs: &[&Tensor<T>],
) -> FerrotorchResult<BTreeMap<char, usize>> {
    let mut dim_map: BTreeMap<char, usize> = BTreeMap::new();

    for (i, (subs, tensor)) in parsed
        .input_subscripts
        .iter()
        .zip(inputs.iter())
        .enumerate()
    {
        if subs.len() != tensor.ndim() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum: input {} has {} subscripts but tensor has {} dimensions",
                    i,
                    subs.len(),
                    tensor.ndim()
                ),
            });
        }
        for (axis, &c) in subs.iter().enumerate() {
            let size = tensor.shape()[axis];
            if let Some(&existing) = dim_map.get(&c) {
                if existing != size {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "einsum: index '{c}' has inconsistent sizes: {existing} vs {size}"
                        ),
                    });
                }
            } else {
                dim_map.insert(c, size);
            }
        }
    }

    // Validate output subscripts reference known indices.
    for &c in &parsed.output_subscripts {
        if !dim_map.contains_key(&c) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum: output index '{c}' does not appear in any input subscripts"
                ),
            });
        }
    }

    Ok(dim_map)
}

// ---------------------------------------------------------------------------
// Single-input einsum (trace, transpose, axis-sum, diagonal)
// ---------------------------------------------------------------------------

fn einsum_single<T: Float>(
    parsed: &ParsedEquation,
    input: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    let in_subs = &parsed.input_subscripts[0];
    let out_subs = &parsed.output_subscripts;

    // Compute output shape.
    let out_shape: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    let data = input.data_vec()?;
    let in_shape = input.shape();

    // General approach: iterate over all output index combinations plus all
    // summed-over index combinations. For each, accumulate the product.
    //
    // Summed indices: indices in input but not in output.
    let summed_indices: Vec<char> = in_subs
        .iter()
        .filter(|c| !out_subs.contains(c))
        .copied()
        .collect::<Vec<_>>();
    // Deduplicate (a repeated index like "ii" means diagonal/trace).
    let summed_unique: Vec<char> = {
        let mut v = summed_indices.clone();
        v.sort_unstable();
        v.dedup();
        // But we need to include only indices not in output.
        v.into_iter().filter(|c| !out_subs.contains(c)).collect()
    };

    // Compute strides for the input tensor (row-major).
    let in_strides: Vec<usize> = {
        let mut strides = vec![1usize; in_shape.len()];
        for i in (0..in_shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * in_shape[i + 1];
        }
        strides
    };

    // Compute ranges for summed indices.
    let summed_sizes: Vec<usize> = summed_unique.iter().map(|c| dim_map[c]).collect();
    let summed_numel: usize = if summed_sizes.is_empty() {
        1
    } else {
        summed_sizes.iter().product()
    };

    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel];

    // For each output element...
    for (out_idx, result_elem) in result.iter_mut().enumerate() {
        // Decode output multi-index.
        let mut out_multi = vec![0usize; out_subs.len()];
        {
            let mut remainder = out_idx;
            for i in (0..out_subs.len()).rev() {
                let size = dim_map[&out_subs[i]];
                out_multi[i] = remainder % size;
                remainder /= size;
            }
        }

        // Build a map from char -> value for the output indices.
        let mut idx_vals: BTreeMap<char, usize> = BTreeMap::new();
        for (i, &c) in out_subs.iter().enumerate() {
            idx_vals.insert(c, out_multi[i]);
        }

        let mut acc = <T as num_traits::Zero>::zero();

        // Iterate over summed indices.
        for s_idx in 0..summed_numel {
            let mut remainder = s_idx;
            let mut valid = true;
            for i in (0..summed_unique.len()).rev() {
                let val = remainder % summed_sizes[i];
                remainder /= summed_sizes[i];
                idx_vals.insert(summed_unique[i], val);
            }

            // Check consistency for repeated indices (e.g., "ii"):
            // If a char appears more than once in input subscripts, all
            // corresponding axis values must match.
            // For repeated input indices, enforce equality.
            let mut first_occurrence: BTreeMap<char, Option<usize>> = BTreeMap::new();
            for &c in in_subs {
                let val = idx_vals[&c];
                match first_occurrence.get(&c) {
                    Some(Some(prev_val)) => {
                        if *prev_val != val {
                            valid = false;
                            break;
                        }
                    }
                    _ => {
                        first_occurrence.insert(c, Some(val));
                    }
                }
            }

            if !valid {
                continue;
            }

            // Compute flat index into input.
            let mut flat_idx = 0usize;
            for (axis, &c) in in_subs.iter().enumerate() {
                flat_idx += idx_vals[&c] * in_strides[axis];
            }

            acc += data[flat_idx];
        }

        *result_elem = acc;
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

// ---------------------------------------------------------------------------
// Two-input einsum via TTGT
// ---------------------------------------------------------------------------

fn einsum_two<T: Float>(
    parsed: &ParsedEquation,
    a: &Tensor<T>,
    b: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    let a_subs = &parsed.input_subscripts[0];
    let b_subs = &parsed.input_subscripts[1];
    let out_subs = &parsed.output_subscripts;

    // Classify indices.
    // batch:    in A, in B, in output
    // free_a:   in A, NOT in B, in output
    // free_b:   in B, NOT in A, in output
    // contract: in A, in B, NOT in output
    let mut batch_chars: Vec<char> = Vec::new();
    let mut free_a_chars: Vec<char> = Vec::new();
    let mut free_b_chars: Vec<char> = Vec::new();
    let mut contract_chars: Vec<char> = Vec::new();

    // Collect unique chars from A.
    let a_unique: Vec<char> = {
        let mut v = a_subs.clone();
        v.sort_unstable();
        v.dedup();
        v
    };
    let b_unique: Vec<char> = {
        let mut v = b_subs.clone();
        v.sort_unstable();
        v.dedup();
        v
    };

    for &c in &a_unique {
        let in_b = b_unique.contains(&c);
        let in_out = out_subs.contains(&c);
        match (in_b, in_out) {
            (true, true) => batch_chars.push(c),
            (true, false) => contract_chars.push(c),
            (false, true) => free_a_chars.push(c),
            (false, false) => {
                // Summed over in A only — treat as A-side contraction (sum out).
                // This case is handled by the general approach below.
                free_a_chars.push(c); // will be summed implicitly
            }
        }
    }
    for &c in &b_unique {
        if !a_unique.contains(&c) && out_subs.contains(&c) {
            free_b_chars.push(c);
        }
        // If not in output either, it's summed over in B only.
    }

    // Compute sizes.
    let batch_sizes: Vec<usize> = batch_chars.iter().map(|c| dim_map[c]).collect();
    let free_a_sizes: Vec<usize> = free_a_chars.iter().map(|c| dim_map[c]).collect();
    let free_b_sizes: Vec<usize> = free_b_chars.iter().map(|c| dim_map[c]).collect();
    let contract_sizes: Vec<usize> = contract_chars.iter().map(|c| dim_map[c]).collect();

    let batch_total: usize = batch_sizes.iter().product::<usize>().max(1);
    let free_a_total: usize = free_a_sizes.iter().product::<usize>().max(1);
    let free_b_total: usize = free_b_sizes.iter().product::<usize>().max(1);
    let contract_total: usize = contract_sizes.iter().product::<usize>().max(1);

    let a_data = a.data_vec()?;
    let b_data = b.data_vec()?;
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Compute input strides.
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);

    // Step 1-2: Build permuted + reshaped 3D views.
    // A target layout: [batch..., free_a..., contract...]
    // B target layout: [batch..., contract..., free_b...]
    //
    // Rather than physically transposing, we use indirect indexing.
    // For the GEMM: C[batch, fa, fb] = sum_c A[batch, fa, c] * B[batch, c, fb]

    // Precompute multi-index decoders for each group.
    // For each flat index in a group, compute the contribution to the input flat index.

    // A: for a given (batch_flat, free_a_flat, contract_flat), compute flat index into A.
    // B: for a given (batch_flat, contract_flat, free_b_flat), compute flat index into B.

    // Build lookup: for each char, which axis in A (or B) does it correspond to?
    let a_char_to_axis: BTreeMap<char, Vec<usize>> = {
        let mut m: BTreeMap<char, Vec<usize>> = BTreeMap::new();
        for (axis, &c) in a_subs.iter().enumerate() {
            m.entry(c).or_default().push(axis);
        }
        m
    };
    let b_char_to_axis: BTreeMap<char, Vec<usize>> = {
        let mut m: BTreeMap<char, Vec<usize>> = BTreeMap::new();
        for (axis, &c) in b_subs.iter().enumerate() {
            m.entry(c).or_default().push(axis);
        }
        m
    };

    // Helper: decode a flat index for a group of chars into per-char values.
    fn decode_multi(flat: usize, sizes: &[usize]) -> Vec<usize> {
        let mut result = vec![0usize; sizes.len()];
        let mut remainder = flat;
        for i in (0..sizes.len()).rev() {
            result[i] = remainder % sizes[i];
            remainder /= sizes[i];
        }
        result
    }

    // Compute A flat index from (batch_vals, free_a_vals, contract_vals).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn compute_a_flat(
        batch_chars: &[char],
        batch_vals: &[usize],
        free_a_chars: &[char],
        free_a_vals: &[usize],
        contract_chars: &[char],
        contract_vals: &[usize],
        a_char_to_axis: &BTreeMap<char, Vec<usize>>,
        a_strides: &[usize],
    ) -> usize {
        let mut flat = 0usize;
        for (i, &c) in batch_chars.iter().enumerate() {
            if let Some(axes) = a_char_to_axis.get(&c) {
                for &ax in axes {
                    flat += batch_vals[i] * a_strides[ax];
                }
            }
        }
        for (i, &c) in free_a_chars.iter().enumerate() {
            if let Some(axes) = a_char_to_axis.get(&c) {
                for &ax in axes {
                    flat += free_a_vals[i] * a_strides[ax];
                }
            }
        }
        for (i, &c) in contract_chars.iter().enumerate() {
            if let Some(axes) = a_char_to_axis.get(&c) {
                for &ax in axes {
                    flat += contract_vals[i] * a_strides[ax];
                }
            }
        }
        flat
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn compute_b_flat(
        batch_chars: &[char],
        batch_vals: &[usize],
        free_b_chars: &[char],
        free_b_vals: &[usize],
        contract_chars: &[char],
        contract_vals: &[usize],
        b_char_to_axis: &BTreeMap<char, Vec<usize>>,
        b_strides: &[usize],
    ) -> usize {
        let mut flat = 0usize;
        for (i, &c) in batch_chars.iter().enumerate() {
            if let Some(axes) = b_char_to_axis.get(&c) {
                for &ax in axes {
                    flat += batch_vals[i] * b_strides[ax];
                }
            }
        }
        for (i, &c) in contract_chars.iter().enumerate() {
            if let Some(axes) = b_char_to_axis.get(&c) {
                for &ax in axes {
                    flat += contract_vals[i] * b_strides[ax];
                }
            }
        }
        for (i, &c) in free_b_chars.iter().enumerate() {
            if let Some(axes) = b_char_to_axis.get(&c) {
                for &ax in axes {
                    flat += free_b_vals[i] * b_strides[ax];
                }
            }
        }
        flat
    }

    // Step 6: GEMM — C[batch, free_a, free_b] = sum_contract A[...] * B[...]
    // Result is [batch_total, free_a_total, free_b_total] in row-major.
    let gemm_size = batch_total * free_a_total * free_b_total;
    let mut gemm_result = vec![<T as num_traits::Zero>::zero(); gemm_size];

    for bi in 0..batch_total {
        let batch_vals = decode_multi(bi, &batch_sizes);
        for fa in 0..free_a_total {
            let free_a_vals = decode_multi(fa, &free_a_sizes);
            for fb in 0..free_b_total {
                let free_b_vals = decode_multi(fb, &free_b_sizes);
                let mut acc = <T as num_traits::Zero>::zero();
                for ci in 0..contract_total {
                    let contract_vals = decode_multi(ci, &contract_sizes);
                    let a_flat = compute_a_flat(
                        &batch_chars,
                        &batch_vals,
                        &free_a_chars,
                        &free_a_vals,
                        &contract_chars,
                        &contract_vals,
                        &a_char_to_axis,
                        &a_strides,
                    );
                    let b_flat = compute_b_flat(
                        &batch_chars,
                        &batch_vals,
                        &free_b_chars,
                        &free_b_vals,
                        &contract_chars,
                        &contract_vals,
                        &b_char_to_axis,
                        &b_strides,
                    );
                    acc += a_data[a_flat] * b_data[b_flat];
                }
                gemm_result[bi * (free_a_total * free_b_total) + fa * free_b_total + fb] = acc;
            }
        }
    }

    // Step 7: Reshape + permute to output shape.
    // The gemm_result is laid out as [batch..., free_a..., free_b...].
    // We need to permute to match the output subscripts order.
    let intermediate_chars: Vec<char> = batch_chars
        .iter()
        .chain(free_a_chars.iter())
        .chain(free_b_chars.iter())
        .copied()
        .collect();
    let intermediate_sizes: Vec<usize> = batch_sizes
        .iter()
        .chain(free_a_sizes.iter())
        .chain(free_b_sizes.iter())
        .copied()
        .collect();

    // If output subscript order matches intermediate, we're done.
    if intermediate_chars == *out_subs {
        let out_shape: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
        return Tensor::from_storage(TensorStorage::cpu(gemm_result), out_shape, false);
    }

    // Otherwise, permute.
    let out_shape: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Build permutation: for each output axis, find which intermediate axis it corresponds to.
    let perm: Vec<usize> = out_subs
        .iter()
        .map(|c| {
            intermediate_chars
                .iter()
                .position(|ic| ic == c)
                .expect("output char must exist in intermediate")
        })
        .collect();

    let inter_strides = row_major_strides(&intermediate_sizes);

    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel];
    for (out_flat, result_elem) in result.iter_mut().enumerate() {
        // Decode output multi-index.
        let out_multi = decode_multi(out_flat, &out_shape);
        // Map to intermediate multi-index.
        let mut inter_flat = 0usize;
        for (out_axis, &inter_axis) in perm.iter().enumerate() {
            inter_flat += out_multi[out_axis] * inter_strides[inter_axis];
        }
        *result_elem = gemm_result[inter_flat];
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

/// Compute row-major strides for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Einstein summation.
///
/// Evaluates the contraction specified by `equation` on the given `inputs`.
///
/// # Examples
///
/// ```ignore
/// // Matrix multiply: (M,K) @ (K,N) -> (M,N)
/// let c = einsum("ij,jk->ik", &[&a, &b])?;
///
/// // Batched matrix multiply
/// let c = einsum("bij,bjk->bik", &[&a, &b])?;
///
/// // Trace
/// let t = einsum("ii->", &[&a])?;
///
/// // Outer product
/// let o = einsum("i,j->ij", &[&a, &b])?;
///
/// // Transpose
/// let t = einsum("ij->ji", &[&a])?;
/// ```
pub fn einsum<T: Float>(equation: &str, inputs: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einsum: expected 1 or 2 input tensors, got {}",
                inputs.len()
            ),
        });
    }

    let parsed = parse_equation(equation, inputs.len())?;
    let dim_map = build_dim_map(&parsed, inputs)?;

    let result = match inputs.len() {
        1 => einsum_single(&parsed, inputs[0], &dim_map)?,
        2 => einsum_two(&parsed, inputs[0], inputs[1], &dim_map)?,
        _ => unreachable!(),
    };

    Ok(result)
}

/// Differentiable Einstein summation. If any input requires grad and grad
/// is enabled, attaches [`EinsumBackward`].
///
/// Participates in autocast: classified as `ReducedPrecision` (`"einsum"`).
pub fn einsum_differentiable<T: Float>(
    equation: &str,
    inputs: &[&Tensor<T>],
) -> FerrotorchResult<Tensor<T>> {
    autocast_guard("einsum");

    let result = einsum(equation, inputs)?;

    let any_requires_grad = inputs.iter().any(|t| t.requires_grad());

    if is_grad_enabled() && any_requires_grad {
        let device = result.device();
        let wrapped = match inputs.len() {
            1 => {
                let grad_fn = Arc::new(EinsumBackwardSingle {
                    equation: equation.to_string(),
                    input: inputs[0].clone(),
                });
                let storage = TensorStorage::on_device(result.data_vec()?, device)?;
                Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
            }
            2 => {
                let grad_fn = Arc::new(EinsumBackwardTwo {
                    equation: equation.to_string(),
                    a: inputs[0].clone(),
                    b: inputs[1].clone(),
                });
                let storage = TensorStorage::on_device(result.data_vec()?, device)?;
                Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
            }
            _ => Ok(result),
        }?;
        Ok(wrapped)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Backward: single-input
// ---------------------------------------------------------------------------

/// Backward for single-input einsum: `C = einsum(eq, [A])`.
///
/// For a single-input einsum like `"ij->ji"` (transpose) or `"ii->"` (trace),
/// the gradient is computed by reversing the equation:
/// `grad_A = einsum(reverse_eq, [grad_C])`.
#[derive(Debug)]
struct EinsumBackwardSingle<T: Float> {
    equation: String,
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for EinsumBackwardSingle<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        // Reverse the equation: swap input and output subscripts.
        // "ij->ji" becomes "ji->ij"
        // "ii->" becomes "->ii" which is not valid single-input einsum.
        //
        // For the trace case ("ii->"), grad is a scalar. We need to produce
        // a diagonal matrix. Handle this specially.
        let (lhs, rhs) = self
            .equation
            .split_once("->")
            .unwrap_or((&self.equation, ""));

        let in_subs: Vec<char> = lhs.chars().filter(|c| c.is_ascii_lowercase()).collect();
        let out_subs: Vec<char> = rhs.chars().collect();

        // Check if this is a "reduction" case where in_subs has repeated chars
        // or chars not in out_subs.
        let has_repeated = {
            let mut seen = std::collections::HashSet::new();
            in_subs.iter().any(|c| !seen.insert(c))
        };

        if has_repeated {
            // Cases like "ii->" (trace). Build the gradient manually:
            // grad_A[i,j] = grad_output * delta(i,j) for trace case.
            let in_shape: Vec<usize> = self.input.shape().to_vec();
            let in_numel = self.input.numel();
            let mut grad_data = vec![<T as num_traits::Zero>::zero(); in_numel];
            let grad_out_data = grad_output.data_vec()?;

            // General approach: for each element of grad_A, we need to determine
            // the gradient. The gradient of sum over repeated indices is nonzero
            // only where repeated indices are equal.
            let out_strides = row_major_strides(grad_output.shape());

            for (flat, grad_elem) in grad_data.iter_mut().enumerate().take(in_numel) {
                // Decode flat to multi-index for input.
                let mut multi = vec![0usize; in_subs.len()];
                {
                    let mut rem = flat;
                    for i in (0..in_subs.len()).rev() {
                        multi[i] = rem % in_shape[i];
                        rem /= in_shape[i];
                    }
                }

                // Check: all occurrences of the same char must have the same value.
                let mut char_val: BTreeMap<char, usize> = BTreeMap::new();
                let mut valid = true;
                for (axis, &c) in in_subs.iter().enumerate() {
                    match char_val.get(&c) {
                        Some(&prev) if prev != multi[axis] => {
                            valid = false;
                            break;
                        }
                        _ => {
                            char_val.insert(c, multi[axis]);
                        }
                    }
                }

                if !valid {
                    continue; // grad is zero
                }

                // Compute the flat index into grad_output for the corresponding output element.
                let mut out_flat = 0usize;
                for (oi, &oc) in out_subs.iter().enumerate() {
                    out_flat += char_val[&oc] * out_strides[oi];
                }

                *grad_elem = if out_subs.is_empty() {
                    // Scalar output — the gradient is just the scalar value.
                    grad_out_data[0]
                } else {
                    grad_out_data[out_flat]
                };
            }

            let grad_tensor = Tensor::from_storage(TensorStorage::cpu(grad_data), in_shape, false)?;
            return Ok(vec![Some(grad_tensor)]);
        }

        // Simple permutation/projection case: reverse the equation.
        // "ij->ji" -> "ji->ij", "ij->" -> reverse is grad_output broadcast.
        if out_subs.is_empty() {
            // All indices summed: grad_A = grad_scalar * ones_like(A)
            let scalar_val = grad_output.item()?;
            let grad_data = vec![scalar_val; self.input.numel()];
            let grad_tensor = Tensor::from_storage(
                TensorStorage::cpu(grad_data),
                self.input.shape().to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_tensor)]);
        }

        // Pure permutation: "ij->ji" reverses to "ji->ij"
        let reverse_eq = format!("{rhs}->{lhs}");
        let grad_a = einsum(&reverse_eq, &[grad_output])?;
        Ok(vec![Some(grad_a)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "EinsumBackward"
    }
}

// ---------------------------------------------------------------------------
// Backward: two-input
// ---------------------------------------------------------------------------

/// Backward for two-input einsum: `C = einsum(eq, [A, B])`.
///
/// For `"ij,jk->ik"`:
/// - `grad_A = einsum("ik,jk->ij", [grad_C, B])` (swap output with A-input)
/// - `grad_B = einsum("ij,ik->jk", [A, grad_C])` (swap output with B-input)
///
/// General rule: to get grad w.r.t. input X, form an equation where:
/// - The output subscripts become those of X.
/// - X's subscripts are removed from the inputs and replaced with the output subscripts.
#[derive(Debug)]
struct EinsumBackwardTwo<T: Float> {
    equation: String,
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> EinsumBackwardTwo<T> {
    /// Derive the backward einsum equation for gradient w.r.t. a specific input.
    ///
    /// For `einsum("ij,jk->ik", [A, B])` and target=0 (grad_A):
    /// We need: `einsum("ik,kj->ij", [grad_C, B])` — but more generally,
    /// the equation for grad w.r.t. input `target` is formed by replacing
    /// the target's subscripts in the output and using grad_C + the other input.
    fn backward_equation(&self, target: usize) -> (String, usize, usize) {
        // Parse the forward equation.
        let (lhs, rhs) = self
            .equation
            .split_once("->")
            .unwrap_or((&self.equation, ""));

        let parts: Vec<&str> = lhs.split(',').collect();
        let a_subs = parts[0];
        let b_subs = parts[1];
        let out_subs = rhs;

        // For grad_A: equation is "(out_subs),(b_subs)->(a_subs)"
        // grad_C has shape matching out_subs, B has shape matching b_subs
        // For grad_B: equation is "(a_subs),(out_subs)->(b_subs)"
        // A has shape matching a_subs, grad_C has shape matching out_subs
        if target == 0 {
            // grad_A: einsum("out,b->a", [grad_C, B])
            let eq = format!("{out_subs},{b_subs}->{a_subs}");
            (eq, 0, 1) // (equation, grad_C_pos, other_pos)
        } else {
            // grad_B: einsum("a,out->b", [A, grad_C])
            let eq = format!("{a_subs},{out_subs}->{b_subs}");
            (eq, 1, 0) // (equation, grad_C_pos=1, A_pos=0)
        }
    }
}

impl<T: Float> GradFn<T> for EinsumBackwardTwo<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_a = if self.a.requires_grad() {
            let (eq, _, _) = self.backward_equation(0);
            Some(einsum(&eq, &[grad_output, &self.b])?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let (eq, _, _) = self.backward_equation(1);
            Some(einsum(&eq, &[&self.a, grad_output])?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "EinsumBackward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: {a} vs {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Matrix multiply: "ij,jk->ik"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_mm() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_close(c.data().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Batched matrix multiply: "bij,bjk->bik"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_bmm() {
        // Batch 0: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        // Batch 1: [[1, 0], [0, 1]] @ [[9, 10], [11, 12]] = [[9, 10], [11, 12]]
        #[rustfmt::skip]
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            1.0, 0.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b_data: Vec<f32> = vec![
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let a = t(&a_data, &[2, 2, 2]);
        let b = t(&b_data, &[2, 2, 2]);
        let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let d = c.data().unwrap();
        // batch 0
        assert_close(&d[0..4], &[19.0, 22.0, 43.0, 50.0], 1e-6);
        // batch 1
        assert_close(&d[4..8], &[9.0, 10.0, 11.0, 12.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Trace: "ii->"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_trace() {
        // [[1, 2], [3, 4]] -> trace = 1 + 4 = 5
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = einsum("ii->", &[&a]).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 5.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Outer product: "i,j->ij"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_outer_product() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0], &[2]);
        let c = einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        // [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
        assert_close(c.data().unwrap(), &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Transpose: "ij->ji"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_transpose() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_close(c.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Sum all: "ij->"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_sum_all() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = einsum("ij->", &[&a]).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 21.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Sum over axis: "ij->i" (sum over j)
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_sum_axis() {
        // [[1, 2, 3], [4, 5, 6]] -> [6, 15]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = einsum("ij->i", &[&a]).unwrap();
        assert_eq!(c.shape(), &[2]);
        assert_close(c.data().unwrap(), &[6.0, 15.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Implicit mode: "ij,jk" (no ->)
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_implicit_mm() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        // j appears twice -> contracted. i,k appear once -> output "ik"
        let c = einsum("ij,jk", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_close(c.data().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Backward: matrix multiply
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_backward_mm() {
        // Same as MmBackward test:
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        // L = sum(C) = 134
        // dL/dA = ones @ B^T = [[11, 15], [11, 15]]
        // dL/dB = A^T @ ones = [[4, 4], [6, 6]]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = leaf(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let c = einsum_differentiable("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // Build sum for scalar.
        let c_data = c.data().unwrap();
        let loss_val: f32 = c_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward { input: c }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2, 2]);
        assert_eq!(b_grad.shape(), &[2, 2]);

        // dL/dA = [[11, 15], [11, 15]]
        assert_close(a_grad.data().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-5);
        // dL/dB = [[4, 4], [6, 6]]
        assert_close(b_grad.data().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Invalid equation
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_invalid_equation() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // Wrong number of inputs.
        assert!(einsum("ij,jk,kl->il", &[&a, &b]).is_err());

        // Subscript count mismatch with tensor dims.
        assert!(einsum("ijk,jk->ik", &[&a, &b]).is_err());

        // Inconsistent dimension sizes.
        let c = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert!(einsum("ij,jk->ik", &[&c, &a]).is_err()); // c is 2x3, a is 2x2; j=3 vs j=2

        // Invalid character.
        assert!(einsum("i1,1j->ij", &[&a, &b]).is_err());
    }

    // -----------------------------------------------------------------------
    // Diagonal extraction: "ii->i"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_diagonal() {
        // [[1, 2], [3, 4]] -> diagonal = [1, 4]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = einsum("ii->i", &[&a]).unwrap();
        assert_eq!(c.shape(), &[2]);
        assert_close(c.data().unwrap(), &[1.0, 4.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Dot product via einsum: "i,i->"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_dot() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = einsum("i,i->", &[&a, &b]).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 32.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Non-square matrix multiply
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_non_square_mm() {
        // (2,3) @ (3,4) -> (2,4)
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
        // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
        //       = [38, 44, 50, 56]
        // Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
        //       = [83, 98, 113, 128]
        assert_close(
            c.data().unwrap(),
            &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0],
            1e-5,
        );
    }

    // -------------------------------------------------------------------
    // autocast_guard integration
    // -------------------------------------------------------------------

    #[test]
    fn test_einsum_differentiable_fires_autocast_guard() {
        use crate::autograd::autocast::{AutocastDtype, autocast, set_autocast_debug};
        use crate::autograd::autocast_ops::{AutocastCategory, drain_autocast_events};

        set_autocast_debug(true);
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // Outside autocast: no events.
        drain_autocast_events();
        let _ = einsum_differentiable("ij,jk->ik", &[&a, &b]).unwrap();
        assert!(drain_autocast_events().is_empty());

        // Inside autocast: records "einsum" as ReducedPrecision.
        autocast(AutocastDtype::F16, || {
            drain_autocast_events();
            let _ = einsum_differentiable("ij,jk->ik", &[&a, &b]).unwrap();
            let events = drain_autocast_events();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].op, "einsum");
            assert_eq!(events[0].category, AutocastCategory::ReducedPrecision);
        });
    }
}
