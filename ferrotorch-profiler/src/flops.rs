//! FLOPS estimation for common tensor operations.
//!
//! Each estimator counts multiply-accumulates (MACs) as 2 FLOPs (one
//! multiply + one add), matching the convention used by `PyTorch`'s
//! `torch.profiler` and the TensorFlow Profiler. The estimates are
//! shape-driven and ignore numerical effects (e.g. masked entries in
//! attention, sparse zeros) — they represent the dense compute cost.
//!
//! When in doubt, the estimator returns `None`. Callers can sum the
//! defined estimates to get a lower bound on total FLOPs. CL-333.

/// Estimate the number of floating-point operations for an op given
/// its name and input shapes.
///
/// Each multiply-accumulate (MAC) is counted as 2 FLOPs, matching the
/// convention used by `PyTorch`'s profiler and `TensorFlow` Profiler.
/// The estimates are shape-driven and represent dense compute cost,
/// ignoring numerical effects such as sparsity or masking.
///
/// # Returns
///
/// Returns `Some(flops)` when the op is recognized and the shapes are
/// sufficient to compute an estimate. Returns `None` when:
/// - The op name is not recognized.
/// - The input shapes are missing or have the wrong rank for the op.
/// - The FLOP count depends on runtime values not available from shapes
///   (e.g. number of unique elements for a `unique()` op).
///
/// # Examples
///
/// ```
/// use ferrotorch_profiler::flops;
///
/// // 2D matrix multiply [4, 5] @ [5, 6]: 2 * 4 * 6 * 5 = 240 FLOPs
/// assert_eq!(flops::estimate("matmul", &[vec![4, 5], vec![5, 6]]), Some(240));
///
/// // Unrecognized op
/// assert_eq!(flops::estimate("my_custom_op", &[vec![3, 4]]), None);
/// ```
#[must_use]
pub fn estimate(op_name: &str, input_shapes: &[Vec<usize>]) -> Option<u64> {
    match op_name {
        // Elementwise binary ops: one FLOP per output element.
        "add" | "sub" | "mul" | "div" => elementwise_binary(input_shapes),
        // Elementwise unary ops: one FLOP per input element.
        "neg" | "abs" | "sqrt" | "exp" | "log" => elementwise_unary(input_shapes),
        // Activations: roughly one FLOP per element (sigmoid/tanh are
        // a few but the order of magnitude is the same and we don't
        // try to be precise about it).
        "relu" | "sigmoid" | "tanh" | "gelu" | "silu" | "leaky_relu" => {
            elementwise_unary(input_shapes)
        }
        // Softmax: 3 FLOPs per element (max, exp, divide) plus the
        // reduction. Approximate as 5 * numel.
        "softmax" | "log_softmax" => {
            let n = numel(input_shapes.first()?);
            Some(5 * n as u64)
        }
        // Reductions: sum/mean require N adds for N elements; output
        // is one element (or smaller along reduced dim).
        "sum" | "mean" | "prod" => {
            let n = numel(input_shapes.first()?);
            Some(n.saturating_sub(1) as u64)
        }
        // Power: numel multiplies for integer powers, ~10 FLOPs for
        // floating-point exponents (transcendental). Use 2 as a
        // conservative estimate.
        "pow" => {
            let n = numel(input_shapes.first()?);
            Some(2 * n as u64)
        }
        // Linalg matmul: 2 * M * N * K MACs.
        "matmul" | "mm" | "bmm" | "linear" => matmul_flops(input_shapes),
        // Convolutions: 2 * C_out * C_in * K * spatial output size MACs.
        "conv1d" => conv_nd_flops(input_shapes, 1),
        "conv2d" => conv_nd_flops(input_shapes, 2),
        "conv3d" => conv_nd_flops(input_shapes, 3),
        // Norm layers: a few FLOPs per element. Approximate.
        "layer_norm" | "rms_norm" | "batch_norm" | "group_norm" => {
            let n = numel(input_shapes.first()?);
            Some(8 * n as u64)
        }
        _ => None,
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

fn elementwise_binary(shapes: &[Vec<usize>]) -> Option<u64> {
    if shapes.len() < 2 {
        return None;
    }
    // Output is the broadcast of the two inputs. We approximate by
    // taking the max numel; for true broadcasting the output is
    // exactly that.
    let n = numel(&shapes[0]).max(numel(&shapes[1]));
    Some(n as u64)
}

fn elementwise_unary(shapes: &[Vec<usize>]) -> Option<u64> {
    let n = numel(shapes.first()?);
    Some(n as u64)
}

/// Matmul FLOPS = 2 * M * N * K. Handles 1D dot, 2D mm, batched bmm,
/// and N-D matmul where the last two dims are M, K and K, N.
fn matmul_flops(shapes: &[Vec<usize>]) -> Option<u64> {
    if shapes.len() < 2 {
        return None;
    }
    let a = &shapes[0];
    let b = &shapes[1];
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (m, k1) = match a.len() {
        1 => (1usize, a[0]),
        n => (a[n - 2], a[n - 1]),
    };
    let (k2, n_dim) = match b.len() {
        1 => (b[0], 1usize),
        n => (b[n - 2], b[n - 1]),
    };
    if k1 != k2 {
        return None;
    }
    // Batch dims are everything except the last two for ndim >= 2.
    let batch_a: usize = if a.len() > 2 {
        a[..a.len() - 2].iter().product()
    } else {
        1
    };
    let batch_b: usize = if b.len() > 2 {
        b[..b.len() - 2].iter().product()
    } else {
        1
    };
    let batch = batch_a.max(batch_b);
    Some(2 * batch as u64 * m as u64 * n_dim as u64 * k1 as u64)
}

/// Conv FLOPS = 2 * `C_out` * `C_in` * (kernel volume) * (output spatial
/// volume). The estimator infers the output spatial size from the
/// input spatial size assuming kernel/stride/padding are part of the
/// op tag — for the standard ops we approximate by using the input
/// spatial dims (correct for stride=1, `padding='same'`).
fn conv_nd_flops(shapes: &[Vec<usize>], n_spatial: usize) -> Option<u64> {
    if shapes.len() < 2 {
        return None;
    }
    let input = &shapes[0];
    let weight = &shapes[1];
    // input shape: [B, C_in, *spatial]
    // weight shape: [C_out, C_in, *kernel]
    if input.len() != 2 + n_spatial || weight.len() != 2 + n_spatial {
        return None;
    }
    let batch = input[0];
    let c_in = input[1];
    let c_out = weight[0];
    let kernel_vol: usize = weight[2..].iter().product();
    let spatial_vol: usize = input[2..].iter().product();
    Some(2 * batch as u64 * c_out as u64 * c_in as u64 * kernel_vol as u64 * spatial_vol as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_add_2d() {
        let shapes = vec![vec![3, 4], vec![3, 4]];
        // 12 output elements, 1 FLOP each = 12 FLOPS.
        assert_eq!(estimate("add", &shapes), Some(12));
    }

    #[test]
    fn test_elementwise_with_broadcast() {
        let shapes = vec![vec![3, 4], vec![1, 4]];
        // Output is [3, 4] = 12 elements.
        assert_eq!(estimate("add", &shapes), Some(12));
    }

    #[test]
    fn test_unary_relu() {
        let shapes = vec![vec![5, 6]];
        assert_eq!(estimate("relu", &shapes), Some(30));
    }

    #[test]
    fn test_softmax_approx() {
        let shapes = vec![vec![2, 5]];
        // 5 * 10 = 50 FLOPS approximation
        assert_eq!(estimate("softmax", &shapes), Some(50));
    }

    #[test]
    fn test_matmul_2d_2d() {
        // [4, 5] @ [5, 6] -> [4, 6]
        // FLOPS = 2 * 4 * 6 * 5 = 240
        let shapes = vec![vec![4, 5], vec![5, 6]];
        assert_eq!(estimate("matmul", &shapes), Some(240));
    }

    #[test]
    fn test_matmul_batched() {
        // [3, 4, 5] @ [3, 5, 6] -> [3, 4, 6]
        // FLOPS = 3 * 2 * 4 * 6 * 5 = 720
        let shapes = vec![vec![3, 4, 5], vec![3, 5, 6]];
        assert_eq!(estimate("bmm", &shapes), Some(720));
    }

    #[test]
    fn test_matmul_dot_1d() {
        // [5] @ [5] -> scalar
        // FLOPS = 2 * 1 * 1 * 5 = 10
        let shapes = vec![vec![5], vec![5]];
        assert_eq!(estimate("matmul", &shapes), Some(10));
    }

    #[test]
    fn test_matmul_inner_mismatch_returns_none() {
        let shapes = vec![vec![4, 5], vec![6, 7]];
        assert_eq!(estimate("matmul", &shapes), None);
    }

    #[test]
    fn test_conv2d() {
        // Input [1, 3, 32, 32], weight [16, 3, 3, 3]
        // FLOPS = 2 * 1 * 16 * 3 * 9 * 1024 = 884736
        let shapes = vec![vec![1, 3, 32, 32], vec![16, 3, 3, 3]];
        assert_eq!(estimate("conv2d", &shapes), Some(884_736));
    }

    #[test]
    fn test_conv1d() {
        // Input [1, 4, 100], weight [8, 4, 3]
        // FLOPS = 2 * 1 * 8 * 4 * 3 * 100 = 19200
        let shapes = vec![vec![1, 4, 100], vec![8, 4, 3]];
        assert_eq!(estimate("conv1d", &shapes), Some(19200));
    }

    #[test]
    fn test_unknown_op_returns_none() {
        let shapes = vec![vec![3, 4]];
        assert_eq!(estimate("totally_made_up_op", &shapes), None);
    }

    #[test]
    fn test_no_shapes_returns_none() {
        assert_eq!(estimate("add", &[]), None);
        assert_eq!(estimate("relu", &[]), None);
    }

    #[test]
    fn test_layer_norm_estimate() {
        let shapes = vec![vec![32, 768]];
        // 8 FLOPS per element * 32 * 768 = 196608
        assert_eq!(estimate("layer_norm", &shapes), Some(196_608));
    }

    #[test]
    fn test_sum_reduction() {
        // sum of 100 elements = 99 adds
        let shapes = vec![vec![10, 10]];
        assert_eq!(estimate("sum", &shapes), Some(99));
    }
}
