use crate::error::{FerrotorchError, FerrotorchResult};

/// Compute the broadcasted shape of two shapes, following NumPy/PyTorch rules.
///
/// Shapes are aligned from the right. Dimensions are compatible when they are
/// equal, or one of them is 1. The output dimension is the maximum of the two.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> FerrotorchResult<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "cannot broadcast shapes {:?} and {:?}: dimension mismatch at axis {} ({} vs {})",
                    a,
                    b,
                    max_ndim - 1 - i,
                    da,
                    db
                ),
            });
        }
    }

    result.reverse();
    Ok(result)
}

/// Total number of elements for a given shape.
#[inline]
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Compute C-contiguous (row-major) strides for a given shape.
pub fn c_contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1isize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

/// Compute channels-last (NHWC) strides for a 4D shape `[N, C, H, W]`.
///
/// The physical memory order becomes `[N, H, W, C]`, so the strides for
/// the logical NCHW dimensions are `[H*W*C, 1, W*C, C]`.
///
/// [CL-309] WU-05: channels-last memory format support
pub fn channels_last_strides(shape: &[usize]) -> Vec<isize> {
    debug_assert_eq!(shape.len(), 4, "channels_last_strides requires a 4D shape");
    let [_n, c, h, w] = [shape[0], shape[1], shape[2], shape[3]];
    vec![
        (h * w * c) as isize, // N stride
        1,                    // C stride (innermost in NHWC)
        (w * c) as isize,     // H stride
        c as isize,           // W stride
    ]
}

/// Compute channels-last-3d (NDHWC) strides for a 5D shape `[N, C, D, H, W]`.
///
/// The physical memory order becomes `[N, D, H, W, C]`, so the strides for
/// the logical NCDHW dimensions are `[D*H*W*C, 1, H*W*C, W*C, C]`.
///
/// [CL-309] WU-05: channels-last memory format support
pub fn channels_last_3d_strides(shape: &[usize]) -> Vec<isize> {
    debug_assert_eq!(
        shape.len(),
        5,
        "channels_last_3d_strides requires a 5D shape"
    );
    let [_n, c, d, h, w] = [shape[0], shape[1], shape[2], shape[3], shape[4]];
    vec![
        (d * h * w * c) as isize, // N stride
        1,                        // C stride (innermost in NDHWC)
        (h * w * c) as isize,     // D stride
        (w * c) as isize,         // H stride
        c as isize,               // W stride
    ]
}

/// Normalize a possibly-negative axis index to a positive one.
///
/// For a tensor with `ndim` dimensions, axis `-1` maps to `ndim - 1`, etc.
pub fn normalize_axis(axis: isize, ndim: usize) -> FerrotorchResult<usize> {
    let ndim_i = ndim as isize;
    if axis >= ndim_i || axis < -ndim_i {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("axis {axis} is out of bounds for tensor with {ndim} dimensions"),
        });
    }
    if axis < 0 {
        Ok((ndim_i + axis) as usize)
    } else {
        Ok(axis as usize)
    }
}

/// Check that two shapes are identical, returning an error if not.
pub fn check_shapes_match(a: &[usize], b: &[usize], op: &str) -> FerrotorchResult<()> {
    if a != b {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("{op}: shapes {a:?} and {b:?} do not match"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_same() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_scalar() {
        assert_eq!(broadcast_shapes(&[2, 3], &[]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[], &[4, 5]).unwrap(), vec![4, 5]);
    }

    #[test]
    fn test_broadcast_expand() {
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]).unwrap(), vec![2, 3]);
        assert_eq!(
            broadcast_shapes(&[5, 1, 4], &[3, 1]).unwrap(),
            vec![5, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_different_ndim() {
        assert_eq!(broadcast_shapes(&[3], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[2, 3], &[3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
        assert!(broadcast_shapes(&[3], &[4]).is_err());
    }

    #[test]
    fn test_c_contiguous_strides() {
        assert_eq!(c_contiguous_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(c_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(c_contiguous_strides(&[]), Vec::<isize>::new());
    }

    #[test]
    fn test_normalize_axis() {
        assert_eq!(normalize_axis(0, 3).unwrap(), 0);
        assert_eq!(normalize_axis(2, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-3, 3).unwrap(), 0);
        assert!(normalize_axis(3, 3).is_err());
        assert!(normalize_axis(-4, 3).is_err());
    }

    #[test]
    fn test_numel() {
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[0, 5]), 0);
    }

    #[test]
    fn test_channels_last_strides() {
        // [N=1, C=3, H=4, W=5] -> NHWC strides: [H*W*C, 1, W*C, C] = [60, 1, 15, 3]
        assert_eq!(channels_last_strides(&[1, 3, 4, 5]), vec![60, 1, 15, 3]);
        // [N=2, C=3, H=2, W=2] -> [12, 1, 6, 3]
        assert_eq!(channels_last_strides(&[2, 3, 2, 2]), vec![12, 1, 6, 3]);
    }

    #[test]
    fn test_channels_last_3d_strides() {
        // [N=1, C=2, D=3, H=4, W=5] -> NDHWC: [D*H*W*C, 1, H*W*C, W*C, C]
        //   = [120, 1, 40, 10, 2]
        assert_eq!(
            channels_last_3d_strides(&[1, 2, 3, 4, 5]),
            vec![120, 1, 40, 10, 2]
        );
    }
}
