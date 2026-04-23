//! Display formatting for Tensor, matching PyTorch's output style.

use std::fmt;

use crate::dtype::Float;
use crate::tensor::Tensor;

impl<T: Float> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape();
        // Use data_vec() so non-contiguous and GPU tensors display correctly.
        let data = match self.data_vec() {
            Ok(d) => d,
            Err(_) => return write!(f, "tensor(<inaccessible>, shape={shape:?})"),
        };

        // Scalar
        if shape.is_empty() {
            if data.is_empty() {
                return write!(f, "tensor([], shape=[])");
            }
            let val = data[0];
            write!(f, "tensor({val}")?;
            if self.grad_fn().is_some() {
                write!(f, ", grad_fn=<{}>", self.grad_fn().unwrap().name())?;
            } else if self.requires_grad() {
                write!(f, ", requires_grad=true")?;
            }
            return write!(f, ")");
        }

        // 1-D
        if shape.len() == 1 {
            write!(f, "tensor([")?;
            let max_show = 6;
            let len = shape[0];
            if len <= max_show {
                for (i, &v) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v:.4}")?;
                }
            } else {
                for (i, val) in data.iter().enumerate().take(3) {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val:.4}")?;
                }
                write!(f, ", ..., ")?;
                for (i, val) in data.iter().enumerate().skip(len - 3) {
                    if i > len - 3 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val:.4}")?;
                }
            }
            write!(f, "]")?;
        }
        // 2-D
        else if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];
            let max_rows = 6;
            write!(f, "tensor([")?;

            let display_row = |f: &mut fmt::Formatter<'_>, row: usize| -> fmt::Result {
                write!(f, "[")?;
                let max_cols = 6;
                if cols <= max_cols {
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", data[row * cols + c])?;
                    }
                } else {
                    for c in 0..3 {
                        if c > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", data[row * cols + c])?;
                    }
                    write!(f, ", ..., ")?;
                    for c in (cols - 3)..cols {
                        if c > cols - 3 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", data[row * cols + c])?;
                    }
                }
                write!(f, "]")
            };

            if rows <= max_rows {
                for r in 0..rows {
                    if r > 0 {
                        write!(f, ",\n        ")?;
                    }
                    display_row(f, r)?;
                }
            } else {
                for r in 0..3 {
                    if r > 0 {
                        write!(f, ",\n        ")?;
                    }
                    display_row(f, r)?;
                }
                write!(f, ",\n        ...")?;
                for r in (rows - 3)..rows {
                    write!(f, ",\n        ")?;
                    display_row(f, r)?;
                }
            }
            write!(f, "]")?;
        }
        // 3-D+: summary
        else {
            let numel = self.numel();
            write!(f, "tensor(<{numel} elements>, shape={shape:?}")?;
            if self.grad_fn().is_some() {
                write!(f, ", grad_fn=<{}>", self.grad_fn().unwrap().name())?;
            } else if self.requires_grad() {
                write!(f, ", requires_grad=true")?;
            }
            return write!(f, ")");
        }

        // Suffix metadata for 1D/2D
        if self.grad_fn().is_some() {
            write!(f, ", grad_fn=<{}>", self.grad_fn().unwrap().name())?;
        } else if self.requires_grad() {
            write!(f, ", requires_grad=true")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary test display value, not π.
    fn test_display_scalar() {
        let t = scalar(3.14f32).unwrap();
        let s = format!("{t}");
        assert!(s.contains("3.14"));
        assert!(s.starts_with("tensor("));
    }

    #[test]
    fn test_display_1d() {
        let t = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let s = format!("{t}");
        assert!(s.contains("1.0000"));
        assert!(s.contains("3.0000"));
    }

    #[test]
    fn test_display_2d() {
        let t = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let s = format!("{t}");
        assert!(s.contains("[1.0000, 2.0000]"));
    }

    #[test]
    fn test_display_with_grad_fn() {
        let a = scalar(2.0f32).unwrap().requires_grad_(true);
        let b = scalar(3.0f32).unwrap().requires_grad_(true);
        let c = (&a + &b).unwrap();
        let s = format!("{c}");
        assert!(s.contains("grad_fn=<AddBackward>"));
    }

    #[test]
    fn test_display_requires_grad() {
        let t = scalar(1.0f32).unwrap().requires_grad_(true);
        let s = format!("{t}");
        assert!(s.contains("requires_grad=true"));
    }

    #[test]
    fn test_display_large_1d_truncated() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let t = from_vec(data, &[100]).unwrap();
        let s = format!("{t}");
        assert!(s.contains("..."));
    }

    #[test]
    fn test_display_3d_summary() {
        let t = zeros::<f32>(&[2, 3, 4]).unwrap();
        let s = format!("{t}");
        assert!(s.contains("24 elements"));
        assert!(s.contains("shape=[2, 3, 4]"));
    }
}
