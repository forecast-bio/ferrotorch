//! Integration tests for the `#[script]` attribute macro.

use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{FerrotorchResult, Tensor};
use ferrotorch_jit::TracedModule;
use ferrotorch_jit_script::script;

fn t1d(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

#[script]
fn weighted_sum(a: Tensor<f32>, w: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let prod = mul(&a, &w)?;
    sum(&prod)
}

#[script]
fn three_arg_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let ab = add(&a, &b)?;
    add(&ab, &c)
}

#[test]
fn script_macro_produces_traced_module() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let w = t1d(&[4.0, 5.0, 6.0]);
    let module: TracedModule<f32> = weighted_sum(a, w).unwrap();

    // Re-execute the captured graph with fresh inputs.
    let a2 = t1d(&[1.0, 2.0, 3.0]);
    let w2 = t1d(&[4.0, 5.0, 6.0]);
    let result = module.forward_multi(&[a2, w2]).unwrap();
    // sum(1*4 + 2*5 + 3*6) = sum(4 + 10 + 18) = 32
    assert_eq!(result.data().unwrap(), &[32.0]);
}

#[test]
fn script_macro_three_args() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    let c = t1d(&[5.0, 6.0]);
    let module: TracedModule<f32> = three_arg_add(a, b, c).unwrap();

    let result = module
        .forward_multi(&[t1d(&[1.0, 2.0]), t1d(&[3.0, 4.0]), t1d(&[5.0, 6.0])])
        .unwrap();
    assert_eq!(result.data().unwrap(), &[9.0, 12.0]);
}

#[test]
fn script_macro_module_save_load_roundtrip() {
    let a = t1d(&[2.0, 3.0]);
    let w = t1d(&[4.0, 5.0]);
    let module = weighted_sum(a, w).unwrap();
    let bytes = module.to_bytes();
    let loaded: TracedModule<f32> = TracedModule::<f32>::from_bytes(&bytes).unwrap();
    let r = loaded
        .forward_multi(&[t1d(&[2.0, 3.0]), t1d(&[4.0, 5.0])])
        .unwrap();
    // 2*4 + 3*5 = 23
    assert_eq!(r.data().unwrap(), &[23.0]);
}
