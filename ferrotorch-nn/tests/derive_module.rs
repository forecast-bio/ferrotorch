//! Integration tests for `#[derive(Module)]`.

use ferrotorch_core::{Float, FerrotorchResult, Tensor};
// Importing `Module` brings in both the trait AND the derive macro (they
// live in different namespaces: type vs macro).
use ferrotorch_nn::{Linear, Module, Parameter};

// ---------------------------------------------------------------------------
// Test struct: params + submodules + skipped fields
// ---------------------------------------------------------------------------

#[derive(Module)]
struct TestModel<T: Float> {
    #[param]
    weight: Parameter<T>,
    #[submodule]
    linear: Linear<T>,
    #[skip]
    #[allow(dead_code)]
    hidden_size: usize,
    training: bool,
}

impl<T: Float> TestModel<T> {
    fn new(in_features: usize, out_features: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            weight: Parameter::zeros(&[in_features, out_features])?,
            linear: Linear::new(in_features, out_features, true)?,
            hidden_size: out_features,
            training: true,
        })
    }

    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.linear.forward(input)
    }
}

// ---------------------------------------------------------------------------
// Test struct: params only, no submodules
// ---------------------------------------------------------------------------

#[derive(Module)]
struct ParamsOnly<T: Float> {
    #[param]
    weight: Parameter<T>,
    #[param]
    bias: Parameter<T>,
    training: bool,
}

impl<T: Float> ParamsOnly<T> {
    fn new() -> FerrotorchResult<Self> {
        Ok(Self {
            weight: Parameter::zeros(&[4, 3])?,
            bias: Parameter::zeros(&[4])?,
            training: true,
        })
    }

    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Ok(input.clone())
    }
}

// ---------------------------------------------------------------------------
// Test struct: submodules only, no direct params
// ---------------------------------------------------------------------------

#[derive(Module)]
struct SubmodulesOnly<T: Float> {
    #[submodule]
    layer1: Linear<T>,
    #[submodule]
    layer2: Linear<T>,
    training: bool,
}

impl<T: Float> SubmodulesOnly<T> {
    fn new() -> FerrotorchResult<Self> {
        Ok(Self {
            layer1: Linear::new(4, 8, true)?,
            layer2: Linear::new(8, 2, false)?,
            training: true,
        })
    }

    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.layer1.forward(input)?;
        self.layer2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Test struct: empty (no params, no submodules)
// ---------------------------------------------------------------------------

#[derive(Module)]
struct EmptyModule<T: Float> {
    #[skip]
    _marker: std::marker::PhantomData<T>,
    training: bool,
}

impl<T: Float> EmptyModule<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Ok(input.clone())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn test_parameters_count() {
    let model = TestModel::<f32>::new(4, 3).unwrap();
    let params = model.parameters();
    // 1 direct #[param] (weight) + 2 from Linear (weight + bias)
    assert_eq!(params.len(), 3);
}

#[test]
fn test_parameters_mut_count() {
    let mut model = TestModel::<f32>::new(4, 3).unwrap();
    let params = model.parameters_mut();
    assert_eq!(params.len(), 3);
}

#[test]
fn test_named_parameters_keys() {
    let model = TestModel::<f32>::new(4, 3).unwrap();
    let named = model.named_parameters();

    let keys: Vec<&str> = named.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, &["weight", "linear.weight", "linear.bias"]);
}

#[test]
fn test_named_parameters_shapes() {
    let model = TestModel::<f32>::new(4, 3).unwrap();
    let named = model.named_parameters();

    // Direct param: weight [4, 3]
    assert_eq!(named[0].0, "weight");
    assert_eq!(named[0].1.shape(), &[4, 3]);

    // Linear weight: [out=3, in=4]
    assert_eq!(named[1].0, "linear.weight");
    assert_eq!(named[1].1.shape(), &[3, 4]);

    // Linear bias: [out=3]
    assert_eq!(named[2].0, "linear.bias");
    assert_eq!(named[2].1.shape(), &[3]);
}

#[test]
fn test_train_eval_toggles() {
    let mut model = TestModel::<f32>::new(4, 3).unwrap();

    assert!(model.is_training());

    model.eval();
    assert!(!model.is_training());
    // Submodule should also be in eval mode.
    assert!(!model.linear.is_training());

    model.train();
    assert!(model.is_training());
    assert!(model.linear.is_training());
}

#[test]
fn test_skip_field_not_in_parameters() {
    let model = TestModel::<f32>::new(4, 3).unwrap();
    // hidden_size is #[skip] — should not appear anywhere.
    let named = model.named_parameters();
    for (key, _) in &named {
        assert!(
            !key.contains("hidden_size"),
            "skipped field should not appear in named_parameters"
        );
    }
}

#[test]
fn test_state_dict_roundtrip() {
    let model = TestModel::<f32>::new(4, 3).unwrap();
    let sd = model.state_dict();

    assert!(sd.contains_key("weight"));
    assert!(sd.contains_key("linear.weight"));
    assert!(sd.contains_key("linear.bias"));
    assert_eq!(sd.len(), 3);
}

#[test]
fn test_params_only_struct() {
    let model = ParamsOnly::<f32>::new().unwrap();

    let params = model.parameters();
    assert_eq!(params.len(), 2);

    let named = model.named_parameters();
    assert_eq!(named[0].0, "weight");
    assert_eq!(named[1].0, "bias");
}

#[test]
fn test_submodules_only_struct() {
    let model = SubmodulesOnly::<f32>::new().unwrap();

    // layer1: Linear(4,8,bias=true) -> 2 params (weight + bias)
    // layer2: Linear(8,2,bias=false) -> 1 param (weight only)
    let params = model.parameters();
    assert_eq!(params.len(), 3);

    let named = model.named_parameters();
    let keys: Vec<&str> = named.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        &["layer1.weight", "layer1.bias", "layer2.weight"]
    );
}

#[test]
fn test_submodules_only_train_eval_propagates() {
    let mut model = SubmodulesOnly::<f32>::new().unwrap();

    model.eval();
    assert!(!model.is_training());
    assert!(!model.layer1.is_training());
    assert!(!model.layer2.is_training());

    model.train();
    assert!(model.is_training());
    assert!(model.layer1.is_training());
    assert!(model.layer2.is_training());
}

#[test]
fn test_empty_module() {
    let model = EmptyModule::<f32> {
        _marker: std::marker::PhantomData,
        training: true,
    };

    assert_eq!(model.parameters().len(), 0);
    assert_eq!(model.named_parameters().len(), 0);
    assert!(model.is_training());
}

#[test]
fn test_empty_module_train_eval() {
    let mut model = EmptyModule::<f32> {
        _marker: std::marker::PhantomData,
        training: true,
    };

    model.eval();
    assert!(!model.is_training());
    model.train();
    assert!(model.is_training());
}

#[test]
fn test_f64_generic() {
    let model = ParamsOnly::<f64> {
        weight: Parameter::zeros(&[2, 2]).unwrap(),
        bias: Parameter::zeros(&[2]).unwrap(),
        training: true,
    };

    assert_eq!(model.parameters().len(), 2);
    assert!(model.is_training());
}

#[test]
fn test_load_state_dict_via_derived_module() {
    let model = TestModel::<f32>::new(4, 3).unwrap();
    let sd = model.state_dict();

    let mut model2 = TestModel::<f32>::new(4, 3).unwrap();
    model2.load_state_dict(&sd, true).unwrap();

    // Verify parameter shapes match after load.
    let named1 = model.named_parameters();
    let named2 = model2.named_parameters();
    assert_eq!(named1.len(), named2.len());
    for ((k1, p1), (k2, p2)) in named1.iter().zip(named2.iter()) {
        assert_eq!(k1, k2);
        assert_eq!(p1.shape(), p2.shape());
    }
}

#[test]
fn test_derived_module_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TestModel<f32>>();
    assert_send_sync::<ParamsOnly<f32>>();
    assert_send_sync::<SubmodulesOnly<f32>>();
    assert_send_sync::<EmptyModule<f32>>();
}
