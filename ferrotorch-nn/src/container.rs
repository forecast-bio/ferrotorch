//! Container modules: [`Sequential`], [`ModuleList`], and [`ModuleDict`].
//!
//! These mirror PyTorch's `nn.Sequential`, `nn.ModuleList`, and
//! `nn.ModuleDict`. They hold sub-modules and propagate `parameters()`,
//! `train()`/`eval()`, and `state_dict()` to all children.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::module::Module;
use crate::parameter::Parameter;

// ===========================================================================
// Sequential
// ===========================================================================

/// A sequential container that chains modules in order.
///
/// The `forward()` method feeds the output of each layer as the input to the
/// next, matching PyTorch's `nn.Sequential` semantics.
///
/// # Named parameters
///
/// Parameters are prefixed by layer index: `"0.weight"`, `"0.bias"`,
/// `"1.weight"`, etc. — matching PyTorch's convention.
///
/// # Examples
///
/// ```ignore
/// let model = Sequential::new(vec![
///     Box::new(Linear::<f32>::new(784, 256, true)?),
///     Box::new(ReLU::new()),
///     Box::new(Linear::<f32>::new(256, 10, true)?),
/// ]);
/// let output = model.forward(&input)?;
/// ```
pub struct Sequential<T: Float> {
    layers: Vec<Box<dyn Module<T>>>,
    training: bool,
}

impl<T: Float> Sequential<T> {
    /// Create a new sequential container from an ordered list of modules.
    pub fn new(layers: Vec<Box<dyn Module<T>>>) -> Self {
        Self {
            layers,
            training: true,
        }
    }

    /// Append a module to the end of the sequence.
    pub fn push(&mut self, layer: Box<dyn Module<T>>) {
        self.layers.push(layer);
    }

    /// Number of layers.
    #[inline]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the container is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<T: Float> Module<T> for Sequential<T> {
    /// Forward pass: chains each layer's forward in order.
    ///
    /// Returns an error if there are no layers, or if any layer's forward
    /// fails.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if self.layers.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "Sequential: cannot forward through empty container".into(),
            });
        }

        let mut output = self.layers[0].forward(input)?;
        for layer in &self.layers[1..] {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| {
                layer
                    .named_parameters()
                    .into_iter()
                    .map(move |(name, param)| (format!("{i}.{name}"), param))
            })
            .collect()
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl<T: Float> std::fmt::Display for Sequential<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sequential(")?;
        for (i, _layer) in self.layers.iter().enumerate() {
            writeln!(f, "  ({i}): <module>")?;
        }
        write!(f, ")")
    }
}

// ===========================================================================
// ModuleList
// ===========================================================================

/// An ordered list of modules, registered for parameter tracking.
///
/// Unlike [`Sequential`], `ModuleList` does **not** define a forward pass.
/// Users iterate over the list manually and call each module's `forward()`
/// as needed. This mirrors PyTorch's `nn.ModuleList`.
///
/// # Named parameters
///
/// Parameters are prefixed by list index: `"0.weight"`, `"1.weight"`, etc.
///
/// # Examples
///
/// ```ignore
/// # use ferrotorch_core::FerrotorchError;
/// fn example(input: &Tensor<f32>) -> Result<Tensor<f32>, FerrotorchError> {
///     let list = ModuleList::<f32>::new(vec![
///         Box::new(Linear::<f32>::new(10, 10, true)?),
///         Box::new(Linear::<f32>::new(10, 10, true)?),
///     ]);
///
///     let mut x = input.clone();
///     for i in 0..list.len() {
///         let module = list.get(i).ok_or_else(|| FerrotorchError::InvalidArgument {
///             message: format!("ModuleList index {i} out of bounds"),
///         })?;
///         x = module.forward(&x)?;
///     }
///     Ok(x)
/// }
/// ```
pub struct ModuleList<T: Float> {
    modules: Vec<Box<dyn Module<T>>>,
    training: bool,
}

impl<T: Float> ModuleList<T> {
    /// Create a new module list from a vector of modules.
    pub fn new(modules: Vec<Box<dyn Module<T>>>) -> Self {
        Self {
            modules,
            training: true,
        }
    }

    /// Create an empty module list.
    pub fn empty() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Get a reference to the module at the given index.
    pub fn get(&self, index: usize) -> Option<&dyn Module<T>> {
        self.modules.get(index).map(|m| m.as_ref())
    }

    /// Get a mutable reference to the module at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut dyn Module<T>> {
        match self.modules.get_mut(index) {
            Some(m) => Some(m.as_mut()),
            None => None,
        }
    }

    /// Append a module to the end of the list.
    pub fn push(&mut self, module: Box<dyn Module<T>>) {
        self.modules.push(module);
    }

    /// Number of modules.
    #[inline]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether the list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl<T: Float> Module<T> for ModuleList<T> {
    /// ModuleList does not implement forward.
    ///
    /// Users should iterate manually and call each module's `forward()`.
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "ModuleList does not implement forward. \
                      Iterate over the list and call each module's forward() manually."
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.modules
            .iter_mut()
            .flat_map(|m| m.parameters_mut())
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.modules
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                m.named_parameters()
                    .into_iter()
                    .map(move |(name, param)| (format!("{i}.{name}"), param))
            })
            .collect()
    }

    fn train(&mut self) {
        self.training = true;
        for m in &mut self.modules {
            m.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for m in &mut self.modules {
            m.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl<T: Float> std::fmt::Display for ModuleList<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ModuleList(")?;
        for (i, _m) in self.modules.iter().enumerate() {
            writeln!(f, "  ({i}): <module>")?;
        }
        write!(f, ")")
    }
}

// ===========================================================================
// ModuleDict
// ===========================================================================

/// An ordered dictionary of named modules, registered for parameter tracking.
///
/// Uses a `Vec<(String, Box<dyn Module<T>>)>` internally to preserve
/// insertion order without requiring an external dependency like `IndexMap`.
///
/// Like [`ModuleList`], `ModuleDict` does **not** define a forward pass.
/// Users look up modules by key and call `forward()` manually. This mirrors
/// PyTorch's `nn.ModuleDict`.
///
/// # Named parameters
///
/// Parameters are prefixed by their dictionary key: `"encoder.weight"`,
/// `"decoder.weight"`, etc.
///
/// # Examples
///
/// ```ignore
/// # use ferrotorch_core::FerrotorchError;
/// fn example(input: &Tensor<f32>) -> Result<Tensor<f32>, FerrotorchError> {
///     let mut dict = ModuleDict::<f32>::new();
///     dict.insert("encoder", Box::new(Linear::<f32>::new(784, 256, true)?));
///     dict.insert("decoder", Box::new(Linear::<f32>::new(256, 784, true)?));
///
///     let encoder = dict.get("encoder").ok_or_else(|| FerrotorchError::InvalidArgument {
///         message: "missing 'encoder' module".into(),
///     })?;
///     let decoder = dict.get("decoder").ok_or_else(|| FerrotorchError::InvalidArgument {
///         message: "missing 'decoder' module".into(),
///     })?;
///     let encoded = encoder.forward(input)?;
///     decoder.forward(&encoded)
/// }
/// ```
pub struct ModuleDict<T: Float> {
    entries: Vec<(String, Box<dyn Module<T>>)>,
    training: bool,
}

impl<T: Float> ModuleDict<T> {
    /// Create an empty module dict.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            training: true,
        }
    }

    /// Insert a module with the given key.
    ///
    /// If a module with the same key already exists, it is replaced
    /// (preserving insertion position).
    pub fn insert(&mut self, key: impl Into<String>, module: Box<dyn Module<T>>) {
        let key = key.into();
        // Replace existing entry if key already exists.
        for entry in &mut self.entries {
            if entry.0 == key {
                entry.1 = module;
                return;
            }
        }
        self.entries.push((key, module));
    }

    /// Get a reference to the module with the given key.
    pub fn get(&self, key: &str) -> Option<&dyn Module<T>> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, m)| m.as_ref())
    }

    /// Get a mutable reference to the module with the given key.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut dyn Module<T>> {
        for (k, m) in &mut self.entries {
            if k == key {
                return Some(m.as_mut());
            }
        }
        None
    }

    /// Return the keys in insertion order.
    pub fn keys(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dict is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<T: Float> Default for ModuleDict<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Module<T> for ModuleDict<T> {
    /// ModuleDict does not implement forward.
    ///
    /// Users should look up modules by key and call `forward()` manually.
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "ModuleDict does not implement forward. \
                      Look up modules by key and call forward() manually."
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.entries
            .iter()
            .flat_map(|(_, m)| m.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.entries
            .iter_mut()
            .flat_map(|(_, m)| m.parameters_mut())
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.entries
            .iter()
            .flat_map(|(key, m)| {
                m.named_parameters()
                    .into_iter()
                    .map(move |(name, param)| (format!("{key}.{name}"), param))
            })
            .collect()
    }

    fn train(&mut self) {
        self.training = true;
        for (_, m) in &mut self.entries {
            m.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for (_, m) in &mut self.entries {
            m.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl<T: Float> std::fmt::Display for ModuleDict<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ModuleDict(")?;
        for (key, _m) in &self.entries {
            writeln!(f, "  ({key}): <module>")?;
        }
        write!(f, ")")
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helper: a simple identity module with one parameter.
    // -----------------------------------------------------------------------

    struct IdentityWithParam<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> IdentityWithParam<T> {
        fn new(size: usize) -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::zeros(&[size])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for IdentityWithParam<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.weight]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("weight".to_string(), &self.weight)]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    // -----------------------------------------------------------------------
    // Sequential tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sequential_forward_chains_layers() {
        // 3 identity layers — output should equal input.
        let seq = Sequential::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(4).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(4).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(4).unwrap()),
        ]);

        let input = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
        let output = seq.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn test_sequential_empty_forward_errors() {
        let seq = Sequential::<f32>::new(vec![]);
        let input = ferrotorch_core::zeros::<f32>(&[1, 4]).unwrap();
        assert!(seq.forward(&input).is_err());
    }

    #[test]
    fn test_sequential_parameter_count() {
        let seq = Sequential::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(5).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(7).unwrap()),
        ]);

        let params = seq.parameters();
        assert_eq!(params.len(), 3);

        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 3 + 5 + 7);
    }

    #[test]
    fn test_sequential_named_parameters_keys() {
        let seq = Sequential::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(2).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(4).unwrap()),
        ]);

        let named = seq.named_parameters();
        let keys: Vec<&str> = named.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, &["0.weight", "1.weight", "2.weight"]);
    }

    #[test]
    fn test_sequential_train_eval_propagation() {
        let mut seq = Sequential::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(2).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
        ]);

        assert!(seq.is_training());

        seq.eval();
        assert!(!seq.is_training());
        // Sub-modules should also be in eval mode.
        for layer in &seq.layers {
            assert!(!layer.is_training());
        }

        seq.train();
        assert!(seq.is_training());
        for layer in &seq.layers {
            assert!(layer.is_training());
        }
    }

    #[test]
    fn test_sequential_state_dict_roundtrip() {
        let seq = Sequential::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(5).unwrap()),
        ]);

        let sd = seq.state_dict();
        assert!(sd.contains_key("0.weight"));
        assert!(sd.contains_key("1.weight"));
        assert_eq!(sd["0.weight"].shape(), &[3]);
        assert_eq!(sd["1.weight"].shape(), &[5]);

        // Load into a new Sequential with the same architecture.
        let mut seq2 = Sequential::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(5).unwrap()),
        ]);
        seq2.load_state_dict(&sd, true).unwrap();

        let sd2 = seq2.state_dict();
        assert_eq!(
            sd["0.weight"].data().unwrap(),
            sd2["0.weight"].data().unwrap()
        );
        assert_eq!(
            sd["1.weight"].data().unwrap(),
            sd2["1.weight"].data().unwrap()
        );
    }

    #[test]
    fn test_sequential_push() {
        let mut seq = Sequential::<f32>::new(vec![]);
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);

        seq.push(Box::new(IdentityWithParam::<f32>::new(4).unwrap()));
        assert_eq!(seq.len(), 1);
        assert!(!seq.is_empty());
    }

    // -----------------------------------------------------------------------
    // ModuleList tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_list_forward_errors() {
        let list =
            ModuleList::<f32>::new(vec![Box::new(IdentityWithParam::<f32>::new(4).unwrap())]);
        let input = ferrotorch_core::zeros::<f32>(&[1, 4]).unwrap();
        assert!(list.forward(&input).is_err());
    }

    #[test]
    fn test_module_list_get() {
        let list = ModuleList::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(5).unwrap()),
        ]);

        assert!(list.get(0).is_some());
        assert!(list.get(1).is_some());
        assert!(list.get(2).is_none());
    }

    #[test]
    fn test_module_list_get_mut() {
        let mut list =
            ModuleList::<f32>::new(vec![Box::new(IdentityWithParam::<f32>::new(3).unwrap())]);

        let m = list.get_mut(0).unwrap();
        m.eval();
        assert!(!list.get(0).unwrap().is_training());
    }

    #[test]
    fn test_module_list_push() {
        let mut list = ModuleList::<f32>::empty();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());

        list.push(Box::new(IdentityWithParam::<f32>::new(4).unwrap()));
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
    }

    #[test]
    fn test_module_list_parameters() {
        let list = ModuleList::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(2).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
        ]);

        assert_eq!(list.parameters().len(), 2);

        let named = list.named_parameters();
        let keys: Vec<&str> = named.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, &["0.weight", "1.weight"]);
    }

    #[test]
    fn test_module_list_train_eval() {
        let mut list = ModuleList::<f32>::new(vec![
            Box::new(IdentityWithParam::<f32>::new(2).unwrap()),
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
        ]);

        list.eval();
        assert!(!list.is_training());
        assert!(!list.get(0).unwrap().is_training());
        assert!(!list.get(1).unwrap().is_training());

        list.train();
        assert!(list.is_training());
        assert!(list.get(0).unwrap().is_training());
        assert!(list.get(1).unwrap().is_training());
    }

    // -----------------------------------------------------------------------
    // ModuleDict tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_dict_forward_errors() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert("enc", Box::new(IdentityWithParam::<f32>::new(4).unwrap()));
        let input = ferrotorch_core::zeros::<f32>(&[1, 4]).unwrap();
        assert!(dict.forward(&input).is_err());
    }

    #[test]
    fn test_module_dict_insert_get() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert(
            "encoder",
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
        );
        dict.insert(
            "decoder",
            Box::new(IdentityWithParam::<f32>::new(5).unwrap()),
        );

        assert!(dict.get("encoder").is_some());
        assert!(dict.get("decoder").is_some());
        assert!(dict.get("missing").is_none());
        assert_eq!(dict.len(), 2);
    }

    #[test]
    fn test_module_dict_insert_replaces() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert("layer", Box::new(IdentityWithParam::<f32>::new(3).unwrap()));
        dict.insert("layer", Box::new(IdentityWithParam::<f32>::new(7).unwrap()));

        // Should still have only 1 entry, with the new parameter size.
        assert_eq!(dict.len(), 1);
        let named = dict.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].1.shape(), &[7]);
    }

    #[test]
    fn test_module_dict_keys_insertion_order() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert(
            "c_layer",
            Box::new(IdentityWithParam::<f32>::new(1).unwrap()),
        );
        dict.insert(
            "a_layer",
            Box::new(IdentityWithParam::<f32>::new(2).unwrap()),
        );
        dict.insert(
            "b_layer",
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
        );

        assert_eq!(dict.keys(), &["c_layer", "a_layer", "b_layer"]);
    }

    #[test]
    fn test_module_dict_get_mut() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert("layer", Box::new(IdentityWithParam::<f32>::new(3).unwrap()));

        let m = dict.get_mut("layer").unwrap();
        m.eval();
        assert!(!dict.get("layer").unwrap().is_training());
    }

    #[test]
    fn test_module_dict_named_parameters_prefixed_by_key() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert(
            "encoder",
            Box::new(IdentityWithParam::<f32>::new(3).unwrap()),
        );
        dict.insert(
            "decoder",
            Box::new(IdentityWithParam::<f32>::new(5).unwrap()),
        );

        let named = dict.named_parameters();
        let keys: Vec<&str> = named.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, &["encoder.weight", "decoder.weight"]);
    }

    #[test]
    fn test_module_dict_train_eval() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert("a", Box::new(IdentityWithParam::<f32>::new(2).unwrap()));
        dict.insert("b", Box::new(IdentityWithParam::<f32>::new(3).unwrap()));

        dict.eval();
        assert!(!dict.is_training());
        assert!(!dict.get("a").unwrap().is_training());
        assert!(!dict.get("b").unwrap().is_training());

        dict.train();
        assert!(dict.is_training());
        assert!(dict.get("a").unwrap().is_training());
        assert!(dict.get("b").unwrap().is_training());
    }

    #[test]
    fn test_module_dict_default() {
        let dict = ModuleDict::<f32>::default();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
    }

    #[test]
    fn test_module_dict_state_dict_roundtrip() {
        let mut dict = ModuleDict::<f32>::new();
        dict.insert("enc", Box::new(IdentityWithParam::<f32>::new(4).unwrap()));
        dict.insert("dec", Box::new(IdentityWithParam::<f32>::new(6).unwrap()));

        let sd = dict.state_dict();
        assert!(sd.contains_key("enc.weight"));
        assert!(sd.contains_key("dec.weight"));

        let mut dict2 = ModuleDict::<f32>::new();
        dict2.insert("enc", Box::new(IdentityWithParam::<f32>::new(4).unwrap()));
        dict2.insert("dec", Box::new(IdentityWithParam::<f32>::new(6).unwrap()));
        dict2.load_state_dict(&sd, true).unwrap();

        let sd2 = dict2.state_dict();
        assert_eq!(
            sd["enc.weight"].data().unwrap(),
            sd2["enc.weight"].data().unwrap()
        );
        assert_eq!(
            sd["dec.weight"].data().unwrap(),
            sd2["dec.weight"].data().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_containers_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Sequential<f32>>();
        assert_send_sync::<ModuleList<f32>>();
        assert_send_sync::<ModuleDict<f32>>();
    }
}
