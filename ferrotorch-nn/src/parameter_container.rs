//! Parameter containers: ParameterList and ParameterDict.
//!
//! These hold parameters that should be registered with a module's
//! `parameters()` method, similar to PyTorch's `nn.ParameterList` and
//! `nn.ParameterDict`.

use std::collections::BTreeMap;

use ferrotorch_core::Float;

use crate::parameter::Parameter;

/// An ordered list of parameters.
///
/// Parameters added to a `ParameterList` are registered and will be
/// included when calling `parameters()` on the containing module.
///
/// Matches PyTorch's `nn.ParameterList`.
#[derive(Debug)]
pub struct ParameterList<T: Float> {
    params: Vec<Parameter<T>>,
}

impl<T: Float> ParameterList<T> {
    /// Create an empty parameter list.
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Create from an existing vector of parameters.
    pub fn from_vec(params: Vec<Parameter<T>>) -> Self {
        Self { params }
    }

    /// Append a parameter.
    pub fn append(&mut self, param: Parameter<T>) {
        self.params.push(param);
    }

    /// Extend with multiple parameters.
    pub fn extend(&mut self, params: impl IntoIterator<Item = Parameter<T>>) {
        self.params.extend(params);
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get a parameter by index.
    pub fn get(&self, index: usize) -> Option<&Parameter<T>> {
        self.params.get(index)
    }

    /// Get a mutable parameter by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Parameter<T>> {
        self.params.get_mut(index)
    }

    /// Iterate over parameters.
    pub fn iter(&self) -> std::slice::Iter<'_, Parameter<T>> {
        self.params.iter()
    }

    /// Iterate mutably over parameters.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Parameter<T>> {
        self.params.iter_mut()
    }

    /// Return all parameters as references (for Module trait integration).
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        self.params.iter().collect()
    }

    /// Return all parameters as mutable references.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.params.iter_mut().collect()
    }

    /// Return named parameters with index-based keys: "0", "1", etc.
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.params
            .iter()
            .enumerate()
            .map(|(i, p)| (i.to_string(), p))
            .collect()
    }
}

impl<T: Float> Default for ParameterList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> std::ops::Index<usize> for ParameterList<T> {
    type Output = Parameter<T>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.params[index]
    }
}

impl<T: Float> std::ops::IndexMut<usize> for ParameterList<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.params[index]
    }
}

/// A dictionary of named parameters.
///
/// Parameters are stored in sorted key order (BTreeMap) for deterministic
/// iteration, matching PyTorch's `nn.ParameterDict`.
#[derive(Debug)]
pub struct ParameterDict<T: Float> {
    params: BTreeMap<String, Parameter<T>>,
}

impl<T: Float> ParameterDict<T> {
    /// Create an empty parameter dict.
    pub fn new() -> Self {
        Self {
            params: BTreeMap::new(),
        }
    }

    /// Insert a named parameter. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: impl Into<String>, param: Parameter<T>) -> Option<Parameter<T>> {
        self.params.insert(key.into(), param)
    }

    /// Get a parameter by name.
    pub fn get(&self, key: &str) -> Option<&Parameter<T>> {
        self.params.get(key)
    }

    /// Get a mutable parameter by name.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Parameter<T>> {
        self.params.get_mut(key)
    }

    /// Remove a parameter by name.
    pub fn remove(&mut self, key: &str) -> Option<Parameter<T>> {
        self.params.remove(key)
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the dict is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Return all parameter keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.params.keys()
    }

    /// Return all parameters as references.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        self.params.values().collect()
    }

    /// Return all parameters as mutable references.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.params.values_mut().collect()
    }

    /// Return named parameters.
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.params.iter().map(|(k, v)| (k.clone(), v)).collect()
    }
}

impl<T: Float> Default for ParameterDict<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(n: usize) -> Parameter<f32> {
        Parameter::zeros(&[n]).unwrap()
    }

    #[test]
    fn test_parameter_list_basic() {
        let mut list = ParameterList::new();
        assert!(list.is_empty());
        list.append(param(3));
        list.append(param(5));
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].tensor().numel(), 3);
        assert_eq!(list[1].tensor().numel(), 5);
    }

    #[test]
    fn test_parameter_list_named() {
        let list = ParameterList::from_vec(vec![param(2), param(4)]);
        let named = list.named_parameters();
        assert_eq!(named[0].0, "0");
        assert_eq!(named[1].0, "1");
    }

    #[test]
    fn test_parameter_list_parameters() {
        let list = ParameterList::from_vec(vec![param(1), param(2), param(3)]);
        assert_eq!(list.parameters().len(), 3);
    }

    #[test]
    fn test_parameter_dict_basic() {
        let mut dict = ParameterDict::new();
        assert!(dict.is_empty());
        dict.insert("weight", param(10));
        dict.insert("bias", param(5));
        assert_eq!(dict.len(), 2);
        assert!(dict.contains_key("weight"));
        assert!(!dict.contains_key("foo"));
        assert_eq!(dict.get("weight").unwrap().tensor().numel(), 10);
    }

    #[test]
    fn test_parameter_dict_replace() {
        let mut dict = ParameterDict::new();
        dict.insert("w", param(3));
        let old = dict.insert("w", param(7));
        assert!(old.is_some());
        assert_eq!(dict.get("w").unwrap().tensor().numel(), 7);
    }

    #[test]
    fn test_parameter_dict_remove() {
        let mut dict = ParameterDict::new();
        dict.insert("a", param(1));
        dict.insert("b", param(2));
        dict.remove("a");
        assert_eq!(dict.len(), 1);
        assert!(!dict.contains_key("a"));
    }

    #[test]
    fn test_parameter_dict_named_sorted() {
        let mut dict = ParameterDict::new();
        dict.insert("z", param(1));
        dict.insert("a", param(2));
        dict.insert("m", param(3));
        let named = dict.named_parameters();
        assert_eq!(named[0].0, "a");
        assert_eq!(named[1].0, "m");
        assert_eq!(named[2].0, "z");
    }
}
