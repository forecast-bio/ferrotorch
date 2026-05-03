/// A unique identifier for IR values (edges in the graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IrValueId(pub usize);

/// A unique identifier for IR nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrNodeId(pub usize);

/// The kind of operation an IR node represents.
/// Matches ferrotorch-core's operation vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub enum IrOpKind {
    // Inputs/outputs
    Input {
        index: usize,
    },
    Constant {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    Output,

    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    /// Raise each element to a fixed exponent. Special-cased separately from
    /// other unary ops because it carries an `exponent` parameter that must be
    /// preserved through optimization passes and codegen.
    Pow {
        exponent: f64,
    },
    Sqrt,
    Abs,
    Exp,
    Log,

    // Reduction
    Sum,
    Mean,
    Prod,

    // Linalg
    Matmul,
    Mm,
    Mv,
    Dot,
    Transpose,
    /// Fused linear: `output = input @ weight^T + bias`.
    /// Inputs: `[input, weight]` or `[input, weight, bias]`.
    Linear,

    // Activation
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Softmax,
    LogSoftmax,

    // Shape
    Reshape {
        shape: Vec<isize>,
    },
    Flatten,
    Squeeze {
        axis: usize,
    },
    Unsqueeze {
        axis: usize,
    },
    Cat {
        axis: usize,
    },

    // Higher-order control flow (must be lowered before interpretation)
    /// Conditional: selects between two sub-graphs based on a predicate.
    /// Must be lowered/inlined before the interpreter can execute the graph.
    Cond,
    /// Sequential scan: applies a step function over a sequence.
    /// Must be lowered/inlined before the interpreter can execute the graph.
    Scan,

    // Fused (created by optimization)
    FusedElementwise {
        ops: Vec<IrOpKind>,
    },

    /// Fused Linear + activation: `activation(input @ weight^T + bias)`.
    /// Created by pattern fusion when Linear is followed by an activation.
    FusedLinearActivation {
        activation: Box<IrOpKind>,
    },

    /// Fused scaled dot-product attention:
    /// `softmax(Q @ K^T / sqrt(d_k)) @ V`.
    /// Created by pattern fusion when the SDPA pattern is detected.
    FusedAttention {
        head_dim: usize,
    },
}

/// An IR value — an edge in the graph carrying shape/dtype metadata.
#[derive(Debug, Clone)]
pub struct IrValue {
    pub id: IrValueId,
    pub shape: Vec<usize>,
    pub producer: Option<IrNodeId>, // None for graph inputs
}

/// An IR node — a single operation in the graph.
#[derive(Debug, Clone)]
pub struct IrNode {
    pub id: IrNodeId,
    pub op: IrOpKind,
    pub inputs: Vec<IrValueId>,
    pub outputs: Vec<IrValueId>,
}

/// The complete IR graph.
#[derive(Debug, Clone)]
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    pub values: Vec<IrValue>,
    pub input_values: Vec<IrValueId>,
    pub output_values: Vec<IrValueId>,
    pub(crate) next_value_id: usize,
    pub(crate) next_node_id: usize,
}

impl IrGraph {
    /// Create an empty IR graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            values: Vec::new(),
            input_values: Vec::new(),
            output_values: Vec::new(),
            next_value_id: 0,
            next_node_id: 0,
        }
    }

    /// Register a graph input with the given shape.
    pub fn add_input(&mut self, shape: Vec<usize>) -> IrValueId {
        let value_id = IrValueId(self.next_value_id);
        self.next_value_id += 1;

        let node_id = IrNodeId(self.next_node_id);
        self.next_node_id += 1;

        let index = self.input_values.len();

        self.values.push(IrValue {
            id: value_id,
            shape,
            producer: Some(node_id),
        });

        self.nodes.push(IrNode {
            id: node_id,
            op: IrOpKind::Input { index },
            inputs: Vec::new(),
            outputs: vec![value_id],
        });

        self.input_values.push(value_id);
        value_id
    }

    /// Register a constant value with the given data and shape.
    pub fn add_constant(&mut self, data: Vec<f64>, shape: Vec<usize>) -> IrValueId {
        let value_id = IrValueId(self.next_value_id);
        self.next_value_id += 1;

        let node_id = IrNodeId(self.next_node_id);
        self.next_node_id += 1;

        self.values.push(IrValue {
            id: value_id,
            shape: shape.clone(),
            producer: Some(node_id),
        });

        self.nodes.push(IrNode {
            id: node_id,
            op: IrOpKind::Constant { data, shape },
            inputs: Vec::new(),
            outputs: vec![value_id],
        });

        value_id
    }

    /// Add an operation node to the graph.
    ///
    /// Returns the node ID and the IDs of its output values.
    pub fn add_node(
        &mut self,
        op: IrOpKind,
        inputs: Vec<IrValueId>,
        output_shapes: Vec<Vec<usize>>,
    ) -> (IrNodeId, Vec<IrValueId>) {
        let node_id = IrNodeId(self.next_node_id);
        self.next_node_id += 1;

        let mut output_ids = Vec::with_capacity(output_shapes.len());
        for shape in output_shapes {
            let value_id = IrValueId(self.next_value_id);
            self.next_value_id += 1;

            self.values.push(IrValue {
                id: value_id,
                shape,
                producer: Some(node_id),
            });

            output_ids.push(value_id);
        }

        self.nodes.push(IrNode {
            id: node_id,
            op,
            inputs,
            outputs: output_ids.clone(),
        });

        (node_id, output_ids)
    }

    /// Mark which values are graph outputs.
    pub fn set_outputs(&mut self, values: Vec<IrValueId>) {
        self.output_values = values;
    }

    /// Return the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of values in the graph.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Compute a topological ordering of nodes (Kahn's algorithm).
    ///
    /// Returns node IDs in an order such that every node appears after
    /// all of its input-producing nodes.
    pub fn topological_order(&self) -> Vec<IrNodeId> {
        use std::collections::{HashMap, VecDeque};

        // Build a map from value_id -> producer node_id.
        let mut value_producer: HashMap<IrValueId, IrNodeId> = HashMap::new();
        for node in &self.nodes {
            for &out in &node.outputs {
                value_producer.insert(out, node.id);
            }
        }

        // Compute in-degree for each node (count of distinct producer nodes feeding inputs).
        let mut in_degree: HashMap<IrNodeId, usize> = HashMap::new();
        let mut dependents: HashMap<IrNodeId, Vec<IrNodeId>> = HashMap::new();

        for node in &self.nodes {
            in_degree.entry(node.id).or_insert(0);

            for &input_val in &node.inputs {
                if let Some(&producer_id) = value_producer.get(&input_val) {
                    *in_degree.entry(node.id).or_insert(0) += 1;
                    dependents.entry(producer_id).or_default().push(node.id);
                }
            }
        }

        // Kahn's algorithm.
        let mut queue: VecDeque<IrNodeId> = VecDeque::new();
        for node in &self.nodes {
            if in_degree[&node.id] == 0 {
                queue.push_back(node.id);
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(nid) = queue.pop_front() {
            order.push(nid);
            if let Some(deps) = dependents.get(&nid) {
                for &dep in deps {
                    let deg = in_degree.get_mut(&dep).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }

        order
    }

    /// Allocate and return the next node ID, advancing the internal counter.
    pub fn alloc_node_id(&mut self) -> IrNodeId {
        let id = IrNodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Allocate and return the next value ID, advancing the internal counter.
    pub fn alloc_value_id(&mut self) -> IrValueId {
        let id = IrValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Check if a value is produced by a `Constant` node.
    pub fn is_constant(&self, value_id: IrValueId) -> bool {
        // Find the value, then check its producer.
        let value = match self.values.iter().find(|v| v.id == value_id) {
            Some(v) => v,
            None => return false,
        };

        let producer_id = match value.producer {
            Some(id) => id,
            None => return false,
        };

        self.nodes
            .iter()
            .find(|n| n.id == producer_id)
            .is_some_and(|n| matches!(n.op, IrOpKind::Constant { .. }))
    }

    /// Remove a node from the graph (for dead code elimination).
    ///
    /// This removes the node itself and any values it produces.
    /// It does **not** recursively remove upstream nodes that may
    /// become dead — the caller should iterate to a fixed point.
    pub fn remove_node(&mut self, node_id: IrNodeId) {
        // Collect output value IDs of the node being removed.
        let output_value_ids: Vec<IrValueId> = self
            .nodes
            .iter()
            .find(|n| n.id == node_id)
            .map(|n| n.outputs.clone())
            .unwrap_or_default();

        // Remove the node.
        self.nodes.retain(|n| n.id != node_id);

        // Remove the values produced by this node.
        self.values.retain(|v| !output_value_ids.contains(&v.id));

        // Clean up input_values and output_values lists.
        self.input_values.retain(|v| !output_value_ids.contains(v));
        self.output_values.retain(|v| !output_value_ids.contains(v));
    }
}

impl Default for IrGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple graph: input -> add(input, input) -> relu -> output
    /// and verify structure.
    #[test]
    fn test_simple_graph_input_add_relu_output() {
        let mut g = IrGraph::new();

        // Single input of shape [2, 3].
        let x = g.add_input(vec![2, 3]);

        // Add: x + x -> shape [2, 3].
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![2, 3]]);
        let add_out = add_outs[0];

        // Relu on the add result.
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_out], vec![vec![2, 3]]);
        let relu_out = relu_outs[0];

        // Mark output.
        g.set_outputs(vec![relu_out]);

        // 3 nodes: Input, Add, Relu.
        assert_eq!(g.node_count(), 3);
        // 3 values: input value, add output, relu output.
        assert_eq!(g.value_count(), 3);
        assert_eq!(g.output_values.len(), 1);
        assert_eq!(g.input_values.len(), 1);
    }

    /// Topological sort must place producers before consumers.
    #[test]
    fn test_topological_order() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![4]);
        let (_input_node_id, ()) = (IrNodeId(0), ()); // Input node is id 0.

        let (add_node_id, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![4]]);
        let (relu_node_id, relu_outs) =
            g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let order = g.topological_order();
        assert_eq!(order.len(), 3);

        // Input must come before Add, Add before Relu.
        let pos = |id: IrNodeId| order.iter().position(|&n| n == id).unwrap();
        assert!(pos(IrNodeId(0)) < pos(add_node_id));
        assert!(pos(add_node_id) < pos(relu_node_id));
    }

    /// Constants should be recognized as constants.
    #[test]
    fn test_add_constant_and_is_constant() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![1.0, 2.0, 3.0], vec![3]);

        assert!(!g.is_constant(x));
        assert!(g.is_constant(c));
    }

    /// Node and value counts update correctly.
    #[test]
    fn test_node_value_counts() {
        let mut g = IrGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.value_count(), 0);

        let x = g.add_input(vec![5]);
        // Input adds 1 node + 1 value.
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.value_count(), 1);

        let c = g.add_constant(vec![0.0; 5], vec![5]);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.value_count(), 2);

        let (_add_id, _add_outs) = g.add_node(IrOpKind::Add, vec![x, c], vec![vec![5]]);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.value_count(), 3);
    }

    /// `remove_node` should remove the node and its output values.
    #[test]
    fn test_remove_node() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![2]);
        let (add_id, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![2]]);
        let (relu_id, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![2]]);
        g.set_outputs(vec![relu_outs[0]]);

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.value_count(), 3);

        // Remove the relu node.
        g.remove_node(relu_id);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.value_count(), 2);
        // Output list was cleaned up.
        assert!(g.output_values.is_empty());

        // Remove the add node.
        g.remove_node(add_id);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.value_count(), 1);
    }
}
