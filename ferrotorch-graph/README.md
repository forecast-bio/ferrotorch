# ferrotorch-graph

Graph neural network composition for ferrotorch.

The first pinned reference is a 2-layer GCN trained for 200 epochs on
the Cora node-classification benchmark (1433-dim features, 2708 nodes,
7 classes), mirrored to `ferrotorch/gcn-cora` and registered in
[`ferrotorch-hub`](../ferrotorch-hub) (#1157).

## What it provides

- **`data::Graph`** — `(num_nodes, num_edges, edge_index [2, E])` plus
  the symmetric-normalized adjacency `Â = D^(-1/2) (A + I) D^(-1/2)`
  pre-computed for the message-passing forward.
- **`GcnConv`** — one Kipf-and-Welling graph convolution layer
  (`Â · X · W + b`, optional bias). Forward: `[N, in_features]` →
  `[N, out_features]` for a fixed graph.
- **`GcnNet`** — two-layer GCN classifier (`GcnConv -> ReLU -> Dropout
  -> GcnConv`) with helpers `in_features()`, `hidden()`,
  `num_classes()`.
- **`load_gcn_net`** — SafeTensors loader for the pinned upstream
  PyTorch-Geometric checkpoint; returns a `DropReport` (#1141
  silent-drop-bug guard).

The message-passing primitive (segmented `scatter_add`) lives in
`ferrotorch-core::ops::scatter::scatter_add_segments` so non-graph
crates can reuse it.

## Quick start

```rust
use ferrotorch_graph::{Graph, GcnNet, load_gcn_net};

// Cora: 2708 nodes, 1433 features, 7 classes
let graph = Graph::cora_from_files("/path/to/cora")?;
let mut net: GcnNet<f32> = GcnNet::new(/*in*/ 1433, /*hidden*/ 16, /*classes*/ 7)?;
let _drop = load_gcn_net(&mut net, "/path/to/gcn-cora.safetensors")?;

let logits = net.forward(&graph, &graph.features)?;
// logits: [2708, 7] — argmax along axis 1 to get predicted class
```

## Real-artifact parity

`scripts/verify_gnn_inference.py` compares this crate's full-graph
forward against the upstream
[`torch_geometric==2.7.0`](https://pyg.org)
`GCNConv`-with-self-loops reference on the pinned Cora checkpoint
(Phase D.1 of real-artifact-driven development; #1157). PASS floor:
`cosine_sim >= 0.999, max_abs <= 0.5`.

## Part of ferrotorch

This crate is one component of the
[ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
