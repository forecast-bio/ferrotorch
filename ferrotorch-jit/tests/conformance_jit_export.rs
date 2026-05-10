//! Conformance Phase C7.4 — `ferrotorch-jit` export + serialize + error paths.
//!
//! Tracking issue: crosslink #806 (C7.4 sub-phase).
//!
//! ## Coverage
//!
//! ### Export API
//! - [`export`] — single-input tracing to [`ExportedProgram`]
//! - [`export_with_dynamic_shapes`] — dynamic batch dim support
//! - [`ExportedProgram::run`] — unchecked execution
//! - [`ExportedProgram::run_with_guards`] — guarded execution with spec validation
//! - [`ExportedProgram::check_inputs`] — standalone guard check
//!
//! ### Shape spec API
//! - [`DimSpec::Static`] / [`DimSpec::Dynamic`]
//! - [`DimSpec::dynamic`] / [`DimSpec::dynamic_range`] / [`DimSpec::is_dynamic`]
//! - [`InputSpec::new`] / [`InputSpec::all_static`] / [`InputSpec::has_dynamic_dims`] / [`InputSpec::rank`]
//!
//! ### Serialize / save / load
//! - [`ExportedProgram::serialize`] + [`ExportedProgram::deserialize`] — binary round-trip
//! - [`ExportedProgram::save`] + [`ExportedProgram::load`] — file I/O round-trip
//! - [`ExportedProgram::parse_json_metadata`] — JSON metadata extraction
//! - [`ExportedProgramMetadata`] — metadata struct fields
//! - [`IrGraph::serialize`] + [`IrGraph::deserialize`] — inner graph round-trip
//!
//! ### Error paths
//! - Bad magic → [`FerrotorchError`]
//! - Unsupported version → [`FerrotorchError`]
//! - Truncated data → [`FerrotorchError`]
//! - Multi-input to `export` → [`FerrotorchError`]
//! - Spec length mismatch → [`FerrotorchError`]
//! - Spec rank mismatch → [`FerrotorchError`]
//! - Static dim mismatch → [`FerrotorchError`]
//! - Guard input count wrong → [`FerrotorchError`]
//! - Guard rank mismatch → [`FerrotorchError`]
//! - Guard static dim mismatch → [`FerrotorchError`]
//! - Guard dynamic dim above max → [`FerrotorchError`]
//!
//! ### JitError
//! - [`JitError::TracingError`] — error message round-trip via Display
//! - [`JitError::UnsupportedOp`] — error message via Display
//! - [`JitError::ShapeMismatch`] — error message via Display
//! - [`JitError::CodegenError`] — error message via Display
//! - [`JitError::SerializationError`] — error message via Display
//! - [`JitError::GraphBreak`] — error message via Display
//! - [`JitError::ExportError`] — error message via Display
//! - [`JitError::ParameterError`] — error message via Display
//! - [`JitError::Unsupported`] — error message via Display
//! - [`JitError`] `From<JitError> for FerrotorchError` — conversion
//!
//! ## Cascade-skip
//!
//! No cascade-skip entries: all tests in this file exercise stable in-tree code.
//! Crosslink-quick for GPU paths is handled by the exclusions TOML.

use std::collections::HashMap;

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_jit::autotune::{AutotuneCandidate, AutotuneKey, Autotuner};
use ferrotorch_jit::codegen::InterpreterBackend;
use ferrotorch_jit::error::JitError;
use ferrotorch_jit::export::{
    DimSpec, ExportedProgram, ExportedProgramMetadata, InputSpec, export,
    export_with_dynamic_shapes,
};
use ferrotorch_jit::graph::{Dtype, IrGraph, IrOpKind};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal `ExportedProgram` without tracing a real module.
/// Graph: input(shape=[4,10]) → relu → output.
fn build_program(input_specs: Vec<InputSpec>) -> ExportedProgram {
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4, 10]);
    let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4, 10]]);
    g.set_outputs(vec![outs[0]]);

    let mut state_dict = HashMap::new();
    state_dict.insert("fc.weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]);
    state_dict.insert("fc.bias".to_string(), vec![0.5f32, 0.5f32]);

    ExportedProgram {
        graph: g,
        state_dict,
        input_shapes: vec![vec![4, 10]],
        input_specs,
        output_shape: vec![4, 10],
    }
}

fn all_static_program() -> ExportedProgram {
    build_program(vec![InputSpec::all_static(&[4, 10])])
}

fn dynamic_batch_program() -> ExportedProgram {
    build_program(vec![InputSpec::new(vec![
        DimSpec::dynamic_range("batch", 1, 32),
        DimSpec::Static(10),
    ])])
}

// ---------------------------------------------------------------------------
// DimSpec API
// ---------------------------------------------------------------------------

#[test]
fn dim_spec_static_not_dynamic() {
    assert!(!DimSpec::Static(4).is_dynamic());
    assert_eq!(DimSpec::Static(4), DimSpec::Static(4));
}

#[test]
fn dim_spec_dynamic_constructor_no_bounds() {
    let d = DimSpec::dynamic("seq");
    assert!(d.is_dynamic());
    match d {
        DimSpec::Dynamic { name, min, max } => {
            assert_eq!(name, "seq");
            assert_eq!(min, None);
            assert_eq!(max, None);
        }
        DimSpec::Static(_) => panic!("expected Dynamic"),
    }
}

#[test]
fn dim_spec_dynamic_range_constructor() {
    let d = DimSpec::dynamic_range("batch", 2, 64);
    assert!(d.is_dynamic());
    match d {
        DimSpec::Dynamic { name, min, max } => {
            assert_eq!(name, "batch");
            assert_eq!(min, Some(2));
            assert_eq!(max, Some(64));
        }
        DimSpec::Static(_) => panic!("expected Dynamic"),
    }
}

// ---------------------------------------------------------------------------
// InputSpec API
// ---------------------------------------------------------------------------

#[test]
fn input_spec_all_static() {
    let s = InputSpec::all_static(&[2, 3, 4]);
    assert_eq!(s.rank(), 3);
    assert!(!s.has_dynamic_dims());
    for d in &s.shape {
        assert!(matches!(d, DimSpec::Static(_)));
    }
}

#[test]
fn input_spec_new_mixed() {
    let s = InputSpec::new(vec![DimSpec::dynamic("batch"), DimSpec::Static(8)]);
    assert_eq!(s.rank(), 2);
    assert!(s.has_dynamic_dims());
}

#[test]
fn input_spec_rank_zero() {
    let s = InputSpec::new(vec![]);
    assert_eq!(s.rank(), 0);
    assert!(!s.has_dynamic_dims());
}

// ---------------------------------------------------------------------------
// ExportedProgram: binary serialize / deserialize round-trip
// ---------------------------------------------------------------------------

#[test]
fn exported_program_binary_roundtrip_all_static() {
    let original = all_static_program();
    let bytes = original.serialize();
    assert!(bytes.starts_with(b"FTEP"), "magic header missing");

    let loaded = ExportedProgram::deserialize(&bytes).expect("deserialize failed");
    assert_eq!(loaded.input_shapes, original.input_shapes);
    assert_eq!(loaded.output_shape, original.output_shape);
    assert_eq!(loaded.state_dict.len(), original.state_dict.len());
    for (k, v) in &original.state_dict {
        assert_eq!(
            loaded.state_dict.get(k),
            Some(v),
            "state_dict mismatch for {k}"
        );
    }
    // Specs preserved
    assert_eq!(loaded.input_specs.len(), 1);
    assert_eq!(loaded.input_specs[0].rank(), 2);
    for dim in &loaded.input_specs[0].shape {
        assert!(matches!(dim, DimSpec::Static(_)));
    }
}

#[test]
fn exported_program_binary_roundtrip_dynamic_dim() {
    let original = dynamic_batch_program();
    let bytes = original.serialize();
    let loaded = ExportedProgram::deserialize(&bytes).expect("deserialize failed");

    assert_eq!(loaded.input_specs.len(), 1);
    match &loaded.input_specs[0].shape[0] {
        DimSpec::Dynamic { name, min, max } => {
            assert_eq!(name, "batch");
            assert_eq!(*min, Some(1));
            assert_eq!(*max, Some(32));
        }
        DimSpec::Static(_) => panic!("expected Dynamic dim at index 0"),
    }
    assert_eq!(loaded.input_specs[0].shape[1], DimSpec::Static(10));
    // Graph round-tripped
    assert_eq!(loaded.graph.node_count(), original.graph.node_count());
    assert_eq!(loaded.graph.input_values, original.graph.input_values);
}

#[test]
fn exported_program_serialize_is_deterministic() {
    let program = all_static_program();
    let a = program.serialize();
    let b = program.serialize();
    assert_eq!(a, b, "serialization must be byte-stable");
}

#[test]
fn exported_program_deserialize_rejects_bad_magic() {
    let mut bad = vec![0u8; 32];
    bad[..4].copy_from_slice(b"XXXX");
    let err = ExportedProgram::deserialize(&bad).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("bad magic"), "unexpected error: {msg}");
}

#[test]
fn exported_program_deserialize_rejects_unsupported_version() {
    let mut bytes = all_static_program().serialize();
    // Overwrite the version u32 at offset 4 with value 99.
    bytes[4] = 99;
    bytes[5] = 0;
    bytes[6] = 0;
    bytes[7] = 0;
    let err = ExportedProgram::deserialize(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported version"),
        "unexpected error: {msg}"
    );
}

#[test]
fn exported_program_deserialize_rejects_truncated_data() {
    let bytes = all_static_program().serialize();
    // Various truncations must all error cleanly.
    for trunc in [0, 4, 8, 12] {
        if trunc < bytes.len() {
            let result = ExportedProgram::deserialize(&bytes[..trunc]);
            assert!(
                result.is_err(),
                "expected error for truncation to {trunc} bytes"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ExportedProgram: file save / load round-trip
// ---------------------------------------------------------------------------

#[test]
fn exported_program_save_load_file_roundtrip() {
    let original = dynamic_batch_program();
    let dir = std::env::temp_dir().join("ferrotorch_c74_conformance_save_load");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.ftep");

    original.save(&path).expect("save failed");
    assert!(path.exists());

    let loaded = ExportedProgram::load(&path).expect("load failed");
    assert_eq!(loaded.input_shapes, original.input_shapes);
    assert_eq!(loaded.output_shape, original.output_shape);
    assert_eq!(loaded.state_dict.len(), original.state_dict.len());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn exported_program_load_nonexistent_file_errors() {
    let result = ExportedProgram::load("/nonexistent/path/to/program.ftep");
    assert!(result.is_err());
}

#[test]
fn exported_program_save_nonexistent_dir_errors() {
    let p = all_static_program();
    // Writing to a path whose parent doesn't exist should fail gracefully.
    let result = p.save("/nonexistent_c74_dir/a/b/c/program.ftep");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// ExportedProgram::parse_json_metadata / ExportedProgramMetadata
// ---------------------------------------------------------------------------

#[test]
fn parse_json_metadata_roundtrip() {
    let json = r#"{"input_shapes":[[4,10]],"output_shape":[4,10],"num_graph_nodes":2,"state_dict_keys":["fc.weight","fc.bias"]}"#;
    let meta: ExportedProgramMetadata =
        ExportedProgram::parse_json_metadata(json).expect("parse failed");
    assert_eq!(meta.num_graph_nodes, 2);
    assert_eq!(meta.input_shapes, vec![vec![4, 10]]);
    assert_eq!(meta.output_shape, vec![4, 10]);
    assert_eq!(meta.state_dict_keys, vec!["fc.weight", "fc.bias"]);
}

#[test]
fn parse_json_metadata_rejects_non_object() {
    let result = ExportedProgram::parse_json_metadata("[1, 2, 3]");
    assert!(result.is_err());
}

#[test]
fn parse_json_metadata_empty_arrays() {
    let json = r#"{"input_shapes":[],"output_shape":[],"num_graph_nodes":0,"state_dict_keys":[]}"#;
    let meta = ExportedProgram::parse_json_metadata(json).expect("parse failed");
    assert_eq!(meta.num_graph_nodes, 0);
    assert!(meta.input_shapes.is_empty());
    assert!(meta.output_shape.is_empty());
    assert!(meta.state_dict_keys.is_empty());
}

// ---------------------------------------------------------------------------
// ExportedProgram::check_inputs + run_with_guards
// ---------------------------------------------------------------------------

#[test]
fn check_inputs_accepts_valid_static() {
    let program = all_static_program();
    let x: Tensor<f32> = ferrotorch_core::zeros(&[4, 10]).unwrap();
    program.check_inputs(&[x]).unwrap();
}

#[test]
fn check_inputs_rejects_wrong_input_count() {
    let program = all_static_program();
    let empty: Vec<Tensor<f32>> = vec![];
    let err = program.check_inputs(&empty).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("expected 1 inputs, got 0"),
        "unexpected: {msg}"
    );
}

#[test]
fn check_inputs_rejects_rank_mismatch() {
    let program = all_static_program();
    let x: Tensor<f32> = ferrotorch_core::zeros(&[4, 10, 2]).unwrap();
    let err = program.check_inputs(&[x]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("rank mismatch"), "unexpected: {msg}");
}

#[test]
fn check_inputs_rejects_static_dim_mismatch() {
    let program = all_static_program();
    // dim 0 expects 4, give 7.
    let x: Tensor<f32> = ferrotorch_core::zeros(&[7, 10]).unwrap();
    let err = program.check_inputs(&[x]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("dim 0"), "unexpected: {msg}");
    assert!(msg.contains("expected 4, got 7"), "unexpected: {msg}");
}

#[test]
fn check_inputs_accepts_dynamic_dim_at_min() {
    let program = dynamic_batch_program();
    // batch=1 is exactly the minimum — should be accepted.
    let x: Tensor<f32> = ferrotorch_core::zeros(&[1, 10]).unwrap();
    program.check_inputs(&[x]).unwrap();
}

#[test]
fn check_inputs_accepts_dynamic_dim_at_max() {
    let program = dynamic_batch_program();
    // batch=32 is exactly the maximum — should be accepted.
    let x: Tensor<f32> = ferrotorch_core::zeros(&[32, 10]).unwrap();
    program.check_inputs(&[x]).unwrap();
}

#[test]
fn check_inputs_rejects_dynamic_dim_above_max() {
    let program = dynamic_batch_program();
    let x: Tensor<f32> = ferrotorch_core::zeros(&[64, 10]).unwrap();
    let err = program.check_inputs(&[x]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("above declared max 32"), "unexpected: {msg}");
}

#[test]
fn check_inputs_rejects_zero_dynamic_dim() {
    let program = dynamic_batch_program();
    // batch=0 should be rejected (zero is special-cased in the guard).
    let x: Tensor<f32> = ferrotorch_core::zeros(&[0, 10]).unwrap();
    let result = program.check_inputs(&[x]);
    // Zero triggers either the "runtime value 0" path or an out-of-bounds for min.
    assert!(result.is_err(), "expected error for batch=0");
}

#[test]
fn run_with_guards_runs_valid_input() {
    let program = dynamic_batch_program();
    // Mixed-sign input so that relu's clamp-at-zero behavior is observable:
    // a stub returning zeros(&[8, 10]) of the right shape would no longer pass.
    // i ∈ 0..80 → values span -40.0..=39.0, guaranteeing both signs.
    let data: Vec<f32> = (0..80).map(|i| (i as f32) - 40.0).collect();
    let x: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![8, 10], false).unwrap();
    let out = program
        .run_with_guards(&[x])
        .expect("run_with_guards failed");
    assert_eq!(out.shape(), &[8, 10]);

    // Elementwise relu verification: out[i] == max(0.0, data[i]).
    let out_slice = out.data().expect("output data on CPU");
    assert_eq!(out_slice.len(), data.len());
    for (i, (&inp, &got)) in data.iter().zip(out_slice.iter()).enumerate() {
        let expected = inp.max(0.0);
        assert_eq!(
            got, expected,
            "relu mismatch at idx {i}: input={inp}, got={got}, expected={expected}"
        );
    }

    // Sabotage probes: input must contain a negative, output must contain
    // both an exact 0.0 (clamp happened) and a strictly positive value
    // (pass-through happened). A shape-only stub returning a fresh zeros
    // tensor would fail the "at least one positive output" check.
    assert!(
        data.iter().any(|&v| v < 0.0),
        "test setup invariant: input must contain at least one negative value"
    );
    assert!(
        out_slice.contains(&0.0),
        "expected at least one clamped-to-zero output (relu of negative input)"
    );
    assert!(
        out_slice.iter().any(|&v| v > 0.0),
        "expected at least one strictly positive output (relu pass-through)"
    );
}

#[test]
fn run_with_guards_rejects_bad_input_without_running_graph() {
    let program = dynamic_batch_program();
    let x: Tensor<f32> = ferrotorch_core::zeros(&[64, 10]).unwrap(); // batch > 32
    assert!(program.run_with_guards(&[x]).is_err());
}

#[test]
fn run_unchecked_executes_relu_graph() {
    // ExportedProgram::run bypasses guards — still must return correct shape
    // AND correct relu values. Mixed-sign input ensures a stub returning a
    // fresh zeros(&[4, 10]) of the right shape would fail this test.
    // i ∈ 0..40 → values span -20.0..=19.0, guaranteeing both signs.
    let program = all_static_program();
    let data: Vec<f32> = (0..40).map(|i| (i as f32) - 20.0).collect();
    let x: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![4, 10], false).unwrap();
    let out = program.run(&[x]).expect("run failed");
    assert_eq!(out.shape(), &[4, 10]);

    // Elementwise relu verification: out[i] == max(0.0, data[i]).
    let out_slice = out.data().expect("output data on CPU");
    assert_eq!(out_slice.len(), data.len());
    for (i, (&inp, &got)) in data.iter().zip(out_slice.iter()).enumerate() {
        let expected = inp.max(0.0);
        assert_eq!(
            got, expected,
            "relu mismatch at idx {i}: input={inp}, got={got}, expected={expected}"
        );
    }

    // Sabotage probes: input must contain a negative, output must contain
    // both an exact 0.0 (clamp happened) and a strictly positive value
    // (pass-through happened). A shape-only stub returning a fresh zeros
    // tensor would fail the "at least one positive output" check.
    assert!(
        data.iter().any(|&v| v < 0.0),
        "test setup invariant: input must contain at least one negative value"
    );
    assert!(
        out_slice.contains(&0.0),
        "expected at least one clamped-to-zero output (relu of negative input)"
    );
    assert!(
        out_slice.iter().any(|&v| v > 0.0),
        "expected at least one strictly positive output (relu pass-through)"
    );
}

// ---------------------------------------------------------------------------
// export() + export_with_dynamic_shapes() error paths
// ---------------------------------------------------------------------------

/// Minimal module that returns x.relu() — sufficient for export tracing.
struct ReluModule;

impl ferrotorch_nn::Module<f32> for ReluModule {
    fn forward(&self, x: &Tensor<f32>) -> ferrotorch_core::error::FerrotorchResult<Tensor<f32>> {
        x.relu()
    }

    fn parameters(&self) -> Vec<&ferrotorch_nn::parameter::Parameter<f32>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut ferrotorch_nn::parameter::Parameter<f32>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &ferrotorch_nn::parameter::Parameter<f32>)> {
        vec![]
    }

    // ReluModule has no parameters or training-mode state; train/eval are
    // no-ops by design (analogous to torch.nn.ReLU which carries no state).
    fn train(&mut self) { /* no training-mode state */
    }
    fn eval(&mut self) { /* no training-mode state */
    }
    fn is_training(&self) -> bool {
        false
    }
}

#[test]
fn export_single_input_produces_program() {
    let module = ReluModule;
    // export() traces via autograd; the example input must have requires_grad=true
    // so the tracer can record a grad_fn on the output.
    let example = ferrotorch_core::zeros::<f32>(&[2, 4])
        .unwrap()
        .requires_grad_(true);
    let program = export(&module, &[example]).expect("export failed");
    assert_eq!(program.input_shapes, vec![vec![2, 4]]);
    // Op-type discrimination: ReluModule::forward calls x.relu(), so the
    // traced graph must contain exactly one IrOpKind::Relu node. A stub
    // tracer that returned an empty graph or a wrong-op graph (e.g.
    // emitting Sigmoid/Identity/Add for relu) would fail this check; a
    // floor like `node_count() >= 1` would pass even for those stubs.
    let relu_node_count = program
        .graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, IrOpKind::Relu))
        .count();
    assert_eq!(
        relu_node_count,
        1,
        "expected exactly one IrOpKind::Relu node in the traced graph, got {relu_node_count}; \
         full op list: {:?}",
        program
            .graph
            .nodes
            .iter()
            .map(|n| &n.op)
            .collect::<Vec<_>>()
    );
    assert_eq!(program.output_shape, vec![2, 4]);
}

#[test]
fn export_rejects_multi_input() {
    let module = ReluModule;
    let a = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
    let b = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
    let err = export(&module, &[a, b]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("single-input") || msg.contains("exactly one"),
        "unexpected error: {msg}"
    );
}

#[test]
fn export_rejects_zero_inputs() {
    let module = ReluModule;
    let inputs: &[Tensor<f32>] = &[];
    assert!(export(&module, inputs).is_err());
}

#[test]
fn export_with_dynamic_shapes_rejects_spec_length_mismatch() {
    let module = ReluModule;
    let example = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
    // 0 specs for 1 example input
    let err = export_with_dynamic_shapes(&module, &[example], vec![]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("length") || msg.contains("mismatch"),
        "unexpected error: {msg}"
    );
}

#[test]
fn export_with_dynamic_shapes_rejects_spec_rank_mismatch() {
    let module = ReluModule;
    let example = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap(); // rank 2
    // spec has rank 3
    let spec = InputSpec::new(vec![
        DimSpec::Static(2),
        DimSpec::Static(4),
        DimSpec::Static(1),
    ]);
    let err = export_with_dynamic_shapes(&module, &[example], vec![spec]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("rank mismatch"), "unexpected error: {msg}");
}

#[test]
fn export_with_dynamic_shapes_rejects_static_dim_mismatch_in_spec() {
    let module = ReluModule;
    let example = ferrotorch_core::zeros::<f32>(&[2, 4]).unwrap();
    // spec says dim 0 is Static(5), but example is 2
    let spec = InputSpec::new(vec![DimSpec::Static(5), DimSpec::Static(4)]);
    let err = export_with_dynamic_shapes(&module, &[example], vec![spec]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("DimSpec::Static") || msg.contains("dim 0"),
        "unexpected error: {msg}"
    );
}

#[test]
fn export_with_dynamic_shapes_succeeds_with_dynamic_batch() {
    let module = ReluModule;
    // requires_grad=true so the tracer can record a grad_fn on the output.
    let example = ferrotorch_core::zeros::<f32>(&[4, 8])
        .unwrap()
        .requires_grad_(true);
    let spec = InputSpec::new(vec![DimSpec::dynamic("batch"), DimSpec::Static(8)]);
    let program = export_with_dynamic_shapes(&module, &[example], vec![spec])
        .expect("export_with_dynamic_shapes failed");
    assert_eq!(program.input_shapes, vec![vec![4, 8]]);
    assert!(program.input_specs[0].has_dynamic_dims());
}

// ---------------------------------------------------------------------------
// IrGraph binary serialize / deserialize
// ---------------------------------------------------------------------------

#[test]
fn ir_graph_serialize_deserialize_simple() {
    let mut g = IrGraph::new();
    let x = g.add_input(vec![3, 4]);
    let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![3, 4]]);
    g.set_outputs(vec![outs[0]]);

    let bytes = g.serialize();
    assert!(bytes.starts_with(b"FTIR"), "IrGraph magic missing");

    let g2 = IrGraph::deserialize(&bytes).expect("IrGraph deserialize failed");
    assert_eq!(g2.node_count(), g.node_count());
    assert_eq!(g2.value_count(), g.value_count());
    assert_eq!(g2.input_values.len(), g.input_values.len());
    assert_eq!(g2.output_values.len(), g.output_values.len());
}

#[test]
fn ir_graph_serialize_empty() {
    let g = IrGraph::new();
    let bytes = g.serialize();
    let g2 = IrGraph::deserialize(&bytes).expect("empty graph round-trip failed");
    assert_eq!(g2.node_count(), 0);
    assert_eq!(g2.value_count(), 0);
}

#[test]
fn ir_graph_deserialize_rejects_bad_magic() {
    let bad = b"NOPE0000extra_data_here";
    let result = IrGraph::deserialize(bad);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("invalid magic"), "unexpected: {msg}");
}

#[test]
fn ir_graph_serialize_preserves_dtype_f64() {
    let mut g = IrGraph::new();
    let _x = g.add_input_with_dtype(vec![4], Dtype::F64);
    let bytes = g.serialize();
    let g2 = IrGraph::deserialize(&bytes).expect("round-trip failed");
    assert_eq!(g2.values[0].dtype, Dtype::F64);
}

#[test]
fn ir_graph_serialize_all_op_kinds() {
    let ops: Vec<IrOpKind> = vec![
        IrOpKind::Add,
        IrOpKind::Sub,
        IrOpKind::Mul,
        IrOpKind::Div,
        IrOpKind::Neg,
        IrOpKind::Pow { exponent: 2.0 },
        IrOpKind::Sqrt,
        IrOpKind::Abs,
        IrOpKind::Exp,
        IrOpKind::Log,
        IrOpKind::Relu,
        IrOpKind::Sigmoid,
        IrOpKind::Tanh,
        IrOpKind::Gelu,
        IrOpKind::Silu,
        IrOpKind::Matmul,
        IrOpKind::Mm,
        IrOpKind::Mv,
        IrOpKind::Dot,
        IrOpKind::Transpose,
        IrOpKind::Linear,
        IrOpKind::Reshape { shape: vec![-1] },
        IrOpKind::Flatten,
        IrOpKind::Squeeze { axis: 0 },
        IrOpKind::Unsqueeze { axis: 0 },
        IrOpKind::Cat { axis: 1 },
        IrOpKind::Sum,
        IrOpKind::Mean,
        IrOpKind::Prod,
        IrOpKind::Softmax,
        IrOpKind::LogSoftmax,
    ];
    let mut g = IrGraph::new();
    for op in &ops {
        g.add_node(op.clone(), vec![], vec![vec![1]]);
    }
    let bytes = g.serialize();
    let g2 = IrGraph::deserialize(&bytes).expect("all-op-kinds round-trip failed");
    assert_eq!(g2.node_count(), ops.len());
    for (orig, deser) in g.nodes.iter().zip(g2.nodes.iter()) {
        assert_eq!(orig.op, deser.op);
    }
}

// ---------------------------------------------------------------------------
// JitError — Display + From conversion
// ---------------------------------------------------------------------------

#[test]
fn jit_error_tracing_error_display() {
    let e = JitError::TracingError {
        message: "test failure".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("tracing error"), "unexpected: {msg}");
    assert!(msg.contains("test failure"), "unexpected: {msg}");
}

#[test]
fn jit_error_unsupported_op_display() {
    let e = JitError::UnsupportedOp {
        op: "some_op".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("unsupported operation"), "unexpected: {msg}");
    assert!(msg.contains("some_op"), "unexpected: {msg}");
}

#[test]
fn jit_error_shape_mismatch_display() {
    let e = JitError::ShapeMismatch {
        traced: vec![2, 3],
        actual: vec![4, 3],
    };
    let msg = e.to_string();
    assert!(msg.contains("shape mismatch"), "unexpected: {msg}");
    assert!(msg.contains("[2, 3]"), "unexpected: {msg}");
    assert!(msg.contains("[4, 3]"), "unexpected: {msg}");
}

#[test]
fn jit_error_codegen_error_display() {
    let e = JitError::CodegenError {
        message: "ptx failed".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("codegen error"), "unexpected: {msg}");
    assert!(msg.contains("ptx failed"), "unexpected: {msg}");
}

#[test]
fn jit_error_serialization_error_display() {
    let e = JitError::SerializationError {
        message: "truncated".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("serialization error"), "unexpected: {msg}");
    assert!(msg.contains("truncated"), "unexpected: {msg}");
}

#[test]
fn jit_error_graph_break_display() {
    let e = JitError::GraphBreak {
        op: "print".to_string(),
        reason: "side effect".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("graph break"), "unexpected: {msg}");
    assert!(msg.contains("print"), "unexpected: {msg}");
}

#[test]
fn jit_error_export_error_display() {
    let e = JitError::ExportError {
        op: "random_op".to_string(),
        reason: "data dependent".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("export error"), "unexpected: {msg}");
    assert!(msg.contains("fullgraph"), "unexpected: {msg}");
}

#[test]
fn jit_error_parameter_error_display() {
    let e = JitError::ParameterError {
        message: "bad lr".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("parameter error"), "unexpected: {msg}");
    assert!(msg.contains("bad lr"), "unexpected: {msg}");
}

#[test]
fn jit_error_unsupported_display() {
    let e = JitError::Unsupported {
        op: "exp".to_string(),
        dtype: "f64".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("GPU codegen"), "unexpected: {msg}");
    assert!(msg.contains("exp"), "unexpected: {msg}");
    assert!(msg.contains("f64"), "unexpected: {msg}");
}

#[test]
fn jit_error_converts_to_ferrotorch_error() {
    use ferrotorch_core::error::FerrotorchError;
    let jit_err = JitError::TracingError {
        message: "boom".to_string(),
    };
    let ft_err: FerrotorchError = jit_err.into();
    let msg = ft_err.to_string();
    assert!(msg.contains("boom"), "unexpected: {msg}");
}

// ---------------------------------------------------------------------------
// AutotuneKey — from_graph
// ---------------------------------------------------------------------------

fn make_simple_graph() -> IrGraph {
    let mut g = IrGraph::new();
    let x = g.add_input(vec![2, 4]);
    let (_, outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![2, 4]]);
    g.set_outputs(vec![outs[0]]);
    g
}

#[test]
fn autotune_key_from_graph_is_stable() {
    let g = make_simple_graph();
    let input_shapes = vec![vec![2usize, 4]];
    let key1 = AutotuneKey::from_graph(&g, &input_shapes);
    let key2 = AutotuneKey::from_graph(&g, &input_shapes);
    assert_eq!(
        key1, key2,
        "AutotuneKey must be stable for the same graph + shapes"
    );
}

#[test]
fn autotune_key_differs_for_different_shapes() {
    let g = make_simple_graph();
    let key_a = AutotuneKey::from_graph(&g, &[vec![2, 4]]);
    let key_b = AutotuneKey::from_graph(&g, &[vec![4, 4]]);
    assert_ne!(
        key_a, key_b,
        "AutotuneKey must differ when input shapes differ"
    );
}

// ---------------------------------------------------------------------------
// AutotuneCandidate — new + name
// ---------------------------------------------------------------------------

#[test]
fn autotune_candidate_name_roundtrip() {
    let candidate = AutotuneCandidate::new("interpreter", Box::new(InterpreterBackend));
    assert_eq!(candidate.name(), "interpreter");
}

// ---------------------------------------------------------------------------
// Autotuner — builder API + accessors
// ---------------------------------------------------------------------------

#[test]
fn autotuner_default_settings() {
    let tuner = Autotuner::new();
    // Default: 5 iterations, 1 warmup, 0 candidates.
    assert_eq!(tuner.iterations(), 5);
    assert_eq!(tuner.warmup(), 1);
    assert_eq!(tuner.candidate_count(), 0);
    assert_eq!(tuner.cache_size(), 0);
}

#[test]
fn autotuner_with_iterations_and_warmup() {
    let tuner = Autotuner::new().with_iterations(10).with_warmup(3);
    assert_eq!(tuner.iterations(), 10);
    assert_eq!(tuner.warmup(), 3);
}

#[test]
fn autotuner_with_candidate_increments_count() {
    let tuner = Autotuner::new()
        .with_candidate("a", Box::new(InterpreterBackend))
        .with_candidate("b", Box::new(InterpreterBackend));
    assert_eq!(tuner.candidate_count(), 2);
}

#[test]
fn autotuner_cached_miss_returns_none() {
    let g = make_simple_graph();
    let key = AutotuneKey::from_graph(&g, &[vec![2, 4]]);
    let tuner = Autotuner::new();
    assert!(tuner.cached(&key).is_none());
}

#[test]
fn autotuner_tune_single_candidate_returns_result() {
    let g = make_simple_graph();
    let tuner = Autotuner::new()
        .with_candidate("interpreter", Box::new(InterpreterBackend))
        .with_iterations(1)
        .with_warmup(0);
    // tune() takes f64 inputs mirroring the graph's constant domain
    let inputs: Vec<Vec<f64>> = vec![vec![1.0f64; 8]];
    let result = tuner.tune(&g, &inputs).expect("tune failed");
    assert_eq!(result.winner_name(), "interpreter");
    assert_eq!(result.all_timings().len(), 1);
    // Winner compiled graph must be usable.
    let _ = result.winner_compiled();
}

#[test]
fn autotuner_tune_caches_result() {
    let g = make_simple_graph();
    let tuner = Autotuner::new()
        .with_candidate("interpreter", Box::new(InterpreterBackend))
        .with_iterations(1)
        .with_warmup(0);
    let inputs: Vec<Vec<f64>> = vec![vec![0.0f64; 8]];
    tuner.tune(&g, &inputs).expect("first tune failed");

    let key = AutotuneKey::from_graph(&g, &[vec![2, 4]]);
    assert!(
        tuner.cached(&key).is_some(),
        "result should be cached after tune()"
    );
}

#[test]
fn autotuner_clear_cache_empties_cache() {
    let g = make_simple_graph();
    let tuner = Autotuner::new()
        .with_candidate("interpreter", Box::new(InterpreterBackend))
        .with_iterations(1)
        .with_warmup(0);
    let inputs: Vec<Vec<f64>> = vec![vec![0.0f64; 8]];
    tuner.tune(&g, &inputs).expect("tune failed");
    assert!(
        tuner.cache_size() > 0,
        "cache should not be empty after tune"
    );
    tuner.clear_cache();
    assert_eq!(
        tuner.cache_size(),
        0,
        "cache_size() must be 0 after clear_cache"
    );
}

#[test]
fn autotuner_result_winner_time_is_nonzero_or_zero() {
    // Verifies the AutotuneResult accessors expose values consistent with
    // the candidates registered. A stub tuner that hardcoded a default
    // result (e.g. `Default::default()` with empty timings or a winner
    // name of `""`) would now fail rather than silently passing.
    let g = make_simple_graph();
    let tuner = Autotuner::new()
        .with_candidate("interpreter", Box::new(InterpreterBackend))
        .with_iterations(1)
        .with_warmup(0);
    let inputs: Vec<Vec<f64>> = vec![vec![0.0f64; 8]];
    let result = tuner.tune(&g, &inputs).expect("tune failed");

    // (1) Sanity bound on winner_time — anything ≥ 10s for a one-iteration
    // run on a tiny graph indicates a broken accessor (e.g. returning a
    // bogus `Duration::MAX` placeholder). Lower bound is implicit:
    // `Duration` is non-negative by construction.
    let winner_time_ns = result.winner_time().as_nanos();
    assert!(
        winner_time_ns < 10_000_000_000,
        "winner_time() {winner_time_ns}ns exceeds 10s sanity ceiling — \
         tuner did not measure a real timing"
    );

    // (2) Exactly one candidate registered → exactly one timing reported,
    // and its name must match what we registered. A stub returning an
    // empty `all_timings()` slice would fail length; one returning a
    // mislabelled entry would fail the name check.
    let timings = result.all_timings();
    assert_eq!(
        timings.len(),
        1,
        "expected exactly 1 timing entry for 1 registered candidate, got {}: {:?}",
        timings.len(),
        timings
    );
    assert_eq!(
        timings[0].0, "interpreter",
        "all_timings()[0] candidate name mismatch: got {:?}",
        timings[0].0
    );

    // (3) Winner name must match the only candidate registered. This is
    // the load-bearing behavioral check — the sabotage probe in the
    // pre-flight (changing the registered name to "typo_name") trips
    // exactly here.
    assert_eq!(
        result.winner_name(),
        "interpreter",
        "winner_name() must match the sole registered candidate"
    );
}
