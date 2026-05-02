//! JIT kernel autotuning.
//!
//! [`Autotuner`] benchmarks a set of candidate [`Codegen`] backends (or
//! configurations of the same backend) against a traced graph and
//! picks the fastest one. Results are cached keyed by a
//! shape-fingerprint so the same problem is only timed once — subsequent
//! calls return the winning compiled graph from cache.
//!
//! # What gets tuned
//!
//! Autotuning is a search problem: the user supplies candidates and the
//! tuner times them. Typical candidates:
//!
//! - Different backends: `InterpreterBackend` vs `NativeBackend` vs
//!   `InductorBackend`.
//! - Different optimization configs: fusion on/off, constant folding
//!   on/off, dead-code elimination on/off.
//! - Different block sizes for GPU codegen (via
//!   [`InductorBackend::with_block_size`]).
//!
//! The tuner doesn't care what the difference is — it runs each
//! candidate against the same inputs, records the median wall-clock
//! time, and picks the fastest.
//!
//! # Caching
//!
//! Tuning results are cached behind a `Mutex<HashMap<AutotuneKey, …>>`.
//! The key combines the graph's structural fingerprint (node count,
//! op-kind sequence, input shapes) with the input-shape slice. This
//! means a graph retraced with a different batch size will trigger a
//! fresh tune, but the same graph + same shapes will hit the cache.
//!
//! ```ignore
//! use ferrotorch_jit::autotune::{Autotuner, AutotuneCandidate};
//! use ferrotorch_jit::codegen::{Codegen, InterpreterBackend, NativeBackend};
//!
//! let tuner = Autotuner::new()
//!     .with_candidate("interpreter", Box::new(InterpreterBackend))
//!     .with_candidate("native", Box::new(NativeBackend))
//!     .with_iterations(10)
//!     .with_warmup(2);
//!
//! let result = tuner.tune(&graph, &inputs)?;
//! println!("Winner: {} ({:?})", result.winner_name(), result.winner_time());
//! let output = result.winner_compiled().execute(&inputs)?;
//! ```
//!
//! CL-369.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};

use crate::codegen::{Codegen, CompiledGraph};
use crate::graph::IrGraph;

// ---------------------------------------------------------------------------
// AutotuneKey — stable cache key derived from graph + input shapes
// ---------------------------------------------------------------------------

/// Structural fingerprint of a graph used as a cache key.
///
/// Combines the shape of every graph input with the sequence of op
/// discriminants in topological order. Two graphs with the same
/// fingerprint are structurally identical up to constant values and
/// node IDs, which is sufficient for autotune cache lookup because
/// tuning operates at the shape-times-op-sequence level.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AutotuneKey {
    /// Concrete input shapes (one `Vec<usize>` per input).
    pub input_shapes: Vec<Vec<usize>>,
    /// Topologically-ordered op-kind fingerprints.
    /// We use a lightweight string representation instead of a full
    /// `IrOpKind` hash because `IrOpKind` contains f64 constants which
    /// don't implement `Hash`.
    pub op_sequence: Vec<String>,
}

impl AutotuneKey {
    /// Build a cache key from an IR graph and the caller-provided
    /// input shapes.
    pub fn from_graph(graph: &IrGraph, input_shapes: &[Vec<usize>]) -> Self {
        let topo = graph.topological_order();
        let node_map: HashMap<_, _> = graph.nodes.iter().map(|n| (n.id, n)).collect();
        let op_sequence: Vec<String> = topo
            .iter()
            .map(|nid| format!("{:?}", node_map[nid].op))
            .collect();
        Self {
            input_shapes: input_shapes.to_vec(),
            op_sequence,
        }
    }
}

// ---------------------------------------------------------------------------
// Candidate
// ---------------------------------------------------------------------------

/// A named candidate backend for the autotuner to benchmark.
pub struct AutotuneCandidate {
    name: String,
    backend: Box<dyn Codegen>,
}

impl AutotuneCandidate {
    /// Create a new candidate.
    pub fn new(name: impl Into<String>, backend: Box<dyn Codegen>) -> Self {
        Self {
            name: name.into(),
            backend,
        }
    }

    /// Human-readable name for this candidate.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// AutotuneResult
// ---------------------------------------------------------------------------

/// Result of a single `Autotuner::tune` call.
///
/// Holds the winning compiled graph, the winning candidate's name and
/// measured time, and the full timing table for every candidate.
#[derive(Debug)]
pub struct AutotuneResult {
    winner_name: String,
    winner_time: Duration,
    winner_compiled: CompiledGraph,
    all_timings: Vec<(String, Duration)>,
}

impl AutotuneResult {
    /// Name of the winning candidate.
    pub fn winner_name(&self) -> &str {
        &self.winner_name
    }

    /// Measured median time of the winning candidate.
    pub fn winner_time(&self) -> Duration {
        self.winner_time
    }

    /// Compiled graph for the winning candidate — ready to execute.
    pub fn winner_compiled(&self) -> &CompiledGraph {
        &self.winner_compiled
    }

    /// Full timing table for every candidate (name, median time).
    /// Entries are in the order the candidates were registered.
    pub fn all_timings(&self) -> &[(String, Duration)] {
        &self.all_timings
    }
}

// ---------------------------------------------------------------------------
// Autotuner
// ---------------------------------------------------------------------------

/// A configurable kernel autotuner that benchmarks candidate backends
/// and caches the winner keyed by graph fingerprint + input shapes.
pub struct Autotuner {
    candidates: Vec<AutotuneCandidate>,
    iterations: usize,
    warmup: usize,
    /// Cache: fingerprint -> (winner name, winner median time).
    ///
    /// We only cache the *decision* (which candidate to use) rather
    /// than the compiled graph itself because `CompiledGraph` is not
    /// `Clone`. On a cache hit, the caller rebuilds the winning
    /// compiled graph by invoking the candidate's backend again —
    /// compilation is cheap relative to tuning.
    cache: Mutex<HashMap<AutotuneKey, (String, Duration)>>,
}

impl Default for Autotuner {
    fn default() -> Self {
        Self::new()
    }
}

impl Autotuner {
    /// Create an empty autotuner with no candidates.
    ///
    /// Default settings: 5 timed iterations, 1 warmup iteration.
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            iterations: 5,
            warmup: 1,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Register a candidate backend under a given name.
    pub fn with_candidate(mut self, name: impl Into<String>, backend: Box<dyn Codegen>) -> Self {
        self.candidates.push(AutotuneCandidate::new(name, backend));
        self
    }

    /// Set the number of timed iterations per candidate (default: 5).
    ///
    /// The median of these iterations is used as the candidate's score.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        assert!(iterations > 0, "iterations must be > 0");
        self.iterations = iterations;
        self
    }

    /// Set the number of warmup iterations per candidate (default: 1).
    ///
    /// Warmup runs are executed and discarded before timing starts;
    /// they exist to prime caches, allocate buffers, and amortize
    /// first-run JIT costs.
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Number of timed iterations per candidate.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Number of warmup iterations per candidate.
    pub fn warmup(&self) -> usize {
        self.warmup
    }

    /// Number of candidates registered.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Look up a cached winner name for the given key, if any.
    ///
    /// Returns `Some((name, median_time))` if a tune has already been
    /// performed for this key.
    pub fn cached(&self, key: &AutotuneKey) -> Option<(String, Duration)> {
        self.cache
            .lock()
            .unwrap()
            .get(key)
            .map(|(n, t)| (n.clone(), *t))
    }

    /// Benchmark every candidate against the graph, record the winner,
    /// cache the decision, and return an [`AutotuneResult`] containing
    /// the winning compiled graph.
    ///
    /// On a cache hit, only the winning candidate is re-compiled and
    /// re-benchmarked (so [`AutotuneResult::all_timings`] on a hit
    /// contains only the winner's row).
    pub fn tune(&self, graph: &IrGraph, inputs: &[Vec<f64>]) -> FerrotorchResult<AutotuneResult> {
        if self.candidates.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "Autotuner: no candidates registered".into(),
            });
        }

        // Derive input shapes from the graph for the cache key.
        // The graph's input_values each have a shape field.
        let input_shapes: Vec<Vec<usize>> = graph
            .input_values
            .iter()
            .map(|vid| {
                graph
                    .values
                    .iter()
                    .find(|v| v.id == *vid)
                    .map(|v| v.shape.clone())
                    .unwrap_or_default()
            })
            .collect();
        let key = AutotuneKey::from_graph(graph, &input_shapes);

        // Cache hit: rebuild winning compiled graph and return.
        if let Some((cached_name, cached_time)) = self.cached(&key) {
            let candidate = self
                .candidates
                .iter()
                .find(|c| c.name == cached_name)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "Autotuner: cached winner '{cached_name}' is not among current candidates"
                    ),
                })?;
            let compiled = candidate.backend.compile(graph)?;
            return Ok(AutotuneResult {
                winner_name: cached_name.clone(),
                winner_time: cached_time,
                winner_compiled: compiled,
                all_timings: vec![(cached_name, cached_time)],
            });
        }

        // Full tune: benchmark every candidate.
        let mut all_timings: Vec<(String, Duration)> = Vec::with_capacity(self.candidates.len());
        let mut best: Option<(usize, Duration)> = None;

        for (i, candidate) in self.candidates.iter().enumerate() {
            let compiled = candidate.backend.compile(graph)?;

            // Warmup.
            for _ in 0..self.warmup {
                let _ = compiled.execute(inputs)?;
            }

            // Timed iterations — record each sample then take median.
            let mut samples: Vec<Duration> = Vec::with_capacity(self.iterations);
            for _ in 0..self.iterations {
                let start = Instant::now();
                let _ = compiled.execute(inputs)?;
                samples.push(start.elapsed());
            }
            samples.sort();
            let median = samples[samples.len() / 2];

            all_timings.push((candidate.name.clone(), median));

            match best {
                None => best = Some((i, median)),
                Some((_, best_time)) if median < best_time => {
                    best = Some((i, median));
                }
                _ => {}
            }
        }

        let (winner_idx, winner_time) = best.expect("candidates is non-empty");
        let winner = &self.candidates[winner_idx];
        let winner_name = winner.name.clone();

        // Record the decision in cache.
        self.cache
            .lock()
            .unwrap()
            .insert(key, (winner_name.clone(), winner_time));

        // Re-compile the winner for the result (we consumed the first
        // compilation during timing).
        let winner_compiled = winner.backend.compile(graph)?;

        Ok(AutotuneResult {
            winner_name,
            winner_time,
            winner_compiled,
            all_timings,
        })
    }

    /// Clear the autotune cache. Primarily useful for tests and for
    /// cases where the user wants to force a re-tune.
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Current size of the autotune cache.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{InterpreterBackend, NativeBackend};
    use crate::graph::{IrGraph, IrOpKind};

    fn build_unary_chain_graph() -> (IrGraph, Vec<Vec<f64>>) {
        // input[3] -> relu -> sqrt -> output
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, relu_out) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![3]]);
        let (_, sqrt_out) = g.add_node(IrOpKind::Sqrt, vec![relu_out[0]], vec![vec![3]]);
        g.set_outputs(vec![sqrt_out[0]]);
        let inputs = vec![vec![1.0, 4.0, 9.0]];
        (g, inputs)
    }

    #[test]
    fn test_autotune_empty_candidates_errors() {
        let tuner = Autotuner::new();
        let (g, inputs) = build_unary_chain_graph();
        let r = tuner.tune(&g, &inputs);
        assert!(r.is_err());
        assert!(format!("{}", r.unwrap_err()).contains("no candidates"));
    }

    #[test]
    fn test_autotune_picks_a_winner_from_two_candidates() {
        let tuner = Autotuner::new()
            .with_candidate("interpreter", Box::new(InterpreterBackend))
            .with_candidate("native", Box::new(NativeBackend))
            .with_iterations(3)
            .with_warmup(1);
        let (g, inputs) = build_unary_chain_graph();

        let result = tuner.tune(&g, &inputs).unwrap();

        // Winner should be one of the two registered candidates.
        assert!(
            result.winner_name() == "interpreter" || result.winner_name() == "native",
            "got winner = {}",
            result.winner_name()
        );
        assert_eq!(result.all_timings().len(), 2);

        // The winning compiled graph should execute and produce the
        // expected output (sqrt of relu([1,4,9]) = [1,2,3]).
        let out = result.winner_compiled().execute(&inputs).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - 2.0).abs() < 1e-10);
        assert!((out[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_autotune_cache_hit_returns_single_timing_row() {
        let tuner = Autotuner::new()
            .with_candidate("interpreter", Box::new(InterpreterBackend))
            .with_candidate("native", Box::new(NativeBackend))
            .with_iterations(2)
            .with_warmup(0);
        let (g, inputs) = build_unary_chain_graph();

        // First call: full tune, 2 timing rows, cache grows to 1.
        let first = tuner.tune(&g, &inputs).unwrap();
        assert_eq!(first.all_timings().len(), 2);
        assert_eq!(tuner.cache_size(), 1);

        // Second call: cache hit — 1 timing row, same winner name.
        let second = tuner.tune(&g, &inputs).unwrap();
        assert_eq!(second.all_timings().len(), 1);
        assert_eq!(second.winner_name(), first.winner_name());
        assert_eq!(tuner.cache_size(), 1);

        // Clearing the cache forces a retune.
        tuner.clear_cache();
        assert_eq!(tuner.cache_size(), 0);
        let third = tuner.tune(&g, &inputs).unwrap();
        assert_eq!(third.all_timings().len(), 2);
    }

    #[test]
    fn test_autotune_key_is_shape_sensitive() {
        // Two graphs with the same ops but different input shapes
        // should produce different cache keys.
        let mut g1 = IrGraph::new();
        let x1 = g1.add_input(vec![4]);
        let (_, r1) = g1.add_node(IrOpKind::Relu, vec![x1], vec![vec![4]]);
        g1.set_outputs(vec![r1[0]]);

        let mut g2 = IrGraph::new();
        let x2 = g2.add_input(vec![8]);
        let (_, r2) = g2.add_node(IrOpKind::Relu, vec![x2], vec![vec![8]]);
        g2.set_outputs(vec![r2[0]]);

        let k1 = AutotuneKey::from_graph(&g1, &[vec![4]]);
        let k2 = AutotuneKey::from_graph(&g2, &[vec![8]]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_autotune_key_is_op_sensitive() {
        // Two graphs with the same input shape but different ops
        // should produce different keys.
        let mut g1 = IrGraph::new();
        let x1 = g1.add_input(vec![3]);
        let (_, r1) = g1.add_node(IrOpKind::Relu, vec![x1], vec![vec![3]]);
        g1.set_outputs(vec![r1[0]]);

        let mut g2 = IrGraph::new();
        let x2 = g2.add_input(vec![3]);
        let (_, r2) = g2.add_node(IrOpKind::Sigmoid, vec![x2], vec![vec![3]]);
        g2.set_outputs(vec![r2[0]]);

        let k1 = AutotuneKey::from_graph(&g1, &[vec![3]]);
        let k2 = AutotuneKey::from_graph(&g2, &[vec![3]]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_autotune_honors_iterations_and_warmup_config() {
        let tuner = Autotuner::new()
            .with_candidate("interp", Box::new(InterpreterBackend))
            .with_iterations(7)
            .with_warmup(3);
        assert_eq!(tuner.iterations(), 7);
        assert_eq!(tuner.warmup(), 3);
        assert_eq!(tuner.candidate_count(), 1);
    }

    #[test]
    #[should_panic(expected = "iterations must be > 0")]
    fn test_autotune_rejects_zero_iterations() {
        let _ = Autotuner::new().with_iterations(0);
    }

    #[test]
    fn test_autotune_with_single_candidate_still_works() {
        let tuner = Autotuner::new()
            .with_candidate("interp", Box::new(InterpreterBackend))
            .with_iterations(2)
            .with_warmup(0);
        let (g, inputs) = build_unary_chain_graph();
        let r = tuner.tune(&g, &inputs).unwrap();
        assert_eq!(r.winner_name(), "interp");
        assert_eq!(r.all_timings().len(), 1);
    }
}
