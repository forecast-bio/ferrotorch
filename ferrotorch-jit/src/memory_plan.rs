//! Static memory planning for the IR graph.
//!
//! Computes buffer-slot assignments by analysing per-value liveness ranges
//! over a topological execution order, so non-overlapping values share the
//! same slot. The resulting [`MemoryPlan`] feeds backend allocators that
//! preallocate one buffer per slot instead of one per IR value.

use std::collections::HashMap;

use crate::graph::{IrGraph, IrNodeId, IrValueId};

/// Result of memory planning: buffer slot assignments for each IR value.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Map from `IrValueId` to buffer slot index.
    pub assignments: HashMap<IrValueId, usize>,
    /// Size of each buffer slot in elements.
    pub slot_sizes: Vec<usize>,
    /// Total slots needed (peak concurrent buffers).
    pub num_slots: usize,
    /// Total memory if no reuse (for comparison), in elements.
    pub naive_total: usize,
    /// Total memory with reuse, in elements.
    pub planned_total: usize,
}

impl MemoryPlan {
    /// Return the savings ratio as a percentage (0.0 to 100.0).
    ///
    /// A value of 50.0 means planned memory is half of naive memory.
    pub fn savings_percent(&self) -> f64 {
        if self.naive_total == 0 {
            return 0.0;
        }
        let saved = self.naive_total.saturating_sub(self.planned_total);
        (saved as f64 / self.naive_total as f64) * 100.0
    }
}

/// Liveness interval for a single IR value.
///
/// `born` is the topological index of the node that produces this value.
/// `last_use` is the topological index of the last node that consumes it.
/// A value that is never consumed (e.g. a graph output with no downstream
/// node) has `last_use == born`.
#[derive(Debug, Clone, Copy)]
struct LiveInterval {
    born: usize,
    last_use: usize,
}

/// Compute the number of elements a value occupies (product of shape dims).
fn value_num_elements(graph: &IrGraph, value: IrValueId) -> usize {
    graph.values.iter().find(|v| v.id == value).map_or(1, |v| {
        if v.shape.is_empty() {
            1
        } else {
            v.shape.iter().product()
        }
    })
}

/// Analyze the IR graph and produce a memory plan that maximizes buffer reuse.
///
/// The algorithm works in three stages:
///
/// 1. **Liveness analysis** — For each value produced by a node in the graph,
///    determine the topological index at which it is produced (`born`) and the
///    last topological index at which it is consumed as an input (`last_use`).
///
/// 2. **Greedy first-fit allocation** — Iterate over values in topological
///    (birth) order. For each value, find the smallest free slot whose size
///    is at least as large as the value's element count. If no suitable free
///    slot exists, allocate a new one. A slot becomes free once the
///    topological index passes the value's `last_use`.
///
/// 3. **Statistics** — Report naive total (sum of all value sizes), planned
///    total (sum of slot sizes), and the number of slots.
pub fn plan_memory(graph: &IrGraph) -> MemoryPlan {
    let topo = graph.topological_order();

    if topo.is_empty() {
        return MemoryPlan {
            assignments: HashMap::new(),
            slot_sizes: Vec::new(),
            num_slots: 0,
            naive_total: 0,
            planned_total: 0,
        };
    }

    // Map from node id to its topological index.
    let topo_index: HashMap<IrNodeId, usize> =
        topo.iter().enumerate().map(|(i, &nid)| (nid, i)).collect();

    // -----------------------------------------------------------------------
    // Step 1: Compute liveness intervals.
    // -----------------------------------------------------------------------

    // Collect all values that we will plan memory for (those with a producer
    // node that appears in the topological order).
    let mut live_intervals: HashMap<IrValueId, LiveInterval> = HashMap::new();

    for val in &graph.values {
        let producer_id = match val.producer {
            Some(id) => id,
            None => continue,
        };
        let &born = match topo_index.get(&producer_id) {
            Some(idx) => idx,
            None => continue,
        };
        live_intervals.insert(
            val.id,
            LiveInterval {
                born,
                last_use: born,
            },
        );
    }

    // Extend last_use for every consuming node.
    for node in &graph.nodes {
        let &node_topo = match topo_index.get(&node.id) {
            Some(idx) => idx,
            None => continue,
        };
        for &input_val in &node.inputs {
            if let Some(interval) = live_intervals.get_mut(&input_val) {
                if node_topo > interval.last_use {
                    interval.last_use = node_topo;
                }
            }
        }
    }

    // Graph output values must stay live until the very end — they cannot be
    // overwritten by later values.
    let max_topo = topo.len().saturating_sub(1);
    for &out_val in &graph.output_values {
        if let Some(interval) = live_intervals.get_mut(&out_val) {
            interval.last_use = max_topo;
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Greedy first-fit allocation.
    // -----------------------------------------------------------------------

    // Sort values by their birth order (topological index of producer).
    // Ties are broken by value id for determinism.
    let mut values_by_birth: Vec<IrValueId> = live_intervals.keys().copied().collect();
    values_by_birth.sort_by(|a, b| {
        let ia = &live_intervals[a];
        let ib = &live_intervals[b];
        ia.born.cmp(&ib.born).then_with(|| a.0.cmp(&b.0))
    });

    let mut assignments: HashMap<IrValueId, usize> = HashMap::new();
    let mut slot_sizes: Vec<usize> = Vec::new();

    // Track the latest last_use of any value currently occupying each slot.
    // A slot is free when its occupancy value is strictly less than the
    // current value's born index.
    let mut slot_occupancy: Vec<usize> = Vec::new();

    let mut naive_total: usize = 0;

    for &val_id in &values_by_birth {
        let interval = live_intervals[&val_id];
        let size = value_num_elements(graph, val_id);
        naive_total += size;

        // Find free slots: a slot is free if its latest occupant's last_use
        // is strictly less than this value's born index.
        // Among free slots, pick the smallest one that fits (>= size).
        let mut best_slot: Option<usize> = None;
        let mut best_slot_size: usize = usize::MAX;

        for (slot_idx, &occupant_last_use) in slot_occupancy.iter().enumerate() {
            if occupant_last_use < interval.born
                && slot_sizes[slot_idx] >= size
                && slot_sizes[slot_idx] < best_slot_size
            {
                best_slot = Some(slot_idx);
                best_slot_size = slot_sizes[slot_idx];
            }
        }

        if let Some(slot_idx) = best_slot {
            assignments.insert(val_id, slot_idx);
            // Update the occupancy to reflect the new value's lifetime.
            slot_occupancy[slot_idx] = interval.last_use;
        } else {
            // Allocate a new slot.
            let slot_idx = slot_sizes.len();
            slot_sizes.push(size);
            slot_occupancy.push(interval.last_use);
            assignments.insert(val_id, slot_idx);
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Statistics.
    // -----------------------------------------------------------------------

    let num_slots = slot_sizes.len();
    let planned_total: usize = slot_sizes.iter().sum();

    MemoryPlan {
        assignments,
        slot_sizes,
        num_slots,
        naive_total,
        planned_total,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};

    /// Simple chain: input -> relu -> sigmoid -> output.
    ///
    /// Each intermediate value is consumed exactly once and sequentially,
    /// so buffer slots can be reused for non-overlapping lifetimes.
    ///
    /// With shape [100], each value is 100 elements.
    /// Values: input(v0), `relu_out(v1)`, `sigmoid_out(v2)`.
    /// Liveness (topo indices: Input=0, Relu=1, Sigmoid=2):
    ///   v0: born=0, `last_use=1`  (consumed by relu at topo index 1)
    ///   v1: born=1, `last_use=2`  (consumed by sigmoid at topo index 2)
    ///   v2: born=2, `last_use=2`  (graph output, pinned to last topo index = 2)
    ///
    /// Allocation order (by birth): v0, v1, v2.
    ///   v0 -> slot 0 (new, size 100)
    ///   v1 -> born=1, slot 0 occupancy=1, 1 < 1 is false -> new slot 1
    ///   v2 -> born=2, slot 0 occupancy=1, 1 < 2 yes -> reuse slot 0
    ///
    /// 2 slots instead of 3. Savings.
    #[test]
    fn test_simple_chain_reuses_buffers() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![100]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![100]]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_outs[0]], vec![vec![100]]);
        g.set_outputs(vec![sig_outs[0]]);

        let plan = plan_memory(&g);

        // Naive: 3 values * 100 = 300 elements.
        assert_eq!(plan.naive_total, 300);

        // We should need fewer slots than values.
        assert!(
            plan.num_slots < 3,
            "chain should reuse at least one slot, got {} slots",
            plan.num_slots
        );

        // Planned total should be less than naive.
        assert!(
            plan.planned_total < plan.naive_total,
            "planned {} should be less than naive {}",
            plan.planned_total,
            plan.naive_total
        );

        // Every value should have an assignment.
        assert_eq!(plan.assignments.len(), 3);

        // Savings should be positive.
        assert!(plan.savings_percent() > 0.0);
    }

    /// Diamond graph: input fans out to two branches, then merges.
    ///
    ///       input (v0)
    ///       /        \
    ///    relu(v1)  sigmoid(v2)
    ///       \        /
    ///        add (v3)  -> output
    ///
    /// v1 and v2 are alive concurrently (both consumed by add),
    /// so they MUST occupy different slots.
    #[test]
    fn test_diamond_needs_two_concurrent_slots() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![50]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![50]]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![x], vec![vec![50]]);
        let (_, add_outs) = g.add_node(
            IrOpKind::Add,
            vec![relu_outs[0], sig_outs[0]],
            vec![vec![50]],
        );
        g.set_outputs(vec![add_outs[0]]);

        let plan = plan_memory(&g);

        // 4 values: x, relu_out, sigmoid_out, add_out.
        assert_eq!(plan.assignments.len(), 4);

        // relu_out and sigmoid_out must be in different slots.
        let relu_slot = plan.assignments[&relu_outs[0]];
        let sig_slot = plan.assignments[&sig_outs[0]];
        assert_ne!(
            relu_slot, sig_slot,
            "concurrent values must be in different slots"
        );

        // Naive: 4 * 50 = 200 elements.
        assert_eq!(plan.naive_total, 200);

        // We should still get some reuse (add_out can reuse a slot freed
        // after relu/sigmoid are consumed).
        assert!(
            plan.planned_total <= plan.naive_total,
            "planned {} should be <= naive {}",
            plan.planned_total,
            plan.naive_total
        );
    }

    /// Verify savings percentage calculation on a longer chain.
    #[test]
    fn test_savings_percentage() {
        let mut g = IrGraph::new();

        // Long chain: input -> relu -> sigmoid -> tanh -> neg -> output.
        // Each step frees the previous intermediate, so heavy reuse.
        let shape = vec![1000];
        let x = g.add_input(shape.clone());
        let (_, v1) = g.add_node(IrOpKind::Relu, vec![x], vec![shape.clone()]);
        let (_, v2) = g.add_node(IrOpKind::Sigmoid, vec![v1[0]], vec![shape.clone()]);
        let (_, v3) = g.add_node(IrOpKind::Tanh, vec![v2[0]], vec![shape.clone()]);
        let (_, v4) = g.add_node(IrOpKind::Neg, vec![v3[0]], vec![shape.clone()]);
        g.set_outputs(vec![v4[0]]);

        let plan = plan_memory(&g);

        // 5 values, 1000 elements each -> naive = 5000.
        assert_eq!(plan.naive_total, 5000);

        // With reuse, we should need far fewer than 5 slots.
        assert!(
            plan.num_slots < 5,
            "got {} slots for a 5-value chain",
            plan.num_slots
        );

        // Savings should be meaningful.
        let pct = plan.savings_percent();
        assert!(pct > 20.0, "expected savings > 20%, got {pct:.1}%");
    }

    /// Empty graph produces an empty plan.
    #[test]
    fn test_empty_graph() {
        let g = IrGraph::new();
        let plan = plan_memory(&g);

        assert!(plan.assignments.is_empty());
        assert_eq!(plan.num_slots, 0);
        assert_eq!(plan.naive_total, 0);
        assert_eq!(plan.planned_total, 0);
        // Bit-exact: with no allocations, savings is constructed as 0.0.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(plan.savings_percent(), 0.0);
        }
    }

    /// Values with different sizes: a small value can be placed in a larger
    /// slot once that slot becomes free.
    #[test]
    fn test_mixed_sizes() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![100]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![100]]);

        // Reshape to [10] — smaller output.
        let (_, reshape_outs) = g.add_node(
            IrOpKind::Reshape { shape: vec![10] },
            vec![relu_outs[0]],
            vec![vec![10]],
        );
        g.set_outputs(vec![reshape_outs[0]]);

        let plan = plan_memory(&g);

        // The reshape output (10 elements) can fit in a slot originally
        // sized for 100.
        assert!(plan.num_slots <= 3);
        assert_eq!(plan.naive_total, 100 + 100 + 10);
    }

    /// Graph output values must remain live until the end — they cannot be
    /// overwritten by later values.
    #[test]
    fn test_graph_outputs_pinned_to_end() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![10]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![10]]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![relu_outs[0]], vec![vec![10]]);

        // Both relu_out and neg_out are graph outputs — they must coexist.
        g.set_outputs(vec![relu_outs[0], neg_outs[0]]);

        let plan = plan_memory(&g);

        let relu_slot = plan.assignments[&relu_outs[0]];
        let neg_slot = plan.assignments[&neg_outs[0]];

        // relu_out is a graph output pinned to the end, so neg cannot reuse it.
        assert_ne!(
            relu_slot, neg_slot,
            "two simultaneously-live graph outputs must be in different slots"
        );
    }

    /// Verify that the plan correctly assigns every value in the graph.
    #[test]
    fn test_all_values_assigned() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![8]);
        let y = g.add_input(vec![8]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![8]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![8]]);
        g.set_outputs(vec![relu_outs[0]]);

        let plan = plan_memory(&g);

        // 4 values total: x, y, add_out, relu_out.
        assert_eq!(plan.assignments.len(), g.value_count());

        // Each assignment must reference a valid slot index.
        for &slot in plan.assignments.values() {
            assert!(slot < plan.num_slots);
        }
    }
}
