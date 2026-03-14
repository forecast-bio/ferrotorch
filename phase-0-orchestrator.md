---
title: "Phase 0 — Build Orchestrator Agent"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-14
updated: 2026-03-14
---


## Design Specification

### Summary

An orchestrator skill (`/orchestrate`) that coordinates the multi-phase ferrotorch build as a swarm of implementation agents. It manages the dependency graph between crates, dispatches work to sub-agents via `/kickoff`, tracks phase completion through crosslink issues, and escalates to the human when decisions or unblocked prerequisites are needed.

### Requirements

- REQ-1: The orchestrator must maintain a machine-readable dependency graph at `.design/dependency-graph.json` encoding all ferrotorch phases, their crate targets, design doc paths, inter-phase dependencies, external prerequisites, and subissue-level file dependencies for hybrid parallelism.
- REQ-2: The orchestrator must dispatch implementation work to sub-agents via `crosslink kickoff run` (the same mechanism used by the existing `/kickoff` skill at `.claude/commands/kickoff.md`) and monitor their completion via tmux session status and crosslink issue state.
- REQ-3: The orchestrator must detect when a sub-agent is blocked (blocker comment on crosslink issue, tmux session exit with non-zero status, hook intervention logged via `crosslink intervene`) and escalate to the human with: the specific blocker, which issue it's on, and what decision is needed.
- REQ-4: The orchestrator must prevent out-of-order work — a phase must not be dispatched until all phases in its `depends_on` list have all subissues closed and `cargo test -p <crate>` passing. External prerequisites must be manually marked resolved by the human.
- REQ-5: The orchestrator must produce a status dashboard showing: which phases are complete, in-progress, blocked, or waiting; subissue completion counts per phase; active tmux sessions; and blockers requiring human attention.
- REQ-6: The orchestrator must use crosslink issues as the single source of truth for progress. It creates one epic per phase and one subissue per implementation unit, with labels matching CHANGELOG categories (`feature`, `bug`, etc.).
- REQ-7: For hybrid parallelism, the orchestrator must read subissue-level `file_deps` from the dependency graph and dispatch independent subissues in parallel (concurrent `kickoff` sessions in separate worktrees) while sequencing subissues that share files.

### Acceptance Criteria

- [ ] AC-1: `.design/dependency-graph.json` exists and encodes all 8 ferrotorch phases with `id`, `name`, `crate`, `design_doc`, `depends_on`, `prerequisites`, and `subissues` (with `file_deps` for parallelism). The orchestrator parses it without error.
- [ ] AC-2: Running `/orchestrate` when Phase 1 is incomplete (open subissues exist) does not dispatch Phase 2 work. Running it when Phase 1 is complete (all subissues closed, `cargo test -p ferrotorch-core` passes) dispatches Phase 2 if its design doc exists.
- [ ] AC-3: When a sub-agent's tmux session exits with non-zero status or a `--kind blocker` comment is logged on a subissue, the orchestrator creates an issue with label `needs-human` containing: the blocker description, the issue ID, and the phase context — within the same `/orchestrate` invocation.
- [ ] AC-4: `crosslink issue list` shows one epic per dispatched phase, with subissues nested under it. Each subissue has a label from the CHANGELOG set (feature, bug, enhancement, etc.).
- [ ] AC-5: `/orchestrate` prints a status table in the format shown in the Architecture section, derived from `crosslink issue list` and `tmux list-sessions`.
- [ ] AC-6: The orchestrator never writes Rust code, modifies Cargo.toml, or makes architectural decisions. It only dispatches, monitors, and escalates.
- [ ] AC-7: Independent subissues within a phase (those with non-overlapping `file_deps`) are dispatched in parallel via separate worktrees. Subissues sharing files (lib.rs, mod.rs, Cargo.toml) are dispatched sequentially after the parallel batch completes.

### Architecture

The orchestrator is a Claude Code skill at `.claude/commands/orchestrate.md`, following the same patterns as the existing `/check` skill (`.claude/commands/check.md`) and `/kickoff` skill (`.claude/commands/kickoff.md`).

### Skill File

`.claude/commands/orchestrate.md` with frontmatter:

```yaml
---
allowed-tools: Bash(crosslink *), Bash(tmux *), Bash(cargo test *), Bash(cat *), Bash(ls *), Bash(git *)
description: Coordinate the multi-phase ferrotorch build — dispatch agents, track progress, escalate blockers
---
```

Dynamic context injection (like `/check` uses):

```markdown
## Context

- Active tmux sessions: !`tmux list-sessions 2>/dev/null || echo "no tmux server running"`
- Current worktrees: !`git worktree list`
- Open issues: !`crosslink issue list -s open --json 2>/dev/null || echo "[]"`
- Dependency graph: !`cat .design/dependency-graph.json 2>/dev/null || echo "not found"`
```

### Dependency Graph

`.design/dependency-graph.json` encodes the full build plan. Phase 1 includes subissue-level dependencies for hybrid parallelism:

```json
{
  "phases": [
    {
      "id": 1,
      "name": "Autograd Engine",
      "crate": "ferrotorch-core",
      "design_doc": ".design/phase-1-autograd-engine.md",
      "depends_on": [],
      "prerequisites": [
        { "id": "bf16", "description": "ferray-core bf16 support merged", "resolved": false },
        { "id": "f32-linalg", "description": "ferray-linalg f32 support merged", "resolved": false }
      ],
      "subissues": [
        {
          "name": "Core types: Tensor, TensorStorage, Device, error",
          "files": ["src/tensor.rs", "src/storage.rs", "src/device.rs", "src/dtype.rs", "src/error.rs", "src/lib.rs", "Cargo.toml"],
          "group": "foundation",
          "order": 1
        },
        {
          "name": "Shape utilities and creation functions",
          "files": ["src/shape.rs", "src/creation.rs", "src/ops/creation.rs"],
          "group": "foundation",
          "order": 2
        },
        {
          "name": "Autograd engine: graph, function trait, no_grad, checkpoint",
          "files": ["src/autograd/mod.rs", "src/autograd/graph.rs", "src/autograd/function.rs", "src/autograd/no_grad.rs", "src/autograd/checkpoint.rs"],
          "group": "autograd",
          "order": 3
        },
        {
          "name": "Elementwise ops bridge to ferray-ufunc",
          "files": ["src/ops/mod.rs", "src/ops/elementwise.rs"],
          "group": "ops",
          "order": 4
        },
        {
          "name": "Linalg ops bridge to ferray-linalg",
          "files": ["src/ops/linalg.rs"],
          "group": "ops",
          "order": 4
        },
        {
          "name": "GradFn: arithmetic (add, sub, mul, div, neg, pow, sqrt, abs)",
          "files": ["src/grad_fns/mod.rs", "src/grad_fns/arithmetic.rs"],
          "group": "grad_fns",
          "order": 5
        },
        {
          "name": "GradFn: reduction (sum, mean, prod)",
          "files": ["src/grad_fns/reduction.rs"],
          "group": "grad_fns",
          "order": 5
        },
        {
          "name": "GradFn: linalg (matmul, bmm, mm, mv, dot)",
          "files": ["src/grad_fns/linalg.rs"],
          "group": "grad_fns",
          "order": 5
        },
        {
          "name": "GradFn: activation (relu, sigmoid, tanh, gelu, silu, softmax, log_softmax)",
          "files": ["src/grad_fns/activation.rs"],
          "group": "grad_fns",
          "order": 5
        },
        {
          "name": "GradFn: shape (reshape, transpose, permute, expand, cat, stack, split, etc.)",
          "files": ["src/grad_fns/shape.rs"],
          "group": "grad_fns",
          "order": 5
        },
        {
          "name": "GradFn: indexing (gather, scatter_add, index_select, masked_fill)",
          "files": ["src/grad_fns/indexing.rs"],
          "group": "grad_fns",
          "order": 5
        },
        {
          "name": "GradFn: comparison (where_)",
          "files": ["src/grad_fns/comparison.rs"],
          "group": "grad_fns",
          "order": 5
        },
        {
          "name": "Test suite: numerical gradient checks, PyTorch reference, edge cases, thread safety",
          "files": ["tests/"],
          "group": "tests",
          "order": 6
        }
      ]
    },
    {
      "id": 2,
      "name": "Neural Network Modules",
      "crate": "ferrotorch-nn",
      "design_doc": ".design/phase-2-nn-modules.md",
      "depends_on": [1],
      "prerequisites": []
    },
    {
      "id": 3,
      "name": "Optimizers + Serialization",
      "crate": "ferrotorch-optim, ferrotorch-serialize",
      "design_doc": ".design/phase-3-optim-serialize.md",
      "depends_on": [1],
      "prerequisites": []
    },
    {
      "id": 4,
      "name": "Data Loading",
      "crate": "ferrotorch-data",
      "design_doc": ".design/phase-4-data-loading.md",
      "depends_on": [2],
      "prerequisites": []
    },
    {
      "id": 5,
      "name": "Vision",
      "crate": "ferrotorch-vision",
      "design_doc": ".design/phase-5-vision.md",
      "depends_on": [2, 4],
      "prerequisites": []
    },
    {
      "id": 6,
      "name": "GPU Backend",
      "crate": "ferrotorch-gpu",
      "design_doc": ".design/phase-6-gpu-backend.md",
      "depends_on": [1],
      "prerequisites": []
    },
    {
      "id": 7,
      "name": "Distributed Training",
      "crate": "ferrotorch-distributed",
      "design_doc": ".design/phase-7-distributed.md",
      "depends_on": [2, 6],
      "prerequisites": []
    },
    {
      "id": 8,
      "name": "JIT / Graph Optimization",
      "crate": "ferrotorch-jit",
      "design_doc": ".design/phase-8-jit.md",
      "depends_on": [1],
      "prerequisites": []
    }
  ]
}
```

### Dispatch Flow

1. Orchestrator reads `.design/dependency-graph.json`
2. For each phase whose `depends_on` phases are all complete:
   a. Check external `prerequisites` — if any have `"resolved": false`, alert human and skip
   b. Check if a design doc exists at the `design_doc` path — if not, alert human: "Phase N needs a design doc before implementation"
   c. Check if the design doc has unresolved `<!-- OPEN -->` markers — if so, alert human
   d. Check if the epic issue exists via `crosslink issue list`; create it if not
   e. Dispatch subissues respecting the `order` field and `files` overlap rules:
      - Subissues at the same `order` with non-overlapping `files` dispatch in parallel (separate worktrees via `crosslink kickoff run`)
      - Subissues sharing files or at a higher `order` wait for the previous batch
3. Monitor via `crosslink issue list` and `tmux list-sessions` — when all subissues for a phase are closed and `cargo test -p <crate>` passes, mark the phase complete

### Parallelism Rules

For Phase 1 as an example:

| Step | Subissues | Parallel? | Reason |
|------|-----------|-----------|--------|
| 1 | Core types (tensor, storage, device, error) | Sequential | Touches lib.rs, Cargo.toml — foundation for everything |
| 2 | Shape utilities + creation functions | Sequential | Depends on core types |
| 3 | Autograd engine | Sequential | Depends on Tensor type |
| 4 | Elementwise ops, Linalg ops | **Parallel** | Separate files (ops/elementwise.rs vs ops/linalg.rs), no overlap |
| 5 | GradFn: arithmetic, reduction, linalg, activation, shape, indexing, comparison | **Parallel** | Each is a separate file in grad_fns/, no shared state |
| 6 | Test suite | Sequential | Depends on all ops and grad_fns being complete |

Steps 4 and 5 each dispatch multiple agents in parallel worktrees. Steps 1-3 and 6 are sequential gates.

### Escalation Rules

The orchestrator escalates to the human when:
- A sub-agent's tmux session exits with non-zero status (detected via `tmux list-sessions` delta)
- A crosslink issue receives a `--kind blocker` comment (detected via `crosslink issue show <id>`)
- A hook rejects a sub-agent's tool call (detected via `crosslink intervene` logs on the issue)
- An external prerequisite has `"resolved": false` in the dependency graph
- A design doc has unresolved `<!-- OPEN -->` markers
- `cargo test -p <crate>` fails after all subissues in a phase are closed

Escalation creates an issue with label `needs-human`:
```bash
crosslink issue create "Human input needed: <summary>" -p critical --label needs-human
crosslink issue comment <id> "Phase <N>, subissue #<id>: <blocker details>" --kind blocker
```

### Status Dashboard

Invoked via `/orchestrate` (or `/orchestrate status`):

```
ferrotorch build status:

  Phase 1: Autograd Engine     [IN PROGRESS] 7/13 subissues closed
    Step 5 (grad_fns):  5 agents running in parallel
      feat-gradfn-arithmetic   [tmux]  Working   Implementing pow backward
      feat-gradfn-reduction    [tmux]  Done      All tests passing
      feat-gradfn-linalg       [tmux]  Working   bmm backward
      feat-gradfn-activation   [tmux]  Blocked   softmax numerical instability
      feat-gradfn-shape        [tmux]  Working   cat backward
      feat-gradfn-indexing     [tmux]  Done      All tests passing
      feat-gradfn-comparison   [tmux]  Done      All tests passing
  Phase 2: NN Modules          [WAITING] depends on Phase 1
  Phase 3: Optim + Serialize   [WAITING] depends on Phase 1
  Phase 4: Data Loading        [WAITING] depends on Phase 2
  Phase 5: Vision              [WAITING] depends on Phase 2, 4
  Phase 6: GPU Backend         [WAITING] depends on Phase 1
  Phase 7: Distributed         [WAITING] depends on Phase 2, 6
  Phase 8: JIT                 [WAITING] depends on Phase 1

  Prerequisites:
    [x] ferray-core bf16 support
    [x] ferray-linalg f32 support

  Blockers requiring human attention:
    #45: softmax numerical instability in grad_fns/activation.rs (Phase 1, step 5)
```

### File Locations
- Dependency graph: `.design/dependency-graph.json`
- Phase design docs: `.design/phase-N-<slug>.md`
- Orchestrator skill: `.claude/commands/orchestrate.md`
- Status: Derived from `crosslink issue list` and `tmux list-sessions` at runtime (no separate state file)

### Out of Scope

- The orchestrator does not implement any Rust code — it only coordinates agents that do
- The orchestrator does not make design decisions — it escalates ambiguity to the human
- CI/CD pipeline setup — that is a separate concern
- Cross-repository coordination (e.g., ferray-core bf16 PR) — the orchestrator only tracks external prerequisites as boolean flags in the dependency graph; the human marks them resolved
- Cost optimization for agent API usage — the orchestrator dispatches work, it does not budget tokens
- Authoring design docs for future phases — the orchestrator checks if they exist and alerts the human if not, but `/design` is a separate skill

### resolved questions

### Q1: Orchestrator as skill vs CLAUDE.md instructions
**Decision**: Formal Claude Code skill at `.claude/commands/orchestrate.md`, invocable via `/orchestrate`.

Follows the same pattern as existing skills: `/check` (`.claude/commands/check.md`) and `/kickoff` (`.claude/commands/kickoff.md`). Uses YAML frontmatter with `allowed-tools` and `description`, plus `!`backtick dynamic context injection for tmux sessions, worktrees, open issues, and the dependency graph.

### Q2: Parallelism within a phase
**Decision**: Hybrid — parallel along logical seams, sequential where dependencies exist.

The dependency graph encodes subissue-level `files` lists. Subissues at the same `order` with non-overlapping files dispatch in parallel (concurrent `crosslink kickoff run` sessions in separate worktrees). Subissues sharing files or at higher `order` values wait. For Phase 1: steps 4 (2 parallel ops agents) and 5 (7 parallel grad_fn agents) are the parallel batches; steps 1-3 and 6 are sequential gates.

