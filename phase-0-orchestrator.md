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

An orchestrator agent that coordinates the multi-phase ferrotorch build as a swarm of implementation agents. It manages the dependency graph between crates, dispatches work to sub-agents, tracks phase completion, and escalates to the human when decisions or unblocked prerequisites are needed.

### Requirements

- REQ-1: The orchestrator must maintain a machine-readable dependency graph of all ferrotorch crates and their build/test prerequisites (e.g., ferrotorch-nn depends on ferrotorch-core being complete and passing tests).
- REQ-2: The orchestrator must be able to dispatch implementation work to sub-agents via `crosslink kickoff` and monitor their completion status.
- REQ-3: The orchestrator must detect when a sub-agent is blocked (hook rejection, failing tests, unresolved design question) and escalate to the human with actionable context — the specific blocker, which issue it's on, and what decision is needed.
- REQ-4: The orchestrator must prevent out-of-order work — a Phase 2 agent must not start until Phase 1's acceptance criteria are met.
- REQ-5: The orchestrator must produce a human-readable status dashboard showing: which phases are complete, which are in-progress, which are blocked, and what the next actionable step is.
- REQ-6: The orchestrator must use crosslink issues as the single source of truth for progress tracking, creating parent issues (epics) per phase and subissues per implementation unit.

### Acceptance Criteria

- [ ] AC-1: A dependency graph file exists at `.design/dependency-graph.json` (or equivalent) that encodes crate → crate dependencies and phase ordering, and the orchestrator reads it to determine what can be dispatched.
- [ ] AC-2: Running the orchestrator when Phase 1 is incomplete does not dispatch Phase 2 work. Running it when Phase 1 is complete (all subissues closed, tests passing) does dispatch Phase 2 work.
- [ ] AC-3: When a sub-agent's `crosslink kickoff` session ends with an unresolved blocker comment, the orchestrator creates a human-attention issue with label `needs-human` summarizing the blocker within 60 seconds.
- [ ] AC-4: `crosslink issue list` shows a hierarchical structure: one epic per phase, subissues per implementation unit, with labels matching the CHANGELOG categories (feature, bug, etc.).
- [ ] AC-5: The orchestrator can be invoked via `/check` and prints a status table showing phase completion percentage, active agents, and blockers.
- [ ] AC-6: The orchestrator does not make architectural decisions — it only dispatches, monitors, and escalates. All design questions are routed to the human.

### Architecture

The orchestrator is not a Rust crate — it is a Claude Code workflow built on crosslink primitives and the existing `/kickoff`, `/check`, and `/maintain` skills.

### Dependency Graph

A static JSON file at `.design/dependency-graph.json` encodes the build order:

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
        "ferray-core bf16 support merged (external)"
      ]
    },
    {
      "id": 2,
      "name": "Neural Network Modules",
      "crate": "ferrotorch-nn",
      "design_doc": ".design/phase-2-nn-modules.md",
      "depends_on": [1]
    }
  ]
}
```

### Dispatch Flow

1. Orchestrator reads dependency graph
2. For each phase whose dependencies are satisfied:
   a. Check if a design doc exists at the `design_doc` path
   b. If not, alert human: "Phase N needs a design doc before implementation"
   c. If yes, check if the epic issue exists; create it if not
   d. Run `crosslink kickoff run "<phase name>" --doc <design_doc>` in a tmux session
3. Monitor via `crosslink issue list` — when all subissues for a phase are closed, mark phase complete

### Escalation Rules

The orchestrator escalates to the human when:
- A sub-agent logs a `--kind blocker` comment
- A sub-agent's tmux session exits with non-zero status
- A hook rejects a sub-agent's tool call (detected via `crosslink intervene` logs)
- An external prerequisite is unmet (e.g., "ferray-core bf16 not yet available")
- A design doc has unresolved `<!-- OPEN -->` questions that block implementation

### Status Dashboard

The `/check` skill already exists. The orchestrator extends it with phase-level awareness:

```
ferrotorch build status:
  Phase 0: Orchestrator        [ACTIVE]
  Phase 1: Autograd Engine     [IN PROGRESS] 12/47 subissues closed, 2 blocked
  Phase 2: NN Modules          [WAITING] depends on Phase 1
  Phase 3: Optim + Serialize   [WAITING] depends on Phase 1
  ...

Blockers requiring human attention:
  - #34: f32 matmul strategy decision needed (Phase 1)
  - #12: ferray-core bf16 PR not merged (external prerequisite)
```

### File Locations
- Dependency graph: `.design/dependency-graph.json`
- Phase design docs: `.design/phase-N-<slug>.md`
- Orchestrator logic: Encoded as a Claude Code skill at `.claude/skills/orchestrate.md`, invocable via `/orchestrate`
- Status: Derived from `crosslink issue list` at runtime (no separate state file)

### Out of Scope

- The orchestrator does not implement any Rust code — it only coordinates agents that do
- The orchestrator does not make design decisions — it escalates ambiguity to the human
- CI/CD pipeline setup — that is a separate concern
- Cross-repository coordination (e.g., ferray-core bf16 PR) — the orchestrator only tracks external prerequisites as boolean flags, it does not manage external repos
- Cost optimization for agent API usage — the orchestrator dispatches work, it does not budget tokens

### resolved questions

### Q1: Orchestrator as skill vs CLAUDE.md instructions
**Decision**: Formal Claude Code skill, invocable via `/orchestrate`.

More structured, has its own prompt template and state, and avoids reliance on agents consistently reading CLAUDE.md. The skill can encode dispatch logic, dependency checking, and escalation rules as a repeatable prompt rather than ambient instructions.

### Q2: Parallelism within a phase
**Decision**: Hybrid — parallel along logical seams, sequential where dependencies exist.

The orchestrator must understand which subissues within a phase are independent (e.g., `grad_fns/arithmetic.rs` and `grad_fns/linalg.rs` touch different files) and dispatch those in parallel via concurrent `kickoff` sessions in separate worktrees. Subissues that modify shared files (lib.rs, mod.rs, Cargo.toml) or that depend on types defined by earlier subissues must be dispatched sequentially. The dependency graph at `.design/dependency-graph.json` encodes both phase-level and subissue-level dependencies to support this.

