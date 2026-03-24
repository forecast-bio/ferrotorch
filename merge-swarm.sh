#!/bin/bash
# Merge all QA'd swarm branches into main
# Run from repo root: bash merge-swarm.sh
# If a merge conflicts, resolve it and run: git merge --continue
# Then re-run this script — it skips already-merged branches.

set -e

cd "$(git rev-parse --show-toplevel)"

merge_if_needed() {
    local branch="$1"
    local desc="$2"

    if ! git rev-parse --verify "$branch" &>/dev/null; then
        echo "SKIP: $branch (not found)"
        return
    fi

    # Check if already merged
    if git merge-base --is-ancestor "$branch" HEAD 2>/dev/null; then
        echo "SKIP: $branch (already merged) — $desc"
        return
    fi

    echo ""
    echo "=========================================="
    echo "MERGING: $branch"
    echo "  $desc"
    echo "=========================================="
    git merge "$branch" --no-edit
    echo "OK: $branch merged"
}

echo "=== Phase 1: Independent crate changes (low conflict risk) ==="
merge_if_needed worktree-agent-a326d1a7  "WU-18: GPU caching allocator"
merge_if_needed worktree-agent-a4d59f02  "WU-05: Channels-last memory format"
merge_if_needed worktree-agent-a797066d  "WU-23: FSDP sharding strategies"

echo ""
echo "=== Phase 2: Optimizer crate ==="
merge_if_needed worktree-agent-a9cc59e2  "WU-15: RAdam, NAdam, Adamax, Adadelta, Rprop, ASGD"
merge_if_needed worktree-agent-a39b87af  "WU-17: EMA, SWA, maximize flag"

echo ""
echo "=== Phase 3: Core crate ==="
merge_if_needed worktree-agent-ac089d65  "WU-02: cumsum, cumprod, cummax, cummin, logcumsumexp"
merge_if_needed worktree-agent-ad9df2ea  "WU-06: Forward-mode AD (DualTensor, jvp, jacfwd)"

echo ""
echo "=== Phase 4: Distribution crate ==="
merge_if_needed worktree-agent-af7d6d30  "WU-26: Transforms, constraints, KL divergence"
merge_if_needed worktree-agent-a7e19478  "WU-27: MultivariateNormal, Dirichlet, Multinomial"

echo ""
echo "=== Phase 5: Mega-worktree (multi-crate, merge last) ==="
merge_if_needed worktree-agent-a81eea52  "WU-16/25/10-14/28/30: Schedulers, distros, nn, train, vision"

echo ""
echo "=== Phase 6: Standalone nn activations ==="
merge_if_needed worktree-wu-09-missing-activations  "WU-09: 10 missing activations"

echo ""
echo "=== Phase 7: Pre-existing QA fixes (optional but recommended) ==="
merge_if_needed worktree-agent-a6c2248c  "QA: GPU profiler CUDA event timing"
merge_if_needed worktree-agent-a6e75a04  "QA: Identity, Flatten, L1Loss, NLLLoss, BatchNorm1d"
merge_if_needed worktree-agent-a71f741c  "QA: JIT fusion safety"
merge_if_needed worktree-agent-aedc9012  "QA: nn, MHA, linalg, GradScaler fixes"
merge_if_needed worktree-agent-af2b4243  "QA: panic-safe no_grad/enable_grad"
merge_if_needed worktree-agent-afa7098f  "QA: DataLoader prefetch and profiler fixes"

echo ""
echo "=== All merges complete ==="
echo ""
git log --oneline -20
