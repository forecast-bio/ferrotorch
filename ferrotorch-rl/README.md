# ferrotorch-rl

Reinforcement-learning policy composition for ferrotorch.

Phase D.2 of real-artifact-driven development: the
[`stable-baselines3`](https://stable-baselines3.readthedocs.io/)
`ActorCriticPolicy` (a.k.a. `MlpPolicy`) for discrete-action
environments, mirrored byte-for-byte from the pinned
[`sb3/ppo-CartPole-v1`](https://huggingface.co/sb3/ppo-CartPole-v1)
checkpoint and verified for forward-pass parity against
`stable-baselines3` via
[`scripts/verify_rl_inference.py`](../scripts/verify_rl_inference.py) (#1158).

## What it provides

- **`MlpPolicyConfig`** — hyperparameters (`obs_dim`, `hidden`,
  `n_actions`); `MlpPolicyConfig::cartpole_v1()` returns the sb3
  CartPole-v1 defaults (`obs_dim=4`, `hidden=64`, `n_actions=2`).
- **`MlpExtractor`** — the two separate Tanh-MLP trunks
  (`policy_net`, `value_net`) that sb3 uses despite the
  `share_features_extractor=True` default — sharing applies only to
  the (identity) flatten layer, not the MLP.
- **`ActionNet`** — `Linear(hidden -> n_actions)` producing
  Categorical logits.
- **`ValueHead`** — `Linear(hidden -> 1)` producing the scalar state
  value.
- **`MlpPolicy`** — the full sb3 `ActorCriticPolicy` (`flatten` →
  `MlpExtractor` → `(ActionNet, ValueHead)` heads) with
  `forward_actor(obs) -> logits` and `forward_critic(obs) -> value`.
- **`load_ppo_policy`** — SafeTensors loader for the upstream sb3
  checkpoint; returns a `DropReport` (#1141 silent-drop-bug guard).

## Architecture (sb3 `MlpPolicy` defaults for CartPole-v1)

```text
MlpPolicy
├── features_extractor: FlattenExtractor (identity on 1-D obs)
├── mlp_extractor: MlpExtractor
│   ├── policy_net: Linear(4 -> 64) -> Tanh -> Linear(64 -> 64) -> Tanh
│   └── value_net:  Linear(4 -> 64) -> Tanh -> Linear(64 -> 64) -> Tanh
├── action_net: Linear(64 -> n_actions=2)   ← Categorical logits
└── value_net:  Linear(64 -> 1)              ← scalar state value
```

## Quick start

```rust
use ferrotorch_rl::{MlpPolicy, MlpPolicyConfig, load_ppo_policy};
use ferrotorch_core::Tensor;

let mut policy = MlpPolicy::new(MlpPolicyConfig::cartpole_v1())?;
let _drop = load_ppo_policy(&mut policy, "/path/to/ppo-cartpole-v1.safetensors")?;

let obs: Tensor<f32> = /* [B, 4] CartPole observation */;
let logits = policy.forward_actor(&obs)?;   // [B, 2]
let value  = policy.forward_critic(&obs)?;  // [B, 1]
```

## Real-artifact parity

`ferrotorch/ppo-cartpole-v1` (HF mirror, Apache 2.0, byte-for-byte
from `sb3/ppo-CartPole-v1`). See
[`scripts/pin_pretrained_rl_weights.py`](../scripts/pin_pretrained_rl_weights.py)
for the pinning recipe. `scripts/verify_rl_inference.py` enforces
`cosine_sim >= 0.999, max_abs <= 0.5` against
`stable_baselines3.PPO.load(...).policy(obs)` forward output.

## Part of ferrotorch

This crate is one component of the
[ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
