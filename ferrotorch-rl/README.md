# ferrotorch-rl

Reinforcement-learning policy composition for ferrotorch.

Phase D.2 of real-artifact-driven development: the
`stable-baselines3` `ActorCriticPolicy` (a.k.a. `MlpPolicy`) for
discrete-action environments, mirrored byte-for-byte from the pinned
`sb3/ppo-CartPole-v1` checkpoint and verified for forward-pass parity
against `stable-baselines3` via
[`scripts/verify_rl_inference.py`](../scripts/verify_rl_inference.py).

## Architecture (matches sb3 `MlpPolicy` defaults for CartPole-v1)

```text
MlpPolicy
├── features_extractor: FlattenExtractor (identity on 1-D obs)
├── mlp_extractor: MlpExtractor
│   ├── policy_net: Linear(4 -> 64) -> Tanh -> Linear(64 -> 64) -> Tanh
│   └── value_net:  Linear(4 -> 64) -> Tanh -> Linear(64 -> 64) -> Tanh
├── action_net: Linear(64 -> n_actions=2)   ← Categorical logits
└── value_net:  Linear(64 -> 1)              ← scalar state value
```

`share_features_extractor=True` in sb3's default, but the policy / value
trunks have *separate* `Linear` weights — the "shared" features
extractor is the flatten layer, not the MLP. The forward path is

```text
features = flatten(obs)              # identity on [B, 4] obs
latent_pi = policy_net(features)     # Tanh-Tanh MLP
latent_vf = value_net(features)      # Tanh-Tanh MLP
action_logits = action_net(latent_pi)   # [B, 2]
value         = value_net_head(latent_vf) # [B, 1]
```

## Weight pin

`ferrotorch/ppo-cartpole-v1` (HF mirror, Apache 2.0, byte-for-byte from
`sb3/ppo-CartPole-v1`). See
[`scripts/pin_pretrained_rl_weights.py`](../scripts/pin_pretrained_rl_weights.py)
for the pinning recipe.

## Loader audit trail

[`load_ppo_policy`](src/safetensors_loader.rs) returns a `DropReport`
listing any upstream key the loader intentionally did not consume.
Same #1141 silent-drop-bug guard as `ferrotorch-bert`,
`ferrotorch-graph`, `ferrotorch-llama`, etc.
