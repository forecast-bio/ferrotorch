//! `MlpPolicy` — `stable-baselines3`'s `ActorCriticPolicy` with the
//! default `FlattenExtractor + MlpExtractor + action_net + value_net`
//! composition.
//!
//! # Per-component math (matches sb3 `ActorCriticPolicy.forward` at eval)
//!
//! Given an observation `obs: [B, obs_dim]`:
//!
//! 1. **Features extractor.** sb3's `FlattenExtractor` returns
//!    `obs.flatten(start_dim=1)` — for a 1-D observation (CartPole's
//!    `[cart_pos, cart_vel, pole_angle, pole_angvel]`) this is the
//!    identity. We expose the same behavior by treating the extractor
//!    as identity at the type level.
//! 2. **MlpExtractor.** Two *separate* Tanh-MLP trunks, one for the
//!    policy latent and one for the value latent:
//!    ```text
//!    latent_pi = tanh(L_pi_1(tanh(L_pi_0(features))))
//!    latent_vf = tanh(L_vf_1(tanh(L_vf_0(features))))
//!    ```
//!    Both trunks use the same `(obs_dim → 64 → 64)` shape under sb3's
//!    `net_arch=[64, 64]` default and the `Tanh` `activation_fn=nn.Tanh`
//!    default, with `ortho_init=True` weight initialization (init is
//!    *not* load-bearing for the harness — the loader overwrites every
//!    parameter from the pinned safetensors before the first forward).
//! 3. **Action head.** `action_net: Linear(64 → n_actions)`, outputting
//!    raw logits for sb3's `CategoricalDistribution.proba_distribution`.
//! 4. **Value head.** `value_net: Linear(64 → 1)`, outputting the
//!    scalar state value `V(s)`.
//!
//! Both heads are unactivated linear projections (sb3's `_build` adds
//! them with `gain=0.01` and `gain=1.0` respectively under
//! `ortho_init`, but again that is only init-time math).
//!
//! # Why a separate `MlpExtractor` type instead of `Sequential`?
//!
//! sb3 deliberately emits `mlp_extractor.policy_net.{0,2}.{weight,bias}`
//! (and `value_net.{0,2}.{weight,bias}`) as the state-dict keys —
//! the `1` / `3` slots are the `Tanh()`s, which have no parameters and
//! occupy a slot only because `nn.Sequential` numbers every child. To
//! load the upstream checkpoint byte-for-byte without remapping we
//! mirror that numbering exactly in [`MlpExtractor::named_parameters`].
//!
//! # `log_std` (discrete vs continuous)
//!
//! sb3's `ActorCriticPolicy` only emits a `log_std` parameter for
//! continuous-action policies (DiagGaussianDistribution). For the
//! discrete `Categorical` distribution used by CartPole-v1 there is
//! no `log_std`, so the state dict has exactly 12 parameter keys:
//! 4 for the policy trunk × 2 (weight+bias of `.0` and `.2`),
//! 4 for the value trunk × 2, 2 for `action_net`, 2 for `value_net`.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_nn::activation::Tanh;
use ferrotorch_nn::linear::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

/// Dimensions for a `MlpPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlpPolicyConfig {
    /// Flattened observation dimension (e.g. 4 for CartPole-v1).
    pub obs_dim: usize,
    /// Width of each hidden Tanh-MLP layer (sb3 default: 64).
    pub hidden: usize,
    /// Number of discrete actions (e.g. 2 for CartPole-v1).
    pub n_actions: usize,
}

impl MlpPolicyConfig {
    /// `MlpPolicyConfig` matching sb3's `MlpPolicy` defaults for
    /// CartPole-v1 (`obs_dim=4, hidden=64, n_actions=2`).
    pub fn cartpole_v1() -> Self {
        Self {
            obs_dim: 4,
            hidden: 64,
            n_actions: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// MlpExtractor — sb3's two-trunk Tanh-MLP.
// ---------------------------------------------------------------------------

/// sb3 `MlpExtractor` — two *separate* Tanh-MLP trunks with no shared
/// weights between the policy and value heads.
///
/// Per-trunk layout: `Linear(obs_dim → hidden) → Tanh → Linear(hidden →
/// hidden) → Tanh`. The `.0` / `.2` slot numbering is preserved so
/// state-dict keys round-trip byte-for-byte against the upstream
/// `nn.Sequential` keys.
#[derive(Debug)]
pub struct MlpExtractor {
    /// `policy_net.0`: `Linear(obs_dim → hidden)`.
    pub policy_net_0: Linear<f32>,
    /// `policy_net.2`: `Linear(hidden → hidden)`.
    pub policy_net_2: Linear<f32>,
    /// `value_net.0`: `Linear(obs_dim → hidden)`.
    pub value_net_0: Linear<f32>,
    /// `value_net.2`: `Linear(hidden → hidden)`.
    pub value_net_2: Linear<f32>,
    tanh: Tanh,
    obs_dim: usize,
    hidden: usize,
    training: bool,
}

impl MlpExtractor {
    /// Construct a fresh `MlpExtractor` with zero-initialized
    /// parameters. The loader replaces them with the pinned
    /// safetensors values before the first forward.
    ///
    /// # Errors
    ///
    /// Forwards [`Linear::new`] errors (e.g. if `obs_dim` or `hidden`
    /// is zero).
    pub fn new(obs_dim: usize, hidden: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            policy_net_0: Linear::<f32>::new(obs_dim, hidden, /* bias = */ true)?,
            policy_net_2: Linear::<f32>::new(hidden, hidden, /* bias = */ true)?,
            value_net_0: Linear::<f32>::new(obs_dim, hidden, /* bias = */ true)?,
            value_net_2: Linear::<f32>::new(hidden, hidden, /* bias = */ true)?,
            tanh: Tanh::new(),
            obs_dim,
            hidden,
            training: false,
        })
    }

    /// Forward the policy trunk on `features: [B, obs_dim]`, returning
    /// `latent_pi: [B, hidden]`.
    ///
    /// # Errors
    ///
    /// Forwards any error from the underlying `Linear` / `Tanh`
    /// forwards (typically a `ShapeMismatch` if `features`'s last
    /// dimension is not `obs_dim`).
    pub fn forward_actor(&self, features: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let h = self.policy_net_0.forward(features)?;
        let h = self.tanh.forward::<f32>(&h)?;
        let h = self.policy_net_2.forward(&h)?;
        self.tanh.forward::<f32>(&h)
    }

    /// Forward the value trunk on `features: [B, obs_dim]`, returning
    /// `latent_vf: [B, hidden]`.
    ///
    /// # Errors
    ///
    /// Forwards any error from the underlying `Linear` / `Tanh`
    /// forwards (typically a `ShapeMismatch` if `features`'s last
    /// dimension is not `obs_dim`).
    pub fn forward_critic(&self, features: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let h = self.value_net_0.forward(features)?;
        let h = self.tanh.forward::<f32>(&h)?;
        let h = self.value_net_2.forward(&h)?;
        self.tanh.forward::<f32>(&h)
    }

    /// `obs_dim` the extractor was constructed with.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// `hidden` the extractor was constructed with.
    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

impl Module<f32> for MlpExtractor {
    fn forward(&self, _input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        // The extractor has *two* outputs; the standard `Module::forward`
        // contract returns one. Keep the trait impl for state-dict
        // plumbing but refuse the call.
        Err(FerrotorchError::InvalidArgument {
            message: "MlpExtractor::Module::forward: call \
                      forward_actor / forward_critic explicitly — the \
                      extractor emits two latents."
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        let mut out = self.policy_net_0.parameters();
        out.extend(self.policy_net_2.parameters());
        out.extend(self.value_net_0.parameters());
        out.extend(self.value_net_2.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
        let mut out = self.policy_net_0.parameters_mut();
        out.extend(self.policy_net_2.parameters_mut());
        out.extend(self.value_net_0.parameters_mut());
        out.extend(self.value_net_2.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
        // Mirror sb3's nn.Sequential numbering: `.0` is the first
        // Linear, `.1` would be Tanh (no params), `.2` is the second
        // Linear, `.3` is Tanh.
        let mut out: Vec<(String, &Parameter<f32>)> = Vec::new();
        for (n, p) in self.policy_net_0.named_parameters() {
            out.push((format!("policy_net.0.{n}"), p));
        }
        for (n, p) in self.policy_net_2.named_parameters() {
            out.push((format!("policy_net.2.{n}"), p));
        }
        for (n, p) in self.value_net_0.named_parameters() {
            out.push((format!("value_net.0.{n}"), p));
        }
        for (n, p) in self.value_net_2.named_parameters() {
            out.push((format!("value_net.2.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.policy_net_0.train();
        self.policy_net_2.train();
        self.value_net_0.train();
        self.value_net_2.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.policy_net_0.eval();
        self.policy_net_2.eval();
        self.value_net_0.eval();
        self.value_net_2.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ActionNet — `Linear(hidden → n_actions)` head producing Categorical logits.
// ---------------------------------------------------------------------------

/// Action head: `Linear(hidden → n_actions)` producing raw Categorical
/// logits. Newtype around [`Linear`] so the state-dict key prefix
/// stays meaningful (`action_net.{weight,bias}` round-trips exactly).
#[derive(Debug)]
pub struct ActionNet {
    /// `Linear(hidden → n_actions)`.
    pub linear: Linear<f32>,
    training: bool,
}

impl ActionNet {
    /// Construct a fresh `ActionNet` with zero-initialized parameters.
    ///
    /// # Errors
    ///
    /// Forwards [`Linear::new`] errors (e.g. if `hidden` or
    /// `n_actions` is zero).
    pub fn new(hidden: usize, n_actions: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            linear: Linear::<f32>::new(hidden, n_actions, /* bias = */ true)?,
            training: false,
        })
    }

    /// Forward: `latent_pi: [B, hidden] → logits: [B, n_actions]`.
    ///
    /// # Errors
    ///
    /// Forwards any error from the inner [`Linear::forward`].
    pub fn forward(&self, latent_pi: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        self.linear.forward(latent_pi)
    }
}

impl Module<f32> for ActionNet {
    fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        self.linear.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        self.linear.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
        self.linear.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
        // sb3 exposes `action_net.weight` / `action_net.bias` — the
        // top-level model puts the `action_net.` prefix itself.
        self.linear.named_parameters()
    }

    fn train(&mut self) {
        self.training = true;
        self.linear.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.linear.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ValueHead — `Linear(hidden → 1)` head producing scalar state values.
// ---------------------------------------------------------------------------

/// Value head: `Linear(hidden → 1)` producing the scalar state value.
/// Newtype around [`Linear`] so the state-dict key prefix stays
/// meaningful (`value_net.{weight,bias}` round-trips exactly). Named
/// `ValueHead` (rather than `ValueNet`) to disambiguate from the
/// `MlpExtractor.value_net` trunk that shares the parameter-name
/// `value_net`.
#[derive(Debug)]
pub struct ValueHead {
    /// `Linear(hidden → 1)`.
    pub linear: Linear<f32>,
    training: bool,
}

impl ValueHead {
    /// Construct a fresh `ValueHead` with zero-initialized parameters.
    ///
    /// # Errors
    ///
    /// Forwards [`Linear::new`] errors (e.g. if `hidden` is zero).
    pub fn new(hidden: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            linear: Linear::<f32>::new(hidden, 1, /* bias = */ true)?,
            training: false,
        })
    }

    /// Forward: `latent_vf: [B, hidden] → value: [B, 1]`.
    ///
    /// # Errors
    ///
    /// Forwards any error from the inner [`Linear::forward`].
    pub fn forward(&self, latent_vf: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        self.linear.forward(latent_vf)
    }
}

impl Module<f32> for ValueHead {
    fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        self.linear.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        self.linear.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
        self.linear.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
        self.linear.named_parameters()
    }

    fn train(&mut self) {
        self.training = true;
        self.linear.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.linear.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// MlpPolicy — the assembled `ActorCriticPolicy`.
// ---------------------------------------------------------------------------

/// Output of [`MlpPolicy::forward`] — Categorical action logits plus
/// scalar state values.
#[derive(Debug, Clone)]
pub struct PolicyOutput {
    /// `[B, n_actions]` raw logits (no softmax applied).
    pub action_logits: Tensor<f32>,
    /// `[B, 1]` scalar state values.
    pub value: Tensor<f32>,
}

/// sb3 `MlpPolicy` (i.e. `ActorCriticPolicy` with
/// `features_extractor_class=FlattenExtractor`,
/// `net_arch=[64, 64]`, `activation_fn=nn.Tanh`, default `ortho_init=True`).
///
/// For a 1-D observation the `FlattenExtractor` is the identity, so
/// the policy's forward path is fully captured by the four submodules
/// stored here. The `share_features_extractor` flag is set in the
/// upstream config, but since the extractor is identity it has no
/// numerical effect; the policy and value trunks remain entirely
/// independent (separate weights).
///
/// # State-dict layout (matches sb3 byte-for-byte for CartPole-v1)
///
/// ```text
/// mlp_extractor.policy_net.0.weight   [hidden, obs_dim]
/// mlp_extractor.policy_net.0.bias     [hidden]
/// mlp_extractor.policy_net.2.weight   [hidden, hidden]
/// mlp_extractor.policy_net.2.bias     [hidden]
/// mlp_extractor.value_net.0.weight    [hidden, obs_dim]
/// mlp_extractor.value_net.0.bias      [hidden]
/// mlp_extractor.value_net.2.weight    [hidden, hidden]
/// mlp_extractor.value_net.2.bias      [hidden]
/// action_net.weight                   [n_actions, hidden]
/// action_net.bias                     [n_actions]
/// value_net.weight                    [1, hidden]
/// value_net.bias                      [1]
/// ```
///
/// Note `value_net` at the top level is the *value head* (`[1, hidden]`),
/// not the same as `mlp_extractor.value_net` (the value-trunk MLP).
/// sb3 reuses the name; we mirror it to keep the state-dict round-trip
/// byte-clean.
#[derive(Debug)]
pub struct MlpPolicy {
    /// Two-trunk Tanh-MLP feature processor.
    pub mlp_extractor: MlpExtractor,
    /// `Linear(hidden → n_actions)` action-logits head.
    pub action_net: ActionNet,
    /// `Linear(hidden → 1)` state-value head. Named `value_head` here
    /// (the underlying sb3 key prefix is `value_net.` — see
    /// [`Module::named_parameters`] below).
    pub value_head: ValueHead,
    cfg: MlpPolicyConfig,
    training: bool,
}

impl MlpPolicy {
    /// Construct a fresh `MlpPolicy` with zero-initialized parameters.
    /// The loader replaces them with the pinned safetensors values
    /// before the first forward.
    ///
    /// # Errors
    ///
    /// Forwards [`MlpExtractor::new`], [`ActionNet::new`], or
    /// [`ValueHead::new`] errors (typically only on zero dims).
    pub fn new(cfg: MlpPolicyConfig) -> FerrotorchResult<Self> {
        Ok(Self {
            mlp_extractor: MlpExtractor::new(cfg.obs_dim, cfg.hidden)?,
            action_net: ActionNet::new(cfg.hidden, cfg.n_actions)?,
            value_head: ValueHead::new(cfg.hidden)?,
            cfg,
            training: false,
        })
    }

    /// `MlpPolicyConfig` the policy was constructed with.
    pub fn config(&self) -> MlpPolicyConfig {
        self.cfg
    }

    /// Forward pass: `obs: [B, obs_dim]` → `(action_logits: [B, n_actions],
    /// value: [B, 1])`.
    ///
    /// Mirrors sb3's `ActorCriticPolicy.forward`:
    ///   1. `features = FlattenExtractor(obs)` (identity for 1-D obs).
    ///   2. `latent_pi = mlp_extractor.forward_actor(features)`.
    ///   3. `latent_vf = mlp_extractor.forward_critic(features)`.
    ///   4. `action_logits = action_net(latent_pi)`.
    ///   5. `value = value_net_head(latent_vf)`.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if `obs.shape()[-1] != cfg.obs_dim` or if
    /// `obs` is 0-D. Higher-rank observations (e.g. image obs) are not
    /// supported by this module — sb3 routes them through a CNN
    /// extractor that ferrotorch-rl does not yet ship.
    pub fn forward(&self, obs: &Tensor<f32>) -> FerrotorchResult<PolicyOutput> {
        if obs.ndim() < 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MlpPolicy::forward: obs must be at least 2-D [B, obs_dim], got shape {:?}",
                    obs.shape()
                ),
            });
        }
        let last = obs.shape()[obs.ndim() - 1];
        if last != self.cfg.obs_dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MlpPolicy::forward: obs.shape()[-1] = {last} != obs_dim = {}",
                    self.cfg.obs_dim
                ),
            });
        }

        // FlattenExtractor is identity on already-1-D obs. (For higher-
        // rank obs sb3 would call .flatten(start_dim=1); we currently
        // only support [B, obs_dim].)
        let features = obs;
        let latent_pi = self.mlp_extractor.forward_actor(features)?;
        let latent_vf = self.mlp_extractor.forward_critic(features)?;
        let action_logits = self.action_net.forward(&latent_pi)?;
        let value = self.value_head.forward(&latent_vf)?;
        Ok(PolicyOutput {
            action_logits,
            value,
        })
    }
}

impl Module<f32> for MlpPolicy {
    fn forward(&self, _input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        // The policy has *two* outputs; the standard `Module::forward`
        // contract returns one. Keep the trait impl so the standard
        // `state_dict / load_state_dict` plumbing works, but refuse
        // the call.
        Err(FerrotorchError::InvalidArgument {
            message: "MlpPolicy::Module::forward: call \
                      MlpPolicy::forward(obs) → PolicyOutput instead — \
                      the policy emits two tensors (action_logits, value)."
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        let mut out = self.mlp_extractor.parameters();
        out.extend(self.action_net.parameters());
        out.extend(self.value_head.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
        let mut out = self.mlp_extractor.parameters_mut();
        out.extend(self.action_net.parameters_mut());
        out.extend(self.value_head.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
        let mut out: Vec<(String, &Parameter<f32>)> = Vec::new();
        for (n, p) in self.mlp_extractor.named_parameters() {
            out.push((format!("mlp_extractor.{n}"), p));
        }
        for (n, p) in self.action_net.named_parameters() {
            out.push((format!("action_net.{n}"), p));
        }
        // Top-level value head uses the sb3 key `value_net.`.
        for (n, p) in self.value_head.named_parameters() {
            out.push((format!("value_net.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.mlp_extractor.train();
        self.action_net.train();
        self.value_head.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.mlp_extractor.eval();
        self.action_net.eval();
        self.value_head.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn cartpole_v1_config_matches_sb3_defaults() {
        let c = MlpPolicyConfig::cartpole_v1();
        assert_eq!(c.obs_dim, 4);
        assert_eq!(c.hidden, 64);
        assert_eq!(c.n_actions, 2);
    }

    #[test]
    fn named_parameters_match_sb3_state_dict_layout() {
        let policy = MlpPolicy::new(MlpPolicyConfig::cartpole_v1()).unwrap();
        let mut names: Vec<String> = policy
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        names.sort();
        let mut expected = vec![
            "action_net.bias",
            "action_net.weight",
            "mlp_extractor.policy_net.0.bias",
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.policy_net.2.bias",
            "mlp_extractor.policy_net.2.weight",
            "mlp_extractor.value_net.0.bias",
            "mlp_extractor.value_net.0.weight",
            "mlp_extractor.value_net.2.bias",
            "mlp_extractor.value_net.2.weight",
            "value_net.bias",
            "value_net.weight",
        ];
        expected.sort_unstable();
        assert_eq!(names, expected);
    }

    #[test]
    fn named_parameter_shapes_match_sb3_layout() {
        let policy = MlpPolicy::new(MlpPolicyConfig::cartpole_v1()).unwrap();
        let map: std::collections::HashMap<String, Vec<usize>> = policy
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.shape().to_vec()))
            .collect();
        assert_eq!(map["mlp_extractor.policy_net.0.weight"], vec![64, 4]);
        assert_eq!(map["mlp_extractor.policy_net.0.bias"], vec![64]);
        assert_eq!(map["mlp_extractor.policy_net.2.weight"], vec![64, 64]);
        assert_eq!(map["mlp_extractor.policy_net.2.bias"], vec![64]);
        assert_eq!(map["mlp_extractor.value_net.0.weight"], vec![64, 4]);
        assert_eq!(map["mlp_extractor.value_net.0.bias"], vec![64]);
        assert_eq!(map["mlp_extractor.value_net.2.weight"], vec![64, 64]);
        assert_eq!(map["mlp_extractor.value_net.2.bias"], vec![64]);
        assert_eq!(map["action_net.weight"], vec![2, 64]);
        assert_eq!(map["action_net.bias"], vec![2]);
        assert_eq!(map["value_net.weight"], vec![1, 64]);
        assert_eq!(map["value_net.bias"], vec![1]);
    }

    #[test]
    fn forward_zero_weights_produces_zero_logits_and_value() {
        // With all parameters at zero (the default init for Linear via
        // Parameter::zeros + zero bias), every internal pre-activation
        // is 0, `tanh(0) = 0`, and both heads produce 0. Confirms the
        // forward path is wired end-to-end without numerical surprises.
        let policy = MlpPolicy::new(MlpPolicyConfig::cartpole_v1()).unwrap();
        // Zero out the weight tensors explicitly (Linear::new uses
        // Kaiming-uniform init; we want all-zero parameters here).
        let zeroed = MlpPolicy::new(MlpPolicyConfig::cartpole_v1()).unwrap();
        let _ = &zeroed; // silence unused warning if we don't fall through

        // Easier: walk parameters_mut and overwrite tensors.
        let mut policy = policy;
        for p in policy.parameters_mut() {
            let shape = p.shape().to_vec();
            let n: usize = shape.iter().product();
            p.set_data(t(&vec![0.0_f32; n], &shape));
        }

        let obs = t(&[0.1, -0.05, 0.02, 0.1], &[1, 4]);
        let out = policy.forward(&obs).unwrap();
        assert_eq!(out.action_logits.shape(), &[1, 2]);
        assert_eq!(out.value.shape(), &[1, 1]);
        for v in out.action_logits.data_vec().unwrap() {
            assert!((v - 0.0).abs() < 1e-7, "expected 0 logit, got {v}");
        }
        for v in out.value.data_vec().unwrap() {
            assert!((v - 0.0).abs() < 1e-7, "expected 0 value, got {v}");
        }
    }

    // Helper to make a [out, in] "identity-like" tensor: rows 0..min(out,in)
    // are one-hot on column i, the rest zero. Lifted out of the test fn
    // so clippy's `items_after_statements` lint doesn't fire.
    #[cfg(test)]
    fn ident(out_dim: usize, in_dim: usize) -> Tensor<f32> {
        let mut data = vec![0.0_f32; out_dim * in_dim];
        for i in 0..out_dim.min(in_dim) {
            data[i * in_dim + i] = 1.0;
        }
        t(&data, &[out_dim, in_dim])
    }

    #[cfg(test)]
    fn zeros_vec(n: usize) -> Tensor<f32> {
        t(&vec![0.0_f32; n], &[n])
    }

    #[test]
    fn forward_identity_weights_yields_tanh_chain() {
        // With every weight set to the identity (where shape allows) and
        // every bias = 0, the policy reduces to two stacked tanh
        // applications on the obs, then a final linear with identity
        // (sub)weight on the heads. We use a hand-computed tanh-chain
        // reference for a specific obs vector to confirm the wiring.
        let mut policy = MlpPolicy::new(MlpPolicyConfig::cartpole_v1()).unwrap();

        // mlp_extractor weights: all identity-like, biases zero.
        policy.mlp_extractor.policy_net_0.weight.set_data(ident(64, 4));
        policy
            .mlp_extractor
            .policy_net_0
            .bias
            .as_mut()
            .unwrap()
            .set_data(zeros_vec(64));
        policy.mlp_extractor.policy_net_2.weight.set_data(ident(64, 64));
        policy
            .mlp_extractor
            .policy_net_2
            .bias
            .as_mut()
            .unwrap()
            .set_data(zeros_vec(64));
        policy.mlp_extractor.value_net_0.weight.set_data(ident(64, 4));
        policy
            .mlp_extractor
            .value_net_0
            .bias
            .as_mut()
            .unwrap()
            .set_data(zeros_vec(64));
        policy.mlp_extractor.value_net_2.weight.set_data(ident(64, 64));
        policy
            .mlp_extractor
            .value_net_2
            .bias
            .as_mut()
            .unwrap()
            .set_data(zeros_vec(64));
        // action_net: identity-on-first-2 of 64, bias zero.
        policy.action_net.linear.weight.set_data(ident(2, 64));
        policy
            .action_net
            .linear
            .bias
            .as_mut()
            .unwrap()
            .set_data(zeros_vec(2));
        // value_head: identity-on-first-1 of 64, bias zero.
        policy.value_head.linear.weight.set_data(ident(1, 64));
        policy
            .value_head
            .linear
            .bias
            .as_mut()
            .unwrap()
            .set_data(zeros_vec(1));

        let obs_data = [0.5_f32, -0.25, 0.1, -0.4];
        let obs = t(&obs_data, &[1, 4]);
        let out = policy.forward(&obs).unwrap();

        // Forward:
        //   h0 = ident(64,4) @ obs => first 4 entries match obs, rest 0.
        //   h0_a = tanh(h0)
        //   h1 = ident(64,64) @ h0_a => h0_a (unchanged).
        //   h1_a = tanh(h1) = tanh(tanh(obs)).
        //   logits = ident(2,64) @ h1_a => first 2 entries.
        //   value  = ident(1,64) @ h1_a => first entry.
        let tanh_tanh = |x: f32| x.tanh().tanh();
        let expected_logits = [tanh_tanh(obs_data[0]), tanh_tanh(obs_data[1])];
        let expected_value = tanh_tanh(obs_data[0]);

        let got_logits = out.action_logits.data_vec().unwrap();
        let got_value = out.value.data_vec().unwrap();
        for (g, e) in got_logits.iter().zip(expected_logits.iter()) {
            assert!(
                (g - e).abs() < 1e-6,
                "logit mismatch: got={g} expected={e}"
            );
        }
        assert!(
            (got_value[0] - expected_value).abs() < 1e-6,
            "value mismatch: got={} expected={expected_value}",
            got_value[0]
        );
    }
}
