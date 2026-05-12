//! Helpers that load a [`MlpPolicy`] from a `model.safetensors` mirror.
//!
//! The pinned `ferrotorch/ppo-cartpole-v1` mirror stores the upstream
//! sb3 `ActorCriticPolicy.state_dict()` verbatim — the keys are
//! exactly:
//!
//! ```text
//! mlp_extractor.policy_net.0.weight   [hidden, obs_dim]
//! mlp_extractor.policy_net.0.bias     [hidden]
//! mlp_extractor.policy_net.2.weight   [hidden, hidden]
//! mlp_extractor.policy_net.2.bias     [hidden]
//! mlp_extractor.value_net.0.weight    [hidden, obs_dim]
//! mlp_extractor.value_net.0.bias      [hidden]
//! mlp_extractor.value_net.2.weight    [hidden, hidden]
//! mlp_extractor.value_net.2.bias      [hidden]
//! action_net.weight                   [n_actions, hidden]
//! action_net.bias                     [n_actions]
//! value_net.weight                    [1, hidden]
//! value_net.bias                      [1]
//! ```
//!
//! These are the same keys [`MlpPolicy::named_parameters`] produces, so
//! the loader is a pure pass-through into the standard
//! `Module::load_state_dict` machinery. The wrapper here exists for
//! parity with the other crate-level `load_*` entry points and to
//! return a [`DropReport`] documenting upstream keys that were
//! intentionally not consumed (per the #1141 audit rail: every key
//! must either land in a parameter or appear in the report).

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};
use ferrotorch_nn::module::Module;
use ferrotorch_serialize::load_safetensors;

use crate::mlp_policy::{MlpPolicy, MlpPolicyConfig};

/// Audit trail returned by [`load_ppo_policy`].
///
/// `unmapped` lists every upstream safetensors key that did NOT match
/// a parameter on the ferrotorch `MlpPolicy`. For the canonical
/// `sb3/ppo-CartPole-v1` pin this is always empty — any non-empty
/// entry on a real pin is a state-dict-drop bug (the #1141 class of
/// failure) and the loader propagates it loudly when `strict=true`.
#[derive(Debug, Default, Clone)]
pub struct DropReport {
    /// Upstream keys present in the safetensors but not mapped to a
    /// parameter on `MlpPolicy`. Empty for a clean pin.
    pub unmapped: Vec<String>,
}

/// Load an [`MlpPolicy`] from `weights_path` (a `model.safetensors`
/// file) using `cfg` to size the model.
///
/// Returns the loaded policy plus a [`DropReport`] for the audit rail.
///
/// `strict=true` errors loudly if any upstream key cannot be mapped;
/// `strict=false` records the unmapped keys in the report and
/// continues. Either way, all *expected* parameter keys must be
/// present in the state dict — a missing key is always fatal.
///
/// # Errors
///
/// Forwards safetensors parse errors, `MlpPolicy` construction errors,
/// and any per-key shape mismatch from `Module::load_state_dict`.
pub fn load_ppo_policy(
    weights_path: &Path,
    cfg: MlpPolicyConfig,
    strict: bool,
) -> FerrotorchResult<(MlpPolicy, DropReport)> {
    let state = load_safetensors::<f32>(weights_path).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!(
                "load_ppo_policy: failed to decode safetensors {}: {e}",
                weights_path.display()
            ),
        }
    })?;

    let mut policy = MlpPolicy::new(cfg)?;
    let expected: std::collections::HashSet<String> = policy
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let mut unmapped: Vec<String> = Vec::new();
    for k in state.keys() {
        if !expected.contains(k) {
            unmapped.push(k.clone());
        }
    }
    unmapped.sort();
    if strict && !unmapped.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "load_ppo_policy: unmapped upstream keys (strict mode): {unmapped:?}"
            ),
        });
    }

    // Filter to only the keys `Module::load_state_dict` knows about
    // (otherwise it would itself reject extras in strict mode and
    // bypass our richer DropReport).
    let filtered: std::collections::HashMap<String, ferrotorch_core::Tensor<f32>> = state
        .into_iter()
        .filter(|(k, _)| expected.contains(k))
        .collect();
    policy.load_state_dict(&filtered, /* strict = */ true)?;
    Ok((policy, DropReport { unmapped }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_serialize::save_safetensors;

    #[test]
    fn round_trip_into_mlp_policy() {
        // Build a tiny MlpPolicy, dump its state_dict, load it back,
        // and confirm the named_parameters' tensor values match exactly.
        let cfg = MlpPolicyConfig {
            obs_dim: 4,
            hidden: 8,
            n_actions: 2,
        };
        let src = MlpPolicy::new(cfg).unwrap();
        // Snapshot expected (name, data) before consuming src.
        let expected: Vec<(String, Vec<f32>)> = src
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().data_vec().unwrap()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        save_safetensors(&src.state_dict(), &path).unwrap();
        let (dst, report) = load_ppo_policy(&path, cfg, /* strict = */ true).unwrap();
        assert!(report.unmapped.is_empty(), "report = {report:?}");
        let dst_params: std::collections::HashMap<String, Vec<f32>> = dst
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().data_vec().unwrap()))
            .collect();
        for (k, vexp) in &expected {
            let v = &dst_params[k];
            assert_eq!(v.len(), vexp.len(), "len mismatch for {k}");
            for (a, b) in v.iter().zip(vexp.iter()) {
                assert!((a - b).abs() < 1e-7, "value mismatch in {k}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn round_trip_forward_matches() {
        use ferrotorch_core::{Tensor, TensorStorage};
        let cfg = MlpPolicyConfig {
            obs_dim: 4,
            hidden: 8,
            n_actions: 2,
        };
        let src = MlpPolicy::new(cfg).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        save_safetensors(&src.state_dict(), &path).unwrap();
        let (dst, _r) = load_ppo_policy(&path, cfg, /* strict = */ true).unwrap();

        let obs = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1_f32, -0.2, 0.3, -0.4]),
            vec![1, 4],
            false,
        )
        .unwrap();
        let a = src.forward(&obs).unwrap();
        let b = dst.forward(&obs).unwrap();
        for (x, y) in a
            .action_logits
            .data_vec()
            .unwrap()
            .iter()
            .zip(b.action_logits.data_vec().unwrap().iter())
        {
            assert!((x - y).abs() < 1e-6);
        }
        for (x, y) in a
            .value
            .data_vec()
            .unwrap()
            .iter()
            .zip(b.value.data_vec().unwrap().iter())
        {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn unmapped_keys_strict_errors() {
        use ferrotorch_core::{Tensor, TensorStorage};
        // Build a state dict with an extra key not present in MlpPolicy.
        let cfg = MlpPolicyConfig {
            obs_dim: 4,
            hidden: 8,
            n_actions: 2,
        };
        let src = MlpPolicy::new(cfg).unwrap();
        let mut sd = src.state_dict();
        sd.insert(
            "log_std".to_string(),
            Tensor::from_storage(TensorStorage::cpu(vec![0.0_f32, 0.0]), vec![2], false).unwrap(),
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        save_safetensors(&sd, &path).unwrap();

        let strict_err = load_ppo_policy(&path, cfg, /* strict = */ true);
        assert!(
            strict_err.is_err(),
            "strict mode must reject unmapped keys"
        );

        let (_p, report) = load_ppo_policy(&path, cfg, /* strict = */ false).unwrap();
        assert_eq!(report.unmapped, vec!["log_std".to_string()]);
    }
}
