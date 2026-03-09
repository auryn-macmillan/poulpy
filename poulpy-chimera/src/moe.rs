//! Mixture-of-Experts (MoE) routing under FHE.
//!
//! MoE transformers activate only a subset of "expert" FFN layers per token,
//! dramatically reducing the compute cost relative to the total parameter count.
//! Under FHE, this sparsity is especially valuable: inactive experts are simply
//! not computed, giving a direct cost reduction proportional to k/E where k is
//! the number of active experts and E is the total number of experts.
//!
//! ## Routing Strategies
//!
//! The challenge is that routing decisions must be made on encrypted data.
//! CHIMERA supports two routing strategies:
//!
//! 1. **Cleartext routing** (semi-honest model): The routing decision is made
//!    by the provider on cleartext router logits. The provider learns which
//!    experts are activated but not the actual activations. This leaks some
//!    information but is much cheaper.
//!
//! 2. **Encrypted routing** (full privacy): The top-k selection is performed
//!    under FHE using comparison circuits from Poulpy's BDD arithmetic. This
//!    reveals nothing to the provider but is expensive.

use crate::params::ModelDims;

/// MoE routing strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Provider sees which experts are active (semi-honest security model).
    /// Cheapest option: only a single plaintext matmul for router logits.
    CleartextRouting,
    /// Full encrypted routing via homomorphic comparison.
    /// Most expensive but reveals nothing about the routing decision.
    EncryptedRouting,
    /// Deterministic routing: experts are assigned by token position.
    /// No routing computation needed, but inflexible.
    DeterministicRouting,
}

/// Configuration for MoE routing.
#[derive(Clone, Debug)]
pub struct MoEConfig {
    /// Total number of experts.
    pub n_experts: usize,
    /// Number of active experts per token.
    pub n_active: usize,
    /// Routing strategy.
    pub strategy: RoutingStrategy,
    /// Router weight matrix dimension: [d_model × n_experts].
    pub router_dim: usize,
}

impl MoEConfig {
    /// Creates a default MoE configuration from model dimensions.
    pub fn from_dims(dims: &ModelDims, strategy: RoutingStrategy) -> Self {
        MoEConfig {
            n_experts: dims.n_experts,
            n_active: dims.n_active_experts,
            strategy,
            router_dim: dims.d_model,
        }
    }
}

/// Cost estimate for MoE routing.
#[derive(Clone, Debug)]
pub struct MoERoutingPlan {
    /// Number of ct-pt multiplications for router logits.
    pub router_muls: usize,
    /// Number of homomorphic comparisons for top-k selection (0 if cleartext routing).
    pub comparison_ops: usize,
    /// Multiplicative depth of the routing computation.
    pub routing_depth: usize,
    /// Cost ratio vs dense model: n_active / n_experts.
    pub sparsity_ratio: f64,
    /// Number of expert FFN evaluations (= n_active).
    pub expert_evals: usize,
}

/// Plans MoE routing computation.
pub fn plan_moe_routing(config: &MoEConfig) -> MoERoutingPlan {
    let router_muls = config.router_dim; // One matmul row per expert

    let (comparison_ops, routing_depth) = match config.strategy {
        RoutingStrategy::CleartextRouting => (0, 0),
        RoutingStrategy::EncryptedRouting => {
            // Top-k selection via comparison circuit:
            // Requires O(n_experts * log(n_experts)) comparisons
            // Each comparison is ~32 depth (using BDD circuits for i32 compare)
            let n = config.n_experts;
            let comparisons = n * ((n as f64).log2().ceil() as usize);
            let depth = 32; // i32 comparison circuit depth
            (comparisons, depth)
        }
        RoutingStrategy::DeterministicRouting => (0, 0),
    };

    let sparsity_ratio = config.n_active as f64 / config.n_experts as f64;

    MoERoutingPlan {
        router_muls,
        comparison_ops,
        routing_depth,
        sparsity_ratio,
        expert_evals: config.n_active,
    }
}

/// Computes router logits on plaintext values (for testing/verification).
///
/// router_logits = x · W_router^T
/// top_k_indices = argsort(router_logits)[:n_active]
pub fn route_plaintext(
    x: &[f64],
    router_weights: &[Vec<f64>],
    n_active: usize,
) -> Vec<usize> {
    let n_experts = router_weights.len();
    assert!(n_active <= n_experts);

    // Compute router logits
    let mut logits: Vec<(f64, usize)> = router_weights
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let logit: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).sum();
            (logit, i)
        })
        .collect();

    // Sort by logit (descending) and take top-k
    logits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    logits.iter().take(n_active).map(|&(_, i)| i).collect()
}

/// Computes expert gating weights using softmax over selected experts' logits.
pub fn compute_gating_weights(
    x: &[f64],
    router_weights: &[Vec<f64>],
    active_experts: &[usize],
) -> Vec<f64> {
    // Compute logits for active experts
    let logits: Vec<f64> = active_experts
        .iter()
        .map(|&i| {
            x.iter()
                .zip(router_weights[i].iter())
                .map(|(&xi, &wi)| xi * wi)
                .sum()
        })
        .collect();

    // Softmax over active expert logits
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();

    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_routing_cleartext() {
        let config = MoEConfig {
            n_experts: 8,
            n_active: 2,
            strategy: RoutingStrategy::CleartextRouting,
            router_dim: 4096,
        };
        let plan = plan_moe_routing(&config);
        assert_eq!(plan.comparison_ops, 0);
        assert_eq!(plan.routing_depth, 0);
        assert_eq!(plan.expert_evals, 2);
        assert!((plan.sparsity_ratio - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_moe_routing_encrypted() {
        let config = MoEConfig {
            n_experts: 8,
            n_active: 2,
            strategy: RoutingStrategy::EncryptedRouting,
            router_dim: 4096,
        };
        let plan = plan_moe_routing(&config);
        assert!(plan.comparison_ops > 0);
        assert_eq!(plan.routing_depth, 32);
        assert_eq!(plan.expert_evals, 2);
    }

    #[test]
    fn test_route_plaintext() {
        // 4 experts, 2-dim input
        let x = vec![1.0, 2.0];
        let router_weights = vec![
            vec![0.1, 0.2], // expert 0: logit = 0.5
            vec![0.5, 0.5], // expert 1: logit = 1.5
            vec![0.3, 0.1], // expert 2: logit = 0.5
            vec![0.9, 0.1], // expert 3: logit = 1.1
        ];

        let active = route_plaintext(&x, &router_weights, 2);
        assert_eq!(active.len(), 2);
        // Experts 1 (logit=1.5) and 3 (logit=1.1) should be selected
        assert_eq!(active[0], 1);
        assert_eq!(active[1], 3);
    }

    #[test]
    fn test_gating_weights() {
        let x = vec![1.0, 2.0];
        let router_weights = vec![
            vec![0.5, 0.5], // expert 0
            vec![0.9, 0.1], // expert 1
        ];
        let active = vec![0, 1];
        let weights = compute_gating_weights(&x, &router_weights, &active);

        assert_eq!(weights.len(), 2);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "gating weights sum = {sum}");
    }

    #[test]
    fn test_sparsity_ratio() {
        let dims = ModelDims::moe_40b();
        let config = MoEConfig::from_dims(&dims, RoutingStrategy::CleartextRouting);
        let plan = plan_moe_routing(&config);
        // 2/8 = 0.25
        assert!((plan.sparsity_ratio - 0.25).abs() < 1e-6);
    }
}
