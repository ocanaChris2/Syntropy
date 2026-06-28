//! Probability mixing in logit space.
//!
//! Final distribution = softmax( (Σ_k w_k · logit_k) / temperature ).
//!
//! The per-model logit vectors come from: the neural Transformer, the local
//! LSTM, and the classical models (their probabilities turned into logits). The
//! mixing weights `w_k` and `temperature` are produced by the exported
//! MetaNetwork (a small MLP over compact per-model features such as entropy,
//! top-1 probability, match length, and model agreement). For the std-only
//! build we ship a static `Gate`; the dynamic, feature-driven gate is described
//! in docs/rust_integration_blueprint.md and slots in behind `Gate::weights_for`.

/// Static gate parameters (the dynamic MetaNetwork overrides `weights_for`).
pub struct Gate {
    pub weights: Vec<f32>,
    pub temperature: f32,
}

impl Gate {
    /// Equal weighting across `k` models, temperature 1.0 — a sane default and
    /// the fallback when no MetaNetwork is loaded.
    pub fn uniform(k: usize) -> Self {
        Gate { weights: vec![1.0 / k as f32; k], temperature: 1.0 }
    }

    /// Hook for the dynamic gate. The static gate ignores `features`; a loaded
    /// MetaNetwork would map features -> (weights, temperature) here.
    pub fn weights_for(&self, _features: &[f32]) -> (&[f32], f32) {
        (&self.weights, self.temperature)
    }
}

/// Turn a probability vector into logits (natural log). Inputs are assumed
/// smoothed (> 0); we clamp to a tiny floor for safety.
pub fn to_logits(probs: &[f32]) -> Vec<f32> {
    probs.iter().map(|&p| p.max(1e-12).ln()).collect()
}

/// Blend per-model logits and return a normalized probability distribution.
pub fn combine(logits: &[Vec<f32>], weights: &[f32], temperature: f32) -> Vec<f32> {
    assert!(!logits.is_empty());
    assert_eq!(logits.len(), weights.len());
    let v = logits[0].len();
    let mut acc = vec![0.0f32; v];
    for (k, lg) in logits.iter().enumerate() {
        debug_assert_eq!(lg.len(), v);
        let w = weights[k];
        for i in 0..v {
            acc[i] += w * lg[i];
        }
    }
    // Numerically stable softmax with temperature.
    let inv_t = 1.0 / temperature;
    let m = acc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for a in acc.iter_mut() {
        *a = ((*a - m) * inv_t).exp();
        sum += *a;
    }
    for a in acc.iter_mut() {
        *a /= sum;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_produces_a_distribution() {
        let l0 = vec![2.0, 0.0, -1.0, 0.5];
        let l1 = vec![0.0, 1.0, 1.0, 0.0];
        let g = Gate::uniform(2);
        let d = combine(&[l0, l1], &g.weights, g.temperature);
        let s: f32 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
        assert!(d.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn temperature_sharpens() {
        let l = vec![vec![3.0, 0.0, 0.0]];
        let hot = combine(&l, &[1.0], 2.0);
        let cold = combine(&l, &[1.0], 0.5);
        // Lower temperature => more peaked on the max.
        assert!(cold[0] > hot[0]);
    }
}
