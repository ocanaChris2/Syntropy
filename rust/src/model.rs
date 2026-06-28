//! Neural model abstraction.
//!
//! The streaming pipeline only needs ONE thing from the neural side: given the
//! context so far, return a logit vector over the vocabulary. That is the
//! `NeuralModel` trait. This keeps the determinism-critical pipeline decoupled
//! from how the logits are produced.
//!
//! * `StubModel` (this file, always available): a deterministic, weight-free
//!   model used to test the full pipeline OFFLINE and bit-exactly. It produces
//!   identical logits for identical context, which is all the round-trip needs.
//!
//! * The REAL model (candle, behind the `neural` feature) loads the exported
//!   safetensors + config and runs the LSTM + ALiBi Transformer with an
//!   incremental KV cache. Its skeleton and the candle forward pass are given
//!   in docs/rust_integration_blueprint.md. It implements the SAME trait, so it
//!   drops straight into the pipeline.

/// Anything that can predict next-token logits from a context.
pub trait NeuralModel {
    fn vocab(&self) -> usize;

    /// Logits over the vocabulary given the realized context (token ids so far).
    /// MUST be a pure function of `ctx` (no hidden RNG) so the encoder and
    /// decoder agree bit-for-bit.
    fn logits(&mut self, ctx: &[u32]) -> Vec<f32>;
}

/// Deterministic, weight-free stand-in for the real network. Good enough to
/// exercise (and bit-exactly verify) the entire compress/decompress pipeline
/// before any training has happened.
pub struct StubModel {
    vocab: usize,
}

impl StubModel {
    pub fn new(vocab: usize) -> Self {
        StubModel { vocab }
    }
}

impl NeuralModel for StubModel {
    fn vocab(&self) -> usize {
        self.vocab
    }

    fn logits(&mut self, ctx: &[u32]) -> Vec<f32> {
        // Mix the last few tokens into a seed, then spread deterministic,
        // mildly-peaked logits. Purely a function of `ctx`.
        let mut seed: u64 = 1469598103934665603;
        for &t in ctx.iter().rev().take(4) {
            seed ^= t as u64;
            seed = seed.wrapping_mul(1099511628211);
        }
        (0..self.vocab)
            .map(|s| {
                let h = seed ^ (s as u64).wrapping_mul(0x9E3779B97F4A7C15);
                // Range roughly [0, 4): gives the coder something non-uniform.
                (((h >> 40) & 0xff) as f32) / 64.0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_is_deterministic() {
        let mut a = StubModel::new(32);
        let mut b = StubModel::new(32);
        let ctx = [1u32, 5, 9, 2, 2];
        assert_eq!(a.logits(&ctx), b.logits(&ctx));
        assert_eq!(a.logits(&ctx).len(), 32);
        // Different context => (almost surely) different logits.
        assert_ne!(a.logits(&ctx), a.logits(&[1, 5, 9, 2, 3]));
    }
}
