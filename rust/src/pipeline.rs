//! The streaming compress / decompress engine.
//!
//! This is where the determinism invariant is enforced: at every position the
//! encoder and decoder compute the EXACT same distribution, because
//!   * the neural model is a pure function of the context,
//!   * the classical models start empty and receive identical updates, and
//!   * the gate/temperature are fixed (or themselves a pure function of state).
//! The distribution is quantized to an integer frequency table and fed to the
//! arithmetic coder. Decompress reconstructs each table BEFORE reading its
//! symbol, so it stays perfectly in step.
//!
//! Container format note: a real file would prepend a small header (magic,
//! config hash, original byte length / token count). Here `compress` returns
//! the token count alongside the bytes so tests can drive `decompress`.

use crate::classical::{MatchModel, Order0, Order1};
use crate::coder::{Decoder, Encoder};
use crate::mixer::{self, Gate};
use crate::model::NeuralModel;
use crate::quant::quantize;
use crate::tokenizer::BpeModel;

/// Hyperparameters for the classical models (these are what the notebook
/// exports as JSON; tables themselves are rebuilt online on both sides).
pub struct ClassicalConfig {
    pub order0_alpha: f32,
    pub order1_alpha: f32,
    pub match_order: usize,
    pub match_max_conf: f32,
}

impl Default for ClassicalConfig {
    fn default() -> Self {
        ClassicalConfig { order0_alpha: 0.02, order1_alpha: 0.02, match_order: 4, match_max_conf: 0.95 }
    }
}

pub struct Pipeline<M: NeuralModel> {
    tok: BpeModel,
    model: M,
    vocab: usize,
    cfg: ClassicalConfig,
    gate: Gate,
    // online classical state (reset at the start of every compress/decompress)
    o0: Order0,
    o1: Order1,
    mm: MatchModel,
}

impl<M: NeuralModel> Pipeline<M> {
    pub fn new(tok: BpeModel, model: M, cfg: ClassicalConfig) -> Self {
        let vocab = tok.vocab_size();
        assert_eq!(vocab, model.vocab(), "tokenizer and model must share vocab size");
        let (o0, o1, mm) = Self::fresh_classical(vocab, &cfg);
        // 4 model streams feed the mixer: neural, order0, order1, match.
        let gate = Gate::uniform(4);
        Pipeline { tok, model, vocab, cfg, gate, o0, o1, mm }
    }

    fn fresh_classical(vocab: usize, cfg: &ClassicalConfig) -> (Order0, Order1, MatchModel) {
        (
            Order0::new(vocab, cfg.order0_alpha),
            Order1::new(vocab, cfg.order1_alpha),
            MatchModel::new(vocab, cfg.match_order, cfg.match_max_conf),
        )
    }

    fn reset(&mut self) {
        let (o0, o1, mm) = Self::fresh_classical(self.vocab, &self.cfg);
        self.o0 = o0;
        self.o1 = o1;
        self.mm = mm;
    }

    /// The shared per-position predictor used identically by both directions.
    fn step_dist(&mut self, ctx: &[u32]) -> Vec<f32> {
        let prev = ctx.last().map(|&x| x as usize);
        let neural = self.model.logits(ctx);
        let p0 = self.o0.predict();
        let p1 = self.o1.predict(prev);
        let pm = self.mm.predict();
        let logits = [neural, mixer::to_logits(&p0), mixer::to_logits(&p1), mixer::to_logits(&pm)];
        let (w, t) = self.gate.weights_for(&[]); // static gate; dynamic gate would pass features
        mixer::combine(&logits, w, t)
    }

    fn observe(&mut self, prev: Option<usize>, sym: usize) {
        self.o0.update(sym);
        self.o1.update(prev, sym);
        self.mm.update(sym);
    }

    /// Returns (coded bytes, number of tokens).
    pub fn compress(&mut self, bytes: &[u8]) -> (Vec<u8>, usize) {
        self.reset();
        let ids = self.tok.encode(bytes);
        let mut enc = Encoder::new();
        let mut ctx: Vec<u32> = Vec::with_capacity(ids.len());
        for &id in &ids {
            let dist = self.step_dist(&ctx);
            let freqs = quantize(&dist);
            enc.encode(id as usize, &freqs);
            let prev = ctx.last().map(|&x| x as usize);
            self.observe(prev, id as usize);
            ctx.push(id);
        }
        (enc.finish(), ids.len())
    }

    pub fn decompress(&mut self, data: &[u8], ntokens: usize) -> Vec<u8> {
        self.reset();
        let mut dec = Decoder::new(data);
        let mut ctx: Vec<u32> = Vec::with_capacity(ntokens);
        for _ in 0..ntokens {
            let dist = self.step_dist(&ctx);
            let freqs = quantize(&dist);
            let id = dec.decode(&freqs) as u32;
            let prev = ctx.last().map(|&x| x as usize);
            self.observe(prev, id as usize);
            ctx.push(id);
        }
        self.tok.decode(&ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StubModel;
    use crate::tokenizer::BpeModel;

    fn pipeline() -> Pipeline<StubModel> {
        let tok = BpeModel::new(vec![
            (b"t".to_vec(), b"h".to_vec()),
            (b"th".to_vec(), b"e".to_vec()),
            (b" ".to_vec(), b"t".to_vec()),
        ]);
        let vocab = tok.vocab_size();
        Pipeline::new(tok, StubModel::new(vocab), ClassicalConfig::default())
    }

    fn roundtrip(input: &[u8]) {
        let mut p = pipeline();
        let (coded, n) = p.compress(input);
        let out = p.decompress(&coded, n);
        assert_eq!(out, input, "roundtrip failed (len {})", input.len());
    }

    #[test]
    fn roundtrip_edge_cases() {
        roundtrip(b"");
        roundtrip(b"a");
        roundtrip(b"the the the the the the the the"); // repetitive
        roundtrip(&[0u8, 255, 128, 1, 254, 7, 7, 7]); // binary
    }

    #[test]
    fn roundtrip_pseudo_random() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut buf = Vec::new();
        for _ in 0..2000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            buf.push((state >> 56) as u8);
        }
        roundtrip(&buf);
    }

    #[test]
    fn roundtrip_textual() {
        roundtrip(b"the quick brown fox jumps over the lazy dog, the the the!");
    }
}
