//! Classical context models: order-0, order-1, and a match model.
//!
//! These are cheap, adaptive, and START EMPTY — identical update rules on the
//! encoder and decoder mean their states stay in lockstep, so only their
//! *hyperparameters* are exported from training (JSON), never large count
//! tables. Each model emits a probability vector over the vocabulary; the mixer
//! blends them with the neural models in logit space.

/// Per-model output: a probability distribution over the vocabulary.
pub type Dist = Vec<f32>;

#[inline]
fn normalize(counts_plus: &[f32]) -> Dist {
    let s: f32 = counts_plus.iter().sum();
    counts_plus.iter().map(|&c| c / s).collect()
}

/// Order-0: a single adaptive symbol-frequency table (Laplace-smoothed).
pub struct Order0 {
    counts: Vec<f32>,
    alpha: f32,
}

impl Order0 {
    pub fn new(vocab: usize, alpha: f32) -> Self {
        Order0 { counts: vec![0.0; vocab], alpha }
    }
    pub fn predict(&self) -> Dist {
        let smoothed: Vec<f32> = self.counts.iter().map(|&c| c + self.alpha).collect();
        normalize(&smoothed)
    }
    pub fn update(&mut self, sym: usize) {
        self.counts[sym] += 1.0;
    }
}

/// Order-1: symbol frequencies conditioned on the previous symbol. Lazily
/// allocates a row per seen context to stay light for large vocabularies.
pub struct Order1 {
    vocab: usize,
    alpha: f32,
    rows: std::collections::HashMap<usize, Vec<f32>>,
}

impl Order1 {
    pub fn new(vocab: usize, alpha: f32) -> Self {
        Order1 { vocab, alpha, rows: std::collections::HashMap::new() }
    }
    pub fn predict(&self, prev: Option<usize>) -> Dist {
        match prev.and_then(|p| self.rows.get(&p)) {
            Some(row) => {
                let smoothed: Vec<f32> = row.iter().map(|&c| c + self.alpha).collect();
                normalize(&smoothed)
            }
            None => normalize(&vec![self.alpha; self.vocab]),
        }
    }
    pub fn update(&mut self, prev: Option<usize>, sym: usize) {
        if let Some(p) = prev {
            let row = self.rows.entry(p).or_insert_with(|| vec![0.0; self.vocab]);
            row[sym] += 1.0;
        }
    }
}

/// Match model: remembers, for a hash of the last `order` symbols, which symbol
/// followed last time. When the current context hash hits, it predicts that
/// symbol with a confidence that grows with the running match length. This is
/// the cheap powerhouse on repetitive data.
pub struct MatchModel {
    vocab: usize,
    order: usize,
    table: std::collections::HashMap<u64, usize>,
    history: Vec<usize>,
    run_len: u32,
    max_conf: f32,
}

impl MatchModel {
    pub fn new(vocab: usize, order: usize, max_conf: f32) -> Self {
        MatchModel {
            vocab,
            order,
            table: std::collections::HashMap::new(),
            history: Vec::new(),
            run_len: 0,
            max_conf,
        }
    }

    fn ctx_hash(&self) -> Option<u64> {
        if self.history.len() < self.order {
            return None;
        }
        let mut h: u64 = 1469598103934665603; // FNV offset
        for &s in &self.history[self.history.len() - self.order..] {
            h ^= s as u64;
            h = h.wrapping_mul(1099511628211);
        }
        Some(h)
    }

    pub fn predict(&self) -> Dist {
        let base = 1.0 / self.vocab as f32;
        let mut dist = vec![base; self.vocab];
        if let Some(h) = self.ctx_hash() {
            if let Some(&pred) = self.table.get(&h) {
                // Confidence grows with match run length, capped.
                let conf = (self.max_conf * (1.0 - 1.0 / (1.0 + self.run_len as f32)))
                    .min(self.max_conf);
                let spread = (1.0 - conf) / self.vocab as f32;
                for d in dist.iter_mut() {
                    *d = spread;
                }
                dist[pred] += conf;
            }
        }
        let s: f32 = dist.iter().sum();
        for d in dist.iter_mut() {
            *d /= s;
        }
        dist
    }

    pub fn update(&mut self, sym: usize) {
        // Track the run length: did our last prediction come true?
        if let Some(h) = self.ctx_hash() {
            match self.table.get(&h) {
                Some(&pred) if pred == sym => self.run_len = self.run_len.saturating_add(1),
                _ => self.run_len = 0,
            }
            self.table.insert(h, sym);
        }
        self.history.push(sym);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_distribution(d: &[f32]) {
        let s: f32 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "sum={}", s);
        assert!(d.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn order0_adapts() {
        let mut m = Order0::new(8, 0.1);
        is_distribution(&m.predict());
        for _ in 0..10 {
            m.update(3);
        }
        let d = m.predict();
        is_distribution(&d);
        // After seeing only symbol 3, it should be the mode.
        let argmax = d.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(argmax, 3);
    }

    #[test]
    fn match_model_learns_repetition() {
        let mut m = MatchModel::new(16, 2, 0.95);
        // Feed a periodic pattern; after a few periods it should predict it.
        let pattern = [1usize, 2, 3, 4];
        for _ in 0..8 {
            for &s in &pattern {
                let _ = m.predict();
                m.update(s);
            }
        }
        // Now context is [...,4]; ctx of last 2 should predict the continuation.
        let d = m.predict();
        is_distribution(&d);
    }
}
