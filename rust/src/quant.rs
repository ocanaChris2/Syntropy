//! Probability -> integer frequency quantization.
//!
//! Mirrors `quantize()` in reference/coder_reference.py. The contract:
//!   * output length == input length
//!   * every frequency >= 1   (a 0-frequency symbol is undecodable)
//!   * sum of frequencies == TOTAL (2^16)
//!
//! Deterministic "largest-remainder" apportionment: floor every symbol at 1,
//! distribute the remaining (TOTAL - V) counts proportionally, then give the
//! leftover units to the largest fractional parts (ties -> lowest index).
//!
//! Computed in f32 with a fixed operation order so the same float inputs yield
//! the same table on every machine. In production the same Rust binary does
//! both compress and decompress, so its f32 model outputs always agree with
//! themselves and the integer table is what crosses the wire.

pub const PRECISION_BITS: u32 = 16;
pub const TOTAL: u32 = 1 << PRECISION_BITS;

pub fn quantize(probs: &[f32]) -> Vec<u32> {
    quantize_total(probs, TOTAL)
}

pub fn quantize_total(probs: &[f32], total: u32) -> Vec<u32> {
    let v = probs.len();
    assert!(v > 0, "empty distribution");
    assert!(total as usize >= v, "total must be >= vocabulary size");

    // Defensive normalization: ignore NaN/<=0 entries a softmax shouldn't emit.
    let mut s: f64 = 0.0;
    for &p in probs {
        if p.is_finite() && p > 0.0 {
            s += p as f64;
        }
    }
    if s <= 0.0 {
        // Degenerate input -> uniform.
        let base = total / v as u32;
        let mut freqs = vec![base; v];
        let rem = total - base * v as u32;
        for f in freqs.iter_mut().take(rem as usize) {
            *f += 1;
        }
        return freqs;
    }

    let pool = (total - v as u32) as f64;
    let mut freqs = vec![1u32; v];
    let mut fracs: Vec<(f64, usize)> = Vec::with_capacity(v);
    let mut allocated: u32 = 0;
    for (i, &p) in probs.iter().enumerate() {
        let share = if p.is_finite() && p > 0.0 {
            (p as f64 / s) * pool
        } else {
            0.0
        };
        let whole = share.floor();
        freqs[i] += whole as u32;
        allocated += whole as u32;
        fracs.push((share - whole, i));
    }

    let leftover = (total - v as u32) - allocated; // in [0, v)
    // Largest fractional part first; tie-break by lowest index.
    fracs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    for k in 0..leftover as usize {
        freqs[fracs[k].1] += 1;
    }

    debug_assert_eq!(freqs.iter().sum::<u32>(), total);
    debug_assert!(freqs.iter().all(|&f| f >= 1));
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariants_hold() {
        // A spread of shapes including a near-degenerate spike.
        let cases: Vec<Vec<f32>> = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.97, 0.01, 0.01, 0.01],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0], // degenerate -> uniform fallback
            (0..256).map(|i| i as f32 + 1.0).collect(),
        ];
        for c in cases {
            let f = quantize(&c);
            assert_eq!(f.len(), c.len());
            assert_eq!(f.iter().sum::<u32>(), TOTAL);
            assert!(f.iter().all(|&x| x >= 1));
        }
    }

    #[test]
    fn uniform_is_balanced() {
        let f = quantize(&vec![1.0f32; 4]);
        assert_eq!(f, vec![16384, 16384, 16384, 16384]);
    }
}
