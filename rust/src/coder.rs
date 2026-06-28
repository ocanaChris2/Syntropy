//! 32-bit Witten-Neal-Cleary arithmetic coder.
//!
//! This is a line-for-line port of `reference/coder_reference.py`. It is pure
//! integer math, therefore bit-for-bit identical across platforms and matches
//! the Python reference exactly (see the golden-vector test at the bottom).
//!
//! FIFO + streaming: each symbol is coded under its own frequency table, which
//! is exactly what an adaptive autoregressive model needs (the distribution
//! changes every token, in stream order). That is why we use arithmetic coding
//! and not ANS/rANS (which is LIFO).

pub const CODE_BITS: u32 = 32;
pub const WHOLE: u64 = 1 << 32;
pub const HALF: u64 = 1 << 31;
pub const QUARTER: u64 = 1 << 30;
pub const THREE_QUARTER: u64 = 3 * QUARTER;
pub const MASK: u64 = WHOLE - 1;

// ---------------------------------------------------------------------------
// Bit I/O (MSB-first, zero-padded). Past the end the reader yields 0 bits,
// which the encoder's two-bit flush guarantees is enough to finish decoding.
// ---------------------------------------------------------------------------

struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { bytes: Vec::new(), cur: 0, nbits: 0 }
    }
    #[inline]
    fn put(&mut self, bit: u8) {
        self.cur = (self.cur << 1) | (bit & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit index
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }
    #[inline]
    fn get(&mut self) -> u64 {
        let byte_idx = self.pos >> 3;
        let bit = if byte_idx >= self.data.len() {
            0
        } else {
            ((self.data[byte_idx] >> (7 - (self.pos & 7))) & 1) as u64
        };
        self.pos += 1;
        bit
    }
}

// ---------------------------------------------------------------------------
// Cumulative helper. freqs must be all >= 1 and sum to <= 2^32 (in practice
// they sum to exactly TOTAL = 2^16 after quantization).
// ---------------------------------------------------------------------------

#[inline]
fn cum_low_high_total(freqs: &[u32], sym: usize) -> (u64, u64, u64) {
    let mut cl: u64 = 0;
    for &f in &freqs[..sym] {
        cl += f as u64;
    }
    let ch = cl + freqs[sym] as u64;
    let mut total = ch;
    for &f in &freqs[sym + 1..] {
        total += f as u64;
    }
    (cl, ch, total)
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

pub struct Encoder {
    low: u64,
    high: u64,
    pending: u64,
    out: BitWriter,
}

impl Encoder {
    pub fn new() -> Self {
        Encoder { low: 0, high: MASK, pending: 0, out: BitWriter::new() }
    }

    #[inline]
    fn emit(&mut self, bit: u8) {
        self.out.put(bit);
        while self.pending > 0 {
            self.out.put(bit ^ 1);
            self.pending -= 1;
        }
    }

    /// Encode one symbol under `freqs` (length = vocabulary size).
    pub fn encode(&mut self, sym: usize, freqs: &[u32]) {
        let (cl, ch, total) = cum_low_high_total(freqs, sym);
        let rng = self.high - self.low + 1;
        self.high = self.low + rng * ch / total - 1;
        self.low = self.low + rng * cl / total;
        self.renorm();
    }

    #[inline]
    fn renorm(&mut self) {
        loop {
            if self.high < HALF {
                self.emit(0);
            } else if self.low >= HALF {
                self.emit(1);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    /// Flush and return the coded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < QUARTER {
            self.emit(0);
        } else {
            self.emit(1);
        }
        self.out.finish()
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

pub struct Decoder<'a> {
    low: u64,
    high: u64,
    value: u64,
    reader: BitReader<'a>,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = BitReader::new(data);
        let mut value: u64 = 0;
        for _ in 0..CODE_BITS {
            value = (value << 1) | reader.get();
        }
        Decoder { low: 0, high: MASK, value, reader }
    }

    /// Decode one symbol; caller must supply the SAME `freqs` used at encode
    /// time for this position.
    pub fn decode(&mut self, freqs: &[u32]) -> usize {
        let total: u64 = freqs.iter().map(|&f| f as u64).sum();
        let rng = self.high - self.low + 1;
        let target = ((self.value - self.low + 1) * total - 1) / rng;

        // Find sym with prefix[sym] <= target < prefix[sym+1] (linear-cum scan;
        // the pipeline can swap in a Fenwick tree for large vocabularies).
        let mut acc: u64 = 0;
        let mut sym = 0usize;
        let mut cl = 0u64;
        let mut ch = 0u64;
        for (i, &f) in freqs.iter().enumerate() {
            let next = acc + f as u64;
            if target < next {
                sym = i;
                cl = acc;
                ch = next;
                break;
            }
            acc = next;
        }

        self.high = self.low + rng * ch / total - 1;
        self.low = self.low + rng * cl / total;
        self.renorm();
        sym
    }

    #[inline]
    fn renorm(&mut self) {
        loop {
            if self.high < HALF {
                // nothing
            } else if self.low >= HALF {
                self.low -= HALF;
                self.high -= HALF;
                self.value -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.low -= QUARTER;
                self.high -= QUARTER;
                self.value -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.value = (self.value << 1) | self.reader.get();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vectors copied from `reference/coder_reference.py` (integer freq
    /// tables, so they are float-independent). The Rust coder MUST reproduce
    /// these exact bytes — this is the strict cross-language contract.
    #[test]
    fn golden_vectors_match_python_reference() {
        let cases: &[(&[u32], &[usize], &str)] = &[
            (&[16384, 16384, 16384, 16384], &[0, 1, 2, 3, 3, 2, 1, 0], "1be440"),
            (&[60000, 2000, 2000, 1536], &[0, 0, 0, 1, 0, 2, 0], "b910"),
            (&[65533, 1, 1, 1], &[0, 0, 0, 0, 1, 0, 3], "fff2004780"),
        ];
        for (freqs, syms, expected_hex) in cases {
            let mut enc = Encoder::new();
            for &s in *syms {
                enc.encode(s, freqs);
            }
            let bytes = enc.finish();
            let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            assert_eq!(&hex, expected_hex, "encoder mismatch for {:?}", syms);

            // ...and it round-trips.
            let mut dec = Decoder::new(&bytes);
            let back: Vec<usize> = (0..syms.len()).map(|_| dec.decode(freqs)).collect();
            assert_eq!(&back, syms, "decoder mismatch for {:?}", syms);
        }
    }

    #[test]
    fn adaptive_roundtrip() {
        // Distribution depends on the realized prefix (FIFO property).
        let v = 29usize;
        let dist = |t: usize, prefix: &[usize]| -> Vec<u32> {
            let mut f = vec![1u32; v];
            let seed = (t as u64).wrapping_mul(2654435761)
                ^ prefix.iter().map(|&x| x as u64).sum::<u64>();
            for (i, fi) in f.iter_mut().enumerate() {
                *fi = 1 + ((seed.wrapping_add(i as u64).wrapping_mul(40503)) % 500) as u32;
            }
            if let Some(&last) = prefix.last() {
                f[last] += 4000;
            }
            // renormalize to sum 2^16 with the same largest-remainder rule the
            // quantizer uses, via floats.
            let probs: Vec<f32> = {
                let s: f64 = f.iter().map(|&x| x as f64).sum();
                f.iter().map(|&x| (x as f64 / s) as f32).collect()
            };
            crate::quant::quantize(&probs)
        };

        let mut state = 12345u64;
        for _ in 0..40 {
            // simple LCG to build a random sequence
            let n = (state % 200 + 1) as usize;
            let mut seq = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                seq.push((state >> 33) as usize % v);
            }
            let mut enc = Encoder::new();
            for (t, &s) in seq.iter().enumerate() {
                enc.encode(s, &dist(t, &seq[..t]));
            }
            let bytes = enc.finish();
            let mut dec = Decoder::new(&bytes);
            let mut back = Vec::with_capacity(n);
            for t in 0..n {
                back.push(dec.decode(&dist(t, &back)));
            }
            assert_eq!(back, seq);
        }
    }
}
