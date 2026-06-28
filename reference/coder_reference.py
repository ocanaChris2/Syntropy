#!/usr/bin/env python3
"""
Syntropy — shared entropy-coder reference (the executable spec).

This file is the *single source of truth* for two things that MUST be bit-for-bit
identical in the Python notebook and the Rust compressor:

  1. PROBABILITY QUANTIZATION  — turning a float probability vector into an
     integer frequency table (freqs >= 1, sum == 2**PRECISION_BITS) using a
     deterministic "largest-remainder" rule.
  2. ARITHMETIC CODING         — a 32-bit Witten-Neal-Cleary (CACM'87) range
     coder with E1/E2/E3 underflow handling. FIFO and streaming, so it matches
     adaptive autoregressive models whose distribution changes every token.

Why arithmetic and not ANS:
  Adaptive coding is FIFO — symbol t is coded under a distribution that depends
  on symbols 0..t-1, in stream order. ANS/rANS is LIFO and would require
  buffering the entire distribution sequence and coding in reverse. WNC codes
  forward, one token at a time, and round-trips bit-identically across
  languages given the same integer frequency tables.

Determinism contract (read this):
  * The coder is PURE INTEGER MATH → identical on every platform/language.
    Its golden vectors (freq table -> coded bytes) are the strict cross-language
    contract; see `golden_vectors()`.
  * The quantizer is specified exactly below. This reference computes in float64
    for clarity; the Rust port computes in f32 with the SAME operation order and
    tie-break. The make-or-break invariant in production is that the *same Rust
    binary* does both compress and decompress, so its floats always agree with
    themselves; the integer freq table is what crosses the wire.

Run me:  python3 reference/coder_reference.py
  (pure stdlib — no numpy/torch required.)
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

# ----------------------------------------------------------------------------
# 1. Quantization: float probabilities -> integer frequency table
# ----------------------------------------------------------------------------

PRECISION_BITS = 16
TOTAL = 1 << PRECISION_BITS  # cumulative frequencies always sum to exactly this


def quantize(probs: Sequence[float], total: int = TOTAL) -> List[int]:
    """Map a probability vector to integer frequencies with these guarantees:

        * len(out) == len(probs)
        * every out[i] >= 1                  (a 0-frequency symbol is undecodable)
        * sum(out) == total                  (the coder requires an exact total)

    Rule (deterministic largest-remainder / Hamilton apportionment):
        reserve 1 count per symbol as a floor, distribute the remaining
        (total - V) counts proportionally, then hand the leftover units to the
        symbols with the largest fractional parts (ties broken by lowest index).

    This is reproducible byte-for-byte as long as `probs` is computed
    identically on both sides (in production: same Rust binary, f32, same op
    order).
    """
    v = len(probs)
    if v == 0:
        raise ValueError("empty distribution")
    if total < v:
        raise ValueError("total must be >= vocabulary size to floor every symbol at 1")

    # Normalize defensively so the proportional pool is exactly (total - v).
    s = 0.0
    for p in probs:
        # clamp away negatives / NaNs that a softmax should never produce but
        # numerical noise occasionally does.
        s += p if (p == p and p > 0.0) else 0.0
    if s <= 0.0:
        # Degenerate input -> uniform.
        base = total // v
        freqs = [base] * v
        for i in range(total - base * v):
            freqs[i] += 1
        return freqs

    pool = total - v
    freqs = [1] * v
    fracs: List[Tuple[float, int]] = []
    allocated = 0
    for i, p in enumerate(probs):
        share = (p / s) * pool if (p == p and p > 0.0) else 0.0
        whole = int(share)  # floor for non-negative share
        freqs[i] += whole
        allocated += whole
        fracs.append((share - whole, i))

    leftover = pool - allocated  # in [0, v)
    # Largest fractional part first; tie-break by lowest index for determinism.
    fracs.sort(key=lambda t: (-t[0], t[1]))
    for k in range(leftover):
        freqs[fracs[k][1]] += 1

    # Invariants (cheap, keep them on — they catch quantizer regressions early).
    assert sum(freqs) == total, (sum(freqs), total)
    assert min(freqs) >= 1
    return freqs


def cumulative(freqs: Sequence[int]) -> List[int]:
    """Exclusive-prefix cumulative table of length len(freqs)+1; cum[-1] == total."""
    cum = [0] * (len(freqs) + 1)
    acc = 0
    for i, f in enumerate(freqs):
        cum[i] = acc
        acc += f
    cum[len(freqs)] = acc
    return cum


# ----------------------------------------------------------------------------
# 2. 32-bit Witten-Neal-Cleary arithmetic coder
# ----------------------------------------------------------------------------

CODE_BITS = 32
WHOLE = 1 << CODE_BITS          # 2**32
HALF = WHOLE >> 1               # 2**31
QUARTER = WHOLE >> 2            # 2**30
THREE_QUARTER = 3 * QUARTER     # 3 * 2**30  (< WHOLE)
MASK = WHOLE - 1                # 0xFFFFFFFF


class _BitWriter:
    __slots__ = ("_bytes", "_cur", "_nbits")

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0

    def put(self, bit: int) -> None:
        self._cur = (self._cur << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def finish(self) -> bytes:
        if self._nbits:
            self._cur <<= (8 - self._nbits)  # pad low bits with 0
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0
        return bytes(self._bytes)


class _BitReader:
    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0  # bit position

    def get(self) -> int:
        # Past the end we feed zero bits (the encoder's flush guarantees this is
        # sufficient to disambiguate the final symbol).
        byte_idx = self._pos >> 3
        if byte_idx >= len(self._data):
            self._pos += 1
            return 0
        bit = (self._data[byte_idx] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit


class ArithmeticEncoder:
    """Encode a stream of symbols, each under its own (possibly changing) freq
    table. Call `encode` per symbol, then `finish()` to get the byte string."""

    def __init__(self) -> None:
        self._low = 0
        self._high = MASK
        self._pending = 0
        self._out = _BitWriter()

    def _emit(self, bit: int) -> None:
        self._out.put(bit)
        while self._pending:
            self._out.put(bit ^ 1)
            self._pending -= 1

    def encode(self, sym: int, freqs: Sequence[int]) -> None:
        cum = cumulative(freqs)
        total = cum[-1]
        cl, ch = cum[sym], cum[sym + 1]
        rng = self._high - self._low + 1
        self._high = self._low + (rng * ch) // total - 1
        self._low = self._low + (rng * cl) // total
        self._renorm()

    def _renorm(self) -> None:
        while True:
            if self._high < HALF:
                self._emit(0)
            elif self._low >= HALF:
                self._emit(1)
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTER:
                self._pending += 1
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1

    def finish(self) -> bytes:
        # Emit two bits identifying the final quarter so the decoder lands inside
        # [low, high]; the pending machinery flushes the rest.
        self._pending += 1
        if self._low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self._out.finish()


class ArithmeticDecoder:
    """Mirror of ArithmeticEncoder. The caller drives it symbol-by-symbol,
    supplying the SAME freq table it used at encode time for that position."""

    def __init__(self, data: bytes) -> None:
        self._reader = _BitReader(data)
        self._low = 0
        self._high = MASK
        self._value = 0
        for _ in range(CODE_BITS):
            self._value = (self._value << 1) | self._reader.get()

    def decode(self, freqs: Sequence[int]) -> int:
        cum = cumulative(freqs)
        total = cum[-1]
        rng = self._high - self._low + 1
        # Target cumulative count the current value maps to.
        target = ((self._value - self._low + 1) * total - 1) // rng
        # Find symbol s with cum[s] <= target < cum[s+1] (binary search).
        lo, hi = 0, len(freqs)
        while lo + 1 < hi:
            mid = (lo + hi) >> 1
            if cum[mid] <= target:
                lo = mid
            else:
                hi = mid
        sym = lo
        cl, ch = cum[sym], cum[sym + 1]
        self._high = self._low + (rng * ch) // total - 1
        self._low = self._low + (rng * cl) // total
        self._renorm()
        return sym

    def _renorm(self) -> None:
        while True:
            if self._high < HALF:
                pass
            elif self._low >= HALF:
                self._low -= HALF
                self._high -= HALF
                self._value -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTER:
                self._low -= QUARTER
                self._high -= QUARTER
                self._value -= QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1
            self._value = (self._value << 1) | self._reader.get()


# ----------------------------------------------------------------------------
# 3. Convenience: code a whole sequence given a per-position distribution fn
# ----------------------------------------------------------------------------

def encode_sequence(symbols: Sequence[int], dist_fn) -> bytes:
    """dist_fn(t, prefix) -> probability list for position t (prefix = symbols[:t])."""
    enc = ArithmeticEncoder()
    for t, sym in enumerate(symbols):
        freqs = quantize(dist_fn(t, symbols[:t]))
        enc.encode(sym, freqs)
    return enc.finish()


def decode_sequence(data: bytes, n: int, dist_fn) -> List[int]:
    dec = ArithmeticDecoder(data)
    out: List[int] = []
    for t in range(n):
        freqs = quantize(dist_fn(t, out))
        out.append(dec.decode(freqs))
    return out


# ----------------------------------------------------------------------------
# 4. Golden vectors (the strict cross-language contract for the coder)
# ----------------------------------------------------------------------------

def golden_vectors() -> List[Tuple[List[int], List[int], bytes]]:
    """Return [(freqs, symbols, expected_bytes)] computed by THIS coder.

    The Rust coder must reproduce `expected_bytes` exactly from the same
    (freqs, symbols). These use EXPLICIT integer freq tables that already sum to
    TOTAL, so they are independent of any float behaviour and pin the coder
    bit-for-bit across languages.
    """
    cases = [
        ([16384, 16384, 16384, 16384], [0, 1, 2, 3, 3, 2, 1, 0]),  # uniform
        ([60000, 2000, 2000, 1536], [0, 0, 0, 1, 0, 2, 0]),        # skewed
        ([65533, 1, 1, 1], [0, 0, 0, 0, 1, 0, 3]),                 # near-degenerate
    ]
    out = []
    for freqs, syms in cases:
        assert sum(freqs) == TOTAL and min(freqs) >= 1
        enc = ArithmeticEncoder()
        for sym in syms:
            enc.encode(sym, freqs)
        out.append((freqs, syms, enc.finish()))
    return out


# ----------------------------------------------------------------------------
# 5. Self-test
# ----------------------------------------------------------------------------

def _random_dist(v: int, rng: random.Random, sharpness: float) -> List[float]:
    raw = [rng.random() ** sharpness for _ in range(v)]
    s = sum(raw)
    return [x / s for x in raw]


def _selftest() -> None:
    rng = random.Random(20240628)

    # (a) Quantizer invariants over many random and degenerate inputs.
    for _ in range(2000):
        v = rng.randint(1, 64)
        if rng.random() < 0.1:
            probs = [0.0] * v
            probs[rng.randrange(v)] = 1.0
        else:
            probs = _random_dist(v, rng, rng.choice([0.3, 1.0, 4.0, 12.0]))
        f = quantize(probs)
        assert len(f) == v
        assert sum(f) == TOTAL
        assert min(f) >= 1
    print("[ok] quantizer: 2000 cases, all freqs>=1 and sum==%d" % TOTAL)

    # (b) Adaptive round-trip: distribution genuinely changes every step,
    #     depending on the realized prefix (this is the FIFO property ANS lacks).
    v = 37
    def adaptive(t, prefix):
        r = random.Random(0xC0DE ^ (t * 2654435761) ^ (sum(prefix) & 0xFFFF))
        d = _random_dist(v, r, 2.0)
        if prefix:
            d[prefix[-1]] *= 5.0  # bias toward repeating last symbol
            s = sum(d)
            d = [x / s for x in d]
        return d

    for trial in range(50):
        n = rng.randint(1, 400)
        seq = [rng.randrange(v) for _ in range(n)]
        data = encode_sequence(seq, adaptive)
        back = decode_sequence(data, n, adaptive)
        assert back == seq, ("adaptive round-trip failed", trial, n)
    print("[ok] adaptive arithmetic round-trip: 50 sequences exact")

    # (c) Coding efficiency: coded size is within ~1% + small constant of the
    #     information content sum(-log2 p). Static distribution for a clean read.
    v = 256
    probs = _random_dist(v, rng, 3.0)
    freqs = quantize(probs)
    qprob = [f / TOTAL for f in freqs]
    n = 20000
    seq = [random.Random(7).choices(range(v), weights=qprob, k=1)[0] for _ in range(n)]
    enc = ArithmeticEncoder()
    ideal_bits = 0.0
    for sym in seq:
        enc.encode(sym, freqs)
        ideal_bits += -math.log2(qprob[sym])
    coded = enc.finish()
    coded_bits = len(coded) * 8
    overhead = (coded_bits - ideal_bits) / ideal_bits
    dec = ArithmeticDecoder(coded)
    assert [dec.decode(freqs) for _ in range(n)] == seq
    print("[ok] efficiency: ideal=%.0f bits, coded=%d bits, overhead=%.3f%%"
          % (ideal_bits, coded_bits, overhead * 100))
    assert overhead < 0.02, overhead

    # (d) Golden vectors are stable.
    gv = golden_vectors()
    for freqs, syms, expected in gv:
        enc = ArithmeticEncoder()
        for s in syms:
            enc.encode(s, freqs)
        assert enc.finish() == expected
        dec = ArithmeticDecoder(expected)
        assert [dec.decode(freqs) for _ in syms] == syms
    print("[ok] golden vectors: %d cases stable + reversible" % len(gv))
    print("    (Rust coder must reproduce these bytes exactly)")
    for i, (freqs, syms, expected) in enumerate(gv):
        print("    gv[%d]: syms=%s -> %s" % (i, syms, expected.hex()))

    print("\nALL REFERENCE TESTS PASSED")


if __name__ == "__main__":
    _selftest()
