# Syntropy — Rust Integration Blueprint (Deliverable 3)

How the Rust streaming compressor consumes the notebook's exported artifacts and
replays the inference pipeline **exactly**, round-tripping bit-for-bit.

This is not vapor: the std-only core in [`../rust/`](../rust/) already builds and
`cargo test`s green (arithmetic coder vs the Python golden vectors, quantizer,
classical models, BPE, mixer, and the full compress→decompress pipeline driven by
a deterministic stub model). What remains for a production build is swapping the
stub for the real candle forward pass behind the `neural` feature. This document
specifies that, plus the format, the determinism contract, and an honest
performance section.

---

## 0. What training exports (the contract)

| File                       | Used by                | Notes                                              |
| -------------------------- | ---------------------- | -------------------------------------------------- |
| `tokenizer.json`           | `tokenizer.rs`         | byte-level BPE: GPT-2 byte↔unicode map + ranked merges |
| `transformer.safetensors`  | `model.rs` (candle)    | ALiBi decoder-only Transformer weights             |
| `local_lstm.safetensors`   | `model.rs` (candle)    | 2-layer LSTM weights                               |
| `meta_network.safetensors` | `mixer.rs` (candle)    | gate MLP weights                                   |
| `model_config.json`        | all                    | d_model, n_layers, n_heads, mem_len, vocab, ALiBi slopes, quant_bits |
| `classical_config.json`    | `classical.rs`         | order-0/1 smoothing, match order + max confidence  |
| `golden_vectors.npz`       | test harness           | context → expected 16-bit cumulative freq table    |
| `MANIFEST.json`            | loader                 | sha256 of every file; verify on load               |

The integer 16-bit frequency table is the **cross-language contract**. Floats may
differ across runtimes; the quantized table does not.

---

## 1. Project setup & dependencies

The core is std-only. Add the neural stack only for the real forward pass:

```toml
[dependencies]
candle-core = "0.8"          # tensors + CPU/CUDA backends
candle-nn   = "0.8"          # embedding, linear, layernorm, lstm
safetensors = "0.4"          # zero-copy weight loading
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"            # tokenizer.json / *_config.json
memmap2     = "0.9"          # mmap large weight files
anyhow      = "1"

[features]
neural = []
```

Gate the candle code with `#[cfg(feature = "neural")]`. `tract`/ONNX is a viable
secondary path but candle's manual forward gives the fine control over the KV
cache and op order that determinism wants.

Module map (already in `rust/src/`):

| Module          | Status in repo            | Role                                            |
| --------------- | ------------------------- | ----------------------------------------------- |
| `coder.rs`      | ✅ real + tested           | 32-bit WNC arithmetic encoder/decoder           |
| `quant.rs`      | ✅ real + tested           | float→integer freq table (largest-remainder)    |
| `classical.rs`  | ✅ real + tested           | order-0, order-1, match model                   |
| `tokenizer.rs`  | ✅ real + tested (in-mem)  | byte-level BPE encode/decode                     |
| `mixer.rs`      | ✅ real + tested           | logit-space gating `combine()`                  |
| `model.rs`      | ✅ trait + stub            | `NeuralModel`; real candle impl goes here        |
| `pipeline.rs`   | ✅ real + tested           | streaming compress/decompress engine             |

---

## 2. Tokenizer (exact byte-level BPE)

`tokenizer.rs` already implements rank-ordered greedy merges over a `Vec<u8>`
alphabet, with an exact-round-trip test. To consume `tokenizer.json`:

1. Parse the `model.vocab` (token-string → id) and `model.merges` (ordered pairs).
2. Reverse the **GPT-2 byte↔unicode map** to turn token strings back into raw
   bytes (the `tokenizers` ByteLevel pre-tokenizer maps each byte to a printable
   unicode code point; invert it so token ids expand to exact `Vec<u8>`).
3. Feed `(left_bytes, right_bytes) → rank` into `BpeModel::new`-style construction.

```rust
#[cfg(feature = "neural")]
pub fn load_tokenizer(path: &str) -> anyhow::Result<BpeModel> {
    let j: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    let byte_decoder = gpt2_unicode_to_byte();           // inverse of ByteLevel map
    let merges = j["model"]["merges"].as_array().unwrap().iter().map(|m| {
        let s = m.as_str().unwrap();
        let (l, r) = s.split_once(' ').unwrap();
        (decode_token(l, &byte_decoder), decode_token(r, &byte_decoder))
    }).collect();
    Ok(BpeModel::new(merges))
}
```

Parity test: encode the same fixtures in Python and Rust; assert identical id
sequences (ship the fixtures from the notebook).

---

## 3. Entropy coder (already done)

`coder.rs` is a line-for-line port of `reference/coder_reference.py` and passes
the golden-vector test (`golden_vectors_match_python_reference`). Nothing to add;
it is FIFO, streaming, and bit-exact across languages. The decoder's linear
cumulative scan can be upgraded to a Fenwick tree for the 8192-wide vocabulary if
profiling shows it matters (it is dwarfed by the neural forward — see §7).

---

## 4. Quantization (already done)

`quant.rs` mirrors the reference `quantize`: floor every symbol at 1, distribute
`TOTAL - V` proportionally, hand leftovers to the largest fractional parts (ties
→ lowest index), `TOTAL = 2^16`. Computed in **f32** with a fixed op order. The
decoder rebuilds the same table from the same logits before reading each symbol.

---

## 5. The real neural model (candle) — `model::NeuralModel`

The pipeline only needs `logits(ctx) -> Vec<f32>`. Implement it with an
incremental KV cache so per-token decode is O(mem_len), not O(n²):

```rust
#[cfg(feature = "neural")]
pub struct CandleTransformer {
    cfg: ModelConfig,
    emb: candle_nn::Embedding,
    blocks: Vec<Block>,           // each: ln1, qkv, proj, ln2, mlp
    ln_f: candle_nn::LayerNorm,
    head: Tensor,                 // tied to emb.weight
    slopes: Vec<f32>,             // ALiBi slopes from model_config.json
    cache: Vec<(Tensor, Tensor)>, // per-layer rolling K, V  (<= mem_len)
    lstm: LstmState,              // local model hidden/cell, advanced per token
}

#[cfg(feature = "neural")]
impl NeuralModel for CandleTransformer {
    fn vocab(&self) -> usize { self.cfg.actual_vocab }

    fn logits(&mut self, ctx: &[u32]) -> Vec<f32> {
        let tok = *ctx.last().unwrap_or(&0);
        // 1) embed the single new token
        let x = self.emb.forward(&Tensor::new(&[tok], &Device::Cpu)?)?; // [1, d]
        // 2) per layer: attention over (cached K,V ++ this token) with ALiBi bias
        let mut h = x;
        for (i, blk) in self.blocks.iter().enumerate() {
            let (q, k, v) = blk.qkv(&blk.ln1(&h));
            let (mut ck, mut cv) = self.cache[i].clone();
            ck = Tensor::cat(&[&ck, &k], 1)?;          // append along time
            cv = Tensor::cat(&[&cv, &v], 1)?;
            if ck.dim(1)? > self.cfg.mem_len {          // roll the window
                ck = ck.narrow(1, ck.dim(1)? - self.cfg.mem_len, self.cfg.mem_len)?;
                cv = cv.narrow(1, cv.dim(1)? - self.cfg.mem_len, self.cfg.mem_len)?;
            }
            let bias = alibi_bias_row(self.slopes[..], ck.dim(1)?);  // [-slope*dist]
            let att = softmax((q.matmul(&ck.t()?)? / scale)? + bias)?;
            h = (&h + blk.proj(&att.matmul(&cv)?)?)?;
            h = (&h + blk.mlp(&blk.ln2(&h)?)?)?;
            self.cache[i] = (ck, cv);
        }
        let h = self.ln_f.forward(&h)?;
        // 3) local LSTM advances one step and contributes its own logits
        let lstm_logits = self.lstm.step(tok);
        // (the pipeline mixes transformer + lstm + classical via the gate; here
        //  we return the transformer logits and expose lstm separately, or fuse)
        to_vec_f32(self.head.matmul(&h.t()?)?)
    }
}
```

Notes:
* Run inference in **fp32** (`DType::F32`). The whole point is reproducibility, not
  raw speed; int8 is a later, carefully-validated optimization (§7).
* The ALiBi bias for incremental decode is a single row:
  `bias[j] = -slope_h * (k_len - 1 - j)` for `j in 0..k_len`, no future to mask
  because the new token is last.
* Weight-tying: `head.weight == emb.weight` (as in the notebook).
* Load weights with `safetensors` via mmap; map names 1:1 with the PyTorch
  `state_dict` keys.

The LSTM is a `candle_nn::LSTM` advanced one token per call, its `(h, c)` state
living alongside the KV cache. The MetaNetwork is a tiny MLP in `mixer.rs` that
turns per-expert features into `(weights, temperature)`; `combine()` already does
the logit-space blend.

---

## 6. Streaming engine (already wired in `pipeline.rs`)

The loop is identical for encode and decode except for the coder call. Per token:

```
ctx, classical(start-empty), kv-cache(empty), lstm-state(zero)
loop:
    neural   = model.logits(ctx)          # transformer (+ lstm) — pure fn of state
    p0,p1,pm = classical.predict(prev)    # order-0, order-1, match
    feats    = features(neural, p0,p1,pm) # entropy + top-1 per expert
    (w, T)   = meta(feats)                # gate weights + temperature
    dist     = softmax((Σ w_k·logit_k)/T) # mixer::combine
    freqs    = quantize(dist)             # 16-bit integer table
    ── ENCODE: enc.encode(token, freqs)   #   (we know the token)
    ── DECODE: token = dec.decode(freqs)  #   (recover it)
    classical.update(prev, token); lstm/kv advance; ctx.push(token); prev = token
```

Because every input to `quantize` is a pure function of the realized prefix and
the (fixed) weights, encoder and decoder produce identical `freqs` at every
position — the round-trip is bit-exact. The container format prepends a header
(magic, sha256 of `model_config.json`, original byte length); decode reads the
length to know how many tokens to pull.

---

## 7. Performance reality (stated, not hidden)

**The brief's ≥ 1 MB/s single-thread decode is not achievable at this model
size.** The math:

* ~30M-param Transformer ≈ **~70 MFLOP per token** (forward, fp32).
* 1 MB/s with ~3–4 bytes/token ⇒ **~85k–330k tok/s** ⇒ **~6–21 TFLOP/s**.
* One modern CPU core delivers **~20–100 GFLOP/s** fp32 → we are **~60–1000×
  short**. Decode is inherently sequential (token *t* needs *t-1*), so you cannot
  thread *across* tokens.

Realistic throughput is **~1–20 KB/s**, which is the NNCP/cmix regime — the price
of state-of-the-art neural compression ratio. Honest mitigations, in order of
payoff:

1. **int8 / SIMD GEMM** for the matmuls (validate bit-exactness against the fp32
   golden vectors; integer inference also *removes* the float-determinism risk).
2. **Smaller model tier** (d_model 256, 4 layers) for a speed/ratio knob.
3. **Top-k vocab coding** with an escape symbol: code over the ~64 most likely
   tokens + escape, shrinking the per-token softmax/coder work.
4. **GPU inference** for batch/offline jobs.
5. **Thread within a forward pass** (heads, GEMM tiles) — not across tokens.

We optimize for **ratio** and report speed truthfully.

---

## 8. Determinism hardening

* Same binary does both directions → its f32 outputs always agree with
  themselves; the 16-bit table is what crosses the wire.
* Pin `DType::F32`, disable any nondeterministic kernels, fix reduction order.
* Verify `MANIFEST.json` sha256s on load so encoder and decoder provably use the
  same weights/config.
* For cross-machine guarantees, the **integer-inference** path (quantized
  weights + integer GEMM) makes the entire pipeline platform-independent; it is
  the documented advanced step, not implemented in the skeleton.

---

## 9. Test harness (what `cargo test` covers, and what to add)

Already green in the repo:
* **Coder golden vectors** — Rust reproduces the exact Python bytes (`1be440`,
  `b910`, `fff2004780`).
* **Quantizer invariants** — freqs ≥ 1, sum = 2¹⁶.
* **Classical / mixer / tokenizer** unit tests.
* **Full pipeline round-trip** — compress→decompress bit-identical on empty, single
  byte, repetitive, binary, pseudo-random, and textual inputs (stub model).

To add once real artifacts exist (all behind `--features neural`):
* **Tokenizer parity** vs Python fixtures (identical id sequences).
* **Golden-vector parity** — load `golden_vectors.npz`; assert Rust quant+coder
  reproduce the expected freq tables/bytes for the exported contexts.
* **End-to-end** — compress a held-out enwik8 chunk, decompress, assert
  `sha256(in) == sha256(out)`, and that bpb < `gzip -9` (matching the notebook).
* **Throughput bench** — report MB/s (expected KB/s; see §7).
