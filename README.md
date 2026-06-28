# Syntropy

A **lossless** data compressor built on a *phased hybrid neural architecture*. A
neural probability predictor (trained offline on Kaggle, ≤30h GPU) drives a
streaming **arithmetic coder** implemented in **Rust**. The Rust tool replays the
inference pipeline exactly and round-trips **bit-for-bit**.

> **Deliverable 1 — Execution Plan & Architecture Summary** lives in this file.
> **Deliverable 2 — Kaggle notebook:** [`notebook/syntropy_compressor.py`](notebook/syntropy_compressor.py)
> **Deliverable 3 — Rust integration blueprint:** [`docs/rust_integration_blueprint.md`](docs/rust_integration_blueprint.md)
> Shared coder spec (runnable): [`reference/coder_reference.py`](reference/coder_reference.py)
> Verifiable Rust core: [`rust/`](rust/) — `cd rust && cargo test`

---

## 1. The five-phase pipeline

```
raw bytes
   │  Phase 1: byte-level BPE (vocab 8192, full 256-byte alphabet, no UNK)
   ▼
token IDs ──────────────────────────────────────────────────────────────┐
   │   For each next-token position t, gather predictions:               │
   │     Phase 2  LocalLSTM(state)               → logits_local           │
   │     Phase 3  Transformer(ALiBi + KV cache)  → logits_global          │
   │     classical: order-0, order-1, match model→ probs_classical        │
   │                         │                                            │
   │     Phase 4  MetaNetwork gates them in logit space → P(token_t)      │
   │                         │                                            │
   │     Phase 5  arithmetic coder encodes token_t under P → bitstream    │
   └──────────────────────────────────────────────────────────────────────┘
```

The coder encodes **BPE token IDs**, so bits-per-byte is
`total_coded_bits / total_original_bytes`.

## 2. Constraints (confirmed)

| Constraint        | Value                                                   |
| ----------------- | ------------------------------------------------------- |
| Training hardware | single Kaggle GPU (P100 16 GB or T4 16 GB)              |
| GPU time budget   | **30 hours** hard limit                                 |
| Precision         | mixed precision (AMP) required + TF32 where available   |
| Dataset           | **enwik8** (100 MB), 90/5/5 split by byte offset        |
| Disk              | ≤ 300 GB (enwik8 is trivially within budget)            |
| Final tool        | **Rust**, candle + safetensors (manual KV-cache decode) |
| Coder             | 32-bit **arithmetic** (Witten–Neal–Cleary), not ANS     |
| Export            | safetensors weights + JSON configs + tokenizer.json     |

## 3. Model sizing

| Component       | Shape                                                  | ≈ Params |
| --------------- | ------------------------------------------------------ | -------- |
| `LocalLSTM`     | 2-layer LSTM, hidden 512, vocab 8192                   | ~5 M     |
| `TransformerLM` | 6 layers, d_model 512, 8 heads, seq 1024, ALiBi memory | ~20–30 M |
| `MetaNetwork`   | small MLP over compact per-model features → gate       | ~0.1 M   |
| classical       | order-0 + order-1 + match model (online, start-empty)  | 0        |

## 4. Key decisions & deliberate deviations (with rationale)

1. **Transformer-XL → decoder-only Transformer + ALiBi + rolling KV cache.**
   Strict TXL relative-position recurrence is painful to export and to step one
   token at a time. ALiBi gives length-extrapolating "memory" with trivial
   incremental decode and a dead-simple Rust port (add a linear bias to attention
   scores). Same benefit, far cleaner — a *documented deviation* from the brief's
   "Transformer-XL style".
2. **Entropy coder = arithmetic (WNC), NOT ANS.** Adaptive autoregressive coding
   is **FIFO** (the distribution changes every token, in stream order); ANS/rANS
   is **LIFO** and would force buffering the entire distribution sequence and
   coding in reverse. Arithmetic coding is naturally forward-streaming and
   bit-reproducible. (The brief permits either coder.)
3. **Mixer = logit-space gating, not an 8192-wide MLP.** The MetaNetwork consumes
   compact per-model features (entropy, top-1 prob, match length, agreement) and
   emits K mixing weights + a temperature; final dist =
   `softmax(Σ_k w_k·logit_k / T)`. Cheap and exportable.
4. **Classical models export only hyperparameters.** Counts update identically on
   both sides, so JSON ships orders/smoothing/match config — never large tables.
5. **Lossless on arbitrary bytes.** Byte-level BPE over the full 256-byte alphabet
   (no UNK); tokenizer round-trip asserted on raw bytes before training.
6. **Determinism is the make-or-break invariant.** fp32 inference; the **same Rust
   binary does both compress and decompress** (identical context ⇒ identical
   floats); the final distribution is quantized to a 16-bit cumulative frequency
   table (freqs ≥ 1, sum = 2¹⁶) via deterministic largest-remainder before the
   coder. Cross-runtime float bit-identity is the known risk → mitigated by
   quantization + golden-vector tests; full integer inference is the documented
   hardening path.

## 5. 30-hour training-budget strategy

| Phase                          | Budget | Notes                                            |
| ------------------------------ | -----: | ------------------------------------------------ |
| Data download + BPE            |   ~1 h | BPE on 100 MB is minutes with Rust-backed HF `tokenizers`; the brief's 10 h is reallocated to training. |
| Stage 1: pretrain `LocalLSTM`  |   ~4 h | Warm start so joint training converges faster.   |
| Stage 2: joint training        |  ~21 h | LSTM + Transformer, next-token CE, AMP, cosine LR.|
| Stage 3: train MetaNetwork     |   ~1 h | Freeze neural; fit the gate on streamed features.|
| Evaluation + export            |   ~2 h | bpb vs gzip, parity checks, golden vectors.      |
| Reserve                        |   ~1 h | Absorbs Kaggle restarts / re-queues.             |

A **wall-clock budget guard** in the notebook force-checkpoints and exports
before the cutoff, so a usable model always survives the 30 h limit.

## 6. Performance reality (stated plainly, not hidden)

The brief's **≥ 1 MB/s single-thread decode is not achievable at this model
size.** A ~30M-param Transformer is ≈ 70 MFLOP/token; 1 MB/s implies
~85k–330k tok/s ⇒ ~6–21 TFLOP/s, while one CPU core delivers ~20–100 GFLOP/s
fp32 — roughly **60–1000× short**. Realistic throughput is **~1–20 KB/s**
(NNCP/cmix territory). Documented mitigations: int8 + SIMD GEMM, top-k vocab
coding with an escape, a smaller model tier, GPU inference, and threading *within*
a forward pass (decode is inherently sequential). We optimize for **compression
ratio**, and report speed honestly.

## 7. What's in this repo today

| Path                                   | Status                                              |
| -------------------------------------- | --------------------------------------------------- |
| `reference/coder_reference.py`         | ✅ runnable: WNC coder + quantizer + self-test       |
| `rust/`                                | ✅ `cargo test` green: coder/quant/classical/BPE/mixer/pipeline, std-only, offline |
| `notebook/syntropy_compressor.py`      | ✅ full Kaggle notebook (run on Kaggle GPU to train) |
| `docs/rust_integration_blueprint.md`   | ✅ integration guide + candle forward + perf math    |
| `artifacts/`                           | schema of the files training will export             |

The Rust core is bit-exact against the Python reference **today** (golden
vectors + full pipeline round-trip with a deterministic stub model). Training on
Kaggle produces the real weights/configs that drop into `model::NeuralModel`.

## 8. Quick verification

```bash
# 1. The shared coder spec (pure stdlib):
python3 reference/coder_reference.py

# 2. The Rust core, fully offline and bit-exact vs the Python golden vectors:
cd rust && cargo test
```
