# artifacts/

This directory holds what the **Kaggle training notebook exports** and the
**Rust tool imports**. The real binary files (`*.safetensors`, `*.onnx`,
`golden_vectors.npz`) only exist after training has run — they are intentionally
**not** committed (see `.gitignore`). What's committed is this schema.

## Expected files

| File                        | Producer                  | Consumer            | Schema |
| --------------------------- | ------------------------- | ------------------- | ------ |
| `tokenizer.json`            | notebook §4 (BPE)         | `tokenizer.rs`      | HF `tokenizers` ByteLevel BPE: `model.vocab` (token→id), ordered `model.merges`. GPT-2 byte↔unicode map makes it exactly byte-reversible. |
| `transformer.safetensors`   | notebook §10              | `model.rs` (candle) | `state_dict` of `TransformerLM` (ALiBi decoder-only). Keys match PyTorch names. |
| `local_lstm.safetensors`    | notebook §10              | `model.rs` (candle) | `state_dict` of `LocalLSTM` (2-layer). |
| `meta_network.safetensors`  | notebook §10              | `mixer.rs` (candle) | `state_dict` of the gate MLP. |
| `model_config.json`         | notebook §10              | all                 | every `Config` field + `actual_vocab`, `avg_bytes_per_token`, `alibi_slopes`. |
| `classical_config.json`     | notebook §10              | `classical.rs`      | `order0_alpha`, `order1_alpha`, `match_order`, `match_max_conf`, `vocab`. Tables rebuilt online. |
| `golden_vectors.npz`        | notebook §10              | Rust test harness   | `freqs`: `uint32 [n_contexts, vocab]` — expected 16-bit cumulative freq tables for fixed contexts. |
| `local_lstm.onnx` (etc.)    | notebook §10 (best-effort)| optional `tract`/ort| step graphs; safetensors+candle is the primary path. |
| `MANIFEST.json`             | notebook §10              | loader              | `created`, `metrics` (bpb vs gzip), and `sha256` of every file. |

## Contracts that must hold

1. **Tokenizer round-trips raw bytes** — asserted in the notebook before training.
2. **`model_config.json` fully reconstructs the architecture** — d_model,
   n_layers, n_heads, mem_len, vocab, ALiBi slopes, quant_bits.
3. **The 16-bit integer frequency table is the cross-language contract** — floats
   may differ across runtimes; the quantized table (and thus the coded bits) does
   not. `golden_vectors.npz` pins this for the Rust harness.
4. **`MANIFEST.json` sha256s** let the Rust loader prove encoder and decoder use
   identical weights/config.

See `../docs/rust_integration_blueprint.md` for how each file is loaded.
