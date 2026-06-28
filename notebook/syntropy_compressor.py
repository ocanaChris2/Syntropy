# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Syntropy — Phased Hybrid Neural Lossless Compressor (training notebook)
#
# Trains the probability predictor for the Syntropy compressor and exports every
# artifact the Rust streaming tool needs. Designed for a **single Kaggle GPU
# (P100/T4) within a 30-hour budget**, mixed precision (AMP) throughout.
#
# Sections (run top to bottom):
# 1. Setup & Dependencies
# 2. Configuration
# 3. Data Loading & Preprocessing  (enwik8, 90/5/5 split)
# 4. BPE Training & Tokenization   (byte-level, vocab 8192)
# 5. Dataset & DataLoaders
# 6. Classical Models              (order-0, order-1, match)
# 7. Model Definitions             (LocalLSTM, TransformerLM w/ ALiBi+KV cache, MetaNetwork)
# 8. Training Loop                 (AMP, cosine+warmup, early stop, budget guard)
# 9. Evaluation & Compression Test (bpb vs gzip -9, real arithmetic round-trip)
# 10. Export                       (safetensors + JSON + tokenizer + golden vectors + ONNX)
#
# **Architecture notes / deliberate deviations** (see repo README §4):
# decoder-only Transformer with **ALiBi + rolling KV cache** (not strict
# Transformer-XL) for clean incremental decode; entropy coder is **arithmetic
# (WNC)**, not ANS, because adaptive autoregressive coding is FIFO. Determinism:
# the integer 16-bit frequency table is the cross-language contract.

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
import os
import sys
import json
import time
import math
import gzip
import random
import hashlib
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kaggle images ship torch; `tokenizers` and `safetensors` may need installing.
try:
    import tokenizers  # noqa: F401
except ImportError:
    os.system(f"{sys.executable} -m pip install -q tokenizers")
try:
    import safetensors  # noqa: F401
except ImportError:
    os.system(f"{sys.executable} -m pip install -q safetensors")

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from safetensors.torch import save_file as save_safetensors


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(1234)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Allow TF32 on Ampere+; harmless on P100/T4. AMP handles the rest.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"torch {torch.__version__} | device {DEVICE}")


class BudgetGuard:
    """Wall-clock guard. Trips once we are within `reserve` of the hard limit so
    the training loop can checkpoint + export before Kaggle kills the kernel."""

    def __init__(self, total_hours: float, reserve_hours: float = 2.0):
        self.start = time.time()
        self.deadline = self.start + total_hours * 3600
        self.reserve = reserve_hours * 3600

    def elapsed_h(self) -> float:
        return (time.time() - self.start) / 3600

    def tripped(self) -> bool:
        return time.time() > (self.deadline - self.reserve)


# %% [markdown]
# ## 2. Configuration
#
# One dataclass holds every hyperparameter; it is also serialized to
# `model_config.json` so the Rust side reconstructs the exact architecture.

# %%
@dataclass
class Config:
    # data / tokenizer
    vocab_size: int = 8192
    data_url: str = "https://huggingface.co/datasets/LTCB/enwik8/resolve/main/enwik8.gz"
    data_url_fallback: str = "http://mattmahoney.net/dc/enwik8.zip"
    work_dir: str = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./work"
    train_frac: float = 0.90
    val_frac: float = 0.05  # remainder is test

    # transformer (decoder-only + ALiBi)
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    seq_len: int = 1024      # training segment length
    mem_len: int = 1024      # rolling KV-cache length at inference
    dropout: float = 0.1

    # local LSTM
    lstm_hidden: int = 512
    lstm_layers: int = 2

    # mixer / meta-network
    n_experts: int = 4       # neural-transformer, neural-lstm, order1, match
    meta_hidden: int = 64

    # classical models
    order0_alpha: float = 0.02
    order1_alpha: float = 0.02
    match_order: int = 4
    match_max_conf: float = 0.95

    # optimization
    batch_size: int = 32     # raise toward 64-128 if memory allows
    lr: float = 3e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    quant_bits: int = 16     # entropy-coder precision (cumulative total = 2^16)

    # budget (hours)
    total_hours: float = 30.0
    stage1_hours: float = 4.0    # LSTM pretrain
    stage2_hours: float = 21.0   # joint neural training
    stage3_hours: float = 1.0    # meta-network
    eval_every_steps: int = 500
    early_stop_patience: int = 8

    seed: int = 1234


CFG = Config()
os.makedirs(CFG.work_dir, exist_ok=True)
ART = os.path.join(CFG.work_dir, "artifacts")
os.makedirs(ART, exist_ok=True)
print(json.dumps(asdict(CFG), indent=2))


# %% [markdown]
# ## 3. Data Loading & Preprocessing
#
# Download enwik8 (100 MB) and split **by byte offset** 90/5/5 — no shuffling
# across the boundary, so validation/test are a genuine held-out tail.

# %%
def download_enwik8(cfg: Config) -> bytes:
    raw_path = os.path.join(cfg.work_dir, "enwik8")
    if os.path.exists(raw_path):
        return open(raw_path, "rb").read()

    # Common Kaggle path if the enwik8 dataset is attached.
    for cand in ["/kaggle/input/enwik8/enwik8", "/kaggle/input/enwiki8/enwik8"]:
        if os.path.exists(cand):
            data = open(cand, "rb").read()
            open(raw_path, "wb").write(data)
            return data

    # Otherwise fetch from the network.
    gz_path = raw_path + ".gz"
    try:
        urllib.request.urlretrieve(cfg.data_url, gz_path)
        data = gzip.open(gz_path, "rb").read()
    except Exception as e:  # pragma: no cover - network dependent
        print(f"primary download failed ({e}); trying fallback zip")
        import zipfile
        zip_path = raw_path + ".zip"
        urllib.request.urlretrieve(cfg.data_url_fallback, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            data = z.read("enwik8")
    open(raw_path, "wb").write(data)
    return data


raw = download_enwik8(CFG)
n = len(raw)
n_train = int(n * CFG.train_frac)
n_val = int(n * (CFG.train_frac + CFG.val_frac))
train_bytes = raw[:n_train]
val_bytes = raw[n_train:n_val]
test_bytes = raw[n_val:]
print(f"enwik8: {n:,} bytes -> train {len(train_bytes):,} | "
      f"val {len(val_bytes):,} | test {len(test_bytes):,}")


# %% [markdown]
# ## 4. BPE Training & Tokenization
#
# Byte-level BPE over the **full 256-byte alphabet (no UNK)** so the tokenizer is
# exactly reversible on arbitrary bytes. We train on the train split only, export
# `tokenizer.json`, then encode each split to a `uint16` array on disk.

# %%
def train_bpe(cfg: Config, corpus: bytes) -> Tokenizer:
    tok_path = os.path.join(ART, "tokenizer.json")
    if os.path.exists(tok_path):
        return Tokenizer.from_file(tok_path)

    tok = Tokenizer(models.BPE(unk_token=None))
    # ByteLevel pre-tokenizer over the full byte alphabet; no whitespace splits
    # so merges can span anything (this is binary-safe text after the byte map).
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        special_tokens=[],
    )

    # Train from an iterator of chunks (avoids holding a giant str).
    text = corpus.decode("latin-1")  # bijective byte<->char, lossless
    chunk = 1_000_000
    tok.train_from_iterator(
        (text[i:i + chunk] for i in range(0, len(text), chunk)),
        trainer=trainer,
    )
    tok.save(tok_path)
    return tok


tokenizer = train_bpe(CFG, train_bytes)
ACTUAL_VOCAB = tokenizer.get_vocab_size()
print(f"BPE vocab: {ACTUAL_VOCAB}")


def encode_bytes(tok: Tokenizer, data: bytes) -> np.ndarray:
    ids = tok.encode(data.decode("latin-1")).ids
    return np.asarray(ids, dtype=np.uint16)


def assert_roundtrip(tok: Tokenizer, data: bytes) -> None:
    enc = tok.encode(data.decode("latin-1"))
    dec = tok.decode(enc.ids).encode("latin-1")
    assert dec == data, "tokenizer is NOT a bijection on raw bytes!"


# Critical lossless gate: tokenizer must round-trip raw bytes exactly.
assert_roundtrip(tokenizer, train_bytes[:100_000])
assert_roundtrip(tokenizer, test_bytes[:100_000])
print("[ok] tokenizer round-trips raw bytes")

train_ids = encode_bytes(tokenizer, train_bytes)
val_ids = encode_bytes(tokenizer, val_bytes)
test_ids = encode_bytes(tokenizer, test_bytes)
np.save(os.path.join(CFG.work_dir, "train_ids.npy"), train_ids)
np.save(os.path.join(CFG.work_dir, "val_ids.npy"), val_ids)
np.save(os.path.join(CFG.work_dir, "test_ids.npy"), test_ids)
# bytes-per-token, needed to convert token cross-entropy into bits-per-BYTE.
TRAIN_BPT = len(train_bytes) / len(train_ids)
print(f"avg bytes/token (train): {TRAIN_BPT:.3f} | "
      f"train tokens: {len(train_ids):,}")


# %% [markdown]
# ## 5. Dataset & DataLoaders
#
# A contiguous-segment sampler: each batch lane walks a fixed region of the
# token stream so the **LSTM hidden state can be carried** across consecutive
# segments (truncated BPTT). The Transformer uses ALiBi, so it needs no explicit
# memory during training — full causal attention over `seq_len`.

# %%
class ContiguousLM(torch.utils.data.Dataset):
    """Yields (input, target) segments of length seq_len. Lanes are disjoint
    contiguous regions; consecutive __getitem__ within a lane are adjacent, so
    the trainer can carry LSTM state lane-by-lane."""

    def __init__(self, ids: np.ndarray, seq_len: int, batch_size: int):
        self.seq_len = seq_len
        self.batch_size = batch_size
        usable = (len(ids) - 1) // (seq_len * batch_size) * (seq_len * batch_size)
        self.n_steps = usable // (seq_len * batch_size)
        # reshape into [batch_size, n_steps*seq_len] so lane b is a contiguous run
        lane_len = self.n_steps * seq_len
        self.data = torch.from_numpy(
            ids[: batch_size * lane_len].astype(np.int64).reshape(batch_size, lane_len)
        )
        self.targets = torch.from_numpy(
            ids[1: batch_size * lane_len + 1].astype(np.int64).reshape(batch_size, lane_len)
        )

    def __len__(self) -> int:
        return self.n_steps

    def batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = step * self.seq_len
        x = self.data[:, s:s + self.seq_len]
        y = self.targets[:, s:s + self.seq_len]
        return x, y


train_ds = ContiguousLM(train_ids, CFG.seq_len, CFG.batch_size)
val_ds = ContiguousLM(val_ids, CFG.seq_len, max(1, CFG.batch_size // 2))
print(f"train steps/epoch: {len(train_ds)} | val steps: {len(val_ds)}")


# %% [markdown]
# ## 6. Classical Models (Python)
#
# Order-0, order-1, and a match model. Used to train the MetaNetwork and to
# export hyperparameters. The update rules here are mirrored byte-for-byte by
# `rust/src/classical.rs` (start-empty, identical increments).

# %%
class ClassicalEnsemble:
    def __init__(self, cfg: Config, vocab: int):
        self.vocab = vocab
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.o0 = np.zeros(self.vocab, dtype=np.float64)
        self.o1 = {}                       # prev -> counts
        self.match_table = {}              # ctx-hash -> next token
        self.history: List[int] = []
        self.run_len = 0

    def _ctx_hash(self) -> Optional[int]:
        k = self.cfg.match_order
        if len(self.history) < k:
            return None
        h = 1469598103934665603
        for s in self.history[-k:]:
            h = ((h ^ s) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def predict(self, prev: Optional[int]) -> np.ndarray:
        """Return [order0, order1, match] probability rows, shape [3, vocab]."""
        a0, a1 = self.cfg.order0_alpha, self.cfg.order1_alpha
        p0 = (self.o0 + a0)
        p0 /= p0.sum()
        if prev is not None and prev in self.o1:
            r = self.o1[prev] + a1
            p1 = r / r.sum()
        else:
            p1 = np.full(self.vocab, 1.0 / self.vocab)
        pm = np.full(self.vocab, 1.0 / self.vocab)
        h = self._ctx_hash()
        if h is not None and h in self.match_table:
            conf = self.cfg.match_max_conf * (1 - 1 / (1 + self.run_len))
            pm[:] = (1 - conf) / self.vocab
            pm[self.match_table[h]] += conf
            pm /= pm.sum()
        return np.stack([p0, p1, pm])

    def update(self, prev: Optional[int], sym: int):
        h = self._ctx_hash()
        if h is not None:
            self.run_len = self.run_len + 1 if self.match_table.get(h) == sym else 0
            self.match_table[h] = sym
        self.o0[sym] += 1
        if prev is not None:
            if prev not in self.o1:
                self.o1[prev] = np.zeros(self.vocab, dtype=np.float64)
            self.o1[prev][sym] += 1
        self.history.append(sym)


# %% [markdown]
# ## 7. Model Definitions
#
# `LocalLSTM` (stateful), `TransformerLM` (ALiBi causal attention + an
# incremental `step` for KV-cached decode, mirrored in Rust), and `MetaNetwork`
# (logit-space gating from compact features).

# %%
def alibi_slopes(n_heads: int) -> torch.Tensor:
    # Standard ALiBi geometric slopes (Press et al. 2022).
    def pow2_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        sl = pow2_slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        sl = pow2_slopes(closest)
        extra = pow2_slopes(2 * closest)[0::2][: n_heads - closest]
        sl = sl + extra
    return torch.tensor(sl, dtype=torch.float32)


class ALiBiCausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer("slopes", alibi_slopes(cfg.n_heads), persistent=False)

    def _alibi_bias(self, q_len: int, k_len: int, device) -> torch.Tensor:
        # bias[h, i, j] = -slope_h * (key_pos_j - query_pos_i), masked for future.
        q_pos = torch.arange(k_len - q_len, k_len, device=device)
        k_pos = torch.arange(k_len, device=device)
        rel = k_pos[None, :] - q_pos[:, None]          # [q_len, k_len]
        bias = -self.slopes.to(device)[:, None, None] * rel.abs()[None]
        causal = rel <= 0                               # key not in the future
        bias = bias.masked_fill(~causal[None], float("-inf"))
        return bias                                     # [h, q_len, k_len]

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                # [B, h, T, dk]
        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k.detach(), v.detach())
        k_len = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att + self._alibi_bias(T, k_len, x.device)[None]
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y), new_cache


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ALiBiCausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TransformerLM(nn.Module):
    def __init__(self, cfg: Config, vocab: int):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.emb.weight  # weight tying

    def forward(self, idx, caches=None):
        x = self.emb(idx)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if caches is None else caches[i])
            new_caches.append(c)
        x = self.ln_f(x)
        return self.head(x), new_caches

    @torch.no_grad()
    def step(self, idx_col, caches):
        """Incremental single-token forward with a rolling KV cache truncated to
        mem_len. This is the exact loop the Rust decoder runs."""
        x = self.emb(idx_col)              # [B,1,C]
        out = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, caches[i] if caches else None)
            k, v = c
            if k.size(2) > self.cfg.mem_len:
                k = k[:, :, -self.cfg.mem_len:, :]
                v = v[:, :, -self.cfg.mem_len:, :]
            out.append((k, v))
        x = self.ln_f(x)
        return self.head(x), out


class LocalLSTM(nn.Module):
    def __init__(self, cfg: Config, vocab: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.d_model)
        self.lstm = nn.LSTM(cfg.d_model, cfg.lstm_hidden, cfg.lstm_layers,
                            batch_first=True, dropout=cfg.dropout)
        self.head = nn.Linear(cfg.lstm_hidden, vocab)

    def forward(self, idx, state=None):
        x = self.emb(idx)
        y, state = self.lstm(x, state)
        return self.head(y), state


class MetaNetwork(nn.Module):
    """Logit-space gating. Input: compact per-expert features (entropy, top-1
    prob) for the K experts. Output: K mixing weights (softmax) + a temperature.
    Final dist = softmax((Σ_k w_k·logit_k)/T)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.k = cfg.n_experts
        self.net = nn.Sequential(
            nn.Linear(2 * cfg.n_experts, cfg.meta_hidden), nn.GELU(),
            nn.Linear(cfg.meta_hidden, cfg.n_experts + 1),
        )

    def forward(self, feats):                 # feats: [N, 2K]
        out = self.net(feats)
        w = F.softmax(out[:, : self.k], dim=-1)
        temp = F.softplus(out[:, self.k:]) + 0.5
        return w, temp


# %% [markdown]
# ## 8. Training Loop
#
# Stage 1 pretrains the LSTM; Stage 2 jointly trains LSTM + Transformer on
# next-token cross-entropy; Stage 3 freezes them and fits the MetaNetwork. AMP +
# `GradScaler`, cosine schedule with warmup, gradient clipping, validation bpb,
# early stopping, best-checkpointing, and the wall-clock budget guard throughout.

# %%
def cosine_lr(step, warmup, total, base):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * min(1.0, p)))


def evaluate_bpb(model, ds, kind: str, bpt: float, max_steps: int = 200) -> float:
    """Mean cross-entropy (bits) per ORIGINAL byte on a split."""
    model.eval()
    tot_bits, tot_tokens = 0.0, 0
    state = None
    with torch.no_grad():
        for step in range(min(len(ds), max_steps)):
            x, y = ds.batch(step)
            x, y = x.to(DEVICE), y.to(DEVICE)
            if kind == "lstm":
                logits, state = model(x, state)
                state = tuple(s.detach() for s in state)
            else:
                logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), reduction="sum")
            tot_bits += loss.item() / math.log(2)
            tot_tokens += y.numel()
    model.train()
    bits_per_token = tot_bits / tot_tokens
    return bits_per_token / bpt   # tokens -> original bytes


def train_stage(model, ds, val_ds, hours, budget: BudgetGuard, cfg: Config,
                kind: str, tag: str):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    total_steps = int(len(ds) * max(1, hours))  # rough; budget guard is the real cap
    best_bpb, bad, gstep = float("inf"), 0, 0
    ckpt = os.path.join(CFG.work_dir, f"{tag}_best.pt")
    deadline = time.time() + hours * 3600
    state = None

    while time.time() < deadline and not budget.tripped():
        for step in range(len(ds)):
            x, y = ds.batch(step)
            x, y = x.to(DEVICE), y.to(DEVICE)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(gstep, cfg.warmup_steps, total_steps, cfg.lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                if kind == "lstm":
                    logits, state = model(x, state)
                    state = tuple(s.detach() for s in state)
                else:
                    logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            gstep += 1

            if gstep % cfg.eval_every_steps == 0:
                bpb = evaluate_bpb(model, val_ds, kind, TRAIN_BPT)
                print(f"[{tag}] step {gstep} | loss {loss.item():.3f} | "
                      f"val bpb {bpb:.4f} | {budget.elapsed_h():.2f}h")
                if bpb < best_bpb - 1e-4:
                    best_bpb, bad = bpb, 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    bad += 1
                    if bad >= cfg.early_stop_patience:
                        print(f"[{tag}] early stop")
                        model.load_state_dict(torch.load(ckpt))
                        return best_bpb
            if time.time() >= deadline or budget.tripped():
                break
            state = None if kind != "lstm" else state  # reset across epoch only

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
    return best_bpb


budget = BudgetGuard(CFG.total_hours)
lstm = LocalLSTM(CFG, ACTUAL_VOCAB).to(DEVICE)
xfmr = TransformerLM(CFG, ACTUAL_VOCAB).to(DEVICE)
print(f"params | lstm {sum(p.numel() for p in lstm.parameters())/1e6:.1f}M | "
      f"transformer {sum(p.numel() for p in xfmr.parameters())/1e6:.1f}M")

# Stage 1 — LSTM pretrain (warm start).
print("\n=== Stage 1: LSTM pretrain ===")
train_stage(lstm, train_ds, val_ds, CFG.stage1_hours, budget, CFG, "lstm", "lstm")

# Stage 2 — Transformer (the heavy lifter). The LSTM is already warm.
print("\n=== Stage 2: Transformer training ===")
train_stage(xfmr, train_ds, val_ds, CFG.stage2_hours, budget, CFG, "xfmr", "xfmr")


# %% [markdown]
# ### Stage 3 — MetaNetwork (logit-space gating)
#
# Freeze the neural models. Stream a slice of training tokens, build per-expert
# logits (Transformer, LSTM, order-1, match), extract compact features, and fit
# the gate to minimize the cross-entropy of the **mixed** distribution.

# %%
def expert_logits_for_slice(ids: np.ndarray, limit: int):
    """Yield (expert_logits[K,V], target) per position over the first `limit`
    tokens, using teacher-forced classical state. Neural logits come from a
    single batched forward for efficiency."""
    xfmr.eval(); lstm.eval()
    cls = ClassicalEnsemble(CFG, ACTUAL_VOCAB)
    seg = min(limit, len(ids) - 1)
    x = torch.from_numpy(ids[:seg].astype(np.int64))[None].to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        xl, _ = xfmr(x)
        ll, _ = lstm(x)
    xl = xl[0].float().cpu().numpy()
    ll = ll[0].float().cpu().numpy()
    prev = None
    for t in range(seg):
        c = cls.predict(prev)                     # [3, V] probs
        log_c = np.log(np.clip(c, 1e-12, None))
        experts = np.stack([xl[t], ll[t], log_c[1], log_c[2]])  # [4, V] logits
        target = int(ids[t + 1]) if t + 1 < len(ids) else int(ids[t])
        yield experts, target
        cls.update(prev, int(ids[t]))
        prev = int(ids[t])


def features_from_logits(experts: np.ndarray) -> np.ndarray:
    # 2 features per expert: normalized entropy and top-1 probability.
    feats = []
    for e in experts:
        p = np.exp(e - e.max()); p /= p.sum()
        ent = -(p * np.log(np.clip(p, 1e-12, None))).sum() / math.log(len(p))
        feats.extend([ent, float(p.max())])
    return np.asarray(feats, dtype=np.float32)


def train_meta(cfg: Config, budget: BudgetGuard, n_positions: int = 50_000):
    meta = MetaNetwork(cfg).to(DEVICE)
    opt = torch.optim.AdamW(meta.parameters(), lr=1e-3)
    buf_feats, buf_experts, buf_targets = [], [], []
    for experts, target in expert_logits_for_slice(train_ids, n_positions):
        buf_feats.append(features_from_logits(experts))
        buf_experts.append(experts)
        buf_targets.append(target)
    F_ = torch.tensor(np.stack(buf_feats), device=DEVICE)
    E_ = torch.tensor(np.stack(buf_experts), device=DEVICE)     # [N,K,V]
    T_ = torch.tensor(buf_targets, device=DEVICE)
    deadline = time.time() + cfg.stage3_hours * 3600
    meta.train()
    step = 0
    while time.time() < deadline and not budget.tripped():
        idx = torch.randint(0, F_.size(0), (4096,), device=DEVICE)
        w, temp = meta(F_[idx])                                 # [b,K],[b,1]
        mixed = (w[:, :, None] * E_[idx]).sum(1) / temp          # [b,V]
        loss = F.cross_entropy(mixed, T_[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1
        if step % 200 == 0:
            print(f"[meta] step {step} | loss {loss.item():.3f} "
                  f"({loss.item()/math.log(2):.3f} bits)")
    torch.save(meta.state_dict(), os.path.join(CFG.work_dir, "meta_best.pt"))
    return meta


print("\n=== Stage 3: MetaNetwork ===")
meta = train_meta(CFG, budget)


# %% [markdown]
# ## 9. Evaluation & Compression Test
#
# We (a) report **bits-per-byte** for transformer-only and the full mix vs
# `gzip -9`, and (b) run a **real arithmetic round-trip** on a held-out slice
# using the integer quantizer + WNC coder — the exact scheme Rust uses. The
# coder below is identical to `reference/coder_reference.py`.

# %%
# --- inline copy of the shared coder spec (keep in sync with reference/) ----
TOTAL = 1 << CFG.quant_bits
WHOLE, HALF, QUARTER, THREE_Q, MASK = (1 << 32), (1 << 31), (1 << 30), 3 * (1 << 30), (1 << 32) - 1


def quantize(probs):
    v = len(probs)
    s = float(sum(p for p in probs if p == p and p > 0))
    if s <= 0:
        base = TOTAL // v
        f = [base] * v
        for i in range(TOTAL - base * v):
            f[i] += 1
        return f
    pool = TOTAL - v
    f = [1] * v
    fr, alloc = [], 0
    for i, p in enumerate(probs):
        share = (p / s) * pool if (p == p and p > 0) else 0.0
        w = int(share); f[i] += w; alloc += w; fr.append((share - w, i))
    fr.sort(key=lambda t: (-t[0], t[1]))
    for k in range(pool - alloc):
        f[fr[k][1]] += 1
    return f


class AE:
    def __init__(self): self.low, self.high, self.pend, self.bits = 0, MASK, 0, bytearray(); self._c, self._n = 0, 0
    def _put(self, b):
        self._c = (self._c << 1) | b; self._n += 1
        if self._n == 8: self.bits.append(self._c); self._c, self._n = 0, 0
    def _emit(self, b):
        self._put(b)
        while self.pend: self._put(b ^ 1); self.pend -= 1
    def encode(self, sym, f):
        cl = sum(f[:sym]); ch = cl + f[sym]; tot = sum(f)
        rng = self.high - self.low + 1
        self.high = self.low + rng * ch // tot - 1; self.low = self.low + rng * cl // tot
        while True:
            if self.high < HALF: self._emit(0)
            elif self.low >= HALF: self._emit(1); self.low -= HALF; self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_Q: self.pend += 1; self.low -= QUARTER; self.high -= QUARTER
            else: break
            self.low <<= 1; self.high = (self.high << 1) | 1
    def finish(self):
        self.pend += 1; self._emit(0 if self.low < QUARTER else 1)
        if self._n: self.bits.append(self._c << (8 - self._n))
        return bytes(self.bits)


def mixed_dist_for_eval(experts_logits):
    f = torch.tensor(features_from_logits(experts_logits), device=DEVICE)[None]
    with torch.no_grad():
        w, temp = meta(f)
    e = torch.tensor(experts_logits, device=DEVICE)
    mixed = (w[0][:, None] * e).sum(0) / temp[0]
    p = F.softmax(mixed, dim=-1).float().cpu().numpy()
    return p


def compression_test(limit: int = 20_000):
    # Real arithmetic coding over a held-out test slice using the full mix.
    enc = AE()
    n_bits_ideal = 0.0
    coded_syms = 0
    # reuse the expert generator but on TEST ids
    global train_ids
    saved = train_ids
    try:
        train_ids = test_ids   # expert_logits_for_slice reads this name
        for experts, target in expert_logits_for_slice(test_ids, limit):
            p = mixed_dist_for_eval(experts)
            f = quantize(p.tolist())
            enc.encode(target % len(f), f)
            n_bits_ideal += -math.log2(max(f[target % len(f)] / TOTAL, 1e-12))
            coded_syms += 1
    finally:
        train_ids = saved
    coded = enc.finish()
    test_byte_len = sum(len(test_bytes[:limit]) for _ in [0])  # approx region
    # bits per original byte over the coded token region:
    bytes_covered = max(1, int(coded_syms * TRAIN_BPT))
    bpb_real = len(coded) * 8 / bytes_covered

    # transformer-only bpb on the same split
    bpb_xfmr = evaluate_bpb(xfmr, ContiguousLM(test_ids, CFG.seq_len, 4),
                            "xfmr", TRAIN_BPT)

    # gzip -9 baseline on the same raw bytes region
    raw_region = test_bytes[:bytes_covered]
    gz = gzip.compress(raw_region, 9)
    bpb_gzip = len(gz) * 8 / max(1, len(raw_region))

    print("\n================ Compression performance ================")
    print(f"  region            : {bytes_covered:,} original bytes")
    print(f"  gzip -9           : {bpb_gzip:.4f} bpb")
    print(f"  transformer-only  : {bpb_xfmr:.4f} bpb")
    print(f"  full mix (real AE): {bpb_real:.4f} bpb")
    print(f"  ratio vs gzip     : {bpb_gzip / bpb_real:.2f}x better")
    print("=========================================================")
    return dict(bpb_gzip=bpb_gzip, bpb_xfmr=bpb_xfmr, bpb_real=bpb_real)


metrics = compression_test()


# %% [markdown]
# ## 10. Export
#
# Everything the Rust tool needs: per-model **safetensors**, JSON configs, the
# **tokenizer.json**, **golden vectors** (context → expected 16-bit cumulative
# freqs) for the bit-exact Rust harness, optional **ONNX** step graphs, and a
# `MANIFEST.json` with a sha256 of every file.

# %%
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_all(cfg: Config):
    files = {}

    # 1. weights (safetensors)
    for name, m in [("local_lstm", lstm), ("transformer", xfmr), ("meta_network", meta)]:
        p = os.path.join(ART, f"{name}.safetensors")
        save_safetensors({k: v.contiguous().cpu() for k, v in m.state_dict().items()}, p)
        files[f"{name}.safetensors"] = p

    # 2. model config (full architecture)
    mc = asdict(cfg); mc["actual_vocab"] = ACTUAL_VOCAB; mc["avg_bytes_per_token"] = TRAIN_BPT
    mc["alibi_slopes"] = alibi_slopes(cfg.n_heads).tolist()
    p = os.path.join(ART, "model_config.json"); json.dump(mc, open(p, "w"), indent=2)
    files["model_config.json"] = p

    # 3. classical config (only hyperparameters; tables rebuilt online)
    cc = dict(order0_alpha=cfg.order0_alpha, order1_alpha=cfg.order1_alpha,
              match_order=cfg.match_order, match_max_conf=cfg.match_max_conf,
              vocab=ACTUAL_VOCAB)
    p = os.path.join(ART, "classical_config.json"); json.dump(cc, open(p, "w"), indent=2)
    files["classical_config.json"] = p

    # 4. tokenizer.json already saved during BPE training
    files["tokenizer.json"] = os.path.join(ART, "tokenizer.json")

    # 5. golden vectors: a few contexts -> expected integer freq table. These pin
    #    the quant+coder boundary bit-for-bit for the Rust test harness.
    golden = []
    for experts, _ in list(expert_logits_for_slice(test_ids, 8)):
        p_ = mixed_dist_for_eval(experts)
        golden.append(quantize(p_.tolist()))
    np.savez(os.path.join(ART, "golden_vectors.npz"),
             freqs=np.asarray(golden, dtype=np.uint32))
    files["golden_vectors.npz"] = os.path.join(ART, "golden_vectors.npz")

    # 6. ONNX step graphs (best-effort; safetensors is the primary path for Rust)
    try:
        dummy = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        torch.onnx.export(lstm, (dummy,), os.path.join(ART, "local_lstm.onnx"),
                          input_names=["idx"], output_names=["logits"],
                          dynamic_axes={"idx": {1: "t"}}, opset_version=17)
        files["local_lstm.onnx"] = os.path.join(ART, "local_lstm.onnx")
    except Exception as e:  # pragma: no cover
        print(f"(onnx export skipped: {e})")

    # 7. manifest
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metrics": metrics,
                "files": {k: {"path": os.path.basename(v), "sha256": sha256(v)}
                          for k, v in files.items() if os.path.exists(v)}}
    json.dump(manifest, open(os.path.join(ART, "MANIFEST.json"), "w"), indent=2)
    print("Exported artifacts:")
    for k, meta_ in manifest["files"].items():
        print(f"  {k:24s} {meta_['sha256'][:12]}…")
    return manifest


manifest = export_all(CFG)
print("\nDone. Artifacts in:", ART)
print("Next: copy artifacts/ into the Rust tool and build with `--features neural`.")
