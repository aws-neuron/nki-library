# GPT-OSS MXFP4 decode golden (standalone, pure-CPU PyTorch)

A self-contained PyTorch reference for the **GPT-OSS MXFP4 decode path**, extracted
from `vllm_neuron/model/gpt_oss/model_mxfp4.py` for use as a **kernel-development
oracle**. Everything here is plain `torch` (plus `safetensors` for the checkpoint
loader) — **no `nki`, no `vllm`, no `transformers`** are imported by the golden
itself. `transformers` is used only by the *tests* as an independent cross-check.

## What this is for

The production `forward_decode` fuses two NKI megakernels (`NF.attention_decode`,
`NF.moe_block_tkg`) with TP/DP/EP collectives. That is great for the accelerator
but hard to iterate a kernel against. This package transcribes the *same math*
into two pure-torch "kernel-oracle" functions with the **same call signatures**
the model uses, so a kernel author can feed identical inputs to their kernel and
to the oracle and diff the outputs.

The structure of `model_mxfp4.py` is **preserved** (not flattened): the same
`GptOss*` `nn.Module` hierarchy and the same numbered sections appear in
`model.py`, minus the parallelism collectives and the FP8/DCP/attention-DP
branches (single device, bf16 KV cache).

## Layout

| file | purpose |
|------|---------|
| `config.py`     | `GptOssConfig` dataclass (mirrors the repo config, no HF/vllm deps). |
| `loader.py`     | Raw HF MXFP4 checkpoint → dense **unpadded, natural-order** weights. Implements `_dequantize_mxfp4_to_bf16` (the repo's ground-truth recipe). |
| `paged_kv.py`   | `PagedKVManager` — a real block-table / slot-mapping paged KV cache (the production contract), so decode is a paging correctness oracle too. |
| `kernels.py`    | `attention_decode(...)` and `moe_block_tkg(...)` — pure-torch oracles mirroring the `NF.*` signatures. |
| `model.py`      | Structured `GptOssRMSNorm` / `GptOssRotaryEmbedding` / `GptOssAttention` / `GptOssExperts` / `GptOssMLP` / `GptOssDecoderLayer` / `GptOssModel` / `GptOssForCausalLM`. |
| `hf_reference.py` | An independent, deliberately-naive eager decode reference (contiguous cache, standard causal/SWA attention, dense MoE) used to differentially cross-check the paged golden. |
| `golden_entry.py` | **Flat whole-model entry point.** `gptoss_mxfp4_decode(...)` drives the `GptOss*` classes internally (does NOT flatten them) and returns logits; optional per-layer kernel-IO capture. |
| `layer_entry.py`  | **Flat single-layer entry point (all plain tensors).** `gptoss_mxfp4_decode_layer(config, ...one tensor per weight/cache...)` — the realistic unit for a decoder-block kernel. Builds a `GptOssDecoderLayer` internally. |
| `test_golden.py`  | pytest suite: dequant correctness, paging equivalence, flat-entry equivalence + capture, golden-vs-HF-reference logits, end-to-end sanity. |

## Whole-model kernel oracle: the flat entry point

For developing a single kernel that computes the entire decode step, use the flat
`weights + inputs + KV caches -> logits` entry in `golden_entry.py`. It is a thin,
stateless boundary around the structured golden (it constructs and drives the
`GptOss*` `nn.Module`s — nothing is flattened), and produces **bit-identical**
logits to `model.forward_decode`.

```python
from gptoss_mxfp4_golden.golden_entry import build_golden_model, gptoss_mxfp4_decode

# Build once (dequantizes MXFP4), decode many times.
model = build_golden_model(cfg, ckpt_dir="/workplace/qieqingy/gptoss-120b-hf", num_layers=4)

# Flat call — caches are per-layer [num_blocks, kv_heads, block_size, head_dim],
# written in place; block_tables/slot_mappings may be one shared tensor or per-layer.
logits = gptoss_mxfp4_decode(
    model, input_ids, positions,
    k_caches, v_caches, block_tables, slot_mappings,
)

# Per-layer kernel-IO capture — the tightest per-stage oracle for your kernel:
logits, io = gptoss_mxfp4_decode(model, ..., capture=True)
attn_in  = io["attention"][layer_idx]["inputs"]   # dict of tensors your kernel takes
attn_out = io["attention"][layer_idx]["output"]    # reference to diff against
moe_in   = io["moe"][layer_idx]["inputs"]          # KV caches snapshotted pre-write
```

You can also pass a `GptOssConfig` plus `weights=<state_dict>` (or `ckpt_dir=`)
instead of a prebuilt model to construct it inline.

# Single decoder layer — the kernel developer's guide

**This is the entry point to use for a decoder-block kernel.** A kernel
environment passes plain tensors, and one decoder block (not all 36 layers) is
the realistic unit to validate — it is the repeating transformer block your
megakernel implements. `layer_entry.py` exposes a flat function whose signature
is **one explicit tensor per weight / cache / input**; scalars come from
`GptOssConfig`. Internally it builds a `GptOssDecoderLayer`, loads your tensors,
and runs its `forward_decode` (→ the `kernels.py` oracles) — nothing is flattened.

It computes one decoder block:

```
h -> RMSNorm -> attention(GQA + sinks + optional SWA + YaRN RoPE, paged KV) -> +residual
  -> RMSNorm -> MoE(router top-k softmax + SwiGLU experts) -> +residual  ==>  out
```

## 1. Signature and tensor contract

```python
from gptoss_mxfp4_golden.layer_entry import gptoss_mxfp4_decode_layer

out = gptoss_mxfp4_decode_layer(
    config,                                            # GptOssConfig (scalars only)
    # ---- inputs -----------------------------------------------------------
    hidden_states, positions, cos, sin,
    # ---- attention weights ------------------------------------------------
    input_layernorm_weight, qkv_proj_weight, qkv_proj_bias,
    o_proj_weight, o_proj_bias, sinks,
    # ---- MoE weights (dense; see §3 for raw MXFP4) ------------------------
    post_attention_layernorm_weight, router_weight, router_bias,
    gate_up_weight, gate_up_bias, down_weight, down_bias,
    # ---- paged KV cache (written IN PLACE) --------------------------------
    k_cache, v_cache, block_table, slot_mapping,
    # ---- per-layer control (keyword) --------------------------------------
    sliding_window=128,                                # None => full-attention layer
    capture=None,                                      # optional dict, see §4
)
# returns: out [T, H]  (post-layer residual stream), and writes new K/V in place
```

Let `B` = sequences in the batch, `S` = active tokens per sequence (1 for plain
decode), `T = B*S`. With GPT-OSS-120B dims: `H = 2880`, `head_dim = 64`,
`num_q_heads = 64`, `num_kv_heads = 8`, `E = 128`, `I = 2880`,
`q_size = 64*64 = 4096`, `kv_size = 8*64 = 512`.

| tensor | shape | dtype | notes |
|--------|-------|-------|-------|
| `hidden_states` | `[T, H]` | bf16/fp32 | pre-layer residual stream |
| `positions` | `[T]` | int | absolute position of each active token |
| `cos`, `sin` | `[T, head_dim//2]` | match hidden | YaRN RoPE tables (see §2) |
| `input_layernorm_weight` | `[H]` | fp32 | pre-attention RMSNorm γ |
| `qkv_proj_weight` | `[H, q_size + 2*kv_size]` | bf16/fp32 | **fused**, row=hidden col=out; `out = X @ W` |
| `qkv_proj_bias` | `[q_size + 2*kv_size]` | | order is `[q | k | v]` |
| `o_proj_weight` | `[q_size, H]` | | `out = attn @ W_out` |
| `o_proj_bias` | `[H]` | | |
| `sinks` | `[num_q_heads]` | fp32 | per-head attention-sink logit |
| `post_attention_layernorm_weight` | `[H]` | fp32 | pre-MoE RMSNorm γ |
| `router_weight` | `[E, H]` | fp32 | `logits = normed @ W.T + b` |
| `router_bias` | `[E]` | fp32 | |
| `gate_up_weight` | `[E, H, 2, I]` | fp32 | dim 2 = `(gate, up)`; contract H → `2I` |
| `gate_up_bias` | `[E, 2, I]` | fp32 | **up already includes +1** (see §3) |
| `down_weight` | `[E, I, H]` | fp32 | contract I → H |
| `down_bias` | `[E, H]` | fp32 | |
| `k_cache`, `v_cache` | `[num_blocks, num_kv_heads, block_size, head_dim]` | match hidden | paged pool; written in place |
| `block_table` | `[B, max_blocks_per_seq]` | int32 | logical→physical block ids; `-1` for unused |
| `slot_mapping` | `[T]` | int64 | write slot for each active token = `physical_block*block_size + offset` |

`sliding_window`: pass the window size for a **sliding-attention** layer (even
layers in GPT-OSS), or `None` for a **full-attention** layer (odd layers). It is
an explicit argument so your kernel controls it per call.

## 2. Building the inputs (RoPE, positions, cache)

RoPE tables come from the model's YaRN embedding — reuse it so your `cos`/`sin`
match the golden exactly:

```python
from gptoss_mxfp4_golden.model import GptOssRotaryEmbedding
rot = GptOssRotaryEmbedding(config)
cos, sin = rot(positions, dtype=hidden_states.dtype)   # each [T, head_dim//2]
```

The KV cache follows the production **paged** contract. The easiest way to build
a valid cache + `block_table` + `slot_mapping` (and to seed prior tokens) is
`PagedKVManager`:

```python
from gptoss_mxfp4_golden.paged_kv import PagedKVManager

mgr = PagedKVManager(num_blocks=64, block_size=16, kv_heads=config.num_key_value_heads,
                     head_dim=config.head_dim, num_layers=1, dtype=hidden_states.dtype)
mgr.allocate_sequence(seq_id=0, num_tokens=prior_len)        # reserve blocks
mgr.prefill_write(0, 0, k_prior, v_prior)                    # k/v_prior: [prior_len, kv_heads, head_dim]
positions    = mgr.current_positions([0])                    # [B]  (== prior_len per seq)
block_table  = mgr.block_table([0], mgr.max_blocks_per_seq([0]))
slot_mapping = mgr.decode_slot_mapping([0])                  # advances length; call once per step
k_cache, v_cache = mgr.k_caches[0], mgr.v_caches[0]
```

`block_table[b, j]` holds the physical block for logical position range
`[j*block_size, (j+1)*block_size)`. Because attention indexes the gathered
buffer by **logical** position, a fragmented cache (scattered physical blocks)
must yield the same result as a contiguous one — proven by `test_paging_equivalence`.

## 3. Weights: dense vs. raw MXFP4

The expert weights are the only quantized ones. `gate_up_weight` / `down_weight`
here are **dense, dequantized, natural-order** tensors. To start from the raw HF
checkpoint tensors instead, dequantize them first:

```python
from gptoss_mxfp4_golden.layer_entry import dequantize_expert_weights

dense = dequantize_expert_weights(
    gate_up_blocks,  # uint8 [E, 2I, H//32, 16]
    gate_up_scales,  # uint8 [E, 2I, H//32]
    gate_up_bias_raw,# bf16  [E, 2I]   interleaved [gate0, up0, gate1, up1, ...]
    down_blocks,     # uint8 [E, H, I//32, 16]
    down_scales,     # uint8 [E, H, I//32]
    down_bias_raw,   # bf16  [E, H]
)
# dense = {gate_up_weight [E,H,2,I], gate_up_bias [E,2,I] (up +1),
#          down_weight [E,I,H], down_bias [E,H]}
out = gptoss_mxfp4_decode_layer(config, ..., **dense, ...)
```

`dequantize_expert_weights` bakes the `+1` into the up bias and de-interleaves
gate/up for you — see the footguns section below. The attention weights
(`q/k/v/o`, `sinks`) and router are already dense bf16 in the checkpoint; the
golden stores `qkv_proj_weight` **fused** as `cat([q, k, v])` transposed to
`[H, q_size+2*kv_size]`, and `o_proj_weight` transposed to `[q_size, H]`.

## 4. Diffing your kernel: whole-layer and per-stage

**Whole-layer**: run your kernel on the same tensors and compare `out` (and the
in-place `k_cache`/`v_cache` writes) against the golden's.

**Per-stage** (recommended — localizes a bug to attention vs. MoE): pass
`capture=<dict>` to record the exact inputs and output of the two internal
kernel calls. Feed the captured inputs to your kernel and diff against the
captured output — the tightest possible per-stage oracle.

```python
io = {}
out = gptoss_mxfp4_decode_layer(config, ..., capture=io)

# io["attention"][0] and io["moe"][0] each = {"inputs": {...}, "output": Tensor}
# The captured KV caches are snapshotted BEFORE the in-place write (i.e. the
# pre-write state your kernel receives).
```

Captured `io["attention"][0]["inputs"]` keys (matches `kernels.attention_decode`):
`X [B,S,H]`, `W_qkv`, `bias_qkv`, `num_q_heads`, `num_kv_heads`, `head_dim`,
`cos [T,hd/2]`, `sin`, `K_cache`, `V_cache`, `block_table`, `slot_mapping`,
`pos_ids [T]`, `sliding_window`, `sink`, `softmax_scale`, `W_out`, `bias_out`,
`update_cache`. Output `[T, H]`.

Captured `io["moe"][0]["inputs"]` keys (matches `kernels.moe_block_tkg`):
`hidden_states [T,H]`, `gamma`, `eps`, `router_weight`, `router_bias`,
`gate_up_weight`, `gate_up_bias`, `down_weight`, `down_bias`, `top_k`,
`swiglu_limit`, `swiglu_alpha`. Output `[T, H]`.

You can also re-run the golden oracle directly on captured inputs:

```python
from gptoss_mxfp4_golden import kernels
ref_moe  = kernels.moe_block_tkg(**io["moe"][0]["inputs"])
ref_attn = kernels.attention_decode(**io["attention"][0]["inputs"])
```

## 5. Precision and acceptance bar

- Run the golden in **fp32** for a clean reference, then compare your (bf16)
  kernel with a bf16-appropriate tolerance. As a scale reference, the whole-model
  golden vs. HuggingFace on real weights is ~2e-4 max in fp32 and ~0.03 mean in
  bf16 (pure rounding).
- The golden math itself is validated bit-for-bit against the structured
  `GptOssDecoderLayer` on **real 120B layer-0 weights**
  (`test_real_weights_single_layer_flat_matches_structured`), and the MoE against
  HuggingFace `transformers`.

## 6. End-to-end example

```python
import torch
from gptoss_mxfp4_golden.config import GptOssConfig
from gptoss_mxfp4_golden.model import GptOssRotaryEmbedding
from gptoss_mxfp4_golden.paged_kv import PagedKVManager
from gptoss_mxfp4_golden.layer_entry import gptoss_mxfp4_decode_layer

cfg = GptOssConfig.from_hf_json("/workplace/qieqingy/gptoss-120b-hf/config.json")
cfg.torch_dtype = torch.float32

# ... obtain your layer's weights as the tensors in the §1 table
# (e.g. dequantize_expert_weights for the experts) ...

B, prior_len = 1, 24
mgr = PagedKVManager(num_blocks=64, block_size=16, kv_heads=cfg.num_key_value_heads,
                     head_dim=cfg.head_dim, num_layers=1, dtype=cfg.torch_dtype)
mgr.allocate_sequence(0, prior_len)
mgr.prefill_write(0, 0, torch.randn(prior_len, cfg.num_key_value_heads, cfg.head_dim),
                        torch.randn(prior_len, cfg.num_key_value_heads, cfg.head_dim))
positions    = mgr.current_positions([0])
block_table  = mgr.block_table([0], mgr.max_blocks_per_seq([0]))
slot_mapping = mgr.decode_slot_mapping([0])

hidden = torch.randn(B, cfg.hidden_size)
cos, sin = GptOssRotaryEmbedding(cfg)(positions, dtype=hidden.dtype)

io = {}
out = gptoss_mxfp4_decode_layer(
    cfg, hidden, positions, cos, sin,
    input_layernorm_weight, qkv_proj_weight, qkv_proj_bias,
    o_proj_weight, o_proj_bias, sinks,
    post_attention_layernorm_weight, router_weight, router_bias,
    gate_up_weight, gate_up_bias, down_weight, down_bias,
    k_cache=mgr.k_caches[0], v_cache=mgr.v_caches[0],
    block_table=block_table, slot_mapping=slot_mapping,
    sliding_window=cfg.sliding_window,   # this is an even (sliding) layer
    capture=io,
)
# diff out (and mgr.k_caches[0]/v_caches[0]) against your kernel;
# or diff io["attention"][0] / io["moe"][0] per stage.
```

## Numerics footguns (all handled, see `loader.py`)

- **FP4 LUT** (E2M1): `[0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]`.
- **Nibble order**: low nibble (`b & 0x0F`) is FIRST (even positions), high nibble second.
- **Scale**: `uint8` biased exponent → value `* 2^(scale - 127)` via `ldexp`.
- **gate/up interleave** on the `2I` axis: `[gate0, up0, gate1, up1, ...]` (even=gate, odd=up).
- **+1 on the up bias**: production bakes `hidden_act_bias=1.0` into the up bias at
  load time and clamps `up ∈ [-limit+1, limit+1]`. HF instead clamps `up ∈ [-limit, limit]`
  then computes `(up+1)*glu`. These are algebraically identical; the golden follows HF's
  `(up+1)` form so both match exactly.

## Validation status

All 17 tests pass (`test_golden.py`). Highlights:

- **`test_moe_matches_huggingface`** — golden MoE == HuggingFace `transformers`
  `GptOssExperts`/`GptOssTopKRouter`/`GptOssRMSNorm` on synthetic MXFP4 (< 1e-3, fp32).
- **`test_paging_equivalence`** — a *fragmented* block table gives **bit-identical**
  logits to a contiguous one (paging contract proven).
- **`test_full_decode_matches_reference`** — full paged decode == an independent
  eager reference (contiguous cache, dense MoE); top-1 agrees.
- **`test_real_weights_decode_matches_huggingface`** — DEFINITIVE: the golden's
  paged decode == HuggingFace's own `GptOssModel` decode on **real 120B weights**,
  seeding the golden's paged cache with HF's own K/V. Measured (2 layers, fp32):
  **max abs 2.0e-4, mean 5.5e-6, rel 1.3e-6** — essentially bit-exact (bf16 gives
  ~0.03 mean, pure rounding).

## Environment

```bash
PY=/local/home/qieqingy/nki-venv/bin/python   # torch 2.11, safetensors, ml_dtypes
# The cross-check tests need `transformers` (HF GptOss modeling). Install once:
$PY -m pip install "transformers>=4.55"       # 5.14.1 verified
```

Real 120B MXFP4 checkpoint: `/workplace/qieqingy/gptoss-120b-hf` (36 layers,
hidden 2880, 64 Q / 8 KV heads, head_dim 64, 128 experts top-4, SWA=128 on even
layers, YaRN θ=150000 factor=32).

## Running

```bash
PY=/local/home/qieqingy/nki-venv/bin/python
$PY -m pytest gptoss_mxfp4_golden/test_golden.py -v         # fast synthetic + cross-checks
GPTOSS_CKPT=/workplace/qieqingy/gptoss-120b-hf \
  $PY -m pytest gptoss_mxfp4_golden/test_golden.py -v -k real   # real-weight tests
```
