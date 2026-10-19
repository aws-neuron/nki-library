# GPT-OSS C128 GIGA kernel — self-contained package

Vendored copy of the GPT-OSS C128 decode **GIGA kernel**
(`gpt_oss_decode_mega_kernel`) and its full transitive dependency
closure, re-rooted under a single importable package `c128_giga_kernel`.
Every dependency the kernel pulls from the library lives under
`c128_giga_kernel/` here, so the kernel imports resolve entirely against this
copy — no dependency on the surrounding `nkilib` source tree, and no
name collision with the real `nkilib` (which the test process also
loads).

## How this folder is maintained

`c128_giga_kernel/` was **initially generated** by `scripts/vendor_giga_kernel.py`,
which walked the import closure of
`nkilib/experimental/gpt_oss/gpt_oss_decode_layer_dp_tp_ep.py`. That entry module
does not exist on `mainline` — the GIGA kernel has no in-tree copy — so the script
is **not re-runnable here** and this folder is now maintained **by hand**. Edit the
files under `c128_giga_kernel/` directly.

That is deliberate, not drift: it is what keeps the GPT-OSS-specific tuning
(the `VLLM_NEURON_MEGA_*` flags, the banded-attention/FlashNorm/PACK64 work)
quarantined from the shared `core/` kernels that every other model uses. The
two trees have since diverged on purpose — for example live
`core/attention/attention_tkg.py` carries a *different* swapped-PV
implementation (`_use_pv_swapped_layout`) than this fork's
(`_use_pv_swapped_column_tiling`).

To port a change from the live tree into the fork (or back), a per-file 3-way
merge works well: `ruff format` all three sides first (the fork is formatted at
the repo's line-length 120, hand-written kernel code often is not), then
`git merge-file <fork-copy> <common-ancestor> <new-version>`. Normalising the
formatting first is what keeps the conflict set down to genuine semantic overlap.

## Feature flags and the packed-scale contract

The decode perf/numerics flags are **ON by default** (the measured-optimal
bundle); set the matching `VLLM_NEURON_MEGA_*` env var to `0` to disable one for
perf iteration. `mega_flag_default_on()` in
`c128_giga_kernel/core/utils/kernel_helpers.py` is the single definition of that
truthiness.

⚠️ **`VLLM_NEURON_MEGA_SCALE_PACK64` is on by default and changes the required HBM
layout of the gate/up MX scale** from `[E_L, 16, 2, H/512, I]` to
`[E_L, 2, 64, H/512, I/4]` (16 scale rows per SBUF quadrant instead of 4 — 2.66x
faster load, 4x less SBUF). Use `test/utils/mx_utils.py::pack64_gate_up_scale()` to
remap; the kernel `kernel_assert`s the layout at trace time. A caller that must keep
the unpacked layout should set `VLLM_NEURON_MEGA_SCALE_PACK64=0`.

## Use it

Importable out of the box through the normal package chain — no
`sys.path` manipulation needed:

```python
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.gpt_oss_decode_layer_dp_tp_ep import (
    gpt_oss_decode_mega_kernel,
)
```

The `c128_giga_kernel/__init__.py` here is a minimal stub — the real `nkilib`
root performs a bundled/standalone swap that would re-pull the full
`nkilib`; the stub keeps this package isolated.

## Runtime dependencies (NOT vendored)

External deps are left as ordinary runtime imports and must be available
on the environment (they are provided by the build/test toolchain):

- `nki`, `neuronxcc` — NKI compiler frontend / Neuron toolchain
- `torch`, `numpy` — tensor libraries
- stdlib: `dataclasses`, `enum`, `math`, `os`, `sys`, `typing`

## Contents (90 modules across 19 packages)

<details><summary>Vendored modules</summary>

- `c128_giga_kernel/core/attention/attention_tkg.py`
- `c128_giga_kernel/core/attention/attention_tkg_seq_packed.py`
- `c128_giga_kernel/core/attention/attention_tkg_torch.py`
- `c128_giga_kernel/core/attention/attention_tkg_utils.py`
- `c128_giga_kernel/core/attention/gen_mask_tkg.py`
- `c128_giga_kernel/core/attention/gen_mask_tkg_torch.py`
- `c128_giga_kernel/core/embeddings/rope.py`
- `c128_giga_kernel/core/embeddings/rope_torch.py`
- `c128_giga_kernel/core/mlp/mlp_parameters.py`
- `c128_giga_kernel/core/mlp/mlp_tkg/mlp_tkg_constants.py`
- `c128_giga_kernel/core/mlp/mlp_tkg/mlp_tkg_layernorm.py`
- `c128_giga_kernel/core/mlp/mlp_tkg/mlp_tkg_rmsnorm.py`
- `c128_giga_kernel/core/mlp/mlp_tkg/mlp_tkg_utils.py`
- `c128_giga_kernel/core/mlp/mlp_tkg/projection_mx_constants.py`
- `c128_giga_kernel/core/moe/moe_tkg/all_expert_impl.py`
- `c128_giga_kernel/core/moe/moe_tkg/all_expert_mx_impl.py`
- `c128_giga_kernel/core/moe/moe_tkg/all_expert_mx_utils.py`
- `c128_giga_kernel/core/moe/moe_tkg/down_projection_mx.py`
- `c128_giga_kernel/core/moe/moe_tkg/down_projection_mx_shard_H.py`
- `c128_giga_kernel/core/moe/moe_tkg/gate_up_projection_mx.py`
- `c128_giga_kernel/core/moe/moe_tkg/gate_up_projection_mx_shard_H.py`
- `c128_giga_kernel/core/moe/moe_tkg/mlp_parameters.py`
- `c128_giga_kernel/core/moe/moe_tkg/mlp_proj_mx_torch.py`
- `c128_giga_kernel/core/moe/moe_tkg/mlp_tkg_constants.py`
- `c128_giga_kernel/core/moe/moe_tkg/mlp_tkg_down_projection.py`
- `c128_giga_kernel/core/moe/moe_tkg/mlp_tkg_down_projection_lhs_rhs_swap.py`
- `c128_giga_kernel/core/moe/moe_tkg/mlp_tkg_gate_up_projection.py`
- `c128_giga_kernel/core/moe/moe_tkg/mlp_tkg_gate_up_projection_lhs_rhs_swap.py`
- `c128_giga_kernel/core/moe/moe_tkg/moe_tkg.py`
- `c128_giga_kernel/core/moe/moe_tkg/moe_tkg_affinity_masking.py`
- `c128_giga_kernel/core/moe/moe_tkg/moe_tkg_torch.py`
- `c128_giga_kernel/core/moe/moe_tkg/moe_tkg_utils.py`
- `c128_giga_kernel/core/moe/moe_tkg/projection_mx_constants.py`
- `c128_giga_kernel/core/moe/moe_tkg/projection_utils.py`
- `c128_giga_kernel/core/moe/moe_tkg/selective_expert_impl.py`
- `c128_giga_kernel/core/moe/moe_tkg/selective_expert_mx_impl.py`
- `c128_giga_kernel/core/moe_block/moe_block_tkg.py`
- `c128_giga_kernel/core/moe_block/moe_block_tkg_torch.py`
- `c128_giga_kernel/core/moe_block/moe_block_tkg_utils.py`
- `c128_giga_kernel/core/output_projection/output_projection_tkg.py`
- `c128_giga_kernel/core/output_projection/output_projection_tkg_mx_impl.py`
- `c128_giga_kernel/core/output_projection/output_projection_tkg_torch.py`
- `c128_giga_kernel/core/output_projection/output_projection_utils.py`
- `c128_giga_kernel/core/qkv/qkv.py`
- `c128_giga_kernel/core/qkv/qkv_cte.py`
- `c128_giga_kernel/core/qkv/qkv_cte_utils.py`
- `c128_giga_kernel/core/qkv/qkv_tkg.py`
- `c128_giga_kernel/core/qkv/qkv_tkg_mx_impl.py`
- `c128_giga_kernel/core/qkv/qkv_tkg_mx_utils.py`
- `c128_giga_kernel/core/qkv/qkv_tkg_torch.py`
- `c128_giga_kernel/core/quantization/constants.py`
- `c128_giga_kernel/core/quantization/fp8_quantize.py`
- `c128_giga_kernel/core/router_topk/router_topk.py`
- `c128_giga_kernel/core/router_topk/router_topk_torch.py`
- `c128_giga_kernel/core/subkernels/layernorm_tkg.py`
- `c128_giga_kernel/core/subkernels/layernorm_torch.py`
- `c128_giga_kernel/core/subkernels/norm_tkg_utils.py`
- `c128_giga_kernel/core/subkernels/norm_torch_dispatch.py`
- `c128_giga_kernel/core/subkernels/rmsnorm_mx_quantize_tkg.py`
- `c128_giga_kernel/core/subkernels/rmsnorm_tkg.py`
- `c128_giga_kernel/core/subkernels/rmsnorm_torch.py`
- `c128_giga_kernel/core/utils/allocator.py`
- `c128_giga_kernel/core/utils/common_types.py`
- `c128_giga_kernel/core/utils/cross_partition_copy.py`
- `c128_giga_kernel/core/utils/dma_names.py`
- `c128_giga_kernel/core/utils/interleave_copy.py`
- `c128_giga_kernel/core/utils/kernel_assert.py`
- `c128_giga_kernel/core/utils/kernel_helpers.py`
- `c128_giga_kernel/core/utils/lnc_sendrecv.py`
- `c128_giga_kernel/core/utils/lnc_subscriptable.py`
- `c128_giga_kernel/core/utils/logging.py`
- `c128_giga_kernel/core/utils/mx_torch_common.py`
- `c128_giga_kernel/core/utils/rotation_matrix.py`
- `c128_giga_kernel/core/utils/stream_shuffle_broadcast.py`
- `c128_giga_kernel/core/utils/tiled_range.py`
- `c128_giga_kernel/core/utils/torch_ref_wrapper.py`
- `c128_giga_kernel/core/utils/tp_broadcast.py`
- `c128_giga_kernel/core/utils/tree_logger.py`
- `c128_giga_kernel/experimental/collectives/distributed_adapter.py`
- `c128_giga_kernel/experimental/gpt_oss/attention_block_tp.py`
- `c128_giga_kernel/experimental/gpt_oss/attention_block_tp_torch.py`
- `c128_giga_kernel/experimental/gpt_oss/gpt_oss_decode_layer_dp_tp_ep.py`
- `c128_giga_kernel/experimental/gpt_oss/gpt_oss_mxfp4_decode_layer.py`
- `c128_giga_kernel/experimental/gpt_oss/moe_block_dp_ep.py`
- `c128_giga_kernel/experimental/gpt_oss/moe_block_dp_ep_torch.py`
- `c128_giga_kernel/experimental/trace_caches.py`
- `c128_giga_kernel/experimental/transformer/attention_block_tkg.py`
- `c128_giga_kernel/experimental/transformer/attention_block_tkg_sharding.py`
- `c128_giga_kernel/experimental/transformer/attention_block_tkg_torch.py`
- `c128_giga_kernel/experimental/transformer/transformer_tkg_torch.py`

</details>
