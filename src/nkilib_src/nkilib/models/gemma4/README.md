# Gemma4 kernels

Model-specific kernels for Gemma4 decode. They live here rather than in `core/` so that tuning that
only pays off for Gemma4's geometry stays out of the shared kernels every other model uses.

## `attention/attention_tkg.py` — token-gen attention with a partitioned block-K load

A fork of `nkilib/core/attention/attention_tkg.py` (same signature, same `AttnTKGConfig` field order)
with two extra config fields for Gemma4's sliding-window layers, whose 1280-token read frame at block
length 16 (80 blocks) is not a multiple of `p_max` and would otherwise be resized to 2-token blocks:

- `batch_shard_sprior_max`: batch-shard threshold override so the layer batch-shards on LNC2 at any
  even batch size instead of splitting 80 blocks over two cores.
- `partitioned_kv_load`: load K with the caller's block length as one indirect DMA per <= 128-block
  chunk, PE-transpose per block position, and scatter into the stock (resized) `k_sb` ordering so the
  mask, softmax, V side and PV path are unchanged.

With both fields at their defaults the kernel is the core kernel. `partitioned_kv_load` is a literal
request; on trn2 it is faster only when the block resize factor exceeds the transpose inflation and
there are at least 4 batches per core (Gemma4 SWA at concurrency >= 8 on LNC2: 0.72x kernel time),
and slower otherwise. `attention_tkg_utils.partitioned_kv_load_recommended` encodes the rule for the
framework; `attention/attention_tkg_design_spec.md` has the measurements.

Companion modules: `attention_tkg_utils.py` (config, sharding and resize helpers taking the override),
`attention_tkg_torch.py` and `gen_mask_tkg_torch.py` (torch references / HBM mask layout aware of the
override). The in-kernel mask generator is shared from `core/attention/gen_mask_tkg.py`.

Maintenance: port core `attention_tkg.py` fixes with a per-file 3-way merge
(`ruff format` all three sides, then `git merge-file <fork> <ancestor> <new core>`), as for
`models/gpt_oss`.

Tests: `test/integration/nkilib/models/gemma4/attention/test_attention_tkg_gemma4.py`.

```python
from nkilib_src.nkilib.models.gemma4.attention.attention_tkg import attention_tkg
from nkilib_src.nkilib.models.gemma4.attention.attention_tkg_utils import AttnTKGConfig
```
