# Gemma4 attention_tkg design notes

This kernel is a fork of `nkilib/core/attention/attention_tkg.py` carrying the two Gemma4 sliding-window
extensions (`AttnTKGConfig.batch_shard_sprior_max`, `AttnTKGConfig.partitioned_kv_load`). Everything else,
including the LNC2 flow, batch tiling and CP support, is documented in the core design spec
(`core/attention/attention_tkg_design_spec.md`); only the fork-specific design is described here.

## Partitioned block-K load (`partitioned_kv_load`)

### Motivation

The stock block-KV K load stages `p_max` (128) blocks per fold, one block per SBUF partition. When
the number of blocks per sequence is not a multiple of `p_max` (times the s_prior shard count),
`resize_cache_block_len_for_attention_tkg_kernel` shrinks the block length to
`gcd(block_len, blocks_per_batch / p_max)` and multiplies the block count so every fold is full.
For a sliding-window layer the prior is small and fixed, so the resize is severe:

| prior / block | blocks | sharding | resized block length | K DMAs per batch |
|---|---|---|---|---|
| 1280 / 16 | 80 | batch (`bs*q_head*s_active >= 128`) | 2 | 5 folds x 128-block gathers, 2 tokens each |
| 1280 / 16 | 80 | s_prior (LNC2, small batch) | 1 | 10 folds x 128-block gathers, 1 token each |

Each gathered row is then only `block_len * d_head` bytes, which underutilizes DMA bandwidth
(the resize helper prints a warning below block length 8).

### Two config fields

- `batch_shard_sprior_max` (default 0): replaces the `curr_sprior <= 2 * p_max` bound in
  `is_batch_sharded` so a small fixed prior batch-shards at any even batch size. All trace-time
  helpers (`is_s_prior_sharded`, `get_total_n_prgs`, `resize_cache_block_len_for_attention_tkg_kernel`,
  `build_full_attention_mask`) take the same value so the framework and the kernel agree.
- `partitioned_kv_load` (default False): load K with the caller's `block_len`. Requires block KV,
  batch sharding (`sprior_n_prgs == 1`, in practice via `batch_shard_sprior_max`), `fp8_packed=False`;
  `use_dma_transpose` is disabled on this path. Dynamic FA early exit (`max_context_len`) is not
  supported. Plain fp8 (per-tensor scale) and bf16 KV both work.

With both fields at their defaults every existing configuration takes exactly the stock code path.

### K load

Per FA tile and batch, blocks are gathered in chunks of `n_c <= p_max` caller-sized blocks:

```
k_blocks_partitioned_sb[p, c*bs + b] = active_blocks_table[b, k_blk_start + c*p_max + p]   (int32, -1 = inactive)

for chunk c:
    k_loaded[n_c_pad, block_len * d_head]  <- ONE indirect dma_copy over n_c partitions (oob_mode.skip)
    for t in range(block_len):           # position inside a block
        for i_d in range(n_d_tiles):
            tp_psum[d_tile, n_c_pad] = nc_transpose(k_loaded[:n_c_pad, t*d_head + i_d*d_tile : +d_tile])
            k_sb[i_d section][strided view]  <- tensor_copy(tp_psum[:, :n_c])
```

`n_c_pad` rounds the transpose input up to a multiple of the SBUF quadrant size (32); the padded
rows are never copied out. For the Gemma4 sliding-window geometry (80 blocks of 16, `d_head=256`,
`bs=32`) this is one K DMA per batch instead of 5 (or 10), at the cost of 32 PE transposes of
80 columns instead of 20 of 128.

### Layout equivalence

The V side, the in-kernel mask (`gen_mask_tkg`), the pre-generated HBM mask, softmax and PV keep
using the resized block length `blv = atp.block_len` with `rf = block_len / blv` sub-blocks per
caller block. The partitioned load writes each transposed column into the stock ordering:

```
token s = blk * block_len + t       ->  resized block rb = blk * rf + t // blv,  j = t % blv
k_sb column (per d-tile)            =   (rb // p_max) * blv * p_max + j * p_max + rb % p_max
```

For fixed `t` the `n_c` blocks of a chunk therefore land on an `[n_folds, p_max / rf]` grid with
stride `rf` inside a fold and stride `blv * p_max` across folds, which is a single strided
`tensor_copy` per `(t, i_d)`. Because the layout is identical, `_stitch_k_active`, the MM1 loop and
everything downstream are unchanged, and the output matches the stock kernel bit-for-bit up to the
transpose/copy rounding of the KV dtype (identical for bf16, one fp8 ulp at most for fp8 KV).

Requirements checked at trace time: `p_max % rf == 0` (holds for power-of-two block lengths) and
the FA tile and `s_prior` being multiples of `block_len`. Every chunk covers whole resized folds
(`n_c % (p_max / rf) == 0`), which follows from the resize helper's own divisibility guarantee.

### When the framework should request it

`partitioned_kv_load=True` is a literal request; the kernel does not decide whether it pays off.
The load trades DMA descriptors for PE work: for `n_blocks` caller blocks per batch it issues
`ceil(n_blocks / p_max)` gathers instead of `n_blocks * resize_factor / p_max`, i.e. it divides the
K descriptor count by `resize_factor`, but it issues `block_len` transposes of `n_blocks` columns
instead of `resized_block_len` transposes of `p_max` columns, i.e. it multiplies the transposes and
strided PSUM-to-SBUF copies by `p_max * ceil(n_blocks / p_max) / n_blocks` (1.6x for 80 blocks,
2x for 64). Measured on trn2 against the stock load with the same sharding over every tested
configuration (kernel time ratio partitioned / stock, 4 to 6 replicates per row, per-row spread
mostly < 5%):

| Regime | Examples | Ratio |
|---|---|---|
| Descriptor reduction > transpose inflation, >= 4 batches per core | 1280/16 (rf 8, x1.6) bs 8-32; 1280/8 (rf 4, x1.6) bs 4-16; 1280/64 (rf 32, x6.4) bs 4; 9216/16 (rf 2, x1.1) | 0.69 - 0.95 |
| Descriptor reduction == transpose inflation | 1024/16 (rf 2, 64 blocks, x2) bs 8-16 | 1.08 - 1.21 |
| No resize needed (rf 1) | 1024/8, 2048/16, 2048/8 | 1.00 - 1.12 |
| Winning geometry but 1 batch per core | bs 1 LNC1 / bs 2 LNC2, 1280/16, s_active 4 | 1.24 |
| Winning geometry, PE-bound | bs 32, 1280/8, d_head 128, s_active 4 | 1.00 - 1.01 |

Profiles show the win is descriptor generation: at bs 32 both arms are GpSimd-bound (SWDGE
descriptor build) and the partitioned load halves the descriptor count. At one batch per core no
engine is saturated and the extra transposes and copies sit on the critical path with nothing to
overlap. The loss is set by the geometry (a gather per fold and 128-token transposes cannot both be
had unless the block count is a multiple of `p_max`), so it is not removable inside the load; a
variant that grouped positions per gather and copied whole PSUM banks with permuted access
patterns measured slower on most rows.

Framework rule, evaluated per (prior length, block_len, batch per core) bucket at trace time and
provided as `attention_tkg_utils.partitioned_kv_load_recommended`:

- request the partitioned load only when the block resize factor
  (`block_len // resized_block_len` from `resize_cache_block_len_for_attention_tkg_kernel`) is
  strictly greater than the transpose inflation `p_max * ceil(n_blocks / p_max) / n_blocks`, AND
- there are at least 4 batches per core (`bs // lnc` under batch sharding; measured threshold),
  together with `batch_shard_sprior_max >= curr_sprior` so the layer batch-shards;
- otherwise leave both fields at their defaults.

For Gemma4-style sliding-window layers (1280-token read frame, block 16: rf 8 vs x1.6) this means
the flag is on from concurrency 8 upwards on LNC2 and off at concurrency 2 to 4. Global layers whose
prior is a multiple of `p_max * block_len` gain nothing and keep the defaults.
