# Dynamic FA Early Exit - Design Spec

## Overview

Dynamic FA enables runtime early exit from the Flash Attention tile loop based on
`max_context_len` — the maximum actual context length across the batch. Tiles beyond
`max_context_len` are skipped, saving compute for short sequences in large KV caches.

## Motivation

In token generation, the KV cache is allocated for the maximum sequence length (e.g., 131K),
but actual context lengths vary per request. Without dynamic FA, all tiles are processed
regardless of actual content. With dynamic FA, only tiles containing valid data are computed.

## Interface

```python
attention_tkg(..., max_context_len: Optional[nl.ndarray] = None)
```

- `max_context_len`: int32 scalar tensor `[1]`. When provided, enables dynamic FA.
  Should be set to `max(pos_ids[:, 0]) + s_active` across the batch.
- When `None`: original static FA loop (all tiles, fully unrolled at compile time).

## Trip Count Computation

```
per_nc_context = max_context_len >> 1  (if interleaved_fa_tiles, else max_context_len)
num_non_last_tiles = max(ceil(per_nc_context / tile_size) - 1, 0)
```

The static last tile always executes (handles K_active append and LNC2 sync).

## Loop Structure

```
┌─────────────────────────────────────────────────────┐
│ Static FA loop (max_context_len = None)             │
│   for fa_tile_idx in range(num_fa_tiles):  # Python │
│       _execute_fa_tile_body(...)                    │
│   Compiler fully unrolls, optimizes across tiles    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Dynamic FA loop (max_context_len provided)          │
│   def body(i): _execute_fa_tile_body(...)           │
│   nl.fori_loop(0, num_non_last_tiles, body)         │
│   # Static last tile (always runs)                  │
│   _execute_fa_tile_body(..., is_last_fa_tile=True)  │
└─────────────────────────────────────────────────────┘
```

> **`nl.fori_loop` semantics.** `nl.fori_loop` follows Pallas `fori_loop`: the
> loop body is a function called once per iteration with the loop index. **Loop-carried
> dependencies are not supported yet** — a value cannot be threaded from one iteration's
> output into the next. All cross-iteration state (running max/sum/output, tile offsets)
> is therefore kept in SBUF and mutated in place, which is exactly why the identity
> initialization and SBUF offset counters described below are required.

## Key Design Decisions

### Identity-initialized FA buffers (dynamic path)

The dynamic loop cannot branch on `fa_tile_idx == 0` (runtime iteration index unknown),
and `fori_loop` carries no per-iteration state. Instead, FA running buffers are
initialized to identity values (and updated in place in SBUF each iteration):

| Buffer | Identity value | Effect on first tile |
|--------|---------------|---------------------|
| `fa_running_max` | -inf (+inf if negated) | max(-inf, tile_max) = tile_max |
| `fa_running_sum` | 0 | 0 * correction + tile_sum = tile_sum |
| `fa_running_output` | 0 | 0 * correction + tile_out = tile_out |

The static path retains the original `fa_tile_idx == 0` first-tile optimization
(direct copy, no correction math) to avoid any perf regression.

### Dynamic tile offset tracking

Two SBUF counters track the current tile position at runtime:
- `dynamic_tile_offset_sbuf`: int32 `(1,1)` — global byte offset for DMA `scalar_offset`
- `dynamic_tile_offset_f32`: float32 `(P_MAX,1)` — for mask generation iota bias

Initialization: NC0 starts at 0, NC1 starts at `fa_tile_s_prior` (interleaved).
Stride: `fa_tile_s_prior * sprior_n_prgs` per iteration (skips other NC's tiles).

### Block table DMA with dynamic offset

Block table loading uses `scalar_offset` with `dge_mode.hwdge` for runtime-offset
indirect DMA. The `scalar_offset` must be pre-multiplied by `partition_resize`
since `indirect_dim` refers to the base tensor dimension (stride 1).

### Mask generation with dynamic offset

The mask uses `nisa.activation(op=nl.copy, bias=offset_f32, scale=1.0)` to add
the runtime tile offset to the iota pattern. Bias shape `(P_MAX, 1)` broadcasts
across the free dimension.

### DMA op naming

All DMA ops inside the `nl.fori_loop` body must have unique names. Since the loop body
is traced once, `fa_tile_idx` (a compile-time constant per call site) differentiates
ops. The static last tile uses `fa_tile_idx = num_fa_tiles - 1`.

## Constraints

- `nl.fori_loop` inserts scheduling barriers at iteration boundaries (~9 µs/iter)
- No loop-carried dependencies (Pallas `fori_loop` semantics) — cross-iteration state must live in SBUF
- No cross-iteration prefetching by the compiler
- SBUF addresses must be compile-time constants (no runtime buffer indexing)
- `dge_mode.none` not allowed inside `nl.fori_loop` — must use `hwdge`
- LNC2 requires both NCs to execute the same number of iterations (`sendrecv` sync)

## Future: Double-Buffered Pipelining

The development branch (`tkg_dynamic`) implements software-pipelined double buffering
to overlap next tile's DMA with current tile's compute, reducing per-tile overhead
from ~9 µs to ~4 µs. See the development branch for details.

## LNC2: Interleaved Tile Sharding

With LNC2 and block KV (`interleaved_fa_tiles=True`), FA tiles alternate between NCs:
- NC0 processes tiles at global offsets: 0, 2T, 4T, ...
- NC1 processes tiles at global offsets: T, 3T, 5T, ...

where T = `fa_tile_s_prior` (tile size).

This ensures both NCs always have equal valid data regardless of `max_context_len`.

**Trip count:** `max_context_len >> 1` (halved, each NC handles half the context)

**Offset initialization:**
- NC0: `dynamic_tile_offset = 0`
- NC1: `dynamic_tile_offset = fa_tile_s_prior`

**Stride:** `fa_tile_s_prior * sprior_n_prgs` (= 2 * tile_size for LNC2)

**Formula:** `max_context_len = max(cache_lens) + s_active` (same for LNC1 and LNC2)
