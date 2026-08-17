## LNC2 Flash Attention Flow Diagrams

### Batch Sharding (bs_n_prgs=2, sprior_n_prgs=1)

Each NC processes different batches. No cross-NC communication during FA loop.
Final sendrecv exchanges output between NCs (if out_in_sb) so both have full result.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         BATCH SHARDING (LNC2)                                  │
│                    NC0: batches [0, bs/2)    NC1: batches [bs/2, bs)           │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐       │
│  │             NC0             │         │             NC1             │       │
│  └─────────────────────────────┘         └─────────────────────────────┘       │
│              │                                       │                         │
│              ▼                                       ▼                         │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐       │
│  │ _allocate_online_softmax_   │         │ _allocate_online_softmax_   │       │
│  │   buffers()                 │         │   buffers()                 │       │
│  │ - running_max = -inf        │         │ - running_max = -inf        │       │
│  │ - running_sum = 0           │         │ - running_sum = 0           │       │
│  │ - running_output = 0        │         │ - running_output = 0        │       │
│  └─────────────────────────────┘         └─────────────────────────────┘       │
│              │                                       │                         │
│              ▼                                       ▼                         │
│  ╔══════════════════════════════════════════════════════════════════════╗      │
│  ║     for fa_tile_idx in range(num_fa_tiles):  [BOTH NCs IN PARALLEL]  ║      │
│  ╠══════════════════════════════════════════════════════════════════════╣      │
│  ║                                                                      ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐     ║      │
│  ║  │ _allocate_qk_buffers()  │         │ _allocate_qk_buffers()  │     ║      │
│  ║  │ _load_mask()            │         │ _load_mask()            │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║              │                                   │                   ║      │
│  ║              ▼                                   ▼                   ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐     ║      │
│  ║  │ _compute_qk_matmul()    │         │ _compute_qk_matmul()    │     ║      │
│  ║  │ QK = Q @ K_tile^T       │         │ QK = Q @ K_tile^T       │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║              │                                   │                   ║      │
│  ║              ▼                                   ▼                   ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐     ║      │
│  ║  │ _cascaded_max_reduce()  │         │ _cascaded_max_reduce()  │     ║      │
│  ║  │ + _update_running_max() │         │ + _update_running_max() │     ║      │
│  ║  │   (NO sendrecv needed)  │         │   (NO sendrecv needed)  │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║              │                                   │                   ║      │
│  ║              ▼                                   ▼                   ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐     ║      │
│  ║  │ _compute_exp_qk()       │         │ _compute_exp_qk()       │     ║      │
│  ║  │ exp(QK - running_max)   │         │ exp(QK - running_max)   │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║              │                                   │                   ║      │
│  ║              ▼                                   ▼                   ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐     ║      │
│  ║  │ _cascaded_sum_reduction │         │ _cascaded_sum_reduction │     ║      │
│  ║  │ + _update_running_sum() │         │ + _update_running_sum() │     ║      │
│  ║  │   (NO sendrecv needed)  │         │   (NO sendrecv needed)  │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║              │                                   │                   ║      │
│  ║              ▼                                   ▼                   ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐     ║      │
│  ║  │ _compute_pv_matmul...() │         │ _compute_pv_matmul...() │     ║      │
│  ║  │ + _accumulate_output()  │         │ + _accumulate_output()  │     ║      │
│  ║  │   (NO sendrecv needed)  │         │   (NO sendrecv needed)  │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║                                                                      ║      │
│  ╚══════════════════════════════════════════════════════════════════════╝      │
│              │                                       │                         │
│              ▼                                       ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │                    _finalize_and_store()                            │       │
│  ├─────────────────────────────────────────────────────────────────────┤       │
│  │ if return_cp_softmax_stats:                                         │       │
│  │     1. export running_max and running_sum to cp_softmax_stats_out   │       │
│  │     2. return out[bs_prg_id portion] in SB (unnormalized)           │       │
│  │ else:                                                               │       │
│  │     1. reciprocal(running_sum)                                      │       │
│  │     2. running_output *= sum_recip  (normalize)                     │       │
│  │     3. Store to out[bs_prg_id portion]                              │       │
│  │                                                                     │       │
│  │     4. ════════════════ SENDRECV (if out_in_sb) ════════════════    │       │
│  │        NC0 ◄──────────────────────────────────────────────► NC1     │       │
│  │             sendrecv(out[0:bs/2] ↔ out[bs/2:bs])                    │       │
│  │        Result: Both NCs have full output                            │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Sequence Sharding (sprior_n_prgs=2, bs_n_prgs=1)

Each NC processes a different portion of `s_prior`. This path runs whenever we shard on
`s_prior`, regardless of whether FA is enabled — the kernel uses the online-softmax running
buffers in both cases (captured by `atp.use_online_softmax = atp.use_fa or atp.sprior_n_prgs > 1`).

The cross-NC max/sum sync runs in `_finalize_and_store`, after the last PV matmul. No
sendrecv runs inside the FA tile loop, so GPSIMD stays available to prefetch V throughout each
tile's PV matmul; the `sendrecv`s for max, sum, and output all execute during finalize when
there is no contention on GPSIMD.

Sink contributions, when present, are loaded inside `_finalize_and_store` (into a
scope-local buffer) and folded into the global max + sum during the sync block.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       SEQUENCE SHARDING (LNC2)                                  │
│              NC0: s_prior [0, s_prior/2)    NC1: s_prior [s_prior/2, s_prior)   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐        │
│  │             NC0             │         │             NC1             │        │
│  │   (processes K[:s_prior/2]) │         │   (processes K[s_prior/2:]) │        │
│  └─────────────────────────────┘         └─────────────────────────────┘        │
│              │                                       │                          │
│              ▼                                       ▼                          │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐        │
│  │ _allocate_online_softmax_   │         │ _allocate_online_softmax_   │        │
│  │   buffers()                 │         │   buffers()                 │        │
│  │ - running_max = -inf        │         │ - running_max = -inf        │        │
│  │ - running_sum = 0           │         │ - running_sum = 0           │        │
│  │ - running_output = 0        │         │ - running_output = 0        │        │
│  └─────────────────────────────┘         └─────────────────────────────┘        │
│              │                                       │                          │
│              ▼                                       ▼                          │
│  ╔═══════════════════════════════════════════════════════════════════════╗      │
│  ║     for fa_tile_idx in range(num_fa_tiles):  [BOTH NCs IN PARALLEL]   ║      │
│  ║     Each NC loops over its OWN tiles of s_prior                       ║      │
│  ╠═══════════════════════════════════════════════════════════════════════╣      │
│  ║                                                                       ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐      ║      │
│  ║  │ _allocate_qk_buffers()  │         │ _allocate_qk_buffers()  │      ║      │
│  ║  │ _load_mask()            │         │ _load_mask()            │      ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘      ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐      ║      │
│  ║  │ _compute_qk_matmul()    │         │ _compute_qk_matmul()    │      ║      │
│  ║  │ QK = Q @ K_tile^T       │         │ QK = Q @ K_tile^T       │      ║      │
│  ║  │ (NC0's portion of K)    │         │ (NC1's portion of K)    │      ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘      ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌───────────────────────────────────────────────────────────────┐    ║      │
│  ║  │              _cascaded_max_reduce()                           │    ║      │
│  ║  │  1. Compute local tile max from QK                            │    ║      │
│  ║  │  2. No cross-NC sendrecv here (runs in _finalize_and_store)   │    ║      │
│  ║  │  3. No sink fold-in here (consumed in finalize)               │    ║      │
│  ║  │  4. _update_running_max()                                     │    ║      │
│  ║  │     - First tile: running_max = tile_max                      │    ║      │
│  ║  │     - Later tiles:                                            │    ║      │
│  ║  │       running_max = max(prev, tile_max);                      │    ║      │
│  ║  │       correction_factor = exp(prev - new) (local only)        │    ║      │
│  ║  └───────────────────────────────────────────────────────────────┘    ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐      ║      │
│  ║  │ _compute_exp_qk()       │         │ _compute_exp_qk()       │      ║      │
│  ║  │ exp(QK - running_max)   │         │ exp(QK - running_max)   │      ║      │
│  ║  │ (uses LOCAL running max │         │ (uses LOCAL running max │      ║      │
│  ║  │  — numerically safe)    │         │  — numerically safe)    │      ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘      ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌───────────────────────────────────────────────────────────────┐    ║      │
│  ║  │              _cascaded_sum_reduction()                        │    ║      │
│  ║  │  1. Compute local tile sum via (exp @ 1_vec)                  │    ║      │
│  ║  │  2. No cross-NC sendrecv here (runs in finalize)              │    ║      │
│  ║  │  3. _update_running_sum()                                     │    ║      │
│  ║  │     - First tile: running_sum = tile_sum                      │    ║      │
│  ║  │     - Later tiles:                                            │    ║      │
│  ║  │       running_sum = running_sum * c_tile + tile_sum           │    ║      │
│  ║  │  4. SKIP reciprocal (done in finalize for online softmax)     │    ║      │
│  ║  └───────────────────────────────────────────────────────────────┘    ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌───────────────────────────────────────────────────────────────┐    ║      │
│  ║  │              _compute_pv_matmul_and_store()                   │    ║      │
│  ║  │  1. PV_tile = exp_qk @ V_tile (local, into PSUM)              │    ║      │
│  ║  │  2. Copy PSUM → exp_v (SKIP per-tile recip multiply —         │    ║      │
│  ║  │     reciprocal is applied in finalize)                        │    ║      │
│  ║  │  3. _accumulate_output() — locally accumulate into            │    ║      │
│  ║  │     running_output with per-tile correction.                  │    ║      │
│  ║  │     No cross-NC sendrecv.                                     │    ║      │
│  ║  └───────────────────────────────────────────────────────────────┘    ║      │
│  ║                                                                       ║      │
│  ╚═══════════════════════════════════════════════════════════════════════╝      │
│              │                                       │                          │
│              ▼                                       ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │                _finalize_and_store()                                │        │
│  │                (cross-NC sync runs here, not in the FA loop)        │        │
│  ├─────────────────────────────────────────────────────────────────────┤        │
│  │  Cross-NC softmax sync block (sprior_n_prgs > 1):                   │        │
│  │                                                                     │        │
│  │  a. If sink present: _prep_sink() loads sink into a scope-local     │        │
│  │     sink_values buffer (only lives during finalize).                │        │
│  │  b. Save local_running_max = running_max (pre-sink, pre-remote).    │        │
│  │  c. Fold sink into running_max (local max) if present.              │        │
│  │  d. ═════════ SENDRECV (running_max) ═══════════════════════        │        │
│  │     NC0 ◄─────────────────────────────────────► NC1                 │        │
│  │     running_max = max(local_with_sink, remote) = M (global).        │        │
│  │  e. c_local = exp(local_running_max - M) → correction_factor.       │        │
│  │  f. running_sum *= c_local.                                         │        │
│  │  g. transpose_broadcast(c_local) → c_local_bc;                      │        │
│  │     running_output *= c_local_bc.                                   │        │
│  │  h. ═════════ SENDRECV (running_sum) ═══════════════════════        │        │
│  │     NC0 ◄─────────────────────────────────────► NC1                 │        │
│  │     running_sum += remote_running_sum.                              │        │
│  │  i. If sink present: sink_values = exp(sink - M);                   │        │
│  │     running_sum += sink_values.                                     │        │
│  │                                                                     │        │
│  │  Normalization and output gather:                                   │        │
│  │  1. if return_cp_softmax_stats:                                     │        │
│  │          export running_max, running_sum to cp_softmax_stats_out    │        │
│  │     else:                                                           │        │
│  │          reciprocal(running_sum)                                    │        │
│  │          running_output *= sum_recip_bc (normalize)                 │        │
│  │                                                                     │        │
│  │  2. ═════════ SENDRECV (running_output) ═══════════════════         │        │
│  │     NC0 ◄─────────────────────────────────────► NC1                 │        │
│  │         Exchange partial outputs                                    │        │
│  │                                                                     │        │
│  │  3. NC0: output = local_output + recv_output                        │        │
│  │     (NC0 combines both halves; NC1 discards unless out_in_sb)       │        │
│  │                                                                     │        │
│  │  4. NC0 stores final output to HBM                                  │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences Summary

| Aspect | Batch Sharding | Sequence Sharding |
|--------|----------------|-------------------|
| Data split | Each NC has different batches | Each NC has different K,V portions |
| FA loop sendrecv | None | None — cross-NC sync runs in `_finalize_and_store` |
| Max reduction | Local only | Local per-tile; cross-NC max folded in during finalize |
| Sum reduction | Local only | Local per-tile; cross-NC sum folded in during finalize |
| PV accumulation | Local only | Local only (no in-loop sendrecv) |
| Sink handling | First FA Tile | Loaded in `_finalize_and_store`, consumed during finalize sync |
| Final combine | sendrecv for out_in_sb | sendrecv for max, sum, and output (all in finalize) |
| Who stores | Both NCs (different batches) | NC0 only (combined result) |

### Function Call Graph (Online-Softmax Path)

Applies when `atp.use_online_softmax = atp.use_fa or atp.sprior_n_prgs > 1`.

```
attention_tkg()
│
├── _compute_tile_params()          # Compute atp.use_fa, sprior_n_prgs, use_online_softmax, ...
├── _allocate_online_softmax_buffers()          # Allocate running max/sum/output/correction_factor
│
├── for fa_tile_idx in range(num_fa_tiles):   # num_fa_tiles=1 when not use_fa
│   │
│   ├── _compute_fa_tile_context()  # Get tile_s_prior, tile_offset, etc.
│   ├── sbm.open_scope()
│   ├── _allocate_qk_buffers()      # Per-tile QK buffer
│   ├── _load_mask()                # Load mask for this tile
│   │
│   ├── _compute_qk_matmul()        # Step 1: QK = Q @ K^T
│   │
│   ├── _cascaded_max_reduce()      # Step 2: Local max reduction (no cross-NC sendrecv)
│   │   ├── _transpose_max_psum()
│   │   ├── _prep_sink()            # [single-NC sink only; sharded sink -> finalize]
│   │   └── _update_running_max()   # Per-tile local correction factor
│   │
│   ├── _compute_exp_qk()           # Step 3: exp(QK - running_max)  (local max)
│   │
│   ├── _cascaded_sum_reduction()   # Step 4: Local sum reduction (no cross-NC sendrecv)
│   │   ├── _tile_sum_reduction()
│   │   └── _update_running_sum()   # Per-tile local correction + accumulate
│   │
│   ├── _compute_pv_matmul_and_store()  # Step 5: PV matmul (skip per-tile recip)
│   │   └── _accumulate_output()  # Local accumulate into running_output
│   │
│   └── sbm.close_scope()
│
└── _finalize_and_store()        # Final sync + normalization + store
    │
    ├── if sprior_n_prgs > 1:       # Cross-NC softmax sync
    │   ├── _prep_sink()             #   sink load (scope-local buffer)
    │   ├── tensor_copy(local_running_max, running_max)
    │   ├── running_max = max(running_max, sink_values)   # fold sink locally
    │   ├── sendrecv(running_max)
    │   ├── running_max = max(local, remote) = M
    │   ├── _update_correction_factor()   # c_local = exp(L - M)
    │   ├── running_sum *= c_local
    │   ├── _s_active_bqh_tile_transpose_broadcast(c_local) -> c_local_bc
    │   ├── running_output *= c_local_bc
    │   ├── sendrecv(running_sum); running_sum += remote
    │   └── sink_values = exp(sink - M); running_sum += sink_values
    │
    ├── if return_cp_softmax_stats:
    │   └── _copy_and_export_softmax_stats(running_max, running_sum)
    ├── else:
    │   ├── reciprocal(running_sum)
    │   └── running_output *= sum_recip_bc
    │
    └── _gather_and_store_output()  # Gather outputs across NCs, store to HBM
```
## Batch tiling

### Motivation

The kernel's SBUF memory usage scales with `bs * q_head * s_active * fa_tile_s_prior`. For large batch sizes, the total SBUF allocation exceeds the hardware limit. The batch outer loop tiles the batch dimension so each tile fits within the SBUF budget.

### Memory Budget

```
8 * tile_bs * q_head * s_active * fa_tile_s_prior <= 16MB
```

Where `tile_bs` is the per-tile batch size (per NC, after LNC sharding). The factor of 8 accounts for the combined size of batch-dependent SBUF buffers (QK, QK exp and mask).

The 16MB budget is a simplified heuristic, not the actual SBM cost model. It is chosen so that when FA is active (`fa_tile_s_prior = 8K`), the effective BQS tile size is 256 (`= 16M / (8 * 8K)`), which is a clean multiple of P_MAX (128). This leaves enough SBUF headroom for K/V buffers during MM1/MM2 to achieve reasonable batch interleave degree.

Trade-offs of the budget value:
- **Too large**: QK/mask buffers consume most of SBUF, starving K/V loads during MM1/MM2. This reduces batch interleave degree and hurts performance, or may cause compilation failure if even a single batch's K/V doesn't fit.
- **Too small**: More batch tiles than necessary, increasing loop overhead.
- **Current assumption**: Supports `q_head * s_active <= 256`. Larger products would benefit from a more accurate model that queries actual SBM free space at allocation time.

Batch tiling only activates when BQS (`bs * q_head * s_active`) per NC exceeds what the budget allows.

### Loop Structure

The batch outer loop wraps around the existing flash attention loop:

```
for batch_tile_idx in range(num_batch_tiles):       # NEW: batch outer loop
    _update_atp_for_batch_tile(atp, tile_bs, TC)    # recompute batch-dependent fields
    _allocate_online_softmax_buffers(...)                        # sized for tile_bs

    for fa_tile_idx in range(num_fa_tiles):          # existing FA loop (unchanged)
        _allocate_qk_buffers(...)
        _load_mask(...)
        _compute_qk_matmul(...)                      # Steps 1-5
        _cascaded_max_reduce(...)
        _compute_exp_qk(...)
        _cascaded_sum_reduction(...)
        _compute_pv_matmul_and_store(...)

    _finalize_and_store(...)                      # per batch tile
```

### Context Parallel (CP) Support

Context Parallel (CP) attention shards the KV cache across ranks. Each rank
runs attention_tkg on its s_prior/CP shard with return_cp_softmax_stats=True to:
    1. Skip output normalization 
    2. Update the cp_softmax_stats_out dict for cross-rank softmax correction:
           "running_max": [s_active_bqh_tile, n_bsq_tiles] @ SBUF
           "running_sum": [s_active_bqh_tile, n_bsq_tiles] @ SBUF
           "max_negated": bool — when True, max is stored as -max
           "atp", "TC": tile params for broadcast in CP correction

See: attention_block_tkg_sharding_design_spec.md
