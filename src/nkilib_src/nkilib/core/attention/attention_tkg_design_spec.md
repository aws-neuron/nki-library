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
│  │           NC0               │         │           NC1               │       │
│  └─────────────────────────────┘         └─────────────────────────────┘       │
│              │                                       │                         │
│              ▼                                       ▼                         │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐       │
│  │ _allocate_fa_buffers()      │         │ _allocate_fa_buffers()      │       │
│  │ - fa_running_max = -inf     │         │ - fa_running_max = -inf     │       │
│  │ - fa_running_sum = 0        │         │ - fa_running_sum = 0        │       │
│  │ - fa_running_output = 0     │         │ - fa_running_output = 0     │       │
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
│  ║  │ + _fa_update_running_max│         │ + _fa_update_running_max│     ║      │
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
│  ║  │ + _fa_update_running_sum│         │ + _fa_update_running_sum│     ║      │
│  ║  │   (NO sendrecv needed)  │         │   (NO sendrecv needed)  │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║              │                                   │                   ║      │
│  ║              ▼                                   ▼                   ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐     ║      │
│  ║  │ _compute_pv_matmul...() │         │ _compute_pv_matmul...() │     ║      │
│  ║  │ + _fa_accumulate_output │         │ + _fa_accumulate_output │     ║      │
│  ║  │   (NO sendrecv needed)  │         │   (NO sendrecv needed)  │     ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘     ║      │
│  ║                                                                      ║      │
│  ╚══════════════════════════════════════════════════════════════════════╝      │
│              │                                       │                         │
│              ▼                                       ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │                    _fa_finalize_and_store()                         │       │
│  ├─────────────────────────────────────────────────────────────────────┤       │
│  │  1. reciprocal(fa_running_sum)                                      │       │
│  │  2. fa_running_output *= sum_recip  (normalize)                     │       │
│  │  3. Store to out[bs_prg_id portion]                                 │       │
│  │                                                                     │       │
│  │  4. ════════════════ SENDRECV (if out_in_sb) ════════════════       │       │
│  │     NC0 ◄──────────────────────────────────────────────────► NC1    │       │
│  │         sendrecv(out[0:bs/2] ↔ out[bs/2:bs])                        │       │
│  │     Result: Both NCs have full output                               │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Sequence Sharding (sprior_n_prgs=2, bs_n_prgs=1)

Each NC processes different portions of s_prior. Cross-NC sendrecv for max/sum
reduction is deferred to the LAST FA tile only (optimization). Sink contributions,
when present, are also deferred and incorporated during this final gather/reduce step
rather than requiring per-tile synchronization.

This deferred synchronization optimization removes unnecessary sendrecvs during FA
loop iterations, improving performance for longer sequence lengths.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       SEQUENCE SHARDING (LNC2)                                  │
│              NC0: s_prior [0, s_prior/2)    NC1: s_prior [s_prior/2, s_prior)   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐        │
│  │           NC0               │         │           NC1               │        │
│  │   (processes K[:s_prior/2]) │         │   (processes K[s_prior/2:]) │        │
│  └─────────────────────────────┘         └─────────────────────────────┘        │
│              │                                       │                          │
│              ▼                                       ▼                          │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐        │
│  │ _allocate_fa_buffers()      │         │ _allocate_fa_buffers()      │        │
│  │ - fa_running_max = -inf     │         │ - fa_running_max = -inf     │        │
│  │ - fa_running_sum = 0        │         │ - fa_running_sum = 0        │        │
│  │ - fa_running_output = 0     │         │ - fa_running_output = 0     │        │
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
│  ║  │  1. Compute local tile max                                    │    ║      │
│  ║  │                                                               │    ║      │
│  ║  │  [sync_softmax_per_fa_tile=False (default for seq sharding)]: │    ║      │
│  ║  │  2-3. SKIP sendrecv - accumulate LOCAL max only               │    ║      │
│  ║  │       (global sync deferred to last FA tile)                  │    ║      │
│  ║  │       Sink prep also deferred to last FA tile                 │    ║      │
│  ║  │                                                               │    ║      │
│  ║  │  4. _fa_update_running_max()                                  │    ║      │
│  ║  │     - On LAST tile: calls                                     │    ║      │
│  ║  │       _fa_gather_and_compute_global_running_max()             │    ║      │
│  ║  │       which incorporates sink (if present) and does           │    ║      │
│  ║  │       the deferred sendrecv                                   │    ║      │
│  ║  └───────────────────────────────────────────────────────────────┘    ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐      ║      │
│  ║  │ _compute_exp_qk()       │         │ _compute_exp_qk()       │      ║      │
│  ║  │ exp(QK - running_max)   │         │ exp(QK - running_max)   │      ║      │
│  ║  │ (uses LOCAL max until   │         │ (uses LOCAL max until   │      ║      │
│  ║  │  last tile when global  │         │  last tile when global  │      ║      │
│  ║  │  sync happens)          │         │  sync happens)          │      ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘      ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌───────────────────────────────────────────────────────────────┐    ║      │
│  ║  │              _cascaded_sum_reduction()                        │    ║      │
│  ║  │  1. Compute local tile sum                                    │    ║      │
│  ║  │                                                               │    ║      │
│  ║  │  [sync_softmax_per_fa_tile=False (default for seq sharding)]: │    ║      │
│  ║  │  2-3. SKIP sendrecv - accumulate LOCAL sum only               │    ║      │
│  ║  │       (global sync deferred to last FA tile)                  │    ║      │
│  ║  │                                                               │    ║      │
│  ║  │  4. _fa_update_running_sum()                                  │    ║      │
│  ║  │     - On LAST tile: calls                                     │    ║      │
│  ║  │       _fa_gather_and_compute_global_running_sum()             │    ║      │
│  ║  │       which does the deferred sendrecv, then                  │    ║      │
│  ║  │       computes sink_exp and adds to global sum                │    ║      │
│  ║  │       (if sink present)                                       │    ║      │
│  ║  └───────────────────────────────────────────────────────────────┘    ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌───────────────────────────────────────────────────────────────┐    ║      │
│  ║  │              _compute_pv_matmul_and_store()                   │    ║      │
│  ║  │  1. PV_tile = exp_qk @ V_tile (local computation)             │    ║      │
│  ║  │  2. _fa_accumulate_output() - accumulate locally              │    ║      │
│  ║  │     (NO sendrecv here - accumulation is local per NC)         │    ║      │
│  ║  └───────────────────────────────────────────────────────────────┘    ║      │
│  ║                                                                       ║      │
│  ╚═══════════════════════════════════════════════════════════════════════╝      │
│              │                                       │                          │
│              ▼                                       ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │                    _fa_finalize_and_store()                         │        │
│  ├─────────────────────────────────────────────────────────────────────┤        │
│  │  1. reciprocal(fa_running_sum)                                      │        │
│  │  2. fa_running_output *= sum_recip  (normalize locally)             │        │
│  │                                                                     │        │
│  │  3. ═══════════ SENDRECV (fa_running_output) ═══════════            │        │
│  │     NC0 ◄──────────────────────────────────────────────────► NC1    │        │
│  │         Exchange normalized partial outputs                         │        │
│  │                                                                     │        │
│  │  4. NC0: output = local_output + recv_output                        │        │
│  │     (NC0 combines both halves, NC1 discards unless out_in_sb)       │        │
│  │                                                                     │        │
│  │  5. NC0 stores final output to HBM                                  │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences Summary

| Aspect | Batch Sharding | Sequence Sharding |
|--------|----------------|-------------------|
| Data split | Each NC has different batches | Each NC has different K,V portions |
| FA loop sendrecv | None | **Only on last tile** (deferred sync) |
| Max reduction | Local only | Local until last tile, then global (incorporates sink if present) |
| Sum reduction | Local only | Local until last tile, then global (incorporates sink_exp if present) |
| PV accumulation | Local only | Local only |
| Sink handling | First FA Tile | Deferred: sink loaded & incorporated during final gather on last FA tile |
| Final combine | sendrecv for out_in_sb | sendrecv + add partial outputs |
| Who stores | Both NCs (different batches) | NC0 only (combined result) |

### Function Call Graph (FA Path)

```
attention_tkg()
│
├── _compute_tile_params()          # Compute atp.use_fa, num_fa_tiles, etc.
├── _allocate_fa_buffers()          # Allocate running max/sum/output
│
├── for fa_tile_idx in range(num_fa_tiles):
│   │
│   ├── _compute_fa_tile_context()  # Get tile_s_prior, tile_offset, etc.
│   ├── sbm.open_scope()
│   ├── _allocate_qk_buffers()      # Per-tile QK buffer
│   ├── _load_mask()                # Load mask for this tile
│   │
│   ├── _compute_qk_matmul()        # Step 1: QK = Q @ K^T
│   │
│   ├── _cascaded_max_reduce()      # Step 2: Max reduction
│   │   ├── _transpose_max_psum()
│   │   ├── _prep_sink()            # [SEQ SHARD: on last FA tile only]
│   │   ├── sendrecv()              # [SEQ SHARD: deferred to last FA tile]
│   │   └── _fa_update_running_max()
│   │       └── _fa_gather_and_compute_global_running_max()  # [last tile: gather sink + remote max]
│   │
│   ├── _compute_exp_qk()           # Step 3: exp(QK - max)
│   │
│   ├── _cascaded_sum_reduction()   # Step 4: Sum reduction
│   │   ├── _tile_sum_reduction()
│   │   ├── sendrecv()              # [SEQ SHARD: deferred to last FA tile]
│   │   └── _fa_update_running_sum()
│   │       └── _fa_gather_and_compute_global_running_sum()  # [last tile: gather remote sum + sink_exp]
│   │
│   ├── _compute_pv_matmul_and_store()  # Step 5: PV matmul
│   │   └── _fa_accumulate_output()
│   │
│   └── sbm.close_scope()
│
└── _fa_finalize_and_store()        # Final normalization & store
    ├── reciprocal(fa_running_sum)
    ├── fa_running_output *= sum_recip
    ├── sendrecv()                  # Combine partial outputs
    └── dma_copy() to out           # Store to HBM (or copy to SBUF)
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
    _allocate_fa_buffers(...)                        # sized for tile_bs

    for fa_tile_idx in range(num_fa_tiles):          # existing FA loop (unchanged)
        _allocate_qk_buffers(...)
        _load_mask(...)
        _compute_qk_matmul(...)                      # Steps 1-5
        _cascaded_max_reduce(...)
        _compute_exp_qk(...)
        _cascaded_sum_reduction(...)
        _compute_pv_matmul_and_store(...)

    _fa_finalize_and_store(...)                      # per batch tile
```