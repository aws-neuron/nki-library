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

Each NC processes different portions of s_prior. Cross-NC sendrecv needed during
FA loop (for max/sum reduction) and at end (to combine partial outputs).

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
│  ║  │  2. ═══════════ SENDRECV (qk_max_buf) ═══════════             │    ║      │
│  ║  │     NC0 ◄────────────────────────────────────────► NC1        │    ║      │
│  ║  │         Exchange local max values                             │    ║      │
│  ║  │  3. Reduce: global_max = max(local_max, recv_max)             │    ║      │
│  ║  │  4. _fa_update_running_max() with global max                  │    ║      │
│  ║  └───────────────────────────────────────────────────────────────┘    ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌─────────────────────────┐         ┌─────────────────────────┐      ║      │
│  ║  │ _compute_exp_qk()       │         │ _compute_exp_qk()       │      ║      │
│  ║  │ exp(QK - running_max)   │         │ exp(QK - running_max)   │      ║      │
│  ║  │ (uses GLOBAL max)       │         │ (uses GLOBAL max)       │      ║      │
│  ║  └─────────────────────────┘         └─────────────────────────┘      ║      │
│  ║              │                                   │                    ║      │
│  ║              ▼                                   ▼                    ║      │
│  ║  ┌───────────────────────────────────────────────────────────────┐    ║      │
│  ║  │              _cascaded_sum_reduction()                        │    ║      │
│  ║  │  1. Compute local tile sum                                    │    ║      │
│  ║  │  2. ═══════════ SENDRECV (exp_sum) ═══════════                │    ║      │
│  ║  │     NC0 ◄────────────────────────────────────────► NC1        │    ║      │
│  ║  │         Exchange local sum values                             │    ║      │
│  ║  │  3. Reduce: global_sum = local_sum + recv_sum                 │    ║      │
│  ║  │  4. _fa_update_running_sum() with global sum                  │    ║      │
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
| FA loop sendrecv | None | Yes (max, sum each tile) |
| Max reduction | Local only | Global via sendrecv |
| Sum reduction | Local only | Global via sendrecv |
| PV accumulation | Local only | Local only |
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
│   │   ├── sendrecv()              # [SEQ SHARD ONLY]
│   │   └── _fa_update_running_max()
│   │
│   ├── _compute_exp_qk()           # Step 3: exp(QK - max)
│   │
│   ├── _cascaded_sum_reduction()   # Step 4: Sum reduction
│   │   ├── _tile_sum_reduction()
│   │   ├── sendrecv()              # [SEQ SHARD ONLY]
│   │   └── _fa_update_running_sum()
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
