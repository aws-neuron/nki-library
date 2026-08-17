# Attention TKG Sharding Modes Design Document

## Overview

This document describes the design and implementation of KV cache sharding strategies in the attention_block_tkg kernel, focusing on KV data parallelism.

For flash attention (tiled KV processing for long context), see [attention_tkg_design_spec.md](attention_tkg_design_spec.md).

## Definitions

    Term        Definition                                                              GPT-OSS 120B (TP64)
    ----        ----------                                                              -------------------
    B           Batch size                                                              1-512
    q_heads     Number of Q heads per rank                                              1 (TP64)
    kv_heads    Number of KV heads per rank                                             1
    s_prior     Prior context length, also called S_ctx                                 up to 128K
    s_active    Active sequence length (tokens being generated), also called S_tkg      1-8
    d_head      Attention head dimension                                                64
    n_layers    Number of transformer layers                                            80
    block_len   Number of tokens per KV cache block                                     16-256
    TP          Tensor Parallelism - shard on attention heads                           64
    KVDP        KV data parallelism - shard KV cache on batch dimension                 8
                (also called batch sharding)
    CP          Context parallelism - shard KV cache on s_prior dimension               8
                (also called flash decode)
    DP          Data parallelism - shard KV cache on batch, replicate Q/out projections 8
                (also called DP attention)
    GQA         Grouped Query Attention - multiple Q heads share one KV head            8 Q heads per KV head
    LNC2        Logical NeuronCore 2 - two physical NCs acting as one logical NC        -
    SBUF        State Buffer - on-chip SRAM for intermediate data                       -
    HBM         High Bandwidth Memory - off-chip DRAM                                   -

## Problem Statement: KV Cache Sharding

KV cache size scales linearly with context (s_prior) and batch. 
For long context inference (s_prior up to 1M tokens) or high throughput (large batch), 
the KV cache becomes too big to fit in HBM (24GB on Trn2, 36GB on Trn3).

Sharding types:
    K/V cache shape: [n_layers, batch, kv_heads, s_prior, d_head]
                                ^^^^^  ^^^^^^^^  ^^^^^^^
                               KV data  vanilla  context
                               parallel  (TP)    parallel
                                (KVDP)            (CP)

In vanilla TP, we shard on attention heads (q_heads) and each rank in a grouped query attention (GQA) group maintains a full copy of the KV cache. 
For example, TP64 on GPT-OSS 120B (64 Q heads and 8 KV heads) will have 1 Q head and 1 KV head per core. Resulting in 8X KV cache replication.
We can eliminate this replication by using TP = KV heads, so TP8 for attention (1 KV head and 8 Q heads per core). 

The remaining 8x parallelism can be used to shard the KV cache:
    TP8 KVDP8 (KV Data Parallel): Shard KV cache along batch dimension. Each rank handles B/8 batches.
    TP8 CP8 (Context Parallelism): Shard KV cache along s_prior dimension. Each rank handles s_prior/8 context length.
Both reduce the KV cache by the same amount.

Sharding Recommendations:
    Use vanilla TP = kv_heads
    B >= global_ranks/kv_heads: Use KV data parallelism
    B < global_ranks/kv_heads: Use context parallelism or combine KV data parallelism with context parallelism

For example, assuming GQA: q_heads=64, kv_heads=8, global_ranks=64
    B=1 Use TP8-CP8
    B=2 Use TP8-CP8 or TP8-KVDP2-CP4
    B=4 Use TP8-CP8 or TP8-KVDP4-CP2
    B=8 and higher: Use TP8-KVDP8

## Out of Scope: Data Parallel (DP) Attention

Both KV data parallelism and DP attention shard KV cache along the batch dimension (each rank has B/KVDP batches). The difference:
- DP Attention (TP8 DP8): 
    Replicates all projection weights (W_q, W_kv, W_out) across DP groups. 
    Each of the 8 DP groups project the TP8 sharded Q heads for B/8 batches. 
    No extra communication across DP ranks is required for the Q projection since each DP group has the full Q projection for its assigned batches.
- KV data parallelism (TP64 KVDP8): 
    Uses standard TP64 for projections (W_kv replicated due to GQA). 
    Each individual rank projects the TP64 sharded Q heads for all batches. 
    In KVDP8 the prior KV is stored batch sharded within the GQA group. 
    Since each KV head within a GQA group must attend to all Q heads within its group, the TP64 sharded Q must be gathered across KVDP ranks before attention.

While DP attention uses larger weights (8x), since the X input activation is batch sharded (8x smaller), the expected number of matrix multiplications is the same across techniques.

KV data parallelism avoids additional weight replication at the cost of two extra collectives: all_gather on Q heads before attention, and all_gather on batch after attention.
This document covers KV data parallelism.
DP attention is out of scope.

### Projection Weight and KV Cache Comparison (GPT-OSS 120B: 64 Q heads, 8 KV heads)

|                     | Vanilla TP64 | KVDP (TP64 KVDP8) | CP (TP64 CP8)        | DP Attention (TP8 DP8) |
|---------------------|--------------|-------------------|----------------------|------------------------|
| W_kv replication    | 8x (GQA)     | 8x (GQA)          | 8x (GQA)             | 8x (DP)                |
| W_q replication     | 1x           | 1x                | 1x                   | 8x (DP)                |
| W_out replication   | 1x           | 1x                | 1x                   | 8x (DP)                |
| X input activation  | [B, S, H]    | [B, S, H]         | [B, S, H]            | [B/8, S, H]            |
| KV cache per rank   | B batches    | B/8 batches       | B batches, s_prior/8 | B/8 batches            |

## Block KV Cache Support

vLLM's PagedAttention divides KV cache into fixed-size blocks that can be allocated and freed independently, 
reducing fragmentation and enabling automatic prefix caching (APC). 

Performance benefits:
- Prefill: Skip computation for cached prefixes (e.g., system prompts), improving TTFT
- Decode: Skip DMA for sequences shorter than the bucket S_ctx size via `oob_mode=skip`, improving OTPS

Block KV Layout:
    Cache shape:  [num_blocks, block_len, d_head]  # pool of blocks
    Active table: [B, num_active_blocks]           # block indices per batch (padded with -1)

Example (B=8, s_prior=131072, block_len=32):

    Contiguous: K_cache[B=8, kv_heads=1, s_prior=131072, d_head=64]
    
    Block KV:
        K_cache[num_blocks=32768, block_len=32, d_head=64]  # pool of blocks
        active_blocks_table[B=8, num_active_blocks=4096]:   # 131072/32 = 4096 blocks per batch
            batch 0: [0, 8, 16, ..., -1, -1]
            batch 1: [1, 9, 17, ..., -1, -1]
            ...

    The active_blocks_table contains block indices into K_cache's first dimension.
    E.g., the second block of batch 1 is at K_cache[9, :, :].

Loading Block KV:
The attention kernel uses `dma_copy` with `vector_offset` (indirect DMA) to load
blocks using indices from `active_blocks_table`. This loads up to 128 blocks at
once into SBUF (one block per partition). Blocks with invalid indices (-1) are
skipped via `oob_mode=skip`.
The K cache is then transposed in SBUF:
    [128blks, block_len * d_head] -> [d_head, block_len * 128blks]
So that d_head on the partition dimension for the Q @ K^T matmul.

### Block KV with KV data parallelism

With KV data parallelism, each KVDP rank computes attention for B/KVDP batches. 
vLLM treats each DP rank as an independent inference endpoint with its own KV cache:
- block pool: K_cache[num_blocks_local, block_len, d_head]
- Its own block table: `active_blocks_table[B/KVDP, num_active_blocks]` with indices local to that rank's cache

Continuing the example above with block KV (B=8, s_prior=131072, block_len=32, KVDP=4):

    Rank 0: K_cache_0[8192, 32, 64], active_blocks_table_0[2, 4096]  -> batches 0,1
    Rank 1: K_cache_1[8192, 32, 64], active_blocks_table_1[2, 4096]  -> batches 2,3
    Rank 2: K_cache_2[8192, 32, 64], active_blocks_table_2[2, 4096]  -> batches 4,5
    Rank 3: K_cache_3[8192, 32, 64], active_blocks_table_3[2, 4096]  -> batches 6,7

## KV data parallelism Implementation

### Purpose

KV data parallelism reduces KV cache memory by distributing cache along the batch dimension across KVDP ranks. 
Each rank stores B/KVDP batches of KV cache instead of B.

    K/V cache shape: [n_layers, batch/KVDP, kv_heads=1, s_prior, d_head]
                                ^^^^^^^^^^^
                                KV data parallelism

### Data Flow

The attention_block_tkg kernel with KV data parallelism wraps the standard attention flow with collective operations:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        ATTENTION BLOCK TKG WITH KV DATA PARALLELISM                            │
│                                                                                                │
│  Per-rank input: X [B, S_tkg, H]     KV cache: [B/KVDP, S_ctx, d] (pre-sharded)                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│  │  RMSNorm X      │-->│ QKV Projection  │-->│ Split QKV->Q,K  │-->│ RMSNorm Q/K     │-->      │
│  │  (optional)     │   │                 │   │  (transpose)    │   │ (optional)      │         │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘         │
│                                                                                                │
│      ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                           │
│   -->│ RoPE Embedding  │-->│ RMSNorm Q/K     │-->│ Quantize K/V    │                           │
│      │ (optional)      │   │ (optional)      │   │ to FP8 (opt.)   │                           │
│      └─────────────────┘   └─────────────────┘   └────────┬────────┘                           │
│                                                           v                                    │
│  ╔════════════════════════════════════════════════════════════════════════════════════════╗    │
│  ║                       KV DATA PARALLELISM INPUT COLLECTIVES                            ║    │
│  ║  ┌─────────────────────────────────────────────────────────────────────────────────┐   ║    │
│  ║  │ 1. Rearrange Q: (q_heads, B, S, d) -> (KVDP*q_heads, B/KVDP, S, d)              │   ║    │
│  ║  │ 2. all_to_all Q dim=0: (KVDP*q_heads, B/KVDP, S, d) — all heads, local batch    │   ║    │
│  ║  │ 3. Transpose back to SBUF: (d, B_attn*q_heads_attn*S)                           │   ║    │
│  ║  │ K: Slice batch to SBUF: (d, B/KVDP*S)                                           │   ║    │
│  ║  │ V: Slice batch in HBM: (B/KVDP, 1, S, d)                                        │   ║    │
│  ║  └─────────────────────────────────────────────────────────────────────────────────┘   ║    │
│  ╚════════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                        │
│                                                       v                                        │
│                   ┌──────────────────────────────────────────────────────────┐                 │
│                   │                    ATTENTION TKG                         │                 │
│                   │  Q: (d, B_attn*q_heads_attn*S) @ SBUF                    │                 │
│                   │  K: (d, B_attn*S) @ SBUF    V: (B_attn, 1, S, d) @ HBM   │                 │
│                   │  softmax(Q @ K^T / sqrt(d)) @ V                          │                 │
│                   │  Output: (d, B_attn*q_heads_attn*S) @ SBUF               │                 │
│                   └───────────────────────────────────┬──────────────────────┘                 │
│                                                       │                                        │
│                                                       v                                        │
│  ╔════════════════════════════════════════════════════════════════════════════════════════╗    │
│  ║                       KV DATA PARALLELISM OUTPUT COLLECTIVES                           ║    │
│  ║  ┌───────────────────────────────────────────────────────────────────────────────────┐ ║    │
│  ║  │ 1. Rearrange attn: (B/KVDP, KVDP*q_heads, d, S) -> (KVDP*q_heads, B/KVDP, d, S)   │ ║    │
│  ║  │ 2. all_to_all attn dim=0: (KVDP*q_heads, B/KVDP, d, S) — local heads, all batches │ ║    │
│  ║  │ 3. Rearrange back: (KVDP*q_heads, B/KVDP, d, S) -> (B, q_heads, d, S)             │ ║    │
│  ║  └───────────────────────────────────────────────────────────────────────────────────┘ ║    │
│  ╚════════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                        │
│                                                       v                                        │
│                   ┌───────────────────────────────────────────────────────────┐                │
│                   │              KV Cache Update (optional)                   │                │
│                   │              Updates B/KVDP batches only                  │                │
│                   └───────────────────────────────────┬───────────────────────┘                │
│                                                       │                                        │
│                                                       v                                        │
│                   ┌───────────────────────────────────────────────────────────┐                │
│                   │              Output Projection (optional)                 │                │
│                   │              W_out @ attn -> [B, S_tkg, H]                │                │
│                   └───────────────────────────────────────────────────────────┘                │
│                                                                                                │
│  Per-rank output: [B, S_tkg, H]                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter              | Type                | Description                                                       |
|------------------------|---------------------|-------------------------------------------------------------------|
| `KVDP`                 | int                 | KV data parallelism degree (1 = disabled)                         |
| `KVDP_replica_group`   | ReplicaGroup        | Rank group for collectives                                        |
| `KVDP_collective_mode` | KVDPCollectiveMode  | Collective mode: `ALL_TO_ALL` (default) or `ALL_GATHER_SLICE`     |
| `KVDP_rank`            | nl.ndarray          | Shape (1,), uint32 @ HBM. This rank's position in its KVDP group  |

### Collective Modes

The `KVDP_collective_mode` parameter selects the collective operation strategy for Q input and attention output redistribution:

- **ALL_TO_ALL (default):** Single `all_to_all` collective that combines gather and slice in one step. Requires Mesh algorithm (≥4 ranks lnc2 or ≥8 ranks lnc1).
- **ALL_GATHER_SLICE:** `all_gather` on heads/batch followed by a `KVDP_rank`-based slice. Works with any rank count (≥2).

Note: K/V batch slicing still uses `KVDP_rank` in both modes (local slice, not a collective).

### Collective Operations: ALL_TO_ALL

The implementation uses a single `all_to_all` collective that combines gather and slice in one step.
Rearranges are required to put KVDP batch/head groups on dim=0 for all_to_all chunking.

**Input Collectives (before attention):**

    SBUF Input               Rearrange + Transpose to HBM                  all_to_all                       Transpose + Rearrange to SBUF
    ──────────               ─────────────────────────────                  ──────────                       ─────────────────────────────
    Q (q==1): (d, B*S)       rearrange in HBM (KVDP, d, B/KVDP*S)          (KVDP, d, B/KVDP*S)              rearrange to (d, B*S) @ SBUF
    Q (q>1):  (d, B*q*S)     rearrange SBUF (d, KVDP*q, B/KVDP*S)          (KVDP*q, B/KVDP, S, d)           (d, B*q*S) @ SBUF
                             tiled transpose to HBM (KVDP*q, B/KVDP, S, d)                                  tiled transpose + rearrange
    K: (d, B*S)              slice in HBM                                  -                                (d, B/KVDP*S) @ SBUF
    V: (B, 1, S, d)          slice in HBM                                  -                                stays in HBM

When q_heads==1, the Q path avoids transpose entirely: d stays on partition dim, rearrange in HBM only.

**Output Collectives (after attention):**

    SBUF Input                 Rearrange + Transpose to HBM                  all_to_all                       Transpose + Rearrange to SBUF
    ──────────                 ─────────────────────────────                 ──────────                       ─────────────────────────────
    attn: (d, B*q*S)           rearrange SBUF (d, KVDP*q, B/KVDP*S)          (KVDP*q, B/KVDP, d, S)           (d, B*q*S) @ SBUF
                               tiled transpose to HBM (KVDP*q, B/KVDP, d, S)                                  tiled transpose + rearrange

**Example with KVDP=4, q_heads=1, B=8, S_tkg=1, d_head=64:**

    Input Collectives (B/KVDP=2, q_heads*KVDP=4):
        Q @ SBUF:  (64, 8)           - d_head=64, B*q_heads*S_tkg=8
        Q @ HBM:   (64, 8)           - copy to HBM (q_heads=1, no transpose)
        Rearranged:(4, 64, 2)        - (KVDP=4, d_head=64, B/KVDP*S=2) — KVDP groups on dim 0
        After a2a: (4, 64, 2)        - rank i gets chunk i from all ranks
        Q @ SBUF:  (64, 8)           - rearrange (q_attn=4, d=64, B_attn=2, S=1) → (d=64, B_attn*q_attn*S=8)

    Output Collectives:
        attn @ SBUF: (64, 8)         - d_head=64, B*q_heads*S_tkg=8
        Rearranged:  (64, 8)         - rearrange (d, B_attn, q_attn, S) → (d, q_attn, B_attn, S)
        attn @ HBM:  (4, 2, 64, 1)   - tiled transpose: (q_attn=4, B_attn=2, d=64, S=1)
        After a2a:   (4, 2, 64, 1)   - rank i gets q_heads from all KVDP ranks
        attn @ SBUF: (64, 8)         - tiled transpose + rearrange (d, KVDP, q, B_attn, S) → (d, B*q*S)

**Example with KVDP=4, q_heads=2, B=8, S_tkg=1, d_head=64 (q_heads>1 path):**

    Input Collectives (B_attn=B/KVDP=2, q_heads_attn=KVDP*q_heads=8):
        Q @ SBUF:  (64, 16)          - (d=64, B*q*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, B, q, S)=(64,8,2,1) → (d, KVDP, q, B_attn, S)=(64,4,2,2,1)
        Q @ HBM:   (8, 64, 2, 1)     - tiled transpose: (KVDP*q=8, d=64, B_attn=2, S=1)
        After a2a: (8, 64, 2, 1)     - rank i gets chunk i (q=2 heads) from all 4 ranks
        Transpose: (64, 16)          - tiled transpose back: (d=64, q_attn*B_attn*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, q_attn, B_attn, S)=(64,8,2,1) → (d, B_attn, q_attn, S)=(64,2,8,1)
        Q @ SBUF:  (64, 16)          - (d=64, B_attn*q_attn*S=16)

    Output Collectives:
        attn @ SBUF: (64, 16)        - (d=64, B_attn*q_attn*S=2*8*1=16)
        Rearrange:   (64, 16)        - rearrange SBUF (d, B_attn, q_attn, S)=(64,2,8,1) → (d, q_attn, B_attn, S)=(64,8,2,1)
        attn @ HBM:  (8, 2, 64, 1)   - tiled transpose: (q_attn=8, B_attn=2, d=64, S=1)
        After a2a:   (8, 2, 64, 1)   - rank i gets q=2 heads from all 4 ranks
        Transpose:   (64, 16)        - tiled transpose back: (d=64, q_attn*B_attn*S=16)
        Rearrange:   (64, 16)        - rearrange SBUF (d, KVDP, q, B_attn, S)=(64,4,2,2,1) → (d, KVDP, B_attn, q, S)=(64,4,2,2,1)
        attn @ SBUF: (64, 16)        - (d=64, B*q*S=8*2*1=16)

### Collective Operations: ALL_GATHER_SLICE

The implementation uses `all_gather` on heads + `slice` on batch. Transposes are required to move the
collective dimension to dim=0 for all_gather.

**Input Collectives (before attention):**

    SBUF Input               Transpose to HBM              all_gather                  slice batch                      Transpose to SBUF
    ──────────               ────────────────              ──────────                  ───────────                      ─────────────────
    Q: (d, B*q_heads*S)      (q_heads, B, S, d) @ HBM      (KVDP*q_heads, B, S, d)     (KVDP*q_heads, B/KVDP, S, d)     (d, B*q_heads*S) @ SBUF
    K: (d, B*S)              slice in HBM                  -                           (d, B/KVDP*S)                    (d, B/KVDP*S) @ SBUF
    V: (B, 1, S, d)          slice in HBM                  -                           (B/KVDP, 1, S, d)                stays in HBM

When q_heads==1, the Q transpose can be skipped: all_gather directly on d_head dimension, then rearrange.

**Output Collectives (after attention):**

    SBUF Input                 Transpose to HBM                 all_gather                 slice heads              Transpose to SBUF
    ──────────                 ────────────────                 ──────────                 ───────────              ─────────────────
    attn: (d, B*q_heads*S)     (B/KVDP, KVDP*q_heads, d, S)     (B, KVDP*q_heads, d, S)    (B, q_heads, d, S)       (d, B*q_heads*S) @ SBUF

**Example with KVDP=4, q_heads=1, B=8, S_tkg=1, d_head=64:**

    Input Collectives (B/KVDP=2, q_heads*KVDP=4):
        Q @ SBUF:  (64, 8)           - d_head=64, B*q_heads*S_tkg=8
        Q @ HBM:   (64, 8)           - no transpose needed (q_heads=1 optimization)
        Gathered:  (256, 8)          - KVDP*d_head=256, B*S_tkg=8
        Sliced:    (4, 64, 2, 1)     - KVDP*q_heads=4, d_head=64, B/KVDP=2, S_tkg=1
        Q @ SBUF:  (64, 8)           - d_head=64, B*q_heads*S_tkg=8

    Output Collectives:
        attn @ SBUF: (64, 8)         - d_head=64, B*q_heads*S_tkg=8
        attn @ HBM:  (2, 4, 64, 1)   - B/KVDP=2, KVDP*q_heads=4, d_head=64, S_tkg=1
        Gathered:    (8, 4, 64, 1)   - B=8, KVDP*q_heads=4, d_head=64, S_tkg=1
        Sliced:      (8, 1, 64, 1)   - B=8, q_heads=1, d_head=64, S_tkg=1
        attn @ SBUF: (64, 8)         - d_head=64, B*q_heads*S_tkg=8

**Example with KVDP=4, q_heads=2, B=8, S_tkg=1, d_head=64 (q_heads>1 path):**

    Input Collectives (B_attn=B/KVDP=2, q_heads_attn=KVDP*q_heads=8):
        Q @ SBUF:  (64, 16)          - (d=64, B*q*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, B, q, S)=(64,8,2,1) → (d, q, B, S)=(64,2,8,1)
        Q @ HBM:   (2, 8, 1, 64)     - tiled transpose: (q=2, B=8, S=1, d=64)
        Gathered:  (8, 8, 1, 64)     - all_gather dim=0: (KVDP*q=8, B=8, S=1, d=64)
        Sliced:    (8, 2, 1, 64)     - KVDP_rank slice on batch: (KVDP*q=8, B_attn=2, S=1, d=64)
        Transpose: (64, 16)          - tiled transpose back: (d=64, q_attn*B_attn*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, q_attn, B_attn, S)=(64,8,2,1) → (d, B_attn, q_attn, S)=(64,2,8,1)
        Q @ SBUF:  (64, 16)          - (d=64, B_attn*q_attn*S=16)

    Output Collectives:
        attn @ SBUF: (64, 16)        - (d=64, B_attn*q_attn*S=2*8*1=16)
        attn @ HBM:  (2, 8, 64, 1)   - tiled transpose: (B_attn=2, q_attn=8, d=64, S=1)
        Gathered:    (8, 8, 64, 1)   - all_gather dim=0: (B=8, q_attn=8, d=64, S=1)
        Sliced:      (8, 2, 64, 1)   - KVDP_rank slice on heads: (B=8, q=2, d=64, S=1)
        attn @ SBUF: (64, 16)        - tiled transpose back: (d=64, B*q*S=8*2*1=16)

See `_KVDP_attention_input_collectives` and `_KVDP_attention_output_collectives` docstrings for pseudocode.


## Context Parallelism (CP) Implementation

### Purpose

CP reduces KV cache memory by sharding along the sequence dimension across CP ranks.

    K/V cache shape: [n_layers, batch, kv_heads, s_prior/CP, d_head]
                                                 ^^^^^^^^^^^
                                                 context parallelism (CP)

KV cache sharded into `s_prior/CP` sequence length per rank.
Requires distributed softmax correction across CP ranks (each rank has partial softmax stats)
Only the owning rank includes `k_active`/`v_active` to avoid counting active tokens CP times
For block KV: each rank attends to num_blocks/CP blocks (default interleave_size=block_len)

Constraints:
- `S_ctx % CP == 0` (sequence length must be divisible by CP degree)
- CP and KVDP can be combined (KVDP slices batch, CP slices sequence) with either KVDP
  collective mode (ALL_GATHER_SLICE or ALL_TO_ALL).
- CP requires a pre-generated (host-supplied) `attention_mask`; in-kernel mask generation
  (`pos_ids`) is not yet supported with CP (see Future Work).

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                   ATTENTION BLOCK TKG WITH KVDP + CP (combined)                              │
│                                                                                              │
│  Per-rank input: X [B, S_tkg, H]                                                             │
│  KV cache: [B/KVDP, S_ctx/CP, d] (batch-sharded by KVDP, seq-sharded by CP)                  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                             │
│  │  RMSNorm X      │──▶│ QKV Projection  │──▶│ Split Q,K,V     │                             │
│  │  (optional)     │   │ W_qkv @ X       │   │ + RoPE + quant  │                             │
│  └─────────────────┘   └─────────────────┘   └────────┬────────┘                             │
│                                                       │                                      │
│  ╔════════════════════════════════════════════════════╧═════════════════════════════════╗    │
│  ║  KVDP INPUT (if KVDP > 1)                                                            ║    │
│  ║  Q: all_to_all batch→heads: [d, B, n, S] → [d, B/KVDP, n*KVDP, S]                    ║    │
│  ║  K, V: slice batch to B/KVDP                                                         ║    │
│  ╚══════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                      │
│  ╔════════════════════════════════════════════════════╧═════════════════════════════════╗    │
│  ║  CP INPUT (if CP > 1)                                                                ║    │
│  ║  Q: all_gather on head dim: [d, B/KVDP, n*KVDP, S] → [d, B/KVDP, n*KVDP*CP, S]       ║    │
│  ║  K, V: unchanged (each rank has its s_prior/CP shard)                                ║    │
│  ╚══════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                      │
│                                                       ▼                                      │
│                        ┌──────────────────────────────────────────────────────┐              │
│                        │                  ATTENTION TKG                       │              │
│                        │  Q: [d, B/KVDP, n*KVDP*CP, S]                        │              │
│                        │  KV: [B/KVDP, S_ctx/CP, d]                           │              │
│                        │  return_cp_softmax_stats=True (if CP)                │              │
│                        └──────────────────────────┬───────────────────────────┘              │
│                                                   │                                          │
│  ╔════════════════════════════════════════════════╧═════════════════════════════════════╗    │
│  ║  CP OUTPUT (if CP > 1)                                                               ║    │
│  ║  1. all_gather softmax stats (max, sum) across CP ranks                              ║    │
│  ║  2. Compute global max, correction, global sum                                       ║    │
│  ║  3. Scale: unnorm_out * exp(local_max - global_max) / global_sum                     ║    │
│  ║  4. all_to_all: redistribute heads back → [B/KVDP, n*KVDP, D, S]                     ║    │
│  ║  5. Sum across CP ranks                                                              ║    │
│  ╚══════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                      │
│  ╔════════════════════════════════════════════════════╧═════════════════════════════════╗    │
│  ║  KVDP OUTPUT (if KVDP > 1)                                                           ║    │
│  ║  all_to_all heads→batch: [d, B/KVDP, n*KVDP, S] → [d, B, n, S]                       ║    │
│  ╚══════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                      │
│                                                       ▼                                      │
│                        ┌──────────────────────────────────────────────────────┐              │
│                        │  KV Cache Update + Output Projection (optional)      │              │
│                        └──────────────────────────────────────────────────────┘              │
│                                                                                              │
│  Per-rank output: [B, S_tkg, H]                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Distributed Softmax Correction Algorithm

Each CP rank runs `attention_tkg` with `return_cp_softmax_stats=True` on its local
`s_prior/CP` shard. The FA loop (or single-tile path) produces local unnormalized
output and local softmax stats. The correction algorithm combines these into the
globally-correct normalized output.

See [attention_tkg_design_spec.md](attention_tkg_design_spec.md) for details on the
`return_cp_softmax_stats` path and exported stats format.

#### Mathematical Basis

Standard softmax attention:

    output = Σ_i [exp(QK_i - global_max) * V_i] / Σ_i [exp(QK_i - global_max)]

Each CP rank `r` computes with its **local** max (over its `s_prior/CP` shard):

    out_r = Σ_tiles [exp(QK - local_max_r) * V]    (FA running_output, unnormalized)
    sum_r = Σ_tiles [exp(QK - local_max_r)]         (FA running_sum)

To combine, rescale everything to the **global** max across all CP ranks:

    correction_r = exp(local_max_r - global_max)        always ≤ 1 (numerically stable)
    sum_r_corrected = sum_r * correction_r              rescale sum
    global_sum = Σ_r sum_r_corrected
    out_r_corrected = out_r * correction_r / global_sum  rescale + normalize output
    final = Σ_r out_r_corrected

This is the same math as FA tile-to-tile correction (`_fa_update_running_max` /
`_fa_update_running_sum` / `_fa_accumulate_output` in `attention_tkg.py`) but
applied **across ranks** instead of across tiles within a single rank.

#### Step-by-Step with `max_negated` Optimization

The exported `local_max` may be negated (see `max_negated` in
[attention_tkg_design_spec.md](attention_tkg_design_spec.md)). The algorithm handles
both cases using the same `nisa.activation` pattern as `_fa_update_running_max`:

```
_CP_attention_output_collectives(unnorm_out, local_max, local_sum, CP, max_negated, ...):

    # Step 1: Exchange stats across CP ranks
    all_max = all_gather(local_max, dim=0)    # (CP, d_head, s_active_bqh)
    all_sum = all_gather(local_sum, dim=0)    # (CP, d_head, s_active_bqh)

    # Step 2: Compute global max
    #   max_negated=True:  global_max = min(all_max)  (negated: min of negated = true max)
    #   max_negated=False: global_max = max(all_max)
    global_max = reduce(all_max, op=min if max_negated else max)

    # Step 3: Compute correction factor for this rank
    #   Same nisa.activation pattern as _fa_update_running_max:
    #   activation(exp, src, bias, scale=-1) computes exp(-(src - bias)) = exp(bias - src)
    #
    #   max_negated=True:  src=local_neg_max, bias=global_neg_max
    #                      → exp(global_neg_max - local_neg_max)
    #                      = exp(-global_true_max + local_true_max)
    #                      = exp(local_true_max - global_true_max)  ✓
    #   max_negated=False: src=global_max, bias=local_max
    #                      → exp(local_max - global_max)            ✓
    correction = nisa.activation(nl.exp,
        src=local_max if max_negated else global_max,
        bias=global_max if max_negated else local_max,
        scale=-1.0)

    # Step 4: Compute corrected sum and global sum
    corrected_sum = local_sum * correction
    global_sum = all_reduce(corrected_sum, op=sum)

    # Step 5: Scale local output
    out_scaled = unnorm_out * correction / global_sum

    # Step 6: Combine across ranks and redistribute heads (ALL_TO_ALL, the default)
    out_recv = all_to_all(out_scaled, dim=heads)   # receive all CP ranks' chunks for this rank's heads
    out_final = sum(out_recv, over=CP)             # local sum across the CP chunks
    # (REDUCE_SCATTER mode instead does sum + head slice in one reduce_scatter collective)

    return out_final
```

### Collective Operations

All collectives currently operate on `shared_hbm` tensors: data is staged SBUF -> HBM,
the collective runs HBM -> HBM, and the result is read back HBM -> SBUF. Using SBUF
directly as the collective src/dst is a future optimization (see Future Work #2).

**Input Collectives (before attention):**

    SBUF Input           all_gather heads         Result @ SBUF
    ──────────           ────────────────         ─────────────
    Q: (d, B*n*S)   →     (d, B*CP*n*S)           Q with all CP ranks' heads

CP's input collective redistributes only Q; the active K/V need no collective
handling — unlike KVDP, whose input collective batch-slices the active K/V.
(Sharding the KV projection across KV heads would extend the CP collective to
gather the active K/V too; see Future Work #7.)

**Output Collectives (after attention):**

    Input                    Operation                    Output
    ─────                    ─────────                    ──────
    unnorm_out (d, B*CP*n*S) ─┐
    local_max  (d, B*CP*n*S)  ├─ all_gather stats      → all_max, all_sum (CP, d, B*CP*n*S)
    local_sum  (d, B*CP*n*S) ─┘
                               ├─ global max + correction → correction (d, B*CP*n*S)
                               ├─ scale output            → out_scaled (d, B*CP*n*S)
                               └─ combine over CP (heads)  → out_final (d, B*n*S)
                                  (all_to_all + local sum [default], or reduce_scatter)

**Example with CP=2, q_heads=8, B=1, S_tkg=1, d_head=128, S_ctx=128K:**

    Input Collectives (q_heads*CP=16):
        Q @ SBUF:  (128, 8)          - d_head=128, B*q_heads*S_tkg=8
        Q gathered: (128, 16)         - d_head=128, B*CP*q_heads*S_tkg=16
        K @ SBUF:  (128, 1)          - d_head=128, B*S_tkg=1 (unchanged)
        K_cache:   (1, 1, 65536, 128) - B=1, kv_heads=1, S_ctx/CP=64K, d_head=128

    Attention TKG:
        Q: (128, 16), K_cache: 64K context, return_cp_softmax_stats=True
        → unnorm_out: (128, 16), local_max: (128, 16), local_sum: (128, 16)

    Output Collectives:
        all_gather stats:  (2, 128, 16)  - CP=2 ranks' max and sum
        correction:        (128, 16)     - exp(local_max - global_max)
        out_scaled:        (128, 16)     - unnorm_out * correction / global_sum
        combine over CP:   (128, 8)      - all_to_all + local sum (default), or reduce_scatter;
                                           either way sums across CP=2 ranks and keeps this rank's heads

See `_CP_attention_input_collectives` and `_CP_attention_output_collectives` in
`attention_block_tkg_sharding.py` for implementation.


### CP Output Collective Modes

The `CP_collective_mode` parameter selects the output collective strategy:

**ALL_TO_ALL (default):**
1. All_gather stats (fp32, tile layout) across CP ranks
2. Compute global max, correction factors in tile layout
3. Broadcast correction to `(d_head, BQS_local)`, scale local output
4. Pack scaled output into `(q_heads_attn, B, d_head, S_tkg)` shared HBM
   (with KVDP head permute if KVDP > 1)
5. All_to_all on dim=0: redistribute heads across ranks
6. Sum across CP chunks locally (each rank accumulates its q_heads)

**REDUCE_SCATTER:**
1. All_gather stats (fp32, tile layout) — same as A2A
2. Compute global max, correction, scale local output — same as A2A
3. Reduce_scatter: sum across ranks + head slice in one collective

Both modes issue two collectives (stats `all_gather` + combine); they
differ only in the combine: A2A redistributes heads then sums locally,
while RS fuses the sum and head slice into one `reduce_scatter`.

### LNC Batch Sharding for CP Output

When `B * q_heads_attn * S_tkg >= 128` (e.g., B=64, CP=4, q_heads_attn=4),
the attention kernel batch-shards across 2 NCs. Each NC produces stats and
output for `B_local = B/2` batches.

**Pre-collective (both RS and A2A):**
- Stats DMA: each NC writes its tiles to its own offset in shared HBM
  (`own_offset = bs_prg_id * n_bsq_tiles`). No race — non-overlapping regions.
- Stats correction: after all_gather, each NC reads only its `nc_offset` slice
- Scale broadcast: uses `BQS_local` via `_BroadcastParams`
- Attn scaling: slices `attn_sb` to per-NC portion
- HBM packing: each NC writes `B_local` batches at `b_offset = bs_prg_id * B_local`

**Collective:** operates on full `(q_heads_attn, B, ...)` shared HBM tensor.
Both NCs contribute their halves before the collective starts.

**Post-collective (RS):** reduce_scatter produces `(q_heads, B, d, S)`.
`BQS_out = B * q_heads * S_tkg` fits in SBUF (q_heads << q_heads_attn),
so no batch sharding needed for the output read-back.

**Post-collective (A2A):** all_to_all produces `(q_heads_attn, B, d, S)`.
Reshaped to `(CP, q_heads, B, d, S)`. Each rank sums across CP chunks.
`BQS_out = B * q_heads * S_tkg` fits in SBUF, so the sum uses full B
(no per-NC slicing needed post-collective).

### Background: Caller-Side Slot Mapping

This section is background on how the caller (the vLLM-Neuron model runner)
distributes tokens across CP ranks. It is **not** part of the kernel: the kernel
receives an already-sharded per-rank cache, `active_blocks_table`, mask, and
`kv_cache_update_idx`, and does not compute token ownership or cache slots itself.
The mapping is documented here only so the per-rank inputs below — and the test's
input builder (`_build_cp_block_kv_rank_inputs`) — can be understood. The
authoritative implementation is `_compute_slot_mapping_cpu` in the vLLM-Neuron model
runner (`neuron_model_runner.py`).

Terms:
- **slot**: a linear index into the block-KV cache pool,
  `block_number * block_len + block_offset`, where `block_offset` is the token's
  position within its block (`0..block_len-1`). This is the value written to
  `kv_cache_update_idx` (vLLM calls it the `slot_mapping`).
- `interleave_size`: number of consecutive positions assigned to one rank before
  rotating to the next. `interleave_size == block_len` assigns whole blocks
  round-robin; `interleave_size == 1` rotates individual tokens. (Neuron targets
  `interleave_size == block_len`.)
- **virtual block**: a run of `CP * block_len` consecutive sequence positions,
  distributed across the `CP` ranks in `interleave_size`-sized chunks (round-robin;
  see the `owner` formula below). `vbo` (virtual-block offset) is a position's index
  within its virtual block: `vbo = position % (CP * block_len)`.

For each position the caller computes:

    owner        = (vbo // interleave_size) % CP          # which rank owns this position
    block_offset = (vbo // (CP * interleave_size)) * interleave_size + vbo % interleave_size
    slot         = block_number * block_len + block_offset   # on the owning rank; skipped otherwise

`block_offset` packs a rank's owned positions into **contiguous** slots within its
blocks. For whole-block interleave (`interleave_size == block_len`) it reduces to
`vbo % block_len` and each rank simply owns whole blocks. Non-owned positions are
mapped to a skipped slot (`oob_mode.skip`).

### Token Ownership

Each CP rank owns a subset of sequence positions (via the `owner` formula in
[Background: Caller-Side Slot Mapping](#background-caller-side-slot-mapping)). The
owning rank determines which rank updates the KV cache for a given token and which
rank's attention mask includes that token as active.

Ownership does not require any interleave-specific logic in the kernel: softmax
attention is permutation-invariant over key/value positions, so the per-rank mask
(not the token ordering in the cache) determines which positions contribute.

See `_cp_owning_rank()` in `test_attention_block_tkg.py` for the test helper.

### Active Tokens

Every rank includes `k_active`/`v_active` in attention: `attention_tkg` writes `k_active`
into the last `S_tkg` positions of `k_prior`, then attends over all of `k_prior`. The
distinction is in the mask: only the owning rank (whose shard contains the active token's
position) leaves that position unmasked, so the active token contributes to softmax.
Non-owning ranks mask the active-token position to `-inf`, so `exp(-inf) = 0` and the
active token contributes nothing. This prevents double-counting active tokens across CP
ranks.

Active-token ownership is per batch element. Each batch element has one active token, and
the active token's sequence position selects the owning rank:

    owner = (vbo // interleave_size) % CP        # vbo = position % (CP * block_len)

So the owner varies per batch element. This is the same `owner` formula used for
prior-token ownership (see [Token Ownership](#token-ownership)). The
**caller** (serving stack) sets the per-rank mask and `kv_cache_update_idx` accordingly.
The test harness mimics the caller in `generate_kernel_inputs` / `_run_multi_rank_test`,
choosing each batch's owner via `_generate_cp_cache_lens` (all in
`test_attention_block_tkg.py`).

### Attention Mask

Each CP rank's mask must cover exactly the `S_local = S_ctx/CP` positions that rank owns,
in the same rank-local order as its KV cache. The mask carries per-position visibility
(`0` = masked to `-inf`), so mask order and cache order must be the same permutation —
otherwise a query would apply the wrong position's visibility.

The caller builds the per-rank mask in three steps (test helpers `_flatten_block_kv_mask`
and `_reblock_cp_mask` in `test_attention_block_tkg.py`):

    1. Flatten:  full block-KV mask -> [B, H, S_tkg, S_ctx]   (undo the block-KV reshape,
                 back to sequence-position order)
    2. Select:   [B, H, S_tkg, S_ctx] -> [B, H, S_tkg, S_local]   (keep the rank's owned
                 positions, in rank-local order — same `owned` order as the KV cache)
    3. Re-block: [B, H, S_tkg, S_local] -> [S_local, B, H, S_tkg]   (re-apply the block-KV
                 reshape over the rank's local S_local shard)

Step 2 is the only rank-specific step: whole-block interleave selects the rank's blocks;
sub-block interleave selects its scattered owned positions. Steps 1 and 3 are the standard
block-KV reshape (over `S_ctx` and `S_local` respectively), which the kernel already
expects — so no CP-specific masking logic lives in the kernel.

### KV Cache Update

All CP ranks compute active K/V from the QKV projection (all ranks have the full input X).
Each rank writes the new active K/V tokens to its own local cache shard at the
appropriate position. On the owning rank, the cache update index
(`kv_cache_update_idx`) is the `slot` from
[Background: Caller-Side Slot Mapping](#background-caller-side-slot-mapping)
(`block_number * block_len + block_offset`). Because `block_offset` packs owned
positions into contiguous slots, each rank sees a dense local cache. The test's
per-rank input builder (`_build_cp_block_kv_rank_inputs` in
`test_attention_block_tkg.py`) reproduces the same packing, so
`kv_cache_update_idx` indexes the local blocks the kernel reads.

Non-owning ranks use an out-of-bounds index so `oob_mode.skip` silently drops the
write (see [Non-Owning Rank KV Cache Update](#non-owning-rank-kv-cache-update)).

The flat (non-paged) KV cache uses a different update index; see
[Flat KV Cache with CP](#flat-kv-cache-with-cp).

### Block KV with CP

With whole-block interleave (`interleave_size == block_len`, the default), all CP
ranks share the **same `active_blocks_table`** — same physical block indices for
all ranks — and each rank attends to the `num_blocks/CP` blocks assigned to it
round-robin (rank `r` owns blocks `r, r+CP, r+2*CP, ...`). This requires
`num_blocks % CP == 0`. With sub-block interleave (`interleave_size < block_len`)
a rank's owned tokens are instead packed into its own local block table (see
[KV Cache Update](#kv-cache-update)).

**Kernel transparency:** `attention_tkg` sees normal block KV — no CP-specific
logic needed inside the kernel.

**Known limitation — active-token overwrite (accuracy):** `attention_tkg` places the
active K/V immediately past the prior context, overwriting the tail of each rank's cache
(see [K/V Cache Padding (Flat KV)](#kv-cache-padding-flat-kv) for the mechanism). For a
**fully-packed** block-KV CP shard those tail positions hold real prior tokens, so the
overwrite corrupts them. Flat KV avoids this via caller-side padding; block KV does not yet
have an equivalent fix — see [Future Work](#future-work) item 8.

Example (B=1, s_prior=128K, block_len=32, CP=4), whole-block interleave:

    All ranks share: active_blocks_table[1, 4096]  (same block indices)
    Rank 0: blocks [0, 4, 8, ...]   → 1024 blocks, 32 tokens each
    Rank 1: blocks [1, 5, 9, ...]   → 1024 blocks, 32 tokens each
    Rank 2: blocks [2, 6, 10, ...]  → 1024 blocks, 32 tokens each
    Rank 3: blocks [3, 7, 11, ...]  → 1024 blocks, 32 tokens each

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `CP` | int | Context parallelism degree (1 = disabled) |
| `CP_replica_group` | ReplicaGroup | Rank group for CP collectives |
| `max_negated` | bool (compile-time) | Whether exported max stats are negated. Determined by `attention_tkg` internals. Passed to CP collectives to select `min` vs `max` for global max reduction. |

### LNC2 Coexistence

Each CP rank runs `attention_tkg` which internally does LNC2 sharding (batch or
s_prior sharding across 2 NCs). The exported stats are already LNC2-combined:
`_gather_and_store_output` handles the LNC2 sendrecv before writing to HBM.

CP collectives operate on post-LNC2 data in HBM. The two levels of parallelism
are orthogonal:

```
┌─────────────────────────────────────────────────────┐
│                    CP Rank 0                        │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │  NC0 (LNC2)  │  │  NC1 (LNC2)  │  ← intra-chip   │
│  │  sendrecv    │  │  sendrecv    │    (fast)       │
│  └──────┬───────┘  └──────┬───────┘                 │
│         └────────┬────────┘                         │
│                  ▼                                  │
│         LNC2-combined stats @ HBM                   │
└─────────────────┬───────────────────────────────────┘
                  │
                  │  ← inter-chip (all_gather / all_reduce)
                  │
┌─────────────────┴───────────────────────────────────┐
│                    CP Rank 1                        │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │  NC0 (LNC2)  │  │  NC1 (LNC2)  │                 │
│  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────┘
```

## CP Implementation Notes

### FA NaN Prevention for All-Masked Tiles

When a CP rank has zero valid prior positions in an FA tile (e.g., rank 3 with only
1 active token and no prior tokens in its shard), the max reduction produces `-Inf`.
Then `exp(QK - max) = exp(-Inf - (-Inf)) = exp(NaN)`, which propagates through FA
running statistics and corrupts the output.

Fix: clamp the infinite max to a finite bound in `attention_tkg.py`
(`_clamp_max_to_finite`): `-Inf -> _MIN_FLOAT32` when `max_negated=False`, or
`+Inf -> _MAX_FLOAT32` when `max_negated=True` (both `= np.finfo(np.float32).min/max`).
With the clamp, `exp(-Inf - _MIN_FLOAT32) = exp(-Inf) = 0`, so fully masked tiles
contribute zero to the running output and sum. This is correct: a tile with no valid
positions should have no effect on the attention output.

### `max_negated` Propagation

The `attention_tkg` kernel's internal softmax stats may store the max in negated form
(`-max` instead of `max`), depending on the code path:

- **Non-FA path**: `max_negated=True` (final reduction uses `negate=True`)
- **FA path with deferred LNC2 sync**: `max_negated=False` (`fa_running_max` is non-negated)

The CP correction in `_CP_attention_output_collectives` needs to know the sign
convention to correctly compute the global max across ranks. With `max_negated=True`,
`nl.minimum` finds the global max (min of negated = true max). With `max_negated=False`,
`nl.maximum` is needed.

`attention_tkg` returns `max_negated` as a compile-time boolean alongside the
unnormalized output and stats. The CP correction uses it to select the correct
reduction op and `nisa.activation` argument order.

### Non-FA Stats Export: Skip Double Reciprocal

The non-FA path originally exported `exp_sum_recip` (reciprocal of the sum) for
normalization. CP correction needs the raw sum (not reciprocal) to compute
`corrected_sum = sum * correction_factor`. Exporting the reciprocal would require
taking the reciprocal again in CP correction (double reciprocal), losing precision.

Fix: when `return_cp_softmax_stats=True`, the non-FA path exports the raw `exp_sum`
directly, skipping the reciprocal computation entirely.

### Non-Owning Rank KV Cache Update

All CP ranks compute K/V from the QKV projection (all ranks have the full input X),
so every rank produces a new active K/V, but only the owning rank should persist it.
The owning rank (whose shard contains the active token's position) updates its cache
at the correct local index. Non-owning ranks use an out-of-bounds index for the cache
update; the kernel issues the update DMA with `oob_mode.skip` (`oob_skip=is_CP` in
`_kv_cache_update`), so the out-of-bounds write is dropped entirely rather than
written anywhere. This ensures the active token is persisted exactly once, by its
owner.

## Flat KV Cache with CP

Flat (non-paged) KV stores each rank's context contiguously as
`[B, kv_heads, s_prior/CP, d_head]`, rather than as a pool of fixed-size blocks
(block KV / PagedAttention). It is **not used by vLLM** — vLLM serving supports
only block KV — and is expected only for non-text models (or non-vLLM callers)
that use a contiguous cache.

The CP collectives (input Q all_gather; output stats all_gather + correction) are
identical for flat and block KV. Flat KV differs from block KV in three places: the
per-rank mask, the cache update index, and the active-token cache padding — all described
below.

Because flat KV stores each rank's context contiguously, the per-rank mask is a plain
contiguous slice `full_mask[r*S_local : (r+1)*S_local]` (`S_local = S_ctx/CP`) — no
`owned`-gather and no block-KV reshape (contrast the block-KV three-step flow in
[Attention Mask](#attention-mask)). The active-token entry within the appended padding is
then set per the [K/V Cache Padding (Flat KV)](#kv-cache-padding-flat-kv) rules below.

### Cache Update Index (Flat KV)

On the owning rank, `kv_cache_update_idx` is the token's local position within the
rank's `s_prior/CP` shard (rather than the block-KV `slot`). Non-owning ranks use
an out-of-bounds index so `oob_mode.skip` drops the write (see
[Non-Owning Rank KV Cache Update](#non-owning-rank-kv-cache-update)).

The position the owner *writes* the token to (its real local offset, above) is
different from the position the token is *read* from during attention (an extra
position past the rank's real context; see
[K/V Cache Padding (Flat KV)](#kv-cache-padding-flat-kv)).

### K/V Cache Padding (Flat KV)

To attend over the new tokens, `attention_tkg` places the active K/V immediately
after the prior K/V in the matmul buffer — the new tokens occupy the positions just
past the end of the prior context. Under CP, a flat-KV rank's cache is a dense
contiguous `s_prior/CP` slice with no spare positions, so when that shard is full
those positions already hold valid prior data that placing the active token would
overwrite.

Fix: extend each rank's cache to `ceil((s_prior/CP + S_tkg) / (lnc * P_MAX)) *
(lnc * P_MAX)` — its real context rounded up to the next `lnc * P_MAX` boundary
(`P_MAX` = 128, the s_prior tiling unit). This leaves room for the `S_tkg` active
tokens past the real context and keeps the padded length a valid s_prior geometry.
The active token's mask entry (in that padding) is set only on the rank that owns the
batch element's position (the owner differs per batch element — see
[Active Tokens](#active-tokens)); other ranks keep it masked out, so the overwritten
token contributes nothing. This is transparent to `attention_tkg` — a slightly larger
cache with the new token at the end, exactly as in the non-CP case.

This is done by the **caller**, not the kernel. In tests, `create_per_rank_input`
(in `_run_multi_rank_test`) allocates the padded per-rank cache/mask and marks the
owning rank. vLLM does not use this path (vLLM CP is block KV); a flat-KV caller must
implement it.

## Future Work

1. **KV projection for batch slice**: Currently each rank computes Q, K, V for all B batches (fused QKV kernel), then slices K/V to B/KVDP for cache update. The extra K/V compute is small relative to attention and avoids unfusing the QKV projection.

2. **SB2SB collectives**: The current implementation round-trips through HBM for all_gather (SBUF -> HBM -> all_gather -> HBM -> SBUF). The all_gather collective supports SBUF tensors as src/dst, which would eliminate the HBM round-trips and reduce latency for the input and output collectives.

3. **Skip cross-NC sendrecv with LNC seq sharding**: When `sprior_n_prgs > 1`, the kernel currently sendrecvs max/sum/output between NCs before exporting stats. Instead, export per-NC stats and partial outputs directly, merging them in the CP output collectives. Stats are small ([s_active_bqh_tile, n_bsq_tiles] per NC), but the partial output ([d_head, s_active_bqh] per NC) doubles the all_gather payload. Whether this is faster than the sendrecv depends on intra-chip sendrecv latency vs. inter-chip all_gather bandwidth — worth benchmarking.

4. **LSE instead of separate max+sum**: Export `lse = max + log(sum)` (1 scalar per batch × q_head × s_active) instead of separate max and sum (2 scalars). In `_copy_and_export_softmax_stats`, add 2 VectorE ops (`nl.log` + `nl.add`) and write a single `lse` tensor. In `_cp_gather_correct_and_scale_to_hbm`, the `local_stats_sb` buffer shrinks from `(s_active_bqh_tile, 2 * n_bsq_tiles)` to `(s_active_bqh_tile, n_bsq_tiles)`, halving the `ncc.all_gather` data size. The correction loop simplifies to `scale = exp(local_lse - global_lse)`, removing the per-rank `rank_sum * rank_correction` accumulation and `max_negated` handling. Same approach used by `ring_attention_fwd.py` (`attention_reduce.py`) and vllm-neuron DCP. Does not reduce collective count — still 1 all-gather + 1 all-to-all.

5. **Coalesced all-to-all for output + stats**: The A2A output path (`_cp_output_all_to_all`) currently issues two collectives — an `all_gather` for the softmax stats and an `all_to_all` for the scaled output. A coalesced all_to_all that accepts multiple src/dst tensors would let us pass attention output and stats as separate tensors through a single collective, eliminating the separate stats `all_gather` and halving the collective count. Distinct from #4 (LSE), which shrinks the stats payload

6. **In-kernel mask generation (`pos_ids`) with CP**: Today CP requires a pre-generated `attention_mask` because `gen_mask_tkg` emits a mask over the full context, whereas each CP rank must mask only the global positions it owns — the per-rank CP-sharded mask cannot be produced in-kernel. Extend `gen_mask_tkg` to accept the rank's CP ownership mapping (`interleave_size`, `CP`, `cp_rank`) and emit only the owned positions in rank-local order, matching the gathered local KV cache. This would let CP configs use `pos_ids` and drop the host-side mask materialization + transfer.

7. **Shard the active KV projection across the CP/KVDP ranks**: Today, when a rank has `kv_heads > 1`, it projects and stores all of them — the KV heads are replicated across the CP/KVDP ranks rather than distributed (the input collective gathers only Q; each rank already holds the full KV-head set). Instead, shard the KV heads across those same ranks: let `shard_degree` be the number of ranks sharing the KV heads (the CP and/or KVDP degree). Each rank runs its active KV projection over only `kv_heads / shard_degree` heads (shrinking the per-rank projection weight and its compute), then gathers the missing active K/V heads through the existing input collective alongside Q, so the per-rank head count grows back to `kv_heads` before attention (`kv_heads_attn = kv_heads`). The per-rank KV cache would then hold only its owned heads. Applies when `kv_heads` is divisible by `shard_degree`.

8. **Active-token cache padding — testing, block KV, and a possible kernel-side fix**: `attention_tkg` overwrites the last `s_active` positions of each rank's `s_prior` with the active K/V. Flat KV reserves those positions via caller-side padding (see [K/V Cache Padding (Flat KV)](#kv-cache-padding-flat-kv)); a fully-packed shard without this padding corrupts real prior tokens (see [Block KV with CP](#block-kv-with-cp)). Three follow-ups:
   - **Test that fails without the padding.** Current kernel-vs-golden tests can't catch this — the torch golden has the same API and clobbers the same positions — so both agree while being wrong. Add a test that compares against an unsharded (CP=1) baseline, which does not clobber.
   - **Add padding for block KV.** Only flat KV is padded today; block KV needs the equivalent caller-side reservation.
   - **Possible kernel-side fix (removes the need for caller padding).** When `CP > 1`, avoid packing the active K/V into the last `s_active` positions of `s_prior` so the clobber can't happen — fixing flat and block KV together. This also needs a separate active-token mask: ownership (owner unmasked, non-owners `-inf`) is currently carried in the `s_prior` tail, so moving the active K/V to a dedicated slot moves that ownership signal too.
