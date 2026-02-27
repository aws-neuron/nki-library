# gen_mask_tkg — Mask Generation Design Specification

Masking logic for the attention TKG kernel: cache structure, mask generation,
sliding window attention (SWA), and LNC2 sharding.

References:
- [Attention TKG Kernel Integration Alignment](https://quip-amazon.com/55tBAbKmae3a)
- [APC BIR-NKI migration + feature addition](https://quip-amazon.com/jzScAbS78ryr) — SWA mask geometry diagrams

---

## 1. Cache Structure

Active tokens are tucked at the END of the prior KV buffer.
`s_prior` includes both cached tokens AND reserved `s_active` slots.

```
k_prior / v_prior buffer (total size = s_prior):
┌────────────────────────────────────────────────┬─────────────────────┐
│         Cached Prior Tokens                    │  Active Tokens      │
│    k_prior[..., :-s_active]                    │ k_prior[...,-s_act:]│
│                                                │    = k_active       │
└────────────────────────────────────────────────┴─────────────────────┘
◄──────────────────────── s_prior ────────────────────────────────────►
                                                 ◄──── s_active ──────►
```

The circular write pointer wraps within `[0, s_prior - s_active)`. The mask
kernel does not special-case the reserved region — it compares `iota` against
caller-provided bounds.

---

## 2. Masking Scenarios

### 2A. Standard Attention (start_pos=None)

All prior positions where `iota < pos_ids[b, i]` are valid. Active region uses causal triangle.

```
pos_id=16, s_prior=16, s_active=4.  # = attend, · = masked

              k/v_prior (cols 0–15)                      │ k/v_active
         0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 │ 16 17 18 19
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┼──┬──┬──┬──┐
   q0   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│ ·│
   q1   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│
   q2   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ ·│
   q3   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

### 2B. SWA Per-Query Banded Mask

Each query has its own window `[start_pos[b,i], end_pos[b,i])`, producing a
BANDED/DIAGONAL pattern (not a uniform rectangle).

```
Per-query positions: rope_pos_ids[b,i] = pos_id[b] + i
  start_pos[b,i] = rope_pos_ids[b,i] - W + 1  (mod cache_len for flat KV)
  end_pos[b,i]   = rope_pos_ids[b,i]           (exclusive via < comparison)
```

#### Small window (W=8, s_prior=16, s_active=4, pos_id=16)

```
              k/v_prior (cols 0–15)                      │ k/v_active
         0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 │ 16 17 18 19
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┼──┬──┬──┬──┐
   q0   │ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│ ·│
   q1   │ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│
   q2   │ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ #│ #│ #│ ·│
   q3   │ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ #│ #│ #│
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

#### Wrap-around (W=8, s_prior=16, s_active=4, pos_id=3 — early positions, window wraps)

```
  q0: sp=12, pos_ids=3 → sp>pos_ids → wrap (OR) → [12,16)∪[0,3)
  q1: sp=13, pos_ids=4 → sp>pos_ids → wrap (OR) → [13,16)∪[0,4)
  q2: sp=14, pos_ids=5 → sp>pos_ids → wrap (OR) → [14,16)∪[0,5)
  q3: sp=15, pos_ids=6 → sp>pos_ids → wrap (OR) → [15,16)∪[0,6)

              k/v_prior (cols 0–15)                      │ k/v_active
         0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 │ 16 17 18 19
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┼──┬──┬──┬──┐
   q0   │ #│ #│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ ·│ ·│ ·│
   q1   │ #│ #│ #│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ ·│ ·│
   q2   │ #│ #│ #│ #│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ ·│
   q3   │ #│ #│ #│ #│ #│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

Column labels 0–15 are cache slot indices (iota values); 16–19 are the active
region columns in the QK matrix. `start_pos` is modded
(`(pos - W + 1) % cache_len`), while `pos_ids` is raw. Wrap-around triggers
when `start_pos > pos_ids` (i.e., early positions where `pos < W - 1`).

#### Block KV (no wrap-around)

`start_pos[b,i] = max(0, rope_pos_ids[b,i] - W + 1)`. Invariant: `end >= start` always.

---

## 3. SWA Branchless Wrap-Around Selection

NKI has no runtime branching. Both normal and wrap-around cases are handled
in a single code path:

```python
ge = (iota >= start_pos[b, i])          # 1 where pos >= start
lt = (iota <  pos_ids[b, i])            # 1 where pos <  end
normal  = ge × lt                       # AND (no wrap)
wrap    = max(ge, lt)                   # OR  (wrap)
is_wrap = (start_pos[b,i] > pos_ids[b,i])
final   = normal + is_wrap × (wrap − normal)
```

Equivalent to the torch reference's explicit `if/else` on `start_val <= end_val`.

---

## 4. Function Call Graph

```
gen_mask_tkg()
│
├── memset(mask_out, 0)
│
├── _generate_iota_tensor()
│   ├── block_len > 0: shuffled iota[p,f] = fold_base + p*blk + f
│   └── block_len = 0: nisa.iota(strided or sequential)
│
├── if start_pos is not None:           ← SWA path (no iota replication)
│   └── _create_batch_masks_swa()
│       └── for batch, sa_idx:
│             8 NKI ops per query (3 comparisons + 5 arithmetic)
│             TensorView triple-select → nisa.tensor_copy to mask_out
│
├── else:                               ← Standard path (.ap() iota replication)
│   ├── Replicate tmp_iota → mask_iota via nisa.tensor_copy + .ap()
│   └── _create_batch_masks()
│       └── for batch: nisa.tensor_scalar(mask_iota < pos_ids[batch])
│
└── if active_mask is not None:
    └── _load_active_mask()
        ├── Block KV: reverse-iota to (p, f) coordinates
        ├── Strided MM1: strided DMA with .ap() patterns
        └── Non-strided: DMA to bottom-right chunk
```

---

## 5. LNC2 Sharding Flow Diagrams

### Batch Sharding (is_batch_sharded=True)

Each NC processes different batches, same s_prior range. No cross-NC communication.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    BATCH SHARDING (LNC2) — gen_mask_tkg                        │
│              NC0: batches [0, bs)    NC1: batches [bs, bs_full)                │
│              Both NCs: full s_prior range, sprior_prg_id = 0                   │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────── NC0 ────────────────┐  ┌──────────────── NC1 ──────────────┐│
│  │ 1. memset(mask_out, 0)              │  │ 1. memset(mask_out, 0)            ││
│  │ 2. _generate_iota(base=0+offset)    │  │ 2. _generate_iota(base=0+offset)  ││
│  │ 3. _create_batch_masks[_swa]()      │  │ 3. _create_batch_masks[_swa]()    ││
│  │    batches [0, bs)                  │  │    batches [0, bs)                ││
│  │ 4. _load_active_mask(batch_start=0) │  │ 4. _load_active_mask(start=bs)    ││
│  └─────────────────────────────────────┘  └───────────────────────────────────┘│
│  No cross-NC communication.                                                    │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Sequence Sharding (is_s_prior_sharded=True)

Each NC processes different s_prior portion, all batches. No cross-NC communication.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  SEQUENCE SHARDING (LNC2) — gen_mask_tkg                        │
│              NC0: s_prior [0, s_prior/2)    NC1: s_prior [s_prior/2, s_prior)   │
│              Both NCs: all batches, batch_start = 0                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────── NC0 ─────────────────┐  ┌──────────────── NC1 ──────────────┐│
│  │ 1. memset(mask_out, 0)               │  │ 1. memset(mask_out, 0)            ││
│  │ 2. _generate_iota(base=0+offset)     │  │ 2. _generate_iota(base=s_p/2+off) ││
│  │ 3. _create_batch_masks[_swa]()       │  │ 3. _create_batch_masks[_swa]()    ││
│  │    all batches, iota covers [0,s_p/2)│  │    all batches, iota [s_p/2,s_p)  ││
│  │ 4. _load_active_mask(batch_start=0)  │  │ 4. _load_active_mask(start=0)     ││
│  │    (placed at end of NC0's tile)     │  │    (placed at end of NC1's tile)  ││
│  └──────────────────────────────────────┘  └───────────────────────────────────┘│
│  No cross-NC communication. Both NCs load same active_mask (same batches).      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences Summary

| Aspect | Batch Sharding | Sequence Sharding |
|--------|----------------|-------------------|
| Data split | Different batches per NC | Different s_prior portion per NC |
| sprior_prg_id | 0 for both NCs | 0 for NC0, 1 for NC1 |
| batch_start | 0 for NC0, bs for NC1 | 0 for both NCs |
| iota_base | Same for both NCs | NC1 offset by s_prior/2 |
| active_mask | Each NC loads its batch portion | Both NCs load same batches |
| Cross-NC comm | None | None |

---

## 6. Key Differences: Standard vs SWA

| Aspect | Standard | SWA |
|--------|----------|-----|
| Valid range | `[0, end_pos)` | `[start_pos[b,i], end_pos[b,i])` per query |
| Mask pattern | Uniform rectangle | Banded/diagonal |
| Wrap-around | N/A | Per-query OR logic |
| Input tensors | `pos_ids` only | `start_pos` + `pos_ids` |
| Code path | `_create_batch_masks()` | `_create_batch_masks_swa()` |
| Iota handling | `.ap()` replication | Direct (per-query loop) |
| SBUF scratch | 1 per batch | 3 per batch (fp32) |
| NKI ops/query | 1 | 8 |
