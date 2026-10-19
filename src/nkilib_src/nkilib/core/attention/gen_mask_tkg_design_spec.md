# gen_mask_tkg — Mask Generation Design Specification

Masking logic for the attention TKG kernel: cache structure, mask generation,
sliding window attention (SWA), and LNC2 sharding.

References:
- Attention TKG Kernel Integration Alignment
- APC BIR-NKI migration + feature addition — SWA mask geometry diagrams

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

All prior positions where `iota < pos_ids[b, 0]` are valid (the kernel reads
only the first element per batch from `pos_ids`). The active mask (causal
triangle) is placed at the last `s_active` positions of the prior buffer,
overwriting whatever the prior mask produced there.

```
cache_lens[b]=10, s_prior=16, s_active=4.  # = attend, · = masked
pos_ids=[10,11,12,13] (prior mask uses pos_ids[b,0]=10 for all queries)

         k/v_prior buffer (total size = s_prior = 16)
         0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
   q0   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│ #│ ·│ ·│ ·│
   q1   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│ #│ #│ ·│ ·│
   q2   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│ #│ #│ #│ ·│
   q3   │ #│ #│ #│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│ #│ #│ #│ #│
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
         ◄──── cached (iota<10) ────►       ◄─ active ─►
                                      ^^^^^^
                                      stale/empty (masked by prior mask)
```

Cols 0–9: valid cached tokens (prior mask: iota < 10).
Cols 10–11: stale/empty slots (prior mask: iota >= 10, masked).
Cols 12–15: last `s_active` positions, overwritten by active mask (causal triangle).

### 2B. SWA Per-Query Banded Mask

Each query has its own window start that shifts per active token, but the
prior mask end is always clamped to `pos_ids[b, 0]` (= `cache_lens[b]`).
Positions `cache_lens[b]` through `cache_lens[b] + s_active - 1` are active
tokens in the active KV buffer, not the prior cache — the active mask handles
those.

```
Input contract:
  pos_ids[b, i]   = cache_lens[b] + i   (but only pos_ids[b, 0] used for prior end)
  start_pos[b, i] = (pos_ids[b, i] - W + 1) mod s_prior   (flat KV)
                   = max(0, pos_ids[b, i] - W + 1)         (block KV)

Prior mask per query:
  end   = pos_ids[b, 0]                 (clamped — same for all queries)
  start = start_pos[b, i]               (shifts right per query)
```

#### Small window (W=8, s_prior=16, s_active=4, cache_lens[b]=10)

The prior mask end is 10 for all queries. Start shifts: 3, 4, 5, 6.
The prior portion shrinks per query; the active mask fills the gap.

```
         k/v_prior buffer (total size = s_prior = 16)
         0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
   q0   │ ·│ ·│ ·│ #│ #│ #│ #│ #│ #│ #│ ·│ ·│ #│ ·│ ·│ ·│  7 prior + 1 active = 8
   q1   │ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ #│ ·│ ·│ #│ #│ ·│ ·│  6 prior + 2 active = 8
   q2   │ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ #│ ·│ ·│ #│ #│ #│ ·│  5 prior + 3 active = 8
   q3   │ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│ ·│ ·│ #│ #│ #│ #│  4 prior + 4 active = 8
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
         ◄── prior mask (start..9) ──►       ◄─ active ─►
                                      ^^^^^^
                                    stale/empty (masked by prior mask)
```

Cols 10–11 are stale slots masked by the prior mask (iota >= 10).
Cols 12–15 are overwritten by the active mask (causal triangle).

#### Wrap-around (flat KV circular buffer)

Flat KV SWA uses a circular buffer of `s_prior - s_active` usable slots
(the last `s_active` slots are reserved for active tokens). The caller is
responsible for computing `pos_ids[b, 0]` and `start_pos[b, i]` as slot
indices with the appropriate modular arithmetic before passing them to the
kernel. The kernel only sees the resulting slot indices.

When the sliding window spans the buffer boundary, `start_pos > pos_ids[b, 0]`,
triggering OR logic: `[start, s_prior) ∪ [0, end)`.

Example: W=5, s_prior=16, s_active=4.
The caller passes pos_ids=[2, 3, 4, 5] and start_pos=[10, 11, 0, 1].

```
  pos_ids[b, 0] = 2 (prior mask end, same for all queries)

  start_pos  end  wrap?
    q0: 10    2   start(10)>end(2) → OR
    q1: 11    2   start(11)>end(2) → OR
    q2:  0    2   start(0)<=end(2) → AND
    q3:  1    2   start(1)<=end(2) → AND

         k/v_prior buffer (total size = s_prior = 16)
         0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
   q0   │ #│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ ·│ ·│ ·│  OR:  [10,16)∪[0,2)
   q1   │ #│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ ·│ ·│  OR:  [11,16)∪[0,2)
   q2   │ #│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ ·│  AND: [0,2)
   q3   │ ·│ #│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ ·│ #│ #│ #│ #│  AND: [1,2)
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
         ◄──►                                ◄─ active ─►
         valid cached                        (causal triangle)
```

Slots 12–15 are the reserved active region, overwritten by the active mask
(causal triangle). The OR logic for q0/q1 marks some of these slots, but
the active mask overwrites them. q2/q3 have `start <= end` so normal AND
logic applies. Note q3 has `start=1, end=2` so only slot 1 is in the prior
window — slot 0 is excluded.

#### Block KV (no wrap-around)

`start_pos[b,i] = max(0, pos_ids[b,i] - W + 1)`. End = `pos_ids[b, 0]`.
Invariant: `end >= start` always (no wrap-around in block KV).

---

## 3. SWA Branchless Wrap-Around Selection

NKI has no runtime branching. Both normal and wrap-around cases are handled
in a single code path. The prior mask end is always `pos_ids[b, 0]`
(= `cache_lens[b]`), clamped to the base position for all queries in a batch:

```python
end = pos_ids[b, 0]                            # base position (same for all queries)
ge = (iota >= start_pos[b, i])                  # 1 where pos >= start
lt = (iota <  end)                              # 1 where pos <  end
normal  = ge × lt                               # AND (no wrap)
wrap    = max(ge, lt)                           # OR  (wrap)
is_wrap = (start_pos[b,i] > end)
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
│             triple-select view chain → nisa.tensor_copy to mask_out
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
| Valid range | `[0, end_pos)` | `[start_pos[b,i], pos_ids[b,0])` per query |
| Mask pattern | Uniform rectangle | Prior mask is shrinking rectangle (start shifts, end fixed) |
| Wrap-around | N/A | Per-query OR logic |
| Input tensors | `pos_ids` only | `start_pos` + `pos_ids` |
| Code path | `_create_batch_masks()` | `_create_batch_masks_swa()` |
| Iota handling | `.ap()` replication | Direct (per-query loop) |
| SBUF scratch | 1 per batch | 3 per batch (fp32) |
| NKI ops/query | 1 | 8 |
