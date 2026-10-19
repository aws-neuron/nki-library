<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Sequence-Packed Token-Generation Attention with Tensor Indirection

## Status

This document describes `attention_tkg_seq_packed.py` and its
framework-to-kernel contract. `attention_block_tkg` dispatches to this kernel
when `seq_packed_attention_mask` is supplied together with the packed block,
Q-index, and sequence-ID tables. The two accumulator route tables are an
optional all-or-none pair. This packed mask is generated from the schedule and
sequence lengths and is the only attention mask consumed by the packed path;
the ordinary `attention_mask` argument may be `None`.

`attention_tkg_seq_packed.py` executes a static schedule of flash-attention
tiles. The schedule may densely mix tiles from requests with different context
lengths. All rows execute the same fixed amount of work; request data does not
control Python loops or trace-time branches.

## Execution Model

Each schedule slot computes one flash-attention tile and produces a tile-local
online-softmax state:

```text
tile_partial = (tile_max, tile_sum, tile_output)
```

This is called a **tile partial** because it contains only one tile's
contribution, not the complete attention result for the sequence. The kernel
maintains one persistent online-softmax state per sequence and merges every tile
partial for that sequence into it. A sequence may therefore appear in several
slots or rows while still updating the same persistent state.

The kernel supports two implementations of this merge. They use the same
schedule, tile partials, persistent state, and associative softmax equations.
Only the routing and reduction strategy differs.

### Ordered Accumulator (Fallback)

When the two accumulator route tables are omitted, the kernel processes the
row's slots in their fixed order:

```text
for slot in 0 .. num_slots - 1:
    sequence = seq_id_table[row, slot]
    state[sequence] = merge(state[sequence], tile_partial[slot])
```

If a sequence occupies multiple slots, each later slot observes the state
written by the earlier slot. This preserves correctness for arbitrary runtime
sequence IDs and repeated sequences without requiring additional routing
metadata. Padded slots produce the online-softmax identity and therefore leave
their safe in-bounds destination unchanged.

This is the general fallback for every valid packed schedule. Its cost is the
serial dependency chain across slots, even when the slots belong to independent
sequences.

See [Ordered Fallback Accumulator Update](#ordered-fallback-accumulator-update)
for its physical data layout and repeated-sequence behavior.

### Grouped Route-Table Accumulator

When both accumulator route tables are supplied, slots belonging to the same
sequence are grouped and merged in parallel. The grouped path:

1. gathers each group's prior sequence state;
2. scatters every tile partial into the group that owns its sequence;
3. reduces all sources within each group; and
4. scatters each reduced group to its persistent sequence state.

The associative online-softmax merge makes this mathematically equivalent to
the ordered accumulator. The caller derives grouping from the runtime
`seq_id_table`; the compiled kernel does not specialize control flow to one
fixed schedule. Dense packing and static execution are preserved.

The caller can construct the hardware-ready Tensor Indirection (TI) routes
with:

```python
from nkilib.core.attention.attention_tkg_seq_packed_torch import (
    build_packed_accumulator_route_tables,
)

routes = build_packed_accumulator_route_tables(
    seq_id_table,
    d_head=d_head,
    s_active_qh=q_head * s_active,
    rows_per_nc=rows_per_nc,
    bs_per_nc=batch // lnc,
)
```

The two route tables are an optional all-or-none pair. Supplying them selects
the grouped accumulator; omitting them selects the ordered fallback.

## Schedule Contract

The framework may construct a logical schedule of `(seq_id, tile_idx)` items.
It must satisfy:

1. Every row has the same number of slots.
2. The same `(seq_id, tile_idx)` item does not appear twice.
3. A sequence may occupy several slots when those slots represent different
   FA tiles.
4. Padding uses `seq_id = -1`.
5. Padded slots have no valid block routes and are fully masked.
6. Rows are contiguous by NeuronCore. With LNC2, all rows for NC0 precede all
   rows for NC1.

`tile_idx` is lowered into block routes and masks. It is not a kernel input.

## Worked Schedule

For sequence lengths:

```text
[4K, 8K, 16K, 64K, 8K, 32K, 16K, 10K]
```

and an 8K FA tile, the per-sequence tile counts are:

```text
[1, 1, 2, 8, 1, 4, 2, 2]
```

With four slots, the 21 work items occupy six logical rows:

```text
logical_schedule =
[
  [(0,0), (1,0), (2,0), (2,1)],
  [(3,0), (3,1), (3,2), (3,3)],
  [(3,4), (3,5), (3,6), (3,7)],
  [(4,0), (5,0), (5,1), (5,2)],
  [(5,3), (6,0), (6,1), (7,0)],
  [(7,1),   pad,   pad,   pad],
]
```

Repeated sequence IDs are different FA tiles, not duplicate work. The
framework lowers this logical schedule into the metadata below and lays out
all rows for each NeuronCore contiguously.

## Kernel Inputs

Let:

- `R`: total schedule rows across all NeuronCores
- `S`: slots per row
- `A`: `s_active_qh = q_head * s_active`
- `BPT = fa_tile_s_prior / block_len`: cache blocks per FA tile
- `C = ceil(S * A / 16)`

The packed inputs are:

| Input | Shape | Dtype | Meaning |
|---|---|---|---|
| `seq_id_table` | `[R, S]` | `int32` | Absolute sequence ID per slot; `-1` means padding |
| `q_index_table` | `[R, d_head, ceil(S * A / 16)]` | `uint16` | Snake-layout TI indices for gathering Q |
| `active_blocks_table` | `[R, S * BPT]` | `int32` | Physical KV-cache block routes |
| `mask` | `[R, p_max, tile_n_sprior * S * A]` | `uint8` | Packed attention masks, including padded-slot masking |
| `accumulator_partial_route_table` | `[R, d_head, C]` | `uint16` | Optional tile-partial to group/source offsets |
| `accumulator_group_state_route_table` | `[R, d_head, C]` | `uint16` | Optional group to persistent sequence-state offsets |

The first four tensors are required. The two optional accumulator tables are
supplied together when the grouped accumulator is requested. Each table is reused
independently for the max, sum, and PV-output fields. The entries are scalar
offsets into one field, not sequence IDs or group labels.

### Sequence and Q Routing

For the worked schedule, the semantic sequence table is:

```text
seq_id_table =
[
  [ 0,  1,  2,  2],
  [ 3,  3,  3,  3],
  [ 3,  3,  3,  3],
  [ 4,  5,  5,  5],
  [ 5,  6,  6,  7],
  [ 7, -1, -1, -1],
]
```

The repeated IDs identify different tiles for the same sequence. The
`seq_id_table` carries semantic ownership, while `q_index_table` is the
pre-expanded TI operand used to gather Q in the hardware-required 16-partition
snake layout.

The kernel uses `seq_id_table` to select active K/V by sequence and to retain
destination IDs in each row handoff for the ordered fallback. The route-table
builder consumes the same semantic table to produce the hardware-ready grouped
accumulator routes.

For `A = 2`, sequence `b` owns Q columns `[2*b, 2*b + 1]`. The logical Q
indices for the worked schedule are:

```text
[
  [ 0,1,  2,3,  4,5,  4,5],
  [ 6,7,  6,7,  6,7,  6,7],
  [ 6,7,  6,7,  6,7,  6,7],
  [ 8,9, 10,11, 10,11, 10,11],
  [10,11, 12,13, 12,13, 14,15],
  [14,15,  0,1,  0,1,  0,1],
]
```

Each eight-value row above is one logical TI row, not one TI column. The pair
at positions `[2*s, 2*s+1]` belongs to slot `s`. Padding selects
sequence zero as an in-bounds Q source; the packed mask prevents those values
from contributing.

For `d_head = 64`, the physical Q route shape is `[R, 64, 1]`. The logical
indices occupy partitions `0:16`, padded to a full TI partition group, and the
same group is repeated at partitions `16:32`, `32:48`, and `48:64`. This lets
every `d_head` row use the same free-axis indices.

### Packed Block Routes

For every `(seq_id, tile_idx)` work item, the framework selects the
corresponding `BPT` physical cache blocks and writes them into:

```text
active_blocks_table[row, slot * BPT : (slot + 1) * BPT]
```

Unused entries are `-1`; indirect DMA skips those out-of-bounds routes. This is
why `tile_idx` does not need its own kernel input.

## Ordered Fallback Accumulator Update

The ordered fallback consumes the same completed-row handoff as the grouped
path:

```text
tile_max:    [S, A]
tile_sum:    [S, A]
tile_output: [d_head, S, A]
sequence_id: [S]
```

Each slot's partial occupies a fixed source range. For `S = 4` and `A = 2`:

```text
slot 0 -> lanes [0,1]
slot 1 -> lanes [2,3]
slot 2 -> lanes [4,5]
slot 3 -> lanes [6,7]
```

This source layout depends only on the slot number, not on the sequence ID or
schedule row. The fallback therefore needs no partial-source route tensor. It
transposes the compact max and sum values into row layout once, broadcasts them
across `d_head`, and then reads each slot's fixed max, sum, and output slices.

`seq_id_table[row, slot]` supplies the destination. The kernel converts the
absolute sequence ID to its NeuronCore-local sequence index. A padded `-1`
sequence ID is clamped to the safe in-bounds sequence-zero destination; its
fully masked tile partial is the online-softmax identity, so the update leaves
sequence zero unchanged.

Conceptually, the fallback performs:

```text
for slot in 0 .. S-1:
    destination = local_sequence_id[slot]
    partial = (
        tile_max[slot],
        tile_sum[slot],
        tile_output[slot],
    )
    prior = gather(running_state, destination)
    merged = online_softmax_merge(prior, partial)
    scatter(running_state, destination, merged)
```

The gathered and scattered record contains all three fields for all `A` query
lanes:

```text
[d_head, 3 fields, A lanes]

field 0 = running maximum
field 1 = running exponential sum
field 2 = running weighted output
```

The scatter completes before the next slot gathers its destination. This
ordering is required when one row contains multiple tiles for the same
sequence. In worked-schedule row 4:

```text
slot:       0      1      2      3
sequence:   5      6      6      7
```

slot 1 first updates sequence 6 with tile `(6,0)`. Slot 2 then gathers that
updated state before merging tile `(6,1)`. Processing both slots from the same
old sequence-6 state would lose one contribution.

This immediate gather-merge-scatter sequence is correct for arbitrary runtime
sequence IDs, repeated sequences, and padding. Its performance limitation is
that all `S` slot updates form a serial dependency chain. The grouped path
removes that chain by deriving same-sequence groups before kernel execution.

## Grouped Accumulator Routes: Worked Example

This section starts from the logical operation and derives the encoded route
tensors afterward.

### What Is a Tile Partial?

A long sequence may require several flash-attention tiles. One schedule slot
processes exactly one `(sequence, FA tile)` work item, not the sequence's entire
attention computation.

For every query lane, QK, masking, softmax preparation, and PV produce this
online-softmax state for that tile:

```text
tile_max    = max(score values in this tile)
tile_sum    = sum(exp(score - tile_max)) over this tile
tile_output = sum(exp(score - tile_max) * V) over this tile
```

The document calls this triple a **tile partial**:

```text
tile_partial(sequence, tile) = (tile_max, tile_sum, tile_output)
```

It is partial because it represents only one cache tile. All tile partials for
a sequence, followed by active K/V and the optional sink, must be merged before
the kernel can normalize and return the final attention output.

The persistent state for one sequence contains the corresponding running
maximum, running exponential sum, and running weighted output accumulated from
all tiles processed so far.

### What Is a Query Lane?

A sequence has one independent online-softmax state for every query head and
active-token position. The flattened `(query_head, active_token)` index is a
**query lane**:

```text
query_lane = query_head * s_active + active_token
A = q_head * s_active = number of query lanes per sequence
```

For this example, assume:

```text
S = 4 slots per row
A = 2 query lanes per sequence
bs_per_nc = 8 sequences on this NeuronCore
d_head = 64
num_rows = 6
```

For example, when `s_active = 1`, the two query lanes are query heads zero and
one. Max and sum have one scalar per lane. `tile_output` has `d_head` values per
lane, but every `d_head` row follows the same sequence routing. The same two
route tables can therefore be reused independently for max, sum, and output.

The ordered fallback reads `seq_id_table` directly. Only the grouped
accumulator needs the route tables described below.

### Step 1: Choose a Concrete Schedule Row

Use row 4, counting from zero, of the schedule introduced above:

```text
slot:        0       1       2       3
work item:  (5,3)   (6,0)   (6,1)   (7,0)
sequence:     5       6       6       7
FA tile:      3       0       1       0
```

The four slots produce:

```text
slot 0 -> tile_partial(5, 3)
slot 1 -> tile_partial(6, 0)
slot 2 -> tile_partial(6, 1)
slot 3 -> tile_partial(7, 0)
```

Slots 1 and 2 belong to the same sequence but cover different FA tiles. Their
tile partials must be merged into one sequence-6 state.

A **group** represents one distinct sequence within this row. Its group number
is the first slot containing that sequence:

| Slot | Work item | Sequence | Group | Reason |
|---:|---:|---:|---:|---|
| 0 | `(5,3)` | 5 | 0 | Sequence 5 first appears in slot 0 |
| 1 | `(6,0)` | 6 | 1 | Sequence 6 first appears in slot 1 |
| 2 | `(6,1)` | 6 | 1 | Sequence 6 already belongs to group 1 |
| 3 | `(7,0)` | 7 | 3 | Sequence 7 first appears in slot 3 |

The slot-to-group mapping is `[0, 1, 1, 3]`. Group 2 is unused, which shows why
group numbers need not be densely packed.

### Step 2: Build One Workspace per Group

Each group has `S + 1` source positions:

```text
source 0:       the sequence's prior persistent state
source slot+1:  the tile partial produced by that slot
```

For this row, the logical workspace is:

| Group | Destination sequence | Source 0 | Source 1, slot 0 | Source 2, slot 1 | Source 3, slot 2 | Source 4, slot 3 |
|---:|---:|---|---|---|---|---|
| 0 | 5 | prior state 5 | `tile_partial(5,3)` | identity | identity | identity |
| 1 | 6 | prior state 6 | identity | `tile_partial(6,0)` | `tile_partial(6,1)` | identity |
| 2 | dummy | dummy state | identity | identity | identity | identity |
| 3 | 7 | prior state 7 | identity | identity | identity | `tile_partial(7,0)` |

`identity` means a maximum that cannot win, a zero exponential sum, and a zero
weighted output. Reducing across the five sources produces the new state for
each used group. The unused group preserves the dummy state.

This workspace exists independently for both query lanes and for each of the
three state fields.

### Step 3: Route Tile Partials into the Workspace

For one state field, the incoming values are flattened in
`[slot, query_lane]` order:

```text
[tile_partial(5,3).lane0, tile_partial(5,3).lane1,
 tile_partial(6,0).lane0, tile_partial(6,0).lane1,
 tile_partial(6,1).lane0, tile_partial(6,1).lane1,
 tile_partial(7,0).lane0, tile_partial(7,0).lane1]
```

Here `.lane0` and `.lane1` mean the selected field for those query lanes. For
example, while routing maximums they mean `tile_max` values; while routing
outputs they mean `d_head` rows of `tile_output`.

The grouped workspace is flattened in `[group, source, query_lane]` order. An
element at `(group, source, lane)` therefore has offset:

```text
workspace_offset = ((group * (S + 1) + source) * A) + lane
```

Because the tile partial from slot `s` always occupies source `s + 1`, this
row gives:

| Slot | Work item | Group | Source | Lane 0 offset | Lane 1 offset |
|---:|---:|---:|---:|---:|---:|
| 0 | `(5,3)` | 0 | 1 | 2 | 3 |
| 1 | `(6,0)` | 1 | 2 | 14 | 15 |
| 2 | `(6,1)` | 1 | 3 | 16 | 17 |
| 3 | `(7,0)` | 3 | 4 | 38 | 39 |

For example, slot 3, lane 0 is written to group 3, source 4:

```text
((3 * 5 + 4) * 2) + 0 = 38
```

Thus the logical partial-route vector for this row is:

```text
[2, 3, 14, 15, 16, 17, 38, 39]
```

Tensor Indirection uses this vector as destination offsets: element `i` of the
flat tile-partial input is scattered to `workspace[partial_route[i]]`.

### Step 4: Gather and Store Persistent Sequence State

Persistent state for one field is flattened in
`[local_sequence, query_lane]` order:

```text
state_offset = local_sequence * A + lane
```

For LNC1, the local sequence IDs in this example equal the absolute IDs. For
LNC2, the route builder subtracts the NeuronCore's batch offset, so each core
uses IDs in `[0, bs_per_nc)`.

This row maps its four groups as follows:

| Group | Sequence state | Lane 0 offset | Lane 1 offset |
|---:|---:|---:|---:|
| 0 | 5 | 10 | 11 |
| 1 | 6 | 12 | 13 |
| 2 | dummy sequence 8 | 16 | 17 |
| 3 | 7 | 14 | 15 |

The persistent allocation contains `bs_per_nc + 1` states. Index
`bs_per_nc`, which is 8 here, is the dummy state used by groups that have no
sequence. The logical group-state route vector for this row is therefore:

```text
[10, 11, 12, 13, 16, 17, 14, 15]
```

The kernel uses this route twice:

1. gather each group's prior state into source 0 before reduction;
2. scatter each reduced group back to its persistent sequence state.

### Step 5: Encode the Logical Routes for Tensor Indirection

Both logical vectors contain `S * A = 8` offsets. Tensor Indirection expects
indices in groups of 16 partitions, so the builder pads each vector to 16
entries:

```text
partial route, schedule row 4:
[2,3, 14,15, 16,17, 38,39, 0,0,0,0,0,0,0,0]

group-state route, schedule row 4:
[10,11, 12,13, 16,17, 14,15, 0,0,0,0,0,0,0,0]
```

The route does not depend on the output's `d_head` row. Hardware still requires
an index row for every partition, so this 16-entry pattern is repeated four
times for `d_head = 64`. With six schedule rows, each physical route tensor has
shape:

```text
[num_rows, d_head, ceil(S * A / 16)] = [6, 64, 1]
```

For completeness, partitions `0:16` for all six schedule rows are:

```text
accumulator_partial_route_table[:, 0:16, 0] =
[
  [2,3, 14,15, 26,27, 28,29, 0,0,0,0,0,0,0,0],
  [2,3,  4, 5,  6, 7,  8, 9, 0,0,0,0,0,0,0,0],
  [2,3,  4, 5,  6, 7,  8, 9, 0,0,0,0,0,0,0,0],
  [2,3, 14,15, 16,17, 18,19, 0,0,0,0,0,0,0,0],
  [2,3, 14,15, 16,17, 38,39, 0,0,0,0,0,0,0,0],
  [2,3, 14,15, 16,17, 18,19, 0,0,0,0,0,0,0,0],
]

accumulator_group_state_route_table[:, 0:16, 0] =
[
  [ 0, 1,  2, 3,  4, 5, 16,17, 0,0,0,0,0,0,0,0],
  [ 6, 7, 16,17, 16,17, 16,17, 0,0,0,0,0,0,0,0],
  [ 6, 7, 16,17, 16,17, 16,17, 0,0,0,0,0,0,0,0],
  [ 8, 9, 10,11, 16,17, 16,17, 0,0,0,0,0,0,0,0],
  [10,11, 12,13, 16,17, 14,15, 0,0,0,0,0,0,0,0],
  [14,15,  0, 1, 16,17, 16,17, 0,0,0,0,0,0,0,0],
]
```

The fifth row of each tensor is the concrete row derived above. In the final
schedule row, three padded slots are grouped under the safe in-bounds
sequence-zero destination. Their tile partials are identities, so they do not
change sequence zero. Unused groups route to the separate dummy state.

## Max, Sum, and Output Data Path

### Logical Layouts

For `S` slots and `A` query lanes, one completed schedule row has:

```text
tile_max:    [S, A]
tile_sum:    [S, A]
tile_output: [d_head, S, A]
sequence_id: [S]
```

`tile_output` is the unnormalized weighted PV numerator. It is divided by the
final sum only after every tile, active K/V, and optional sink contribution has
been merged.

The grouped accumulator represents the row and persistent state as:

```text
partial_records: [d_head, 3 fields, S slots, A lanes]
staged_records:  [d_head, 3 fields, S groups, S+1 sources, A lanes]
running_state:   [d_head, 3 fields, bs_per_nc+1 sequences, A lanes]

field 0 = maximum
field 1 = exponential sum
field 2 = weighted output
```

The extra persistent sequence is the dummy state for unused groups. The ordered
fallback uses the same three real fields without the dummy group routing.

### Workspace Footprint

The grouped layout is intentionally dense: `staged_records` reserves every
`[group, source]` position even though most positions contain identities. Its
size grows as `O(S^2)`, while the number of real prior states and tile partials
is `O(S)`.

This is usually not a significant SBUF-capacity cost for the supported slot
counts. For `d_head=64`, `A=8`, and FP32 intermediates:

| Slots | `staged_records` | All per-row grouped temporaries | Fraction of Trn3 32 MiB SBUF |
|---:|---:|---:|---:|
| 4 | 120 KiB | about 312 KiB | about 1% |
| 8 | 432 KiB | about 1 MiB | about 3% |

The temporary scope is released after each row accumulation and its storage is
reused, so the peak footprint does not multiply by the number of schedule rows.
The dense layout still performs operations over identity source positions;
whether that instruction work matters should be determined from profiles rather
than inferred from the allocation size alone.

### Why Max and Sum Are Broadcast

Max and sum contain one scalar per query lane, whereas output contains one value
per `d_head` row and query lane. The row handoff therefore stores max and sum in
a compact column layout. Before accumulation they are transposed into partition
zero and broadcast across `d_head` so all three fields share the layouts above.

This replication does not change the softmax semantics; it only allows max,
sum, and output to use the same elementwise merge and routing operations. One
combined max-and-sum broadcast is sufficient. `tile_output` already has every
`d_head` row and requires no equivalent broadcast.

### How the Route Tables Apply to the Three Fields

The route-table inputs contain no max, sum, or output field identifier. Their
offsets address only:

```text
partial route:     [group, source, query_lane]
group-state route: [local_sequence, query_lane]
```

The kernel selects one field at a time and applies the same offsets independently
to maximum, sum, and output. Conceptually:

```text
for field in [maximum, sum, output]:
    route partial_records[field] with the same partial_route
    route running_state[field] with the same group_state_route
```

The framework therefore generates two route tensors from `seq_id_table`, not
three field-specific versions. This is also why each route entry is an offset
within one selected field rather than an offset across the complete three-field
record.

### Where Tensor Indirection Is Used

The grouped path has three Tensor Indirection stages:

1. **Gather prior state.** Use `group_state_route` as source indices to gather
   each group's persistent sequence state. Copy it into source zero of
   `staged_records`.
2. **Scatter tile partials.** Use `partial_route` as destination indices to place
   each slot's tile partial into its `[group, source=slot+1, lane]` position.
3. **Store reduced state.** After ordinary tensor reductions combine the
   sources, reuse `group_state_route` as destination indices to scatter each
   group's result back to its persistent sequence state.

Each stage is applied separately to maximum, sum, and output. The source
reduction and correction-factor arithmetic between stages two and three use
ordinary tensor operations, not Tensor Indirection.

The ordered fallback does not use these route tensors. It obtains a destination
sequence from `seq_id_table`, gathers the complete max/sum/output state for one
slot, merges it, and stores it back before processing the next slot.

### Merge Equations

For group sources `i`, including source zero's prior state, the merge computes:

```text
new_max = max(max_i)
correction_i = exp(max_i - new_max)
new_sum = sum(sum_i * correction_i)
new_output = sum(output_i * correction_i)
```

When maxima are stored negated, the equivalent minimum/subtraction direction is
used. This is the standard associative online-softmax merge, so combining all
same-sequence slots in parallel is mathematically equivalent to processing them
serially in slot order.

The route builder does not assume a particular length distribution. It uses
only `seq_id_table`; therefore arbitrary request ordering, duplicate sequence
IDs within a row, padded slots, and different schedules across invocations are
handled by the same compiled kernel.

## Row Pipeline

For each schedule row, the kernel:

1. loads the packed block routes, mask, sequence IDs, and Q routes;
2. gathers Q and the scheduled K/V cache tiles;
3. runs QK, softmax, and PV for the fixed slot geometry;
4. stores compact per-slot max, sum, output, and sequence metadata; and
5. merges the preceding row using either the grouped route-table accumulator or
   the ordered fallback.

Two compact SBUF handoffs alternate between rows:

```text
row 0 result -> ping
row 1 QK runs while ping is accumulated
row 1 result -> pong
row 2 QK runs while pong is accumulated
```

Each handoff retains only max, sum, output, and sequence IDs. Large QK scratch
buffers remain row-local.

Ping and pong are separate persistent SBUF allocations with the same fixed slot
layout. They are internal kernel storage, not caller inputs.

## Active K/V and Sink

After all scheduled cache tiles have been accumulated, finalization folds the
active K/V for each sequence into the same persistent online-softmax state.
The attention sink is then folded once as another valid online-softmax source.
Active positions remain masked in the packed cache-tile masks, so they are not
counted twice.

A one-key active contribution and an 8K cache tile use the same associative
merge equations. This avoids assigning a special terminal slot or passing an
`active_slot_mask`. Causality among multiple active tokens is applied while the
active keys are folded.

## Metadata the Kernel Derives

The schedule should express each fact once. Semantic ownership comes from
`seq_id_table`; the remaining tensors are either direct data-access lowerings
of the schedule or values the kernel can derive from fixed geometry.

- There is no caller-visible `tile_idx` tensor. Before kernel invocation, the
  framework lowers each tile index into its portion of `active_blocks_table`
  and the packed mask.
- There is no `partial_index_table`. The kernel derives every tile-partial
  source range from `slot * A`; padded slots already produce the identity
  through the packed mask.
- There is no separate semantic accumulator-index or destination-mask input.
  The ordered fallback derives each destination directly from
  `seq_id_table`. When the grouped path is requested, the caller helper derives
  the two hardware-ready TI route tables from that same sequence table. The
  routes are a performance lowering, not a second ownership definition.
- There is no `active_slot_mask`. Active K/V is folded once per sequence during
  finalization instead of being assigned to a terminal packed slot.

This distinction matters for correctness: schedule semantics remain in the
sequence, block, Q, and mask metadata, while optional route tables only change
how an equivalent accumulator update is executed.

## Padding

A padded slot:

- has sequence ID `-1`;
- selects sequence zero only as an in-bounds routing target;
- has no valid cache-block routes;
- is fully masked; and
- produces identity max, zero sum, and zero output.

It therefore leaves sequence zero unchanged even when it shares that group with
a real sequence-zero slot. No extra sequence is appended to
`seq_id_table`; the grouped path uses a separate internal dummy state for
unused groups.

## DMA Batching

The packed K and V paths use hardware-validated batching caps:

- K gather-transpose: cap 32.
- V gather-copy: cap 64.
- Both paths also cap one DMA operation at 64 indirect descriptors. This matters
  when cache-block resizing reduces the effective block length to one token:
  the byte-volume cap alone could otherwise admit 128 descriptors.

Larger caps reduced DMA instruction count but regressed the broad packed
performance matrix, so they are not retained. The selected caps do not change
the packed schedule or accumulator semantics.

## Slot-Count Guidance

Slot count is a framework scheduling choice, not a kernel branch. More slots
reduce row count and expose larger QK/PV work, but require more padding when a
row cannot be filled.

For the measured batch-48, 32K-context matrix, eight slots per row consistently
outperformed four slots and four-slot compute combined across two accumulator
rows. The recommended starting point for that regime is therefore S8. Smaller
batches and sparse schedules should still choose the slot count that minimizes
total padded rows.

## Alternatives Evaluated

The following designs are not implemented:

- **Batch-by-source accumulation, optionally across multiple rows.** This
  design staged state as
  `[d_head, 3 fields, bs_per_nc, N+1 sources, A lanes]`, where source zero was
  every sequence's prior state and the remaining `N` sources held routed tile
  partials. Grouping several schedule rows reduced the number of accumulator
  invocations but increased `N` from `S` to `rows_per_group * S`. Reduction
  work and SBUF use therefore scaled as `bs_per_nc * (N+1)`, including identity
  cells for sequences not touched by those rows. Larger batches and row groups
  made that wasted work substantial, while cross-row grouping also extended
  partial lifetimes and added scheduling dependencies. The implemented
  slot-local layout instead scales as `S * (S+1)` per row and routes only the
  sequences represented by that row.
- **S4 compute with a G8 accumulator.** Combining two four-slot rows added more
  accumulator work and was materially slower than native S8 compute.
- **Combined max/sum/output routes.** A route three times as wide reduced TI
  call count, but the wider indirection records were generally slower than
  reusing one narrow route independently for the three fields.
- **Compile-time fixed routing.** Specializing control flow to one schedule is
  incorrect when runtime request lengths change.
- **Swapped QK.** This remains a separate possible optimization and is not
  required by the accumulator design.

A possible future direction is to decouple compute width from accumulator width
with additional framework-provided row-group metadata. That should be pursued
only if a future sweep finds a geometry where the simpler single-row routing is
insufficient.

## Limitations and Dispatch Constraints

- Sliding-window `start_pos_ids` are not consumed by packed attention; the
  fused block rejects packed SWA.
- LNC1 runs the full batch on one NeuronCore. LNC2 assigns each NeuronCore a
  disjoint, equally sized batch slice. Sequence-prior sharding is not supported.
- Packed attention requires block KV cache routing, a positive `block_len`, an
  FA tile size divisible by `block_len`, an even slot count, and
  `num_slots <= bs_per_nc`. Schedule rows must divide evenly across logical
  NeuronCores.
- FP8 E4M3 KV cache is supported when K-prior, V-prior, K-active, and V-active
  all use the same FP8 E4M3 variant. `fp8_packed` requires all four tensors to
  be FP8. FP8 E5M2 is not supported.
- The accumulator route tables must be supplied together. The route builder
  requires `d_head` to be a positive multiple of 16 and every generated offset
  to fit in `uint16`. Omitting both tables selects the ordered slot fallback.
- Every `(seq_id, tile_idx)` work item must be unique; repeating one would
  accumulate the same cache tile twice.
- The `attention_block_tkg` packed dispatch additionally requires `KVDP=1`,
  `CP=1`, an even batch, `S_tkg=1`, one KV head, block KV cache, and enabled FA
  prior-sequence tiling. Supplying `seq_packed_attention_mask` requires the
  other three base packed tensors, while the two accumulator routes remain an
  optional all-or-none pair. The ordinary `attention_mask` may be `None`.
