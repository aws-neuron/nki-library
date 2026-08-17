# NeuroTile User Guide


This guide onboards you to NeuroTile, a tile-based programming model for writing NKI
kernels on AWS Trainium / Inferentia. It advances from basics to expert material; each
section is anchored to a runnable example under `examples/` and illustrated with diagrams.
The exhaustive, normative API contract lives in the `nt.*` docstrings (read with
`help(nt.tiles)`); Appendix A collects the cross-cutting behavior and architecture the
docstrings do not cover.



---

## Table of contents


- [Part 0 — Introduction](#part-0--introduction)
- [Part 1 — Basics: your first kernel](#part-1--basics-your-first-kernel)
- [Part 2 — Intermediate: the optimization vocabulary](#part-2--intermediate-the-optimization-vocabulary)
- [Part 3 — Advanced: production concerns](#part-3--advanced-production-concerns)
- [Part 4 — Indirect indexing](#part-4--indirect-indexing)
- [Part 5 — Expert: full kernels](#part-5--expert-full-kernels)
- [Part 6 — Reference & survival kit](#part-6--reference--survival-kit)
- [Appendix A — Architecture & Cross-Cutting Behavior](#appendix-a--architecture--cross-cutting-behavior)



---

# Part 0 — Introduction



## What & why


A production NKI kernel is roughly **70% hardware-aware boilerplate and 30% algorithm**. The
boilerplate — tile-index arithmetic, HBM↔SBUF DMA orchestration, access-pattern math,
SBUF/PSUM allocation, software pipelining, LNC sharding, remainder handling — is each
individually subtle, hardware-specific, and rewritten per kernel, which is what makes NKI
kernels long, slow to write, and hard to change. NeuroTile collapses that 70% into a small set
of composable, **zero-cost** primitives so you write (and read) mostly the 30% that is the
algorithm, and in practice kernels shrink ~5–8× in line count at parity performance. The
deeper payoff, though, is a shift in what you spend your attention on: instead of hand-managing
tile indices, DMA descriptors, buffer rotation, and bank placement — each easy to get subtly
wrong — you reason about the kernel the way you reason about the algorithm, declaring the
tiles, moving them on-chip, computing, and storing. The boilerplate that used to demand months
of NeuronCore expertise to write correctly is absorbed into a handful of named primitives, so
the code you read back mirrors the algorithm itself and is far easier to reason about and to
change. That lower cognitive load is the central benefit; the rest follows from it.


The programming model will feel familiar if you have written Triton or Helion — you declare a
tile grid over a tensor, index into tiles, load them on-chip, compute, and store — which also
lowers the on-ramp for authors new to NKI, and none of it costs performance, because every
primitive lowers to the same `nisa.*` ISA a hand-written kernel would emit and the perf ceiling
is unchanged even as the line count drops. The common optimizations that authors otherwise
re-derive per kernel — coalesced multi-tile DMA, double and triple buffering, LNC sharding,
remainder handling — are on by default and hard to get wrong, so you cannot accidentally ship
an un-coalesced DMA or a single-buffered weight stream; and nothing is hidden, with no secret
compute, no secret DMA, no rewritten loops, and a `.data` escape hatch that hands you the raw
NKI tensor to drop into `nisa.dma_copy` whenever the high-level methods don't express
what you need.


This guide assumes NKI fluency. If you are new to these, the NKI architecture docs are the
prerequisite — NeuroTile makes NKI easier to *write*, but it is still NKI underneath.



## API summary table



### View factories — turn a tensor into an addressable view (no data moved)


| API | Purpose | Returns | Section | Contract |
|---|---|---|---|---|
| `nt.tiles(source, tile_size=, ...)` | Decompose a tensor (or a transformed/sliced view of one) into a tile grid | `NDSlice` | [§1](#tiles) | the `nt.tiles` docstring |
| `nt.blocks(source, block_size, tile_size=, ...)` | Group tiles into coalesced-DMA blocks | `NDSlice` | [§2](#blocks) | the `nt.blocks` docstring |
| `src.reshape/.permute/.flatten_dims/...` | Transform a tensor's layout (no DMA) before tiling | `NkiTensor` | [§3](#transform-before-tiling) | NkiTensor view-op docstrings |



### Allocation — reserve SBUF / HBM / PSUM


| API | Purpose | Returns | Section | Contract |
|---|---|---|---|---|
| `nt.alloc_tiles(tile_size, grid=/element_shape=, ...)` | Allocate a tiled SBUF/HBM buffer | `NDSlice` | [§2](#allocating-sbufhbm-buffers-alloc_tiles--alloc_blocks) | the `nt.alloc_tiles` / `nt.psum_pool` docstrings |
| `nt.alloc_blocks(tile_size, block_size, ...)` | Allocate a block-structured buffer | `NDSlice` | [§2](#allocating-sbufhbm-buffers-alloc_tiles--alloc_blocks) | the `nt.alloc_tiles` / `nt.psum_pool` docstrings |
| `nt.psum_pool(tile_size, ..., bank_axis=, bank_ids=)` | PSUM accumulator bank pool | `NDSlice` | [§2](#psum-accumulators-psum_pool-intro) / [§5](#walkthrough-mlp-cte--swiglu) | the `nt.alloc_tiles` / `nt.psum_pool` docstrings |



### Data movement — methods on an `NDSlice`


| API | Purpose | Returns | Section | Contract |
|---|---|---|---|---|
| `.load(...)` | Single coalesced DMA, HBM → SBUF | `NDSlice` | [§1](#data-movement-load-and-store) | the data-movement docstrings |
| `.store(data, ...)` | Single coalesced DMA, SBUF → HBM | — | [§1](#data-movement-load-and-store) | the data-movement docstrings |
| `.stream(dim=, buffer_count=, ...)` | Rotating-buffer DMA/compute overlap | `BlockStream` | [§2](#streaming) / [§3](#streaming-in-depth) | the data-movement docstrings |
| `.data` | Underlying NKI tensor view — compute operand and escape hatch | `nl.ndarray` | [§1](#data-movement-load-and-store) | the data-movement docstrings |
| `.tolist(dim=)` | Materialize sub-views as a Python list | `list` | — | the data-movement docstrings |
| `.whole_tiles()` / `.remainder_tiles()` | Split clean vs boundary sub-views | `list` | [§3](#remainder-handling) | the data-movement docstrings |



### View transforms — metadata-only, zero runtime cost


| API | Purpose | Section | Contract |
|---|---|---|---|
| `.reshape_dim(dim, shape)` | Split one dimension | [§3](#view-transforms) | the view-transform docstrings |
| `.flatten_dims(start, end)` | Merge contiguous dims | [§3](#view-transforms) | the view-transform docstrings |
| `.permute(dims)` | Reorder dims | [§3](#view-transforms) | the view-transform docstrings |
| `.rearrange(src, dst, sizes)` | einops split+reorder+merge in one call | [§3](#view-transforms) | the view-transform docstrings |
| `.split(dim, n)` | Split a dim into n chunks | [§3](#view-transforms) | the view-transform docstrings |
| `.slice(dim, start, end)` | Narrow an element range | [§3](#view-transforms) | the view-transform docstrings |
| `.broadcast(dim, size)` | Stride-0 broadcast of a size-1 dim | [§3](#view-transforms) | the view-transform docstrings |
| `.expand_dim(dim)` / `.squeeze_dim(dim)` | Insert / remove a size-1 dim | [§3](#view-transforms) | the view-transform docstrings |
| `.reshape(new_shape)` | Full contiguous reshape | [§3](#view-transforms) | the view-transform docstrings |
| `.fold(src_dim, into_dim, position=)` | Merge non-adjacent dims via DMA recipe | [§3](#fold) | the view-transform docstrings |



### Tile operations


| API | Purpose | Section | Contract |
|---|---|---|---|
| `.load(transpose=True, transpose_axes=)` | DMA transpose (static / gather) | [§3](#transpose) / [§4](#gather-transpose) | the view-transform docstrings |
| indirect `view[idx, ...]` | Gather / scatter / dynamic select | [§4](#part-4--indirect-indexing) | the `NDSlice.__getitem__` docstring |



### Sharding & trace-time utilities


| API | Purpose | Returns | Section | Contract |
|---|---|---|---|---|
| `nt.block_range(rank, num_shards, total)` | Contiguous LNC shard slice | `slice` | [§3](#multi-core-lnc-sharding) | the shard-helper docstrings |
| `nt.uneven_block_range(...)` | Contiguous, remainder to early ranks | `slice` | [§3](#multi-core-lnc-sharding) | the shard-helper docstrings |
| `nt.interleaved_range(...)` | Round-robin shard slice | `slice` | [§3](#multi-core-lnc-sharding) | the shard-helper docstrings |
| `nt.get_shard_info(...)` | Diagnostic shard summary (dict) | `dict` | [§3](#multi-core-lnc-sharding) | the shard-helper docstrings |
| `nt.ceiling_div(a, b)` | Trace-time `ceil(a/b)` | `int` | [§3](#multi-core-lnc-sharding) | the `nt.ceiling_div` / `nt.largest_divisor` docstrings |
| `nt.largest_divisor(n, max_val)` | Largest divisor ≤ max_val | `int` | — | the `nt.ceiling_div` / `nt.largest_divisor` docstrings |



## Install & import


```python
import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.experimental import neurotile as nt
```


NeuroTile depends only on the standard Neuron SDK toolchain (NKI, compiler, runtime) and already
part of `NKILib/experimental`.



## The core abstraction: NDSlice


Everything in NeuroTile is one composite type, the **`NDSlice`**:


```
  NDSlice
  ├─ Grid     — the logical iteration schedule
  │            (per-dim level stack: block step / tile step / element step,
  │             a cursor, and the remaining addressable region)
  └─ Layout   — the physical memory descriptor for the region it lives in
               (HBM / SBUF / PSUM: source, offset, strides, indirect parameters)
```


The split mirrors the two questions a tile-based kernel constantly answers — *which* tiles to
visit and *where* their bytes live. The **Grid** owns the first: it is the logical iteration
schedule, tracking how the tensor is carved into blocks, tiles, and elements, a cursor for how
far indexing has advanced, and how much addressable region remains. Indexing and slicing
(`x[i, j]`, `x[:, m]`) and the shape attributes (`.shape`, `.tile_shape`) are all Grid
concerns; no memory is touched. The **Layout** owns the second: it is the physical memory
descriptor — the source tensor, the byte offset and strides into it, the buffer region
(HBM / SBUF / PSUM), and any indirect-access parameters. It is what `.load()` / `.store()` /
`.data` consult to issue an actual DMA. Keeping the two separate is what lets a transform like
`.permute(...)` or a shard slice rewrite the iteration plan without moving data, and lets the
same view describe a tensor in HBM before a load and in SBUF after one. The full structure of
each is in [A.1](#a1--architecture--core-types); for everyday use, "Grid = which tiles, Layout
= where they live" is enough.


Every factory (`nt.tiles`, `nt.blocks`, `nt.alloc_*`) returns an
`NDSlice`. Every slice (`x[i, j]`), every transform (`.permute(...)`) of an NDSlice, every
`.load()` result, and every stream child is also an `NDSlice`. One type, used everywhere —
learn its behavior once and it applies uniformly. (`BlockStream`, returned by `.stream()`, is the one
extra type: it owns a rotating SBUF pool and hands out `NDSlice` children.)


The end-to-end shape of every kernel:


```
   HBM tensor
      │  nt.tiles(tile_size=(P, F))
      ▼
   tile grid (NDSlice)  ──[i, j]──▶  view (NDSlice)
                                        │  .load()   ── one coalesced DMA ──▶
                                        ▼
                                   SBUF tile (.data)
                                        │  nisa.*  (your algorithm)
                                        ▼
                                   SBUF result
                                        │  dst[i, j].store(result.data)
                                        ▼
                                   HBM tensor
```


With that picture in mind, Part 1 builds the first kernel.



---

# Part 1 — Basics: your first kernel


This part covers the irreducible core of NeuroTile: declaring a tile grid, addressing
tiles, moving them between HBM and SBUF, and looping over them. By the end you will have
written a complete matmul. Everything here is exercised in
`examples/_01_iteration/_01_tiles.py` and `examples/_02_matmul/_01_matmul_patterns.py` —
run those alongside reading.



## Tiles


`nt.tiles()` decomposes a tensor into a logical grid of fixed-size tiles. No data is moved —
what you get back is a lightweight view that knows how to address each tile by its grid
coordinate, and the data stays in HBM until you load it.


```python
src_tiles = nt.tiles(src, tile_size=(128, 256))   # src.shape == (512, 1024) -> 4x4 tile grid
src_tiles.shape                                    # (4, 4) -- tile grid, not elements
```


A `(512, 1024)` tensor tiled at `(128, 256)` becomes a 4×4 grid:


```
              256       256       256       256
          ┌─────────┬─────────┬─────────┬─────────┐
     128  │ (0, 0)  │ (0, 1)  │ (0, 2)  │ (0, 3)  │
          ├─────────┼─────────┼─────────┼─────────┤
     128  │ (1, 0)  │ (1, 1)  │ (1, 2)  │ (1, 3)  │
          ├─────────┼─────────┼─────────┼─────────┤
     128  │ (2, 0)  │ (2, 1)  │ (2, 2)  │ (2, 3)  │
          ├─────────┼─────────┼─────────┼─────────┤
     128  │ (3, 0)  │ (3, 1)  │ (3, 2)  │ (3, 3)  │
          └─────────┴─────────┴─────────┴─────────┘

      src.shape       = (512, 1024)   -- element dimensions
      src_tiles.shape = (4, 4)        -- tile grid dimensions
```


The first entry of `tile_size` is the tile's **partition (P) extent** and the rest is its
**free (F) extent**, matching the SBUF layout a tile lands in once loaded. `tile_size` must
have at least two dimensions: there is no 1-D tile. If you want a tile that spans the full
partition axis but a single free column, write `(P, 1)`; for a single partition row across
the free axis, write `(1, F)`. This keeps the P/F orientation explicit rather than inferred.


**Choosing `tile_size`.** `tile_size` is the **compute grain** — the shape one `nisa.*`
instruction operates on once the tile is loaded. Pick it from how the tile will be consumed,
so it lands ready for its op and you avoid re-tiling later. Size each axis to the consuming
op's per-operand limit rather than to a whole logical dimension — e.g. for `nisa.nc_matmul`
the stationary operand maxes at `128×128` but the moving operand can be `128×512`, so a tile
feeding the moving side wants `(128, 512)` and one feeding the stationary side wants
`(128, 128)`. Don't set the F extent to a full hidden/intermediate dimension just because the
tensor is that shape — that overflows the op's operand limit or wastes SBUF; let `nt.blocks()`
group several right-sized tiles when you want to move more per DMA.



## `.shape` vs `.element_shape` vs `.tile_size`


A view carries three shape-like attributes, and keeping them straight removes most early
confusion:


- **`.shape`** — the *iteration* count per dimension: how many tiles you walk along each
  axis from where indexing currently stands. For the grid above it is `(4, 4)`.
- **`.element_shape`** — the *element* extent the view still addresses: `(512, 1024)` here.
- **`.tile_size`** — the per-tile granularity you passed in: `(128, 256)`.


```
   src_tiles  (the 4x4 grid above)

   .shape         = (4, 4)        ◀── how many tiles to iterate
   .element_shape = (512, 1024)   ◀── total elements addressed
   .tile_size     = (128, 256)    ◀── one tile's extent
```


Read tile counts from `.shape[d]` rather than recomputing `dim // TILE`. The floor division
hides remainders and duplicates information the view already holds; `.shape` is always
correct, including when the last tile is a partial (covered in Part 3).



## Slicing & indexing


Indexing a tile view is NumPy-style, but it operates at *tile* granularity and always
returns another `NDSlice` — never raw data, never a DMA. The common patterns:


| Pattern | Selects |
|---|---|
| `tiles[i, j]` | a single tile |
| `tiles[i]` / `tiles[i, :]` | one row of tiles (identical — trailing `:` is a no-op) |
| `tiles[:, j]` | one column of tiles |
| `tiles[a:b, c:d]` | a rectangular sub-grid |
| `tiles[-1]` | the last tile-row (negative indices allowed) |
| `tiles[::2]` | every other tile-row (stepped slices; step ≥ 1) |
| `tiles[0][2]` | chained indexing — the tile at `(0, 2)` |


```
   tiles[1, 2]          tiles[1, :]          tiles[:, 2]          tiles[1:3, 1:3]
   ┌──┬──┬──┬──┐        ┌──┬──┬──┬──┐        ┌──┬──┬██┬──┐        ┌──┬──┬──┬──┐
   │  │  │  │  │        │  │  │  │  │        │  │  │██│  │        │  │  │  │  │
   ├──┼──┼──┼──┤        ├──┼──┼──┼──┤        ├──┼──┼██┼──┤        ├──┼██┼██┼──┤
   │  │  │██│  │        │██│██│██│██│        │  │  │██│  │        │  │██│██│  │
   ├──┼──┼──┼──┤        ├──┼──┼──┼──┤        ├──┼──┼██┼──┤        ├──┼██┼██┼──┤
   │  │  │  │  │        │  │  │  │  │        │  │  │██│  │        │  │  │  │  │
   └──┴──┴──┴──┘        └──┴──┴──┴──┘        └──┴──┴██┴──┘        └──┴──┴──┴──┘
   one tile             one row              one column           2x2 sub-grid
```


Keys may be compile-time `int`s, `slice`s with step ≥ 1, and (as we will see in Part 4) NKI
runtime expressions for indirect access. Booleans, lists, dicts, strings, floats, fancy
indexing, boolean masks, and `...` are rejected — these reflect what a strided DMA can
actually express.


Indexing a dimension away behaves as it does for any multi-dimensional array: once
`row = tiles[i, :]` consumes dim 0, `row` is a lower-rank view and `row[j]` indexes the
surviving dimension. The precise rules for the trickier cases — block views and batch dims —
are in the `NDSlice.__getitem__` docstring.



## Data movement: `load()` and `store()`


A view describes *where* tiles live; `.load()` and `.store()` move them. `.load()` issues a
DMA from HBM into a freshly allocated SBUF tile and returns an SBUF-backed `NDSlice`.
`.store()` is its mirror, writing SBUF data back to the view's HBM region.


```python
tile = src_tiles[i, j].load()                 # one DMA: HBM -> SBUF
nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
dst_tiles[i, j].store(tile.data)              # one DMA: SBUF -> HBM
```


A loaded view exposes its underlying NKI tensor as `.data`: the same `nl.ndarray` you pass to
`nisa.*` compute ops, hand to `.store()`, or drop into a raw `nisa.dma_copy`. `.data` is also
the escape hatch behind the "transparent and overridable" promise — when the high-level methods
don't express what you need, it gives you the raw tensor to pair with a hand-written DMA.


Loading a *slice* of tiles coalesces them into a single DMA. This is **hoisting**: the whole
row or column lands in SBUF at once, packed contiguously along the free axis, and you index
the loaded view in SBUF with no further DMA.


```python
col = src_tiles[:, m].load()    # one coalesced DMA gathers the whole column
tile = col[k]                   # SBUF index -- no DMA
```


```
   src_tiles[:, m].load()   (column m of a 4xN grid, 128xF tiles)

     HBM (row-major)                         SBUF
     ┌──┬──┬──┬────┬──┐          r0 ┌────┬────┬────┬────┐
     │  │  │  │ t0 │  │             │ t0 │ t1 │ t2 │ t3 │  128 partitions
     ├──┼──┼──┼────┼──┤             └────┴────┴────┴────┘
     │  │  │  │ t1 │  │             │◀──── free dim ────▶│
     ├──┼──┼──┼────┼──┤
     │  │  │  │ t2 │  │          A single coalesced DMA gathers the
     ├──┼──┼──┼────┼──┤          non-contiguous column via a strided
     │  │  │  │ t3 │  │          access pattern; tiles pack along F.
     └──┴──┴──┴────┴──┘
```


Hoisting trades SBUF footprint for fewer DMAs and lets you reuse a stationary operand across
a loop — a pattern we lean on heavily in the matmul below and again in Part 2.


**Prefer one coalesced DMA over a per-tile loop.** NeuroTile coalesces a multi-tile slice
into a single DMA, so when you need several tiles resident at once, load the slice
(`tiles[:, m].load()`, `nt.blocks(...)[bi, bj].load()`) rather than looping `tiles[i, j].load()`
one tile at a time into the same SBUF region — far fewer DMA instructions for the same bytes.
This assumes the slice fits in SBUF; when it does not (or PSUM / other resources are the
binding constraint), fall back to streaming (Part 2) or per-tile loads. Coalesce by default;
loop per tile only when a resource limit forces it.



## The single-DMA contract


Every `.load()` and `.store()` (and every `stream.load(k)` / `stream.store(k)` in Part 2)
emits **exactly one `nisa.dma_copy`**. This is a deliberate guarantee, not a best effort.


> **The single-DMA contract.** If the region a call describes cannot be expressed as one
> coalesced DMA — for instance a multi-tile slice whose HBM layout needs more strided levels
> than a single NKI access pattern allows — the call raises at trace time. It never silently
> expands into a multi-DMA loop behind your back. The one documented exception is a
> partition-dimension `.fold()`, which issues one DMA per partition slice by hardware
> necessity (Part 3).


The contract is what makes NeuroTile's performance legible: the DMA count in your kernel is
the DMA count you wrote. When a coalesced load legitimately cannot be expressed as one DMA,
the fix is to drop down a level and load each tile in a loop — see Part 3 for the workaround.



## Iteration


NeuroTile views are not Python iterables in the usual sense — the NKI tracer rejects a bare
`for x in view:`. Instead you loop over `view.shape[d]` and index, which gives the body the
coordinate it needs to pair tiles with a destination, compute an offset, or cross-reference
another view:


```python
rows, cols = src_tiles.shape
for i in range(rows):
    for j in range(cols):
        tile = src_tiles[i, j].load()
        dst_tiles[i, j].store(tile.data)
```


The loop *primitive* you choose controls scheduling, and NeuroTile leaves that decision to
you — it does not wrap or rewrite your loops:


| Range | Behavior |
|---|---|
| `nl.affine_range(n)` | compiler may reorder / parallelize iterations (the usual choice) |
| `nl.sequential_range(n)` | strict in-order execution (needed for accumulation, pipelines) |
| `range(n)` | compile-time unroll — each iteration emits a distinct trace |



## A complete matmul


We now have enough to write a complete matmul. The inputs follow the NKI convention of a
transposed LHS: `lhsT` is `[K, M]`, `rhs` is `[K, N]`, and the result `C` is `[M, N]`. Each
output tile is the sum over the K dimension of `lhsT[k]ᵀ @ rhs[k]`, accumulated in PSUM.


```python
TILE_M, TILE_K, TILE_N = 128, 128, 512

@nki.jit
def matmul_baseline(lhsT_hbm, rhs_hbm):
    K, M = lhsT_hbm.shape
    _, N = rhs_hbm.shape
    C = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    lhsT_tiles = nt.tiles(lhsT_hbm, tile_size=(TILE_K, TILE_M))
    rhs_tiles  = nt.tiles(rhs_hbm,  tile_size=(TILE_K, TILE_N))
    C_tiles    = nt.tiles(C,        tile_size=(TILE_M, TILE_N))

    for m in range(C_tiles.shape[0]):
        for n in range(C_tiles.shape[1]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in nl.affine_range(lhsT_tiles.shape[0]):
                lhs_tile = lhsT_tiles[k, m].load()   # tile: [128, 128]
                rhs_tile = rhs_tiles[k, n].load()    # tile: [128, 512]
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C
```


Everything here is Part 1 material: three tile grids, `.load()` for data movement, `.shape`
to drive the loops, `nl.affine_range` over the K reduction, and `.store()` for the result.
The compute is plain `nisa.*` — `nc_matmul` into a PSUM accumulator, then a copy to SBUF
before the store, since PSUM is the matmul-output memory.


This version is correct but deliberately naive: it re-loads every LHS and RHS tile from HBM
on every step of the K loop, so a tile that could be reused is fetched many times. That
redundant traffic is exactly what Part 2's optimization vocabulary — hoisting, streaming,
and blocked accumulation — removes, using the very same kernel as the running example.



---

# Part 2 — Intermediate: the optimization vocabulary


Part 1's matmul works but moves far more data than it needs to. This part introduces the
vocabulary that closes that gap — blocks, hoisting, streaming, dedicated accumulator buffers,
and pipelining — and applies it to turn the naive matmul into a coalesced, double-buffered one.
The anchors are `examples/_01_iteration/{_02_blocks,_03_streaming,_04_pipeline}.py` and
`examples/_02_matmul/_02_matmul_coalesced.py`.



## Blocks


`nt.blocks()` adds a second level of grouping above the tile grid: a **block** is a
rectangular group of tiles that load and store together in one coalesced DMA. Its purpose is
to **decouple the DMA grain from the compute grain** — you move data a block at a time but
still compute one tile at a time. Think of `tile_size` as the slice of data you feed to an
`nisa.*` op — sized for the engine that consumes it — and `block_size` as how many of those
tiles you move per DMA, sized for transfer efficiency.


That separation matters because the same kernel typically runs across many tensor sizes. The
degree of parallelism (LNC core count, batch, sequence length) varies per deployment, which
changes how much SBUF a kernel can spend and what DMA grain is most efficient. `block_size`
lets you tune that grain directly — a larger block amortizes DMA overhead when SBUF is
plentiful, a smaller one keeps the footprint down when it isn't — without touching the compute
loop, which keeps iterating one tile at a time regardless. So a kernel author can dial in the
DMA grain for a specific model or mode by adjusting a single argument (or threading it through
config), rather than rewriting the loop structure or hardcoding model-specific sizes.


**The discipline: write kernels at two levels.** `tile_size` is irreducible — it is fixed by
the consuming `nisa.*` op (see *Choosing `tile_size`* in Part 1) — so you tune at the
**block** level, not by reshaping the compute. `block_size` is the general data-movement and
grouping knob: how many tiles move per DMA, how many you stage in SBUF at once, the unit you
shard across cores. Structure the kernel as an outer loop over blocks and an inner loop over
the tiles a loaded block exposes; that two-level scoping is what lets one kernel re-tune for
different shapes, SBUF budgets, and core counts by changing `block_size` alone.


```python
src_blocks = nt.blocks(src, tile_size=(128, 128), block_size=(2, 2))
src_blocks.shape          # (4, 2) -- block grid
src_blocks.block_size     # (2, 2) -- tiles per block
```


A `(1024, 512)` tensor with `tile_size=(128, 128)` is an 8×4 tile grid; grouping into 2×2
blocks gives a 4×2 block grid, where each block carries four tiles:


```
   tile grid: 8x4                         block grid: 4x2
   (each cell = one 128x128 tile)         (each cell = a 2x2 block of tiles)

   ┌──┬──┬──┬──┐                          ┌─────────┬─────────┐
   │  │  │  │  │                          │ blk(0,0)│ blk(0,1)│
   ├──┼──┼──┼──┤                          │ 2x2     │ 2x2     │
   │  │  │  │  │                          ├─────────┼─────────┤
   ├──┼──┼──┼──┤                          │ blk(1,0)│ blk(1,1)│
   │  │  │  │  │           group 2x2      │ 2x2     │ 2x2     │
   ├──┼──┼──┼──┤          ───────────▶    ├─────────┼─────────┤
   │  │  │  │  │                          │ blk(2,0)│ blk(2,1)│
   ├──┼──┼──┼──┤                          │ 2x2     │ 2x2     │
   │  │  │  │  │                          ├─────────┼─────────┤
   ├──┼──┼──┼──┤                          │ blk(3,0)│ blk(3,1)│
   │  │  │  │  │                          │ 2x2     │ 2x2     │
   └──┴──┴──┴──┘                          └─────────┴─────────┘
```


A block view supports the same indexing and slicing surface as a tile view from
[§1](#slicing--indexing) — it just operates at *block* granularity. You can take a single
block, a row or column of blocks, or a rectangular sub-grid, and `.load()` each as one
coalesced DMA; the wider the slice, the fewer and larger the DMAs (the grain knob from above):


```python
block      = src_blocks[bi, bj].load()   # a single block
block_row  = src_blocks[bi, :].load()    # a whole row of blocks, one coalesced DMA
block_col  = src_blocks[:, bj].load()    # a whole column of blocks
sub_blocks = src_blocks[0:2, 0:2].load() # a 2x2 sub-grid of blocks
```


What the result looks like depends on how far you indexed. Indexing **down to a single block**
auto-descends to that block's interior tile grid: `src_blocks[bi, bj].shape` reports the
*tile* grid inside the block, and the loaded view is tile-addressable as `block[ti, tj]`. A
**multi-block slice** stays block-structured — you index the surviving block dimension first,
then the interior tiles (`block_row[bj][ti, tj]`). Either way the loaded view already carries
the right structure; index it directly rather than passing it back through `nt.tiles()`.
(Part 3's view algebra covers explicitly changing the grain — promoting, demoting, and
re-tiling.)


So the canonical block loop is two-level: outer over blocks, inner over the tiles a single
loaded block exposes.


```python
for bi in range(src_blocks.shape[0]):
    for bj in range(src_blocks.shape[1]):
        block = src_blocks[bi, bj].load()      # one coalesced DMA for the 2x2 block
        for ti in range(block.shape[0]):       # block.shape is the interior tile grid
            for tj in range(block.shape[1]):
                nisa.tensor_scalar(block[ti, tj].data, block[ti, tj].data, nl.multiply, 2.0)
        dst_blocks[bi, bj].store(block.data)   # one coalesced DMA back
```



## Streaming


`.load()` brings data on-chip and *then* you compute on it — the DMA and the compute happen
one after the other. **Streaming** overlaps them. `.stream()` allocates a small pool of
rotating SBUF buffers and walks a view one step at a time, so while you compute on the data in
one buffer, the DMA for the next step fills another. This is the standard way to hide memory
latency behind compute for an operand you consume in order (the K-dimension of a matmul, a
sequence of weight blocks, an element-wise pass over a tensor).



### The rotating buffer


`.stream(buffer_count=N)` pre-allocates `N` SBUF buffers ("slots"), each sized to hold **one
step** of the walk. `stream.load(k)` issues the DMA for step `k` into slot `k % N`;
`stream[k]` returns a slot you loaded earlier without issuing a DMA. With `buffer_count=2`
(double-buffering), step `k` computes on one slot while step `k + 1`'s DMA fills the other:


```
   buffer_count=2 -- pipeline over a 4-step walk (time flows right →)

   time:       t0        t1        t2        t3        t4
            ┌─────────┬─────────┬─────────┬─────────┬─────────┐
   load     │ ld k0   │ ld k1   │ ld k2   │ ld k3   │         │
            │ →slot0  │ →slot1  │ →slot0  │ →slot1  │         │
            ├─────────┼─────────┼─────────┼─────────┼─────────┤
   compute  │         │ cm k0   │ cm k1   │ cm k2   │ cm k3   │
            │         │ @slot0  │ @slot1  │ @slot0  │ @slot1  │
            └─────────┴─────────┴─────────┴─────────┴─────────┘
                          ▲
        steady state: load(k+1) and compute(k) run in the same step,
        on the two different slots (k % 2). The first load fills the
        pipeline (prologue); the last compute drains it (epilogue).
```


```python
stream = rhs_tiles[:, n].stream(buffer_count=2)   # 2 slots, one tile each
for k in nl.affine_range(K_TILES):
    tile = stream.load(k)                          # DMA step k into slot k % 2
    nisa.nc_matmul(acc, lhs_tile.data, tile.data)  # compute overlaps the next load
```


`buffer_count=2` is the default and the right choice almost always; `3` (triple-buffering)
helps only when one stage can't hide the DMA latency, at the cost of an extra slot of SBUF.



### What one step covers — choosing the stream granularity


The single most important thing to understand about streaming is that **what one slot holds
is decided by the view you call `.stream()` on**. The same `.stream()` call streams a single
tile, a whole packed row or column, or an entire block — you choose by shaping the parent
view. The patterns:


**Stream the tiles of one column** — `tiles[:, m].stream()` walks down column `m`, one tile
per slot:


```
   tiles[:, 0].stream(buffer_count=2)        one tile per slot

        col 0
      ┌───────┐
   k0 │  t0   │ ──▶ slot0
      ├───────┤
   k1 │  t1   │ ──▶ slot1
      ├───────┤
   k2 │  t2   │ ──▶ slot0   (reused)
      ├───────┤
   k3 │  t3   │ ──▶ slot1
      └───────┘
```


```python
col_stream = tiles[:, m].stream(buffer_count=2)
for k in range(col_stream.count):
    tile = col_stream.load(k)        # slot holds one tile (tile_size)
```


**Stream the tiles of one row** — `tiles[i, :].stream()` walks across row `i`, one tile per
slot (the mirror of the column case):


```
   tiles[0, :].stream(buffer_count=2)        one tile per slot

       k0      k1      k2      k3
   ┌───────┬───────┬───────┬───────┐
   │  t0   │  t1   │  t2   │  t3   │   row 0
   └───┬───┴───┬───┴───┬───┴───┬───┘
       ▼       ▼       ▼       ▼
     slot0   slot1   slot0   slot1
```


**Stream entire columns (packed)** — `tiles.stream(dim=1)` makes each slot hold one whole
column of tiles, gathered in a single coalesced DMA. Use this when the column is your natural
work unit and the larger per-slot footprint is acceptable:


```
   tiles.stream(dim=1, buffer_count=2)       one packed COLUMN per slot

   ┌─────┬─────┬─────┬─────┐
   │(0,0)│(0,1)│(0,2)│(0,3)│      k=0  col 0 = (0,0)(1,0)(2,0)(3,0) ──▶ slot0
   ├─────┼─────┼─────┼─────┤      k=1  col 1 = (0,1)(1,1)(2,1)(3,1) ──▶ slot1
   │(1,0)│(1,1)│(1,2)│(1,3)│      k=2  col 2                        ──▶ slot0
   ├─────┼─────┼─────┼─────┤      k=3  col 3                        ──▶ slot1
   │(2,0)│(2,1)│(2,2)│(2,3)│
   ├─────┼─────┼─────┼─────┤      one coalesced DMA per column;
   │(3,0)│(3,1)│(3,2)│(3,3)│      index the loaded slot per tile: col[k]
   └─────┴─────┴─────┴─────┘
```


```python
col_stream = tiles.stream(dim=1, buffer_count=2)
for j in range(col_stream.count):
    col = col_stream.load(j)         # slot holds a whole column, packed
    for k in range(col.shape[0]):    # index the slot per tile
        nisa.nc_matmul(acc, lhs.data, col[k].data)
```


`tiles.stream(dim=0)` is the symmetric form — each slot holds one whole **row** of tiles,
packed.


**Stream blocks** — calling `.stream()` on a block view makes each slot hold one whole block
(`block_size` tiles), loaded in a single coalesced DMA. Walk a block-column with `blocks[:, b]`
or a block-row with `blocks[bi]`:


```
   blocks[:, 0].stream(buffer_count=2)       one BLOCK per slot

   block-col 0
   ┌────────────┐
   │  blk(0,0)  │ ──▶ slot0     each slot = block_size tiles,
   │  2x2 tiles │               one coalesced DMA
   ├────────────┤
   │  blk(1,0)  │ ──▶ slot1
   │  2x2 tiles │
   └────────────┘
```


```python
blk_stream = blocks[:, b].stream(buffer_count=2)
for bi in range(blk_stream.count):
    block = blk_stream.load(bi)      # slot holds a whole block
    for ti in range(block.shape[0]): # index the loaded block per interior tile
        for tj in range(block.shape[1]):
            nisa.tensor_scalar(block[ti, tj].data, block[ti, tj].data, nl.multiply, 2.0)
```


The table summarizes the choices; the wider the per-slot grain, the fewer (and larger) the
DMAs, at a proportionally larger SBUF footprint:


| Parent view | One slot holds | DMAs |
|---|---|---|
| `tiles[:, m]` | one tile | one per tile |
| `tiles[i, :]` | one tile (row walk) | one per tile |
| `tiles.stream(dim=1)` | one whole column of tiles, packed | one per column |
| `tiles.stream(dim=0)` | one whole row of tiles, packed | one per row |
| `blocks[:, b]` / `blocks[bi]` | one block (`block_size` tiles) | one per block |



### Output streaming


Streaming also works for results: write into a slot, then store it back. `stream[k]` gives
you a slot to fill, and `stream.store(k)` DMAs it to HBM at position `k`:


```python
out = out_tiles[:, n].stream(buffer_count=2)
for k in nl.affine_range(N):
    nisa.tensor_copy(out[k].data, result_k)   # fill slot k % 2
    out.store(k)                               # DMA the slot back to HBM
```


Part 3's [Streaming in depth](#streaming-in-depth) covers the remaining details — the `dim=`
default, the SBUF cost model, the load-type lock, and transpose streams.



## Hoisting vs streaming


With both data-movement strategies in hand, the choice between them comes down to how an
operand is used. **Hoist** (`tiles[:, m].load()`, from Part 1) a **stationary** operand —
one reused across a loop — so it stays resident and every reuse is a free SBUF index.
**Stream** a **flowing** operand — one consumed once in order — so its DMA overlaps compute
and only a couple of slots are resident at a time. They are complementary, and a typical
matmul uses both: hoist the stationary operand, stream the flowing one.


|  | `.load()` (hoisting) | `.stream(buffer_count=N)` |
|---|---|---|
| **SBUF cost** | every tile in the slice resident | `N` slots, one step each |
| **DMA/compute overlap** | no | yes |
| **Data reuse** | any tile, any time | each slot valid for one iteration |
| **Use for** | stationary operand, small total bytes | flowing operand, K-dimension iteration |


`.stream()` walks rows, columns, or blocks depending on the parent view you call it on; the
slot granularity and the full streaming surface are covered in [Part 3](#streaming-in-depth).
For now, the per-tile column stream above is all the matmul needs.



## Allocating SBUF/HBM buffers: `alloc_tiles` & `alloc_blocks`


`.load()` allocates its SBUF destination for you. When you need a buffer you own — an output
accumulator written across many iterations, then stored once — allocate it explicitly with
`nt.alloc_tiles` (tile-structured) or `nt.alloc_blocks` (block-structured). Both return an
`NDSlice` over freshly allocated memory. The `buffer_type=` argument chooses where it lives —
`nl.sbuf` for an on-chip buffer or `nl.shared_hbm` / `nl.private_hbm` for an HBM scratch
buffer — so the same factory serves both an SBUF accumulator and an HBM intermediate.


The motivation is to keep the *physical* allocation hidden behind a clean tile or block view.
Memory on the device is rarely laid out the way the grid suggests. A block-grid on SBUF, for
example, cannot be a literal 2-D grid: SBUF is partition-major, so the two-level (block, tile)
structure is flattened along the free dimension with strides chosen so that `acc[m, n]` still
resolves to the correct tile. `alloc_blocks` computes that layout, and you simply allocate by
grid shape and index by grid coordinate — never a flat offset by hand.


The `element_shape=` argument extends the same idea to ragged sizes. When an extent is not a
multiple of `tile_size`, the trailing tiles along P or F are allocated as remainders and
clamped for you, so the partial-tile bookkeeping that usually clutters a kernel never appears
in your code. Together these let an accumulator read the way the algorithm reads: declare it
by shape, `memset` it, write `acc[m, n].data` inside the natural loop, and store it once with
`acc.data`.


```python
acc = nt.alloc_tiles(
    tile_size=(128, 512),
    element_shape=(M, 1024),    # exact extent; trailing partial tiles auto-clamp
    buffer_type=nl.sbuf,
    dtype=nl.float32,
)
nisa.memset(acc.data, 0.0)
acc[m, n].data                  # a tile slot -- an nl.ndarray for nisa.*
acc.data                        # view over the whole buffer (for one coalesced store)
```


You size the buffer one of two ways, and they are mutually exclusive:


- **`grid=`** gives a tile-aligned extent (`grid × tile_size`). Use it when the extent is
  exactly tile-aligned.
- **`element_shape=`** gives the exact element extent and ceiling-divides into the tile grid,
  auto-clamping a partial trailing tile. Prefer this in general — the same code path then
  handles both tile-aligned and remainder cases.



## Mixed precision


Matmul takes bf16/fp8 inputs and accumulates in fp32 PSUM, so a kernel mixes dtypes — and you
cast at the DMA boundary, not with a separate op. `load(dtype=)` allocates the SBUF
destination at the requested dtype and the DMA hardware casts on the mismatch; `store(dtype=)`
mirrors it on the way out. The cast is allocation-driven — free at the copy.


```python
x   = src_tiles[i, j].load(dtype=nl.bfloat16)                    # HBM fp32 -> SBUF bf16, cast in the DMA
acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)   # accumulate in fp32
dst_tiles[i, j].store(acc, dtype=out.dtype)                     # fp32 -> output dtype on store
```


See `examples/_04_precision/_01_mixed_precision.py`.



## PSUM accumulators: `psum_pool` (intro)


PSUM is the matmul-output memory: `nisa.nc_matmul` writes there with hardware-accumulate
semantics, across eight hardware banks. The `alloc_*` factories above deliberately reject
`nl.psum` — PSUM has its own factory, `nt.psum_pool`, because placing tiles onto banks is its
own concern. For a single one-off accumulator you don't need it; allocate a plain
`nl.ndarray(..., buffer=nl.psum)`, as the matmuls in this guide do.


When you need *several* PSUM tiles addressed as a grid, `nt.psum_pool` allocates them across
the banks and returns an `NDSlice` — so a pool is a first-class grid you index just like any
other view (`psums[s, i].data`), not a flat list you track by hand:


```python
psums = nt.psum_pool(tile_size=(128, 512), element_shape=(256, 2048))
for s in range(2):
    for i in range(4):
        nisa.nc_matmul(psums[s, i].data, x, w[s, i])
```


The motivation is the same readability win as `alloc_tiles`: you index the pool by its natural
grid coordinate (`psums[s, i]`) and let the factory map that to the underlying bank, instead of
flattening the tile/block structure into a single list and recomputing a linear index
(`psums[s * 4 + i]`) at every use. The loop reads the way the algorithm does.


One difference from `alloc_tiles` / `alloc_blocks` is worth noting: although the pool is an
`NDSlice`, **each tile is its own `nl.ndarray`** — there is no single contiguous buffer behind
it (PSUM banks aren't one linear region the way an SBUF allocation is). So you operate on the
pool one tile at a time via `psums[s, i].data`; there is no whole-pool `.data` view to store in
one shot the way an SBUF accumulator has.


How tiles map onto banks is controlled by two optional arguments. With neither, the compiler
picks banks for you. Passing `bank_ids=` (a tuple of bank indices) pins placement: by default
every tile gets its own bank, and adding `bank_axis=` fans that axis across the listed banks
while packing the remaining tiles as slots within each bank — useful for small tiles (e.g.
128×128), where several fit in one of the eight banks. Production kernels usually pass explicit
`bank_ids` to control accumulator placement; the MLP CTE kernel does exactly this for its
gate/up, down, and transpose pools. The placement modes and their sizing rules are detailed in
the `nt.alloc_tiles` / `nt.psum_pool` docstrings and illustrated in the
[MLP CTE walkthrough](#walkthrough-mlp-cte--swiglu); for now, treat `psum_pool` as "a grid of
PSUM tiles."



## The canonical matmul


We can now rewrite Part 1's matmul as a blocked, coalesced kernel. Kernel makes use of `nt.blocks`
for coalesced DMAs for load as well as for storing the result back. The left operands is loaded
whereas right operand is streamed (doubled-buffered). `element_shape=` arg on `nt.alloc_tiles`,
`nt.psum_pool` handles the remainder for free.


```python
@nki.jit
def matmul_blocked(lhsT, rhs, BLOCK_M=2, BLOCK_N=2, BLOCK_K=4):
    TILE_M, TILE_K, TILE_N = 128, 128, 512
    K, M = lhsT.shape
    lhsT_blocks = nt.blocks(lhsT, tile_size=(TILE_K, TILE_M), block_size=(BLOCK_K, BLOCK_M))
    rhs_blocks  = nt.blocks(rhs,  tile_size=(TILE_K, TILE_N), block_size=(BLOCK_K, BLOCK_N))
    out_blocks  = nt.blocks(C,    tile_size=(TILE_M, TILE_N), block_size=(BLOCK_M, BLOCK_N))

    for m_blk in range(lhsT_blocks.shape[1]):
        # load M-column, demote to tiles
        lhsT_tiles = nt.tiles(lhsT_blocks[:, m_blk].load())
        for n_blk in range(rhs_blocks.shape[1]):
            out_block = out_blocks[m_blk, n_blk]
            psums = nt.psum_pool(tile_size=(TILE_M, TILE_N), element_shape=out_block.element_shape)
            acc   = nt.alloc_tiles(tile_size=(TILE_M, TILE_N), element_shape=out_block.element_shape,
                                    buffer_type=nl.sbuf, dtype=C.dtype)
            n_m_tiles, n_n_tiles = acc.shape

            # Stream the N-block's K-blocks (double-buffered: DMA of k+1 overlaps matmul on k).
            rhs_stream = rhs_blocks[:, n_blk].stream(buffer_count=2)
            for k_blk in nl.affine_range(rhs_blocks.shape[0]):
                rhs_tiles = nt.tiles(rhs_stream.load(k_blk))
                for bm in range(n_m_tiles):
                    for bn in range(n_n_tiles):
                        for bk in range(rhs_tiles.shape[0]):
                            nisa.nc_matmul(psums[bm, bn].data,
                                            lhsT_tiles[k_blk * BLOCK_K + bk, bm].data,
                                            rhs_tiles[bk, bn].data)

            # Evict PSUM -> SBUF, then one coalesced block store.
            for bm in range(n_m_tiles):
                for bn in range(n_n_tiles):
                    nisa.tensor_copy(acc[bm, bn].data, psums[bm, bn].data)
            out_block.store(acc.data)
    return C
```



## Software pipelining


Streaming with `nl.affine_range` already lets the compiler overlap DMA and compute. When you
want explicit control over the overlap, you write the pipeline phases yourself with
`nl.sequential_range`. The deepest common form is a **3-stage pipeline** that keeps a load, a
compute, and a store in flight at once, on three different slots — so it uses an input *and* an
output stream with `buffer_count=3`. Each steady-state step loads `k`, computes `k-1`, and
stores `k-2`; a prologue fills the first two slots and an epilogue drains the last two:


```python
src_stream = src_tiles[:, 0].stream(buffer_count=3)
dst_stream = dst_tiles[:, 0].stream(buffer_count=3)

# PROLOGUE: load tiles 0 and 1, compute tile 0
src_stream.load(0)
src_stream.load(1)
nisa.tensor_scalar(dst_stream[0].data, src_stream[0].data, nl.multiply, 4.0)

# STEADY STATE: load(k), compute(k-1), store(k-2) -- three slots in flight
for k in nl.sequential_range(2, N_TILES):
    src_stream.load(k)                                              # stage 1: load
    nisa.tensor_scalar(dst_stream[k - 1].data, src_stream[k - 1].data, nl.multiply, 4.0)  # stage 2: compute
    dst_stream.store(k - 2)                                         # stage 3: store

# EPILOGUE: drain the last two compute/store stages
dst_stream.store(0)
nisa.tensor_scalar(dst_stream[N_TILES - 1].data, src_stream[N_TILES - 1].data, nl.multiply, 4.0)
dst_stream.store(N_TILES - 1)
dst_stream.store(N_TILES - 2)
```


Here `stream.load(k)` issues the DMA into slot `k % 3`, while `stream[k]` returns a slot view
*without* a DMA — used to read a tile loaded on a previous step or to fill an output slot
before its store. With three slots, the load, compute, and store of three consecutive steps
never touch the same buffer, so all three overlap:


```
   triple buffer (buffer_count=3), steady state:

   k    load(k) ─▶ slotₖ     compute(k-1) on slotₖ₋₁     store(k-2) from slotₖ₋₂
        └──────────── three different slots active simultaneously ───────────┘
```


A shallower **2-stage** form (`buffer_count=2`) drops the store stage — prime the first load,
then prefetch `k + 1` while computing `k` — which is the explicit-prefetch shape the matmul
above uses for its `rhs` stream.


| Pattern | `buffer_count` | Loop primitive | Use case |
|---|---|---|---|
| Implicit streaming | 2 | `affine_range` | simple streaming, no manual control |
| Explicit prefetch | 2 | `sequential_range` | matmul with hoist + stream |
| 3-stage pipeline | 2 | `sequential_range` | element-wise load/compute/store |
| Triple buffer | 3 | `sequential_range` | high-latency DMA, deeper overlap |


Reach for `buffer_count=3` only when two stages cannot hide the DMA latency — the extra slot
costs SBUF.



---

# Part 3 — Advanced: production concerns


These sections rest on the same fact: a view is an `NDSlice`, and `nt.tiles` / `nt.blocks`
accept an `NDSlice` as their source, so a view can be re-shaped into another view without
ever moving data.



## View algebra: re-tile, promote, demote


Because `nt.tiles()` and `nt.blocks()` accept an existing view as their `source`, you can
restructure a grid in place.


- **Re-tile** — `nt.tiles(view, tile_size=...)` rebuilds the grid at a new tile granularity.
- **Promote** — `nt.blocks(view, block_size=...)` groups an existing tile grid into blocks
  (the tile size is inherited; only `block_size=` is needed).
- **Demote** — `nt.tiles(view)` with *no* `tile_size=` drops the block grouping and
  iterates the underlying tile grid directly.


```python

# Promote a tile view to a block view.
src_tiles  = nt.tiles(src, tile_size=(128, 128))         # (8, 4) tile grid
src_blocks = nt.blocks(src_tiles, block_size=(2, 2))     # (4, 2) block grid, tile_size inherited

# Demote a block view back to its tile grid (drop the block axis).
src_tiles  = nt.tiles(src_blocks)                        # (8, 4) tiles again

# Re-tile a view at a finer grain (block grouping dropped, F halved 128 -> 64).
src_tiles  = nt.tiles(src_blocks, tile_size=(128, 64))   # (8, 8) tiles of (128, 64)
```


```
   re-tile (128,128) -> (128,64)        promote (tiles -> blocks)        demote (blocks -> tiles)

   ┌────┬────┐      ┌──┬──┬──┬──┐        ┌──┬──┐    ┌───────┐            ┌───────┐    ┌──┬──┐
   │    │    │      │  │  │  │  │        │  │  │    │ blk   │            │ blk   │    │  │  │
   ├────┼────┤  ──▶ ├──┼──┼──┼──┤        ├──┼──┤──▶ │ 2x2   │            │ 2x2   │──▶ ├──┼──┤
   │    │    │      │  │  │  │  │        │  │  │    └───────┘            └───────┘    │  │  │
   └────┴────┘      └──┴──┴──┴──┘        └──┴──┘                                      └──┴──┘
   grid rebuilt at finer F              tiles grouped into blocks      block grouping dropped
```


Both demote and re-tile drop the block grouping (`block_size` becomes `None`), and both
**preserve any outer shard or broadcast axes** the view carried — which is what makes the
shard-then-promote pattern below work. One restriction worth knowing now: re-tiling a view
that was sharded round-robin can only *subdivide* (a smaller tile that evenly divides the
old), never *aggregate* across the shard gaps; the full rule is in the `nt.tiles`
docstring.


> Do not confuse this with re-tiling a *loaded* view. A loaded SBUF view from `.load()` or
> `stream.load(k)` already exposes its interior tile grid — index it directly
> (`block[ti, tj]`); don't pass it back through `nt.tiles()`.



## Shard-then-promote


A common production composition combines sharding and blocking: first narrow a tile view to
this core's owned tiles by slicing, then promote the owned tiles into a block grid for
coalesced DMA. This composes even under **interleaved (round-robin) sharding**, where a core's
owned tiles are *non-contiguous* in HBM — `nt.blocks` groups the owned tiles densely, ignoring
the shard gap, because promotion preserves the outer shard axis:


```python
own    = nt.interleaved_range(rank=nl.program_id(0), num_shards=nl.num_programs(0),
                              total=nt.ceiling_div(M, 128))
view   = nt.tiles(hbm, tile_size=(128, 512))[own, :]     # round-robin owned tiles (rows 0, 2, ...)
blocks = nt.blocks(view, block_size=(2, 1))              # group owned tiles into blocks
```


```
   full tile grid (4x1)      [own, :] -> core 0's           nt.blocks(block_size=(2,1))
   shared across 2 cores     every-other row (gapped)       groups owned tiles densely

   ┌──┐                      ┌──┐                            ┌─────┐
   │  │  row 0               │█0│  core 0  ◀ owned           │ blk │  rows 0+2,
   ├──┤                      ├──┤                            │     │  packed dense
   │  │  row 1               │·1│  core 1                    └─────┘
   ├──┤        ──▶           ├──┤              ──▶            core 0's 2 owned
   │  │  row 2               │█0│  core 0  ◀ owned            (non-adjacent) tiles
   ├──┤                      ├──┤                             -> 1 block; the gap
   │  │  row 3               │·1│  core 1                     is skipped, not stored
   └──┘                      └──┘
```


The promoted block holds core 0's owned tiles (rows 0 and 2) packed contiguously — the
round-robin gap left by core 1's rows is not allocated. The same works for SBUF sources and
for the contiguous helpers; sharding itself — the `*_range` helpers — is covered in full below.



## Sub-tile narrowing


Once indexing reaches a single tile, you can slice *into* it. Sub-tile slicing narrows the
view's `element_shape` but leaves `tile_size` unchanged — and `.load()` allocates SBUF sized
to the narrowed `element_shape`, not the declared tile.


```python
tile = src_tiles[i, j]              # element_shape == tile_size == (128, 256)
narrow = tile[:, 0:64]             # element_shape (128, 64); tile_size still (128, 256)
narrow.load()                      # allocates (128, 64) in SBUF
```


```
   tile (128, 256)            tile[:, 0:64]
   ┌───────────────┐          ┌────┬──────────┐
   │               │          │////│          │
   │  element_shape│   ──▶    │////│  ignored │   element_shape = (128, 64)
   │  = (128, 256) │          │////│          │   tile_size      = (128, 256)  (unchanged)
   └───────────────┘          └────┴──────────┘   .load() allocates (128, 64)
```


This is the mechanism behind per-slice compute on a higher-rank tile, and it underpins the
pre-load composition pattern in [Part 4](#sub-tile-indexing-pre-load-vs-post-load).



## Higher-dimensional tiles


Nothing so far required a tensor to be 2-D. NeuroTile tiles tensors of **any rank**, and a
`tile_size` can itself be higher-rank — a 3-D or 4-D tile carves the source the same way a 2-D
tile does, just with more axes. The first `tile_size` dim is always the partition (P) axis;
the remaining dims are F-axis subdivisions, kept as logical structure so you can address them
independently.


```python
src_tiles = nt.tiles(src, tile_size=(128, 4, 32))   # src.shape == (256, 4, 32) -> 3-D tile grid
src_tiles.shape                                      # (2, 1, 1) -- ceil-div per dim
tile = src_tiles[i, j, k].load()                     # one DMA -> a [128, 4, 32] tile in SBUF
```


The grid and indexing generalize exactly as you'd expect: an N-D `tile_size` gives an N-D tile
grid, you index it with N coordinates, and a loaded tile keeps its N-D logical shape. The
F-subdivisions are then addressable with the sub-tile slicing from above — e.g. operate on each
`F1` slab of a `(P, F1, F2)` tile separately:


```python
for i in range(src_tiles.shape[0]):
    for j in range(src_tiles.shape[1]):
        for k in range(src_tiles.shape[2]):
            tile = src_tiles[i, j, k].load()    # [128, 4, 32]
            for f1 in range(tile.element_shape[1]):
                slab = tile[:, f1, :]           # [128, 1, 32] -- one F1 subdivision
                nisa.tensor_scalar(slab.data, slab.data, nl.multiply, 2.0)
            dst_tiles[i, j, k].store(tile.data)
```


A higher-rank `tile_size` and a higher-rank *source* are independent choices. A 2-D
`tile_size` on a 3-D source treats the leading dim as a batch axis (see
[Batch dimensions](#batch-dimensions) below); a 3-D `tile_size` on a 3-D source produces a 3-D
tile per grid position. Use the rank that matches how you want to address the data — and
reshape or permute the source tensor (see [Transform before tiling](#transform-before-tiling),
later in this part) into the layout your tile rank expects before tiling.


The reason a higher-rank tile can keep its logical shape while still being a normal SBUF tile
is covered next.



## Logical rank vs physical 2-D SBUF backing


A tile can be logically higher-rank — `tile_size=(P, F1, F2)` gives a 3-D tile you can slice
on `F1` and `F2` independently. Physically, SBUF is always 2-D: the tile is stored as
`(P, F1*F2)`, and `tile.data.shape` reports that 2-D shape. Logical slices flatten through
the 2-D backing internally; you address the logical structure, the hardware sees the flat F.


```python
src_tiles = nt.tiles(src, tile_size=(128, 2, 32))   # logical 3-D tile
tile = src_tiles[i, j, k].load()                     # logical shape (128, 2, 32)
slice0 = tile[:, 0, :]                                # (128, 1, 32) -- logical narrow
nisa.tensor_scalar(slice0.data, slice0.data, nl.multiply, 2.0)
```


```
   logical tile (P, F1, F2) = (128, 2, 32)        physical SBUF (P, F1*F2) = (128, 64)
   ┌──────────┬──────────┐                        ┌─────────────────────────┐
   │  F1=0    │  F1=1    │   128 partitions       │  [F1=0 | F1=1]  64 wide │  128 partitions
   │  (32)    │  (32)    │                        └─────────────────────────┘
   └──────────┴──────────┘                         tile.data.shape == (128, 64)
   tile[:, 0, :] addresses the left half; the backing stays 2-D.
```


Keeping the logical rank lets the F subdivisions stay addressable without flattening at the
source. To map a *different* source dim onto P, reshape or permute the source first (next
sections), or transpose in-DMA.



## Batch dimensions


When `tile_size` has fewer dimensions than the source, the leading source dimensions become
**batch dimensions**. NeuroTile left-pads `tile_size` with `1`s to match the source rank, so
each batch dim contributes one iteration level that you index down with a single coordinate
before loading.


```python

# src.shape == (B, M, N); tile_size (P, F) is auto-padded to (1, P, F).
src_tiles = nt.tiles(src, tile_size=(128, 128))      # shape (B, M//128, N//128)
for b in range(B):
    for i in range(M // 128):
        for j in range(N // 128):
            tile = src_tiles[b, i, j].load()          # batch consumed before load
```


```
   src (B, M, N), tile_size=(128,128) auto-padded to (1, 128, 128)

   b=0   ┌──┬──┐      batch dim is a slab selector:
         │  │  │      src_tiles[b, i, j] picks slab b, then tile (i, j)
   b=1   ├──┼──┤
         │  │  │      .load() asserts every batch dim is consumed first
   b=2   └──┴──┘
```


Two rules follow:


- **`tile_size` must be at least 2-D.** A 1-D `tile_size=(N,)` is rejected — write `(N, 1)`
  for a partition-column tile or `(1, N)` for a free-row tile so the P/F orientation is
  explicit.
- **Batch dims must be consumed before `.load()` / `.store()` / `.stream()`**, which assert
  this. Index them away first.


There is one ergonomic shortcut. When the *only* iteration level is the batch dim and the
trailing tile grid is `(1, ..., 1)` — e.g. `tile_size=(P, F)` on `src.shape=(B, P, F)` —
indexing the batch lands directly at single-tile element level. Call `.load()` on the slab;
do not subscript further:


```python
src_tiles = nt.tiles(src, tile_size=(128, 16))   # src.shape == (4, 128, 16)
slab = src_tiles[0]                               # already a single tile
tile = slab.load()                                # NOT slab[0, 0].load()
```



## Data movement in depth


Part 1 used the bare `.load()` / `.store()`. Both carry a full option surface for production
use. The options, with the rule each obeys:


| Option (`.load`) | Effect |
|---|---|
| `dtype=` | allocate the SBUF destination at this dtype; the DMA hardware casts on a mismatch (allocation-driven, no separate cast op) |
| `transpose=` / `transpose_axes=` | DMA-transpose path (see [Transpose](#transpose)); `transpose_axes` is gather-only |
| `dst=` | load into a caller-provided SBUF destination — a raw `nl.ndarray` *or* a `.data` view |
| `oob_mode=` / `oob_value=` | suppress out-of-bounds DMA faults / pre-fill (see [Remainder](#remainder-handling)) |
| `priority=` | DMA QoS level, int in `[0, 3]` (lower = higher priority) |
| `pattern_override=` / `out_shape=` | custom HBM access pattern + its SBUF shape (see [Access patterns](#access-patterns)) |
| `dge_mode=` | DMA generation-engine hint (`hwdge` / `swdge` / `unknown`) |


`.store()` mirrors the relevant ones (`dtype=`, `oob_mode=`, `priority=`, `pattern_override=`,
`dge_mode=`). Its `data` argument must be an `nl.ndarray` or a `.data` view — not a bare
`NDSlice`; pass `view.data`.


The `dst=` option deserves special attention because it composes: a `dst=` target can be the
`.data` slot of a pre-allocated buffer, which lets you assemble a larger buffer one
transpose-load at a time. For example, transpose-loading each owned tile into its slot of an
`alloc_blocks` buffer:


```python
xblk = nt.alloc_blocks(tile_size=(H0, H1 * TS), block_size=(BS, 1), grid=(1, 1),
                       buffer_type=nl.sbuf, dtype=x.dtype)
for ti in range(BS):
    src_tiles[ti, 0].load(transpose=True, dst=xblk[0, 0][ti].data)   # one tile -> one slot
```


```
   per-tile transpose-load into block slots (dst=)

   src tile 0 ──load(transpose=True, dst=slot0)──▶ ┌────┐
   src tile 1 ──load(transpose=True, dst=slot1)──▶ │slot│  assembled block buffer
   src tile 2 ──load(transpose=True, dst=slot2)──▶ │ s  │  (one alloc_blocks, filled
                                                   └────┘   tile-by-tile)
```


Recall the [single-DMA contract](#the-single-dma-contract) from Part 1: each of these calls
is one `nisa.dma_copy`. When a multi-tile `.load()` cannot be expressed as one coalesced DMA,
it raises at trace time rather than silently fanning out. The workaround is to drop a level
and load each tile in a loop:


```python

# If a coalesced column load raises (layout won't fit one DMA), loop per tile:
col = tiles[:, m]
for k in range(col.shape[0]):
    tile = col[k].load()    # one single-DMA load each
```


The lone exception is a partition-dimension `.fold()`, which issues K DMAs by hardware
necessity — covered under [Fold](#fold).



## Streaming in depth


Part 2's [Streaming](#streaming) section covered the rotation mechanics and the granularity
patterns — streaming tiles of a row/column, whole packed rows/columns, and blocks. This
section adds the three details that matter once you push streaming hard: how the streamed
axis is chosen, what the SBUF footprint actually is, and the constraint that one stream
serves one load type.


**The `dim=` default.** `.stream()` walks the **outer dimension still being iterated** — the
first one not yet indexed away. On a fresh view that is dim 0 (or the first non-batch dim); on
a child of `parent[i, :]` it is the surviving dim. This is why `blocks[bi].stream(...)` walks
the surviving block dimension with no explicit `dim=`, and why `tiles[:, m].stream()` walks the
column. Pass `dim=` only to override that default — e.g. `tiles.stream(dim=1)` to walk whole
columns of a full grid instead.


**SBUF cost.** The footprint is `buffer_count × (one step's elements)`, and "one step" is
exactly the per-slot grain from Part 2's table:


```
   total SBUF = buffer_count × (one step's elements)

   tiles[:, m].stream(buffer_count=2)       2 × prod(tile_size)
   tiles.stream(dim=1, buffer_count=2)      2 × (rows × tile_size[0]) × tile_size[1]
   blocks[:, b].stream(buffer_count=2)      2 × prod(block_size × tile_size)
```


A per-tile stream is cheap; a packed-column or block stream can be large. Use `buffer_count=2`
by default and size the slot against your SBUF budget before reaching for `3`.


**Load-type lock.** A stream's `(transpose, transpose_axes)` is fixed either at construction
(`stream(transpose=...)`) or by the first `stream.load()`, and every rotating slot is sized
once accordingly. All later loads on that stream must pass the *same* transpose settings —
you cannot mix a transpose and a non-transpose load into one pool. Use one stream per load
type. This is why transpose streams (below) declare `transpose=True` up front.



## Multi-core (LNC) sharding


On Logical NeuronCore (LNC) mode, multiple cores run the same kernel over different slices of
the data. In NeuroTile, **sharding is slicing**: you narrow a view to this core's owned tiles
with a slice helper, and the rest of the kernel is unchanged.


The payoff is that **the algorithm code is identical whether it iterates a sharded view or the
full one.** A sharded view's `.shape` reports this core's tile count and its indexing addresses
this core's tiles, so the loop body is byte-for-byte what a single-core kernel would write — the
only difference is the slice applied at view construction. There is no separate multi-core code
path, no per-rank offset arithmetic threaded through the loops, and no rank-conditional
branching in the body: drop the slice and the same kernel runs single-core; add it and the same
kernel shards across LNC cores. Sharding becomes a one-line decision at view construction rather
than a rewrite of the compute.



### The shard helpers


Three helpers build the slice, each returning a plain Python `slice`:


```python
nt.block_range(rank, num_shards, total)        # contiguous block per core
nt.uneven_block_range(rank, num_shards, total)  # contiguous, remainder to early cores
nt.interleaved_range(rank, num_shards, total)   # round-robin (stride = num_shards)
```


```
   4 tiles, 2 cores

   block_range:                    interleaved_range:
   core 0: [0, 1]  core 1: [2, 3]  core 0: [0, 2]  core 1: [1, 3]
   ┌──┬──┬··┬··┐                   ┌──┬··┬──┬··┐
   │█0│█1│  │  │                   │█0│  │█2│  │   core 0 owns every 2nd tile
   └──┴──┴──┴──┘                   └──┴──┴──┴──┘
```


`total` counts **iteration units** on the shard dimension: tiles for a `nt.tiles` view,
blocks for a `nt.blocks` view (use `nt.ceiling_div` to compute it). Apply the slice and loop
over the narrowed view with *local* indices — the slice carries the offset:


```python
shard = nt.block_range(rank=nl.program_id(0), num_shards=nl.num_programs(0),
                       total=nt.ceiling_div(M, TILE_P))
A_tiles = nt.tiles(a, tile_size=(TILE_P, TILE_F))[shard, :]   # this core's rows
for i in range(A_tiles.shape[0]):                              # i is LOCAL
    for j in range(A_tiles.shape[1]):
        a_tile = A_tiles[i, j].load()
```



### Shard vs replicate


Only the views that touch the sharded axis get sliced; the rest stay whole. In a matmul sharded
on M, the output (`M` on dim 0) and `lhsT` (`M` on dim 1) are sliced, but `rhs` (`K × N`) is
replicated because every core needs the full reduction space:


```python
shard = nt.block_range(rank=nl.program_id(0), num_shards=nl.num_programs(0),
                       total=nt.ceiling_div(M, TILE_M))
lhsT_tiles = nt.tiles(lhsT, tile_size=(TILE_K, TILE_M))[:, shard]   # M on dim 1
rhs_tiles  = nt.tiles(rhs,  tile_size=(TILE_K, TILE_N))             # not sharded
C_tiles    = nt.tiles(C,    tile_size=(TILE_M, TILE_N))[shard, :]   # M on dim 0

```



### Custom slicing


The helpers are conveniences — each just returns a plain Python `slice`, and any `slice` works.
`slice(start, stop, step)` is Python's built-in for the `start:stop:step` indexing syntax
(`slice(0, 4)` is `[0:4]`, `slice(1, None, 2)` is `[1::2]`), so when none of the three
distributions fits, build the shard range yourself with the same arithmetic:


```python
own = M // 128 // nl.num_programs(0)                    # tiles per core
start = nl.program_id(0) * own
view = nt.tiles(src, tile_size=(128, 512))[slice(start, start + own), :]
```


This is the same thing `block_range` computes internally; the helpers exist so you rarely have
to. Because a `slice` is just data, you can also build one from a runtime expression (a
`program_id`-derived `start`) and apply it the same way — the view handles the offset.



### Runtime vs compile-time rank


The `rank` you pass to a shard helper can be either a compile-time `int` or a **runtime**
scalar such as `nl.program_id(0)`, and both are fully supported — including runtime-rank
sharding, which is the usual case for an SPMD launch where each core only learns its id at
run time.


The difference is purely internal. With a compile-time `int` rank, the slice's `start` is a
constant, so the offset folds into the view's layout at trace time — the DMA addresses are
fixed in the emitted code. With a runtime rank, `start` is a runtime expression, so there is no
constant to fold; NeuroTile instead routes that offset through the **indirect-indexing path**
(the same `scalar_offset` mechanism from [Part 4](#dynamic-selection)), and the DMA resolves
the per-core base address at run time. You do not write any of that — passing `nl.program_id(0)`
is the entire difference:


```python
# Compile-time rank: offset folds into the layout (e.g. a fixed shard known at trace time).
view = nt.tiles(src, tile_size=(128, 512))[nt.block_range(0, num_shards=2, total=n_tiles), :]

# Runtime rank: program_id is known only at run time; the library resolves the base
# address via indirect indexing -- the kernel body is otherwise unchanged.
shard = nt.block_range(rank=nl.program_id(0), num_shards=nl.num_programs(0), total=n_tiles)
view  = nt.tiles(src, tile_size=(128, 512))[shard, :]
```


Both produce a view you iterate with local indices exactly as above; the runtime case simply
emits an indirect DMA instead of a fixed-offset one. One caveat: `uneven_block_range` with a
runtime rank requires `total % num_shards == 0` — the uneven distribution gives different ranks
different owned counts, which can't be derived from a runtime scalar, so use it with a
compile-time rank when the split is ragged.


A few more practical notes, with the full contract in the shard-helper docstrings:


- **Build the slice in the kernel body.** Module-level `slice` constants do not trace.
- `nt.get_shard_info(tensor_shape, tile_size, shard_dim=)` returns a diagnostic dict
  (`total_tiles`, `tiles_per_shard`, …) for asserting the expected partition structure — it is
  not on the runtime DMA path.
- Runnable: `examples/_06_multicore/_01_tensor_add.py` and `_02_matmul.py`.



## View transforms


Every `NDSlice` carries a family of **metadata-only** transforms that restructure the logical
shape and strides without moving data. They work uniformly on HBM views (before `.load()`)
and SBUF views (after), and they chain.


| Method | Effect | Example |
|---|---|---|
| `.reshape_dim(dim, shape)` | split one dimension | `(128, 512)` → `(128, 4, 128)` |
| `.flatten_dims(start, end)` | merge contiguous dims | `(128, 4, 128)` → `(128, 512)` |
| `.permute(dims)` | reorder dimensions | `(128, 4, 8)` → `(8, 128, 4)` |
| `.rearrange(src, dst, sizes)` | einops split+reorder+merge in one call | `(512, 128)` → `(128, 4, 128)` via `(ni p) q → p ni q` |
| `.split(dim, n)` | split a dim into n chunks (shorthand for `reshape_dim`) | `(128, 512)` → `(128, 2, 256)` |
| `.slice(dim, start, end)` | narrow an element range | `(128, 512)` → `(128, 256)` |
| `.broadcast(dim, size)` | stride-0 broadcast of a size-1 dim | `(128, 1, 64)` → `(128, 16, 64)` |
| `.expand_dim(dim)` / `.squeeze_dim(dim)` | insert / remove a size-1 dim | `(128, 512)` ↔ `(128, 1, 512)` |
| `.reshape(new_shape)` | full contiguous reshape | `(128, 512)` → `(128, 2, 256)` |


**A transform collapses the view to a single tile.** Every transform above resets
`tile_size` to the new `element_shape` and makes the grid `(1, ...)` — **any prior tile or
block grid is discarded** (you can no longer index the old tiles). This is why the pattern is
*transform first, then tile*: to grid the transformed layout, call `nt.tiles()` / `nt.blocks()`
on the result. The lone exception is `.fold()`, which carries a DMA recipe instead of
collapsing (see [Fold](#fold)). The collapse is also the point of a coalesced store: a
transformed tile-column is one tile, so it stores in a single DMA (see [Transpose](#transpose)).


A chain restructures the layout before a single DMA does the move. The classic case turns
`[BxS, H]` into the partition layout a normalization kernel wants:


```python
view = (src_tiles[0, 0]
        .reshape_dim(dim=1, shape=[H1, H0])      # (4, 1024) -> (4, 8, 128)
        .flatten_dims(start_dim=0, end_dim=1))   # -> (32, 128)
tile = view.load(transpose=True)                  # one DMA executes the strided read
```


```
   reshape_dim(1, [8,128])      flatten_dims(0,1)
   (4, 1024)  ──────────────▶   (4, 8, 128)  ──────────────▶   (32, 128)
   shape/strides updated as metadata; data is untouched until .load()
```


`.broadcast()` is worth calling out: it sets a stride of 0 on a size-1 dimension so the same
source element is reused `size` times with no copy — for example, broadcasting a per-channel
scale or an inverse-RMS factor across a tile in a normalization kernel.


**`.rearrange(src_pattern, dst_pattern, fixed_sizes=None)`** is the einops-style one-liner for
the whole `reshape_dim` → `permute` → `flatten_dims` chain: name the axes on each side and let
matching names drive the split/reorder/merge. A grouped tuple on the `src` side splits that
dim; a grouped tuple on the `dst` side merges its members. It reads as the intent, not as
strides:


```python
# (ni p) q -> p ni q : split the row axis into nc bands of 128, lift the band axis out front.
dst = out_tiles[:, mi].rearrange((("ni", "p"), "q"), ("p", "ni", "q"), {"p": 128})
```


This is the readable way to express a coalesced-store layout regroup (see [Transpose](#transpose)),
equivalent to `.reshape_dim(0, (nc, 128)).permute((1, 0, 2))` but self-documenting. The SBUF
P-axis rule below still applies — a `rearrange` that would move dim 0 on an SBUF view is rejected
by the underlying `permute`.


See `examples/_03_tile_ops/_01_reshape_permute.py`.



## The SBUF P-axis constraint


On an SBUF-backed view, dimension 0 is the hardware partition axis and is **fixed**. The
free-dimension transforms above all work, but on SBUF:


- `.permute(dims)` requires `dims[0] == 0` — P cannot be rotated onto another dim.
- `.reshape_dim(0, ...)` / `.split(0, ...)` are not supported.
- `.squeeze_dim(0)` / changing P's size are not supported.
- An element-level `.slice(0, ...)` (narrowing P) is fine, and all free-dim (dim ≥ 1)
  transforms behave as they do on HBM.


```
   SBUF tile                 P (dim 0) is hardware-fixed:
   ┌────────────────┐          ✗  permute dim 0, reshape_dim(0), split(0), squeeze_dim(0)
   │ P (partition)  │          ✗  changing P's size
   │ F (free)       │        Free dims (>= 1) are unrestricted:
   └────────────────┘          ✓  reshape_dim / permute / slice / broadcast
```


A related rule: a **broadcast** axis (stride 0) is a decorator, not an iteration level — you
cannot index into it or iterate it; it only repeats a value during compute.



## Transform before tiling

To build a transform chain *before* deciding the tile grid, call the layout ops on the
**tensor itself** — a tensor (the `nl.ndarray` you were given, or any SBUF/PSUM buffer)
self-describes its layout, so `flatten_dims` / `reshape` / `reshape_dim` / `permute` /
`slice` return a new view sharing storage. Hand the transformed tensor to `nt.tiles()`:


```python
view = (
    src.flatten_dims(0, 1)        # [B*S, H]
    .reshape_dim(1, (H0, H1))     # [B*S, H0, H1]
    .permute((1, 0, 2))           # [H0, B*S, H1]
)
src_tiles = nt.tiles(view, tile_size=(H0, B * S, H1))
tile = src_tiles[0, 0, 0].load()  # one DMA reads the transformed layout
```


This works for HBM, SBUF, and PSUM sources alike (the transforms are pure metadata, no DMA).
Two cases that are *not* tiling and so skip `nt.tiles()` entirely:

- **Untiled load** — transform then DMA the whole thing in one shot:
  `nisa.dma_copy(dst=sbuf, src=w.reshape((H0, H1)).permute((1, 0)))`.
- **Broadcast operand** — a stride-0 view fed straight to a compute op:
  `nisa.tensor_tensor(out, x, gamma.expand_dim(1).broadcast(1, n), nl.multiply)`.


See `examples/_03_tile_ops/_03_tensor_view.py`.



## Access patterns


`tile_size` controls the tile grain; **`access_pattern`** controls how the source tensor is
walked. It is a list of `[stride, count]` levels — one per logical dimension — that describes
the read layout. Omitted, NeuroTile derives the contiguous pattern from the source. Supplied,
it can express strided or windowed reads:


```python

# Equivalent to the default contiguous layout:
nt.tiles(src, access_pattern=[[N, M], [1, N]], tile_size=(128, 512))

# Strided: partition stride 2*N skips every other source row.
nt.tiles(src, access_pattern=[[2 * N, M // 2], [1, N]], tile_size=(64, N))
```


```
   strided AP [[2*N, M//2], [1, N]] -- read every other row

   row 0 ◀──── tile reads
   row 1        (skipped)
   row 2 ◀──── tile reads
   row 3        (skipped)
   ...
```


The AP rank is **decoupled** from the source rank: a 3-level AP on a 2-D source produces a
3-D logical view (exposing structure not in `source.shape`), and a 1-level AP flattens an N-D
source. The exact rules and the largest-offset bound are in the `nt.tiles` / `NDSlice.ap`
docstrings.


Two related knobs:


- **`pattern_override=` + `out_shape=`** on `.load()` / `.store()` replace the auto-generated
  pattern for a single transfer, when a per-tile layout differs from the natural strides.
  `out_shape` gives the SBUF `(P, F)` with `P ≤ 128`; the override bypasses NeuroTile's
  emitter, so the compiler validates the descriptor depth downstream.


  ```python
  tile = src_tiles[0, 0].load(
      pattern_override=[[F_src, P], [2, F_dst]],   # stride-2 column gather
      out_shape=(P, F_dst),
  )                                                 # tile.data[p, f] == src[p, 2*f]
  ```


- **Sliced source** — a slice self-describes its strides and offset (it shares the parent's
  storage), so tile it directly; no extra argument is needed:


  ```python
  window = src[0:P, 0:F]
  src_tiles = nt.tiles(window, tile_size=(P, F))   # strides/offset from the slice itself
  ```


- **Runtime-indexed handle is not a source.** A handle that already carries a *runtime*
  (gather / dynamic-select) offset — for example `t[k]` with a traced `k`, or a gathered
  view — cannot be passed to `nt.tiles()` / `nt.blocks()`: only the compile-time offset is
  read, so the runtime offset would be silently dropped (it raises instead). Tile the base
  tensor and apply the runtime index on the *view*: `nt.tiles(t, ...)[k]` (see Part 4).


See `examples/_01_iteration/_05_access_patterns.py`; sliced sources in
`examples/_01_iteration/_06_sliced_sources.py`.



## Remainder handling


When a tensor extent is not a multiple of `tile_size`, the trailing tile along that axis is a
**remainder** — fewer elements than a full tile. A `(300, 500)` tensor at `tile_size=(128, 128)`
has a 3×4 grid with partials on the last row and column:


```
   (300, 500) at tile_size (128, 128)

   P per row-tile:  [128, 128, 44]         ◀ row 2 is a P-remainder (300 - 2*128)
   F per col-tile:  [128, 128, 128, 116]   ◀ col 3 is an F-remainder (500 - 3*128)

   ┌────┬────┬────┬──┐
   │    │    │    │░░│   ░ = partial F
   ├────┼────┼────┼──┤
   │    │    │    │░░│
   ├────┼────┼────┼──┤
   │▒▒▒▒│▒▒▒▒│▒▒▒▒│▓▓│   ▒ = partial P, ▓ = partial P and F
   └────┴────┴────┴──┘
```


Individual concrete tiles size themselves correctly. For **coalesced** loads (a row, column,
or sub-grid) and for **indirect** loads (Part 4), a single access pattern cannot express
"128 rows for two tiles, then 44," so you guard the boundary with `oob_mode` / `oob_value`:


| Parameter | Effect |
|---|---|
| `oob_mode=nisa.oob_mode.skip` | suppress OOB DMA faults; skipped positions keep their SBUF contents |
| `oob_value=<float>` | pre-fill the remainder SBUF slots with this value before the DMA (requires `oob_mode`) |


The `.is_remainder` attribute tells you when a view touches the boundary, so you only pay for
OOB handling where it is needed:


```python
for i in range(tiles.shape[0]):
    row = tiles[i, :]
    if row.is_remainder:
        data = row.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
    else:
        data = row.load()                      # interior fast path -- no OOB overhead
    out_tiles[i, :].store(data.data, oob_mode=nisa.oob_mode.skip)
```


For bulk handling, three options pre-partition clean tiles from boundary ones:


- **`remainder="skip"`** on `nt.tiles` / `nt.blocks` floor-divides the grid, excluding
  boundary tiles entirely.
- **`view.whole_tiles()`** returns only the non-remainder sub-views.
- **`view.remainder_tiles()`** returns only the boundary sub-views.


```python
for clean in tiles[0, :].whole_tiles():
    clean.load()                                            # fast path
for rem in tiles[0, :].remainder_tiles():
    rem.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
```


Internally NeuroTile is frugal: memset only touches the SBUF slots that actually have a
remainder (full tiles are never zeroed), and for a column slice where all tiles share the
same F extent, the SBUF allocation shrinks F to the actual width. The full
`is_remainder`-by-access-type matrix and the memset-targeting rules are in
the `NDSlice.load` / `whole_tiles` docstrings.



## Transpose


`view.load(transpose=True)` is the only NeuroTile transpose path (HBM→SBUF), wrapping
`nisa.dma_transpose`. Transposing is a tile-coordinate swap: source tile `[mi, ni]` loads
transposed into output tile `[ni, mi]`. Hardware requires a 2-byte dtype (bf16/fp16) on both
sides.


```python
xposed = src_tiles[mi, ni].load(transpose=True)   # one DMA per tile
out_tiles[ni, mi].store(xposed.data)               # swapped coords == the transpose
```


For the **static** (non-indirect) path, the result **preserves the tile grid, axis-swapped**:
the transposed tiles are indexable by coordinate, exactly like a non-transpose load — no
re-tiling. The free dim F maps onto SBUF partition rows (≤ 128), which sets the single-DMA rule:


| F dimension | Result | DMA |
|---|---|---|
| F ≤ 128 | grid of `(F, P)` tiles | one `dma_transpose` |
| F % 128 == 0 | grid of `(128, P)` tiles, F/128 chunks packed on the free axis | one batched `dma_transpose` |
| F > 128 and F % 128 ≠ 0 | **rejected** — would need 2 DMAs (single-DMA contract) | split with `whole_tiles()` / `remainder_tiles()` |


A whole tile-row transposes in one coalesced DMA and the packed result keeps its chunk grid,
so you index each transposed chunk directly:


```python
packed = src_tiles[mi, :].load(transpose=True)   # ONE DMA: tile-row [mi, :] -> grid (1, N/128)
for ni in range(packed.shape[1]):
    out_tiles[ni, mi].store(packed[0, ni].data)  # chunk ni == src(mi, ni).T -> out tile [ni, mi]
```


```
   transpose of a tile-row, F = 4 chunks of 128, packed on SBUF free axis

   src row [mi, :]  ──one dma_transpose──▶  ┌──────┬──────┬──────┬──────┐
   (128, 512)                               │ c0.T │ c1.T │ c2.T │ c3.T │  128 partitions
                                            └──────┴──────┴──────┴──────┘
                                            grid (1, 4): index packed[0, ni]
```


**Storing the coalesced result in one DMA.** The per-tile store above is `N/128` small DMAs.
To store the whole row-column in ONE DMA, note the asymmetry: the *load* is a hardware
transpose, but the *store* is a plain copy that maps SBUF-partition → HBM-row and
SBUF-free → HBM-col. `packed.data` holds each chunk on the **free** axis, while its home in
`out` is a **partition row-band** — a plain store cannot scatter free → row-band. Rearrange
the destination column so its iteration order matches `packed.data`:


```python
packed = src_tiles[mi, :].load(transpose=True)   # ONE DMA -> (128, N), grid (1, N/128)
dst = out_tiles[:, mi].rearrange((("ni", "p"), "q"), ("p", "ni", "q"), {"p": 128})
dst.store(packed.data)                            # ONE DMA: free chunks -> their row-bands
```


The `(ni p) q -> p ni q` regroup makes the `out` column iterate `(p, ni, q)`, matching
`packed.data`'s `(p, [ni, q])` layout, so the copy pairs element-for-element in one strided
DMA. Prefer the per-tile store for clarity; reach for the `rearrange` form when the extra
store DMAs matter.


Transpose also composes with **streaming** — declare `stream(transpose=True)` so the rotating
slots are sized to the transposed output up front (the [load-type lock](#streaming-in-depth)
applies; every `.load()` repeats `transpose=True`). For a contiguous block this is one
transpose DMA per block; for a tile-sharded gapped block it is one transpose per owned tile
into the slot's sub-tiles (the gather-transpose form and `transpose_axes` are in
[Part 4](#gather-transpose)).


For a tile already in SBUF or PSUM, NeuroTile does not wrap `nisa.nc_transpose` — call it
directly:


```python
psum_tmp = nl.ndarray((tile_k, tile_m), dtype=dtype, buffer=nl.psum)
nisa.nc_transpose(psum_tmp, tile.data)
```


See `examples/_03_tile_ops/_04_transpose.py`.



## Fold


`.fold(src_dim, into_dim)` merges one dimension into another, reducing rank by one. Unlike
`flatten_dims` (adjacent dims, pure metadata), `fold` records a DMA recipe that the next
`.load()` / `.store()` applies — so non-adjacent dims merge at DMA time. It has two modes:


| Mode | Trigger | DMA cost |
|---|---|---|
| **Free-dim fold** | `src_dim > 0` and `into_dim > 0` | one coalesced DMA |
| **Partition fold** | `src_dim == 0` or `into_dim == 0` | K separate DMAs (one per partition slice) |


```python
src_tiles = nt.tiles(src_hbm, tile_size=(P, F, K))

folded = src_tiles[0, 0, 0].fold(2, 1)    # free-dim: (P, F, K) -> (P, F*K), one DMA
tile = folded.load()

folded = src_tiles[0, 0, 0].fold(2, 0)    # partition: (P, F, K) -> (P*K, F), K DMAs
tile = folded.load()
```


```
   free-dim fold (2 -> 1)                  partition fold (2 -> 0)
   (P, F, K) -> (P, F*K)                   (P, F, K) -> (P*K, F)

   ┌────┬────┬────┐                        ┌────┐ k=0
   │ k0 │ k1 │ k2 │                        │ k0 │
   └────┴────┴────┘                        ├────┤ k=1
                                           │ k1 │
   one DMA packs the K slices              ├────┤ k=2
   along the free axis.                    │ k2 │
                                           └────┘
                                           K separate DMAs stack the K
                                           slices along the partition axis
                                           (one AP can't address disjoint
                                           partition ranges).
```


The partition fold is the documented exception to the single-DMA contract. Folds compose —
`step1 = view.fold(3, 2); step2 = step1.fold(0, 1)` — and the recipe round-trips, so the same
fold is valid on both `.load()` and `.store()`.


See `examples/_03_tile_ops/_02_fold_pattern_override.py` (fold and `pattern_override`).



---

# Part 4 — Indirect indexing


Everything so far addressed tiles at compile-time-known positions. Production kernels also
need **runtime-dynamic** access: gathering rows by an index vector, selecting an expert by a
runtime id, shifting a KV-cache window by a computed offset. NeuroTile expresses all of these
through the same `[]` operator — you pass a *loaded index* (an SBUF view or raw `nl.ndarray`)
where you would otherwise pass an `int`. The anchors are `examples/_05_indirect/*`.


The key distinction is the index's shape, which selects the NKI mechanism:


- a **vector** index (shape `[P>1, 1]`) triggers `vector_offset` — each partition entry picks
  its own source row (a gather);
- a **scalar** index (shape `[1, 1]`) triggers `scalar_offset` — one value shifts the access
  window uniformly.



## Gather & scatter


A loaded index tile of shape `[K, 1]` in position 0 of `[]` gathers `K` rows: each entry
selects which source row to fetch. The data tensor lives in HBM, the index tile in SBUF.


```python
data_iter = nt.tiles(data, tile_size=(N, T_D))    # full N rows, T_D-wide column tiles
idx_iter  = nt.tiles(indices, tile_size=(T_K, 1))
out_iter  = nt.tiles(out, tile_size=(T_K, T_D))

for i in range(idx_iter.shape[0]):
    idx_tile = idx_iter[i, 0].load()               # index tile -> SBUF [T_K, 1]
    for j in range(out_iter.shape[1]):
        gathered = data_iter[idx_tile, j].load()    # vector_offset gather -> [T_K, T_D]
        out_iter[i, j].store(gathered.data)
```


```
   index tile (SBUF)        data (HBM)              result (SBUF)
   idx = [2, 5, 0, 7]
                            row 0 │a0 a1 ...│        ┌─────────────────┐
                            row 2 │c0 c1 ...│        │ row 2 (c0 c1..) │ ◀ idx[0]=2
                            row 5 │f0 f1 ...│   ──▶  │ row 5 (f0 f1..) │ ◀ idx[1]=5
                            row 7 │h0 h1 ...│        │ row 0 (a0 a1..) │ ◀ idx[2]=0
                                                     │ row 7 (h0 h1..) │ ◀ idx[3]=7
                                                     └─────────────────┘
```


**Scatter** is the mirror — the same vector index in a `.store()` writes rows to the indexed
positions:


```python
out_iter[idx_tile, j].store(src_tile.data)          # vector_offset scatter
```



## Dynamic selection


A scalar index (shape `[1, 1]`) selects a single slab or shifts a window for *all* rows at
once. Its **position** in `[]` sets which source dimension is indirect (`indirect_dim`). The
canonical case is MoE expert selection — a 3-D weight tensor `[E, P, F]` indexed by a runtime
expert id yields the chosen expert's `(P, F)`:


```python
w_iter   = nt.tiles(weights, tile_size=(P, F))      # 2-D tile on 3-D tensor; dim 0 = expert
eid_tile = nt.tiles(expert_id_tensor, tile_size=(1, 1))[0, 0].load()  # [1, 1]
expert_data = w_iter[eid_tile].load()               # scalar_offset on dim 0 -> [P, F]
```


```
   indirect_dim is set by position in []:

   data_iter[scalar, :, :]   -> indirect_dim 0   (select a slab)
   data_iter[:, scalar, :]   -> indirect_dim 1   (shift a window on dim 1)
   data_iter[i, scalar, :]   -> indirect_dim 1   (static i sets base, scalar shifts dim 1)
```


`indirect_dim` is **source-tensor-relative**: it tracks the original tensor dimension and
does not shift as you consume dims during indexing. At most one indirect index per `[]`;
`vector_offset` is restricted to position 0, while `scalar_offset` works at any position.

## Runtime tile positions and element offsets


Runtime indexing shows up whenever the next tile is chosen by data rather than by the
Python loop nest: a dynamic sequence tile, an expert id, a generated block id, or a scalar
counter maintained in SBUF. In raw NKI you usually turn that value into an address offset
by hand before using it in the DMA. NeuroTile keeps the same mental model as the rest of
the guide: index the view by the thing your algorithm means, and let the view's metadata
connect that index to the source tensor layout.


The most direct form is a **logical coordinate**. If `tiles` is a tile view, then
`tiles[m, h]` means "tile row `m`, tile column `h`" even when `m` is a runtime scalar:


```python
tiles = nt.tiles(src, tile_size=(128, 512))
m_tile_idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)

tile = tiles[m_tile_idx, h].load()                 # m_tile_idx is a tile coordinate
```


At the DMA boundary, hardware still needs a source-element offset. NeuroTile derives that
offset from the current view: one step in dimension 0 advances by the view's dim-0 index
stride, one step in dimension 1 advances by the dim-1 stride, and so on. If the runtime
coordinate is already stride-1, no arithmetic is emitted. If it is an SBUF scalar and the
stride is larger than 1, NeuroTile inserts the scalar multiply needed to convert the
logical coordinate into source elements before the load or store.


That compact form is the right default when the index is used once. When the same runtime
position drives several tensors, compute the source-element offset once and reuse it. Mark
the value with `nt.element_offset(...)` to tell NeuroTile that the runtime value is already
scaled:


```python
q_tiles = nt.tiles(q, tile_size=(128, 512))
k_tiles = nt.tiles(k, tile_size=(128, 512))
v_tiles = nt.tiles(v, tile_size=(128, 512))

m_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
nisa.memset(dst=m_offset, value=0)

def body(_):
    q_tile = q_tiles[nt.element_offset(m_offset), h].load()
    k_tile = k_tiles[nt.element_offset(m_offset), h].load()
    v_tile = v_tiles[nt.element_offset(m_offset), h].load()

    nisa.tensor_scalar(
        dst=m_offset,
        data=m_offset,
        op0=nl.add,
        operand0=q_tiles.index_stride_elements[0],
    )
```


`nt.element_offset(m_offset)` changes only the interpretation of that one bracket entry.
The bracket position still selects the dimension. The offset is relative to the current
view, and NeuroTile does not scale it again. The stride you need to update such a counter
is available as metadata on the view:


```python
q_tiles.index_stride_elements[0]    # source elements per logical step in dim 0
```


Reading `index_stride_elements` emits no instruction; it is a compile-time value derived
from the view. That matters for sliced and blocked views. If the view already starts at an
offset, keep your runtime counter relative to that view and let NeuroTile carry the static
origin in the layout metadata:


```python
tail = tiles[2:, :]
tile = tail[nt.element_offset(m_offset), h].load()  # m_offset is relative to tail
```


So the choice is simple:


- pass the runtime scalar directly when your algorithm has a logical tile or block
  coordinate and the value is used once;
- pass `nt.element_offset(counter)` when you have already converted the runtime value into
  source elements, especially when that counter is shared by several loads or stores.


Today, keep scaled runtime loop counters in SBUF when you need them as dynamic tile
coordinates. If you maintain the counter in source-element units, use
`nt.element_offset(counter)`.


Concrete examples live under `examples/_05_indirect/`: `_02_dynamic_select.py` shows
stride-1 logical runtime indexing, while `_07_runtime_element_offset.py` shows SBUF scalar
logical tile indexing side by side with a reusable element-offset counter.



## Mixed static & indirect indexing


Static indices (an `int` or loop variable) compute the base offset; the indirect index
supplies the dynamic part. A KV-cache read selects a static batch and a dynamic sequence
position:


```python
kv_iter  = nt.tiles(kv_cache, tile_size=(1, D))     # kv_cache: [B, S, D]
seq_tile = nt.tiles(seq_offset_tensor, tile_size=(1, 1))[0, 0].load()
kv_data  = kv_iter[batch_id, seq_tile].load()       # batch_id static, seq_tile -> scalar_offset dim 1
```


The index need not be wrapped by `nt.tiles`. If you already have an index in SBUF — for
instance one you computed with `nisa.*` — pass the raw `nl.ndarray` directly; the same shape
rule applies (`(1, 1)` → scalar, `[K, 1]` → vector):


```python
seq_sbuf = nl.ndarray((1, 1), dtype=seq_offset_tensor.dtype, buffer=nl.sbuf)
nisa.dma_copy(seq_sbuf, seq_offset_tensor)
kv_data = kv_iter[batch_id, seq_sbuf].load()        # raw nl.ndarray as the index
```



## Sub-tile indexing: pre-load vs post-load


There are two distinct ways to "narrow" with an index, and they cost differently:


- **Pre-load composition** — compose the view *before* `.load()` so the DMA moves only what
  you need. This narrows the data transferred.
- **Post-load extraction** — slice the SBUF tile *after* `.load()` to rearrange on-chip. The
  full tile was already moved; this is a free SBUF view.


```python

# Pre-load: narrow to one row before the DMA (less data moved).
slab_view = data_iter[slab_idx]
row_view  = slab_view[row_idx, :]      # narrow to one row -- still an HBM view
tile = row_view.load()                  # DMA moves only [1, F]

# Post-load: load the whole tile, then extract on-chip (rearrange, no extra DMA).
tile = data_iter[0, 0].load()           # DMA moves [P, F]
sub  = tile[row_idx]                     # [1, F] -- an SBUF view, no DMA
```


```
   pre-load composition              post-load extraction
   view[slab][row, :].load()         tile = view.load(); tile[row]

   narrow ──▶ DMA [1, F]             DMA [P, F] ──▶ slice ──▶ [1, F]
   (moves less)                      (rearrange on-chip)
```


The two compose with dynamic selection — e.g. select an expert dynamically, then extract a
static row from the loaded tile: `expert = w_iter[eid_tile].load(); sub = expert[row_idx]`.



## Indirect remainder handling


Because an indirect index is unknown at compile time, NeuroTile cannot prove the gathered
rows are in bounds — so an indirect view always reports `is_remainder == True`. Guard it with
the same `oob_mode` / `oob_value` from [Part 3](#remainder-handling):


```python
gathered = data_iter[idx_tile, 0].load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
```


`oob_mode.skip` suppresses the out-of-bounds fault; `oob_value` pre-fills the SBUF slot so
skipped lanes hold a deterministic value (without it, skipped positions keep stale SBUF
contents). Scatter takes `oob_mode.skip` too, silently dropping OOB writes. The
`is_remainder` guard reads the same as the coalesced case — and for an indirect view the
`if` branch is always taken:


```python
view = data_iter[idx_tile, 0]
if view.is_remainder:                  # always True for an indirect view
    gathered = view.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
else:
    gathered = view.load()
```


Indirect indexing also composes with `nt.blocks`: a scalar offset shifts the whole block in
one coalesced DMA, while a vector offset issues one indirect DMA per tile in the block.



## Gather transpose


`load(transpose=True)` on a view carrying an index gathers the indexed rows *and* transposes
them in a single `nisa.dma_transpose` with a `vector_offset` access pattern. The
`transpose_axes=` permutation states the rank:


| Input view | `transpose_axes` | Result |
|---|---|---|
| 2-D `(N, D)` | `(1, 0)` (default) | `(D, N)` |
| 3-D `(d0, d1, d2)` | `(2, 1, 0)` | `(d2, d1, d0)` |
| 3-D + size-1 dummy dim | `(3, 1, 2, 0)` | `(P, 1, f_tiles, rows)` (4-D reshape trick) |


```python
idx = nt.tiles(indices, tile_size=(N, 1))[0, 0].load()      # uint32 index tile in SBUF
xposed = data_iter[idx, 0].load(transpose=True)             # 2-D: gather + transpose -> (D, N)
xposed = data_iter[idx, 0, 0].load(transpose=True, transpose_axes=(2, 1, 0))   # 3-D
```


`transpose_axes` is omittable on a 2-D gather (defaults to `(1, 0)`) but **required** on a
>2-D gather, since the 3-D `(2,1,0)` and 4-D reshape-trick `(3,1,2,0)` forms are otherwise
ambiguous. The 4-D form synthesizes the size-1 dummy dimension the hardware needs to gather a
wide free dimension a plain 2-D transpose cannot reach.


> **A streaming caveat.** A rotating *gather-transpose* stream is not supported — it hits a
> compiler bug where `nisa.dma_transpose` with `vector_offset` drops the DMA when a slot
> buffer is reused across steps. Use a non-streamed gather transpose
> (`view[idx].load(transpose=True, ...)`) instead. (Static transpose streams from
> [Part 3](#transpose) are fine.)



---

# Part 5 — Expert: full kernels


This part walks a complete production kernel end to end, showing how the primitives from
Parts 1–4 compose. It lives under `examples/kernels/` and is validated against a PyTorch
reference. MLP CTE is the richest example in the library and exercises nearly all of it.



## Walkthrough: MLP CTE / SwiGLU


The MLP kernel computes `y = (silu(x @ gate) · (x @ up)) @ down`. It composes nearly the whole
library — M-axis sharding, batch-level streaming, K-streamed weights, PSUM-bank management, and
coalesced block I/O. We first read the top-level structure, then drill into one representative
stage end to end: the **gate projection**, `silu(x @ gate)`. The up projection is identical, and
the down projection follows the same accumulate-in-PSUM-then-evict shape.


**Top-level structure.** The kernel flattens `[B, S, H]` to `[M, H]`, shards `M` across cores by
slicing the block views, and streams the activation blocks so each batch's DMA overlaps the
previous batch's compute. Per batch it runs the three projections and stores the result:


```python
# Tile / block sizes (a real kernel derives these from a config; literals here for clarity).
TILE_M, TILE_K, TILE_I, TILE_H = 128, 128, 512, 512
M_BLK, K_BLK = 2, 4          # activation block: M-subtiles x K-tiles per coalesced DMA

@nki.jit
def mlp_cte(x, gate_proj, up_proj, down_proj):
    B, S, H = x.shape
    M = B * S
    x = x.reshape((M, H))                                          # real reshape -> [M, H]

    output = nl.ndarray((M, H), dtype=x.dtype, buffer=nl.shared_hbm)
    x_blocks      = nt.blocks(x,      tile_size=(TILE_M, TILE_K), block_size=(M_BLK, K_BLK))
    output_blocks = nt.blocks(output, tile_size=(TILE_M, TILE_H), block_size=(M_BLK, H // TILE_H))

    # Shard the M-block axis across cores; total = the block count the view already knows.
    shard = nt.uneven_block_range(rank=nl.program_id(0), num_shards=nl.num_programs(0),
                                  total=x_blocks.shape[0])
    x_blocks      = x_blocks[shard, :]
    output_blocks = output_blocks[shard, :]

    # Weights are HBM block views, hoisted once across all batches.
    gate = nt.blocks(gate_proj, tile_size=(TILE_K, TILE_I), block_size=(K_BLK, 1))
    up   = nt.blocks(up_proj,   tile_size=(TILE_K, TILE_I), block_size=(K_BLK, 1))
    down = nt.blocks(down_proj, tile_size=(TILE_I, TILE_H), block_size=(K_BLK, 1))

    x_stream = x_blocks.stream(buffer_count=2)                     # prefetch batch k+1 while computing k
    for m_batch in range(x_blocks.shape[0]):
        x_block   = x_stream.load(m_batch)
        x_T       = transpose_in_place(x_block)                    # layout for the gate/up matmuls
        gated     = compute_swiglu(x_T, gate, up)                  # silu(x@gate) * (x@up)
        gated_T   = transpose_in_place(gated)                      # layout for the down matmul
        out_block = compute_down_matmul(gated_T, down)             # gated @ down
        output_blocks[m_batch].store(out_block.data)               # one coalesced block store

    return output.reshape((B, S, H))
```


`transpose_in_place` relays activations between the matmuls (it fans `nc_transpose` across the
PSUM banks — a detail we won't expand here). The interesting work is inside `compute_swiglu`,
which runs two projections of identical shape — `silu(x @ gate)` and `x @ up` — and multiplies
them. We trace the gate half, factored out below as `gate_projection`; the up half is the same
code without the `silu`.


**The gate projection.** This is where the Part 2 vocabulary lands at production scale.
`silu(x @ gate)` is a K-reduction matmul whose `I` (output) dimension is too wide for PSUM's
eight banks at once, so the columns are processed in **groups**, each group's accumulators
sized to fit the bank budget. Per group: narrow the weight columns to the group, stream those
weight K-blocks while the resident `x` block is just indexed, accumulate the full K reduction in
a `psum_pool`, then evict through `silu` (with `bias`, a zero-initialized SBUF bias vector) into
the SBUF output slice:


```python
def gate_projection(x_block, gate, bias):
    m_tiles = x_block.tile_shape[0]
    I_tiles = gate.shape[1]

    # SBUF output for silu(x @ gate) -- full I width, written one column-group at a time.
    gated = nt.alloc_tiles(tile_size=(TILE_M, TILE_I), buffer_type=nl.sbuf, dtype=gate.dtype,
                           element_shape=(x_block.element_shape[0], gate.element_shape[1]))

    # Split the I columns so each group's m_tiles x i_count accumulators fit PSUM's 8 banks.
    MAX_I_PER_GROUP = 8 // m_tiles
    for i_start in range(0, I_tiles, MAX_I_PER_GROUP):
        i_count     = min(MAX_I_PER_GROUP, I_tiles - i_start)
        gate_cols   = gate[:, i_start : i_start + i_count]          # narrow weights to this group
        gated_slice = gated[:, i_start : i_start + i_count]         # matching output slice

        # One PSUM accumulator per (m_tile, i_tile) in the group, on distinct banks.
        gate_psums = nt.psum_pool(tile_size=(TILE_M, TILE_I), grid=(m_tiles, i_count),
                                  bank_ids=tuple(range(m_tiles * i_count)))

        # Stream the weight K-blocks; x_block's K-blocks are resident, just indexed.
        gate_k_stream = gate_cols.stream(buffer_count=2)
        for kb in range(x_block.shape[0]):
            x_k    = nt.tiles(x_block[kb])                          # resident: index, no DMA
            gate_k = nt.tiles(gate_k_stream.load(kb))              # streamed: overlapped DMA
            for m in range(m_tiles):
                for kt in range(gate_k.shape[0]):
                    for it in range(i_count):
                        nisa.nc_matmul(gate_psums[m, it].data,      # accumulates across kb and kt
                                       x_k[m, kt].data, gate_k[kt, it].data)

        # Evict: silu(gate_psums) -> the gated SBUF slice, freeing the banks for the next group.
        for m in range(m_tiles):
            for it in range(i_count):
                nisa.activation(gated_slice[m, it].data, op=nl.silu,
                                data=gate_psums[m, it].data, bias=bias)
    return gated
```


```
   gate projection: silu(x @ gate), one I-column-group shown

   x_block (resident) ─┐
                       ├─▶ nc_matmul ──accumulate over K──▶ gate_psums ──silu──▶ gated[:, group]
   gate_cols ──stream──┘     (one PSUM bank per output tile)            (SBUF, this group's slice)

   next I-group reuses the same 8 PSUM banks, writes the next slice of `gated`
```


Reading it against the primitives:


- **`nt.blocks` + sharding** — the activation and weight block views set the DMA grain; slicing
  `x_blocks` by `shard` narrows to this core's batches (Part 3 sharding), and the whole
  `gate_projection` body is the same code a single-core kernel would run.
- **`alloc_tiles`** — `gated` is the SBUF output the projection fills, declared by
  `element_shape` and written one column-group slice at a time.
- **Weight-column narrowing** — `gate[:, i_start : i_start + i_count]` selects just this group's
  output columns, so the matmul and the `psum_pool` only cover what fits the banks.
- **`psum_pool`** — `m_tiles × i_count` accumulators on explicit `bank_ids`; the matmul writes
  the full K reduction there (no per-step SBUF add), and `silu` evicts the result, freeing the
  banks for the next group to reuse — the bank-placement story from
  [Part 2](#psum-accumulators-psum_pool-intro), made concrete.
- **Stream vs resident** — the weight K-blocks are streamed (overlapped DMA) while the `x` block
  is resident and only indexed, so the inner loop issues no blocking load — exactly the
  hoist-one/stream-one shape from the [canonical matmul](#the-canonical-matmul).


The excerpts above are trimmed for the walkthrough — the complete, runnable kernel (all three
projections, both transposes, the config-driven block sizing, and its test) lives in
[`examples/kernels/mlp_cte/`](examples/kernels/mlp_cte/). The payoff: that full kernel is under
400 lines versus ~3,800 for the hand-written baseline, and it *outperforms* that baseline
because the single-DMA contract on the block loads and stores issues coalesced multi-tile DMAs
the hand-written version did not.


For a second full kernel of a different shape — a broadcast-heavy normalization rather than a
matmul chain — see RMSNorm TKG in `examples/kernels/rmsnorm/tkg/`.



## Composing the patterns & the perf contract


The walkthrough reduces to the same handful of primitives applied in layers: a tile or block
view over each tensor, sharding by slicing, hoisting the stationary operands and streaming the
flowing ones, accumulating in PSUM, and a single coalesced store. Nothing is hidden between
those steps.


That is the contract worth internalizing before you write your own:


- **No hidden compute** — you write every `nisa.*` op; the algorithm is yours.
- **No hidden loops** — you write every `nl.affine_range` / `nl.sequential_range`.
- **No hidden DMAs** — every `.load()` / `.store()` / `stream` step is exactly one
  `nisa.dma_copy` (the single documented exception being a partition-dim `.fold()`).
- **`.data` is always there** — when a high-level method doesn't express what you need, drop
  to the raw NKI tensor and hand-write the DMA.


Because the lowered instruction stream is the one you wrote, NeuroTile keeps NKI's "no
surprises" performance model while removing the boilerplate — which is how these kernels reach
~5–8× fewer lines at parity-or-better performance.



---

# Part 6 — Reference & survival kit


The day-to-day quick reference: the anti-patterns to avoid and a task→API lookup. For the
per-API contract see the `nt.*` docstrings; for cross-cutting behavior and the NKI parser
constraints see Appendix A.



## Anti-patterns catalog


Each of these has a shorter, more correct NeuroTile form. The left column is what to stop
writing; the right is what the library already gives you.


| Anti-pattern | Use instead | Why |
|---|---|---|
| `for i in range(N): tiles[i*TS:(i+1)*TS]` | `tiles[i]` | the factory computes the offset; manual arithmetic is fragile |
| `K_TILES = K // TILE_K` | `view.shape[d]` | floor division hides remainders and re-derives known info |
| `view[i, :]` when `view[i]` suffices | `view[i]` | the trailing default `:` is a no-op |
| `row[:, j]` / `block_row[:, bj]` referencing a consumed dim | `row[j]` / `block_row[bj]` | a single int targets the surviving dim (the `NDSlice.__getitem__` docstring) |
| hand-rolled two-buffer swap | `.stream(buffer_count=2)` | declarative double-buffer |
| PSUM as a generic accumulator | FP32 SBUF for non-matmul reductions | PSUM is matmul-output memory; an SBUF FP32 accumulator saves a copy and a scarce bank |
| `psums[i*N + j, 0]` flat indexing | shape the pool to natural axes, `psums[i, j]` | clearer and less error-prone |
| `nt.tiles(loaded_block)` to re-tile a loaded slot | index it directly (`block[ti, tj]`) | a loaded view already exposes its tile grid |
| recomputing `block_size[1] * tile_size[1]` | `view.element_shape[1]` | the API already has the extent |



## Cheat sheet


| Task | API |
|---|---|
| Tile a tensor | `nt.tiles(t, tile_size=(P, F))` |
| Group tiles for coalesced DMA | `nt.blocks(t, tile_size=, block_size=)` |
| Build a transform chain before tiling | `t.reshape_dim/.permute/...` then `nt.tiles(view, ...)` |
| Untiled transform-then-load (no tiling) | `nisa.dma_copy(dst, t.reshape(...).permute(...))` |
| Move one tile / slice HBM→SBUF | `view.load()` |
| Move SBUF→HBM | `view.store(data.data)` |
| Overlap DMA with compute | `view.stream(buffer_count=2)` |
| Allocate an SBUF accumulator | `nt.alloc_tiles(...)` / `nt.alloc_blocks(...)` |
| Allocate PSUM accumulator tiles | `nt.psum_pool(tile_size=, element_shape=, bank_ids=)` |
| Shard across cores | slice the view with `nt.block_range(...)` etc. |
| Cast dtype | `load(dtype=)` / `store(dtype=)` (allocation-driven) |
| Transpose HBM→SBUF | `view.load(transpose=True)` |
| Gather / scatter | `view[idx_tile, :].load()` / `.store(...)` |
| Merge non-adjacent dims at DMA time | `view.fold(src_dim, into_dim)` |
| Handle a partial trailing tile | `view.is_remainder` + `oob_mode` / `oob_value` |
| Raw NKI tensor for a manual DMA | `view.data` |
| Trace-time tile-count math | `nt.ceiling_div(a, b)` / `nt.largest_divisor(n, max)` |



## Debugging & testing


- **`nki.simulate(kernel)(inputs)`** runs the kernel on CPU, deterministically — it catches
  logic and indexing bugs before you reach the device.
- **Use torch end to end for bf16** (inputs, reference, and the matmul); numpy has no native
  bf16, so upcast both sides to fp32 at the compare site.
- **bf16 rounding can look like a compiler bug.** Before blaming the stack, repro with both
  small and large magnitudes — integer test data quantizes cleanly to bf16 and can mask a
  layout or fold mismatch.


```python
out = nki.simulate(my_kernel)(x_np, w_np)   # CPU run, no device -- check logic first
```



## Glossary


- **NDSlice** — the one view type; `Grid + Layout`. Returned by every factory, slice, load,
  and stream child.
- **Grid** — the logical iteration schedule: per-dim level stack (block/tile/element step), a
  cursor, and the remaining addressable region.
- **Layout** — the physical memory descriptor: `HBMLayout`, `SBUFLayout`, or `PSUMLayout`
  (source, offset, strides, indirect parameters).
- **BlockStream** — the rotating-buffer object `.stream()` returns; owns the SBUF slot pool.
- **Hoisting** — loading a whole slice resident in SBUF for reuse (`tiles[:, m].load()`).
- **Streaming** — flowing data through a small rotating buffer for DMA/compute overlap.
- **P / F axes** — the SBUF partition axis (dim 0, ≤ 128) and free axis (the rest).
- **Tile vs block** — a tile is the compute grain; a block groups tiles for coalesced DMA.
- **Remainder** — a trailing tile that is smaller than `tile_size` because the extent didn't
  divide evenly.
- **Shard** — this core's slice of a view under LNC; produced by slicing with a shard helper.
- **indirect_dim** — the source-tensor dimension an indirect index addresses; set by the
  index's position in `[]`, and source-relative (it does not shift as dims are indexed away).



---

# Appendix A — Architecture & Cross-Cutting Behavior


> The per-API contract (signatures, parameters, validation rules) lives in the `nt.*`
> docstrings — read with `help(nt.tiles)`. This appendix collects what the docstrings do not:
> the architecture of the view type, the library-wide behavioral guarantees, the access-pattern
> emission mechanism, the NKI parser constraints, and an error-message → fix index.



## A.0 — How this appendix works


The per-API contract is in the docstrings ([A.2](#a2--per-api-contract-see-the-docstrings)).
This appendix covers the cross-cutting material that is not tied to a single API: the view-type
architecture ([A.1](#a1--architecture--core-types)), the library-wide behavioral contracts
([A.3](#a3--library-wide-behavioral-contracts)), the NKI parser constraints
([A.4](#a4--nki-parser-constraints)), and an error-message index
([A.5](#a5--error-message--causefix-index)).


The library enforces its rules with named asserts. [A.5](#a5--error-message--causefix-index)
maps an assert excerpt (and the hardware errors some front for) back to the rule and the fix.



### Hard limits


Every fixed limit in the library, in one place:


| Constant | Value | Meaning |
|---|---|---|
| `MIN_TILED_DIMS` | 2 | a tile/block view always keeps ≥ 2 dims (P, F); cleanup stops before dropping below this |
| `MAX_SBUF_PARTITION_ROWS` | 128 | maximum P-axis extent in SBUF |
| `PSUM_BANK_SIZE` | 2048 | elements per PSUM bank |
| `NUM_HW_BANKS` | 8 | PSUM banks on the hardware |
| `P_DIM` | 0 | the partition axis is always dimension 0 |



## A.1 — Architecture & core types


The `NDSlice = Grid + Layout` model is introduced in
[The core abstraction: NDSlice](#the-core-abstraction-ndslice) (Part 0) — Grid is the logical
iteration schedule, Layout the physical memory descriptor (`HBMLayout` / `SBUFLayout` /
`PSUMLayout`). The one detail that section defers is what the package actually exports.


**Public surface.** The top-level package (`nt`) exports only the factories and helpers —
`tiles`, `blocks`, `alloc_tiles`, `alloc_blocks`, `psum_pool`, the `*_range`
shard helpers, `get_shard_info`, `ceiling_div`, `largest_divisor`. `NDSlice`, `Grid`, and the
`*Layout` types are internal: you receive and operate on `NDSlice` values but do not construct
or import them directly. (`BlockStream`, returned by `.stream()`, is likewise handed to you,
not constructed.)



## A.2 — Per-API contract: see the docstrings

The complete, normative contract for every public API — exact signature, every
parameter and its accepted values, every validation rule with its assert text,
source modes, and return shapes — lives in the **`nt.*` docstrings**, which are the
single source of truth (kept in lock-step with the code by a build-time drift
test). Read them with `help(nt.tiles)` (etc.). This appendix does **not** restate
them, so it cannot drift from the code.

Quick map from API to its docstring:

| Topic | APIs (read the docstring of each) |
|---|---|
| View factories | `nt.tiles`, `nt.blocks` |
| Allocation & pools | `nt.alloc_tiles`, `nt.alloc_blocks`, `nt.psum_pool` |
| Data movement | `NDSlice.load`, `NDSlice.store`, `NDSlice.stream` (→ `BlockStream`), `NDSlice.ap`, `NDSlice.tolist` |
| Indexing & remainder split | `NDSlice.__getitem__`, `NDSlice.whole_tiles`, `NDSlice.remainder_tiles` |
| View transforms | `NDSlice.reshape_dim` / `reshape` / `permute` / `flatten_dims` / `split` / `slice` / `broadcast` / `expand_dim` / `squeeze_dim` / `fold` |
| Sharding & trace-time math | `nt.block_range`, `nt.uneven_block_range`, `nt.interleaved_range`, `nt.get_shard_info`, `nt.ceiling_div`, `nt.largest_divisor` |

The `NDSlice` class docstring catalogs the view attributes (`.shape`,
`.element_shape`, `.tile_size`, `.tile_shape`, `.block_size`, `.is_remainder`, …).
One distinction it draws is worth repeating because it causes real bugs:
**`.shape` shrinks as you index dimensions away, while `.tile_shape` records how
many tiles span the view and does not** — use `.tile_shape[d]` when you need the
tile count after indexing.

The rest of this appendix covers only what the docstrings do not: the architecture
of the view type ([A.1](#a1--architecture--core-types)), the cross-cutting
behavioral contracts ([A.3](#a3--library-wide-behavioral-contracts)), the NKI
parser constraints ([A.4](#a4--nki-parser-constraints)), and the
error-message → fix index ([A.5](#a5--error-message--causefix-index)).


## A.3 — Library-wide behavioral contracts


The cross-cutting guarantees, collected:


- **Zero-cost abstractions.** Every NeuroTile call lowers to standard NKI ISA — no hidden
  compute, no hidden DMAs, no hidden loops.
- **Allocation-driven dtype cast.** `load(dtype=)` / `store(dtype=)` allocate the destination
  at the requested dtype; the DMA hardware casts on a mismatch. There is no separate cast op.
- **Metadata transforms vs the fold recipe.** `reshape_dim` / `flatten_dims` / `permute` / etc.
  are pure metadata; `fold` records a DMA recipe applied at the next load/store.
- **Single-DMA contract** with its one partition-`fold` exception (see the
  `NDSlice.load` / `NDSlice.store` docstrings).
- **Load-type lock** on streams (see the `NDSlice.stream` / `BlockStream` docstrings).
- **Logical rank vs physical 2-D SBUF backing.** A `(P, F1, F2)` tile is stored `(P, F1*F2)`;
  `.data.shape` reports the 2-D shape; logical slices flatten through internally.



## A.4 — NKI parser constraints


The static NKI parser rejects or mistraces several ordinary Python constructs inside an
`@nki.jit` function. The normative checklist:


- No exceptions (`raise` / `try` / `except`); use `assert cond, "msg"`.
- No `is` / `is not`; use `==` / `!=`.
- No `set()`, `sorted()`, `getattr`, `hasattr`, `import`, `**kwargs` in kernel bodies.
- No list literals; use tuples.
- No string buffer args; use `nl.sbuf` / `nl.psum` / `nl.shared_hbm`.
- `int + object` trap: route runtime values through indirect / `nl.ds`, not raw arithmetic on
  a `LoopVar`.
- Module-level `slice` constants don't trace — build them in the body.
- LoopVar tiers: `affine_range` / `sequential_range` are full expressions; `dynamic_range` is a
  scalar (no arithmetic, no `.ap(offset=)`).
- `nl.fori_loop(start, stop, body, step)` is the parser-frontend path for runtime trip counts:
  it follows Pallas `fori_loop` semantics (body is a function called per index). No loop-carried
  dependencies yet — keep cross-iteration state in SBUF and mutate in place.
- Output tensors use `nl.shared_hbm`, not `nl.hbm`.



## A.5 — Error-message → cause/fix index


A lookup from an assert excerpt or compiler error to the rule and the fix.


| Error / assert excerpt | Cause | Fix |
|---|---|---|
| `tile_size= must have at least 2 dims (P, F)` | 1-D `tile_size` | write `(N, 1)` or `(1, N)` (see the `nt.tiles` docstring) |
| `buffer_type=nl.psum is not supported` | PSUM passed to a view factory | use `nt.psum_pool` for PSUM; transform a bank's `.data` directly |
| `source already carries a runtime (gather / dynamic-select) offset` | passed a runtime-indexed handle (`t[k]`, a gathered view) as the `source` | tile the base and index the view: `nt.tiles(t, ...)[k]` |
| `load() requires all batch dims to be consumed first` | `.load()` on a view with unconsumed batch dims | index the batch dims first (the `NDSlice.__getitem__` docstring) |
| `expects ndarray or .data view, not NDSlice` | `.store(ndslice)` | pass `.data`: `.store(view.data)` |
| `pattern_override= requires out_shape= or dst=` | override without an SBUF shape | add `out_shape=(P, F)` or `dst=` |
| trace-time layout / descriptor-depth error on a multi-tile `.load()` | region won't coalesce to one AP (too many strided levels for one DMA) | re-tile / merge / drop dims, or load per tile (`NDSlice.load` docstring) |
| `cannot aggregate across interleaved-shard gaps` | re-tiling a round-robin shard to a larger tile | only subdivide; build from the root to re-shard |
| `all loads on a stream must use the same transpose/...` | mixed transpose/non-transpose loads on one stream | use one stream per load type ([A.3](#a3--library-wide-behavioral-contracts)) |
| `transpose_axes is only supported on the indirect (gather) transpose` | `transpose_axes` on a static transpose | drop it; static transpose infers axes |
| `is ambiguous (3-D (2,1,0) vs the 4-D ...)` | >2-D gather transpose without `transpose_axes` | pass the explicit permutation (the `NDSlice.load` docstring) |
| `... exceeding the 2048-element PSUM bank capacity` | slot-packed pool over budget | fewer slots/bank, or all-fanout (the `nt.psum_pool` docstring) |
| `grid tiles exceed the 8 available PSUM banks` | all-fanout pool needs > 8 banks | reduce the grid or use slot-packing |
| a PSUM bank-conflict / scheduling compile error | PSUM tiles closer than `slot_stride` | honor `slot_stride = max(512, 4*tile_f)` (the `nt.psum_pool` docstring) |
| `Tensor Engine transpose limited to [128, 128]` | `nc_transpose` on a tile with F > 128 | use `load(transpose=True)` (DMA transpose) or tile to ≤ 128 |
| `(int, object)` | `int + LoopVar` arithmetic in a traced branch | route through indirect / `nl.ds` ([A.4](#a4--nki-parser-constraints)) |
| `failed to specialize` | a shape/bound isn't compile-time resolvable | make the value static, or consult the specialization-debugging guidance |
| "entry function not found" / "unsupported expression" | a rejected Python construct in the kernel body | check the parser checklist ([A.4](#a4--nki-parser-constraints)) |
