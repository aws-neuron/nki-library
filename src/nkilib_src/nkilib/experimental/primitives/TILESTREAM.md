# TileStream

TileStream is a tiling and iteration abstraction for NKI kernels. It decouples **how data is laid out in SBUF** from **how operations consume it tile-by-tile**, providing a uniform interface that DMA and BLAS primitives use to iterate over tiles without knowing the underlying buffer geometry.

## Core Concepts

### Logical vs Container Shape

NKI hardware has a **partition dimension** (P) with a fixed hardware size (`pdim_size`, typically 128). Tensors that are logically larger than `pdim_size` in P must be stored with an extra "p_tile" dimension.

| Shape type | Format | Example |
|---|---|---|
| **Logical** | `(P, *F)` | `(512, 64)` — 512 partition rows, 64 free elements |
| **Container** | `(pdim_size, n_p_tiles, *F)` | `(128, 4, 64)` — 4 tiles of 128 rows each |

The conversion is: `P = pdim_size * n_p_tiles`. The `alloc_logical` helper allocates SBUF in container shape from a logical shape specification.

### Tile Grid

Given a logical tensor and a `tile_shape`, TileStream computes a **tile grid** — the number of tiles along each dimension:

```
grid[dim] = ceil(logical_size[dim] / tile_size[dim])    # for tiled dims
grid[dim] = logical_size[dim]                            # for grid-only dims
```

### Tile Dims vs Grid Dims

- **Tile dims**: Dimensions that are sliced into tiles. Each tile gets a contiguous sub-range of the dimension.
- **Grid-only dims**: Dimensions not in `tile_dims`. Each grid position selects a single index (like `select`), removing the dimension from the tile.

Example: For a tensor of logical shape `(128, 4, 2, 64)` with `tile_dims=(0, 3)` and `tile_shape=(128, 64)`:
- Dims 0 and 3 are tiled (sliced into 128- and 64-element chunks)
- Dims 1 and 2 are grid-only (iterated one index at a time)
- Grid: `(1, 4, 2, 1)` — 8 total tiles

### ViewSpec

`ViewSpec` composes lazy transformations that are applied to each tile when extracted. Operations are chained immutably — each method returns a new `ViewSpec`.

```python
from nkiprimitives.view_spec import view

spec = view().reshape_dim(1, (4, 16)).permute((0, 2, 1))
ts = tile_stream.tile(buf, tile_shape=(128, 64), tile_view=spec)
```

**Available operations:**

| Operation | Description |
|---|---|
| `.slice(dim, start, end)` | Narrow dimension to `[start, end)` |
| `.select(dim, index)` | Pick single index, removing the dimension |
| `.reshape_dim(dim, shape)` | Split one dimension into multiple |
| `.permute(dims)` | Reorder dimensions |
| `.broadcast(dim, size)` | Broadcast: if dim is size 1, expand to `size`; otherwise reshape into `(original, 1)` then broadcast the new dim |
| `.expand(dim)` | Insert a size-1 dimension (unsqueeze) |
| `.stride(dim, stride)` | Override stride for a dimension |

ViewSpec affects the **effective tile shape** (what `get_tile_shape()` returns) and the actual `nl.NkiTensor` returned by `get_tile()`.

ViewSpec is used in two contexts: as a `tile_view` parameter (transforms each tile after extraction), and as part of `ViewOrder` iteration orders (reshapes the grid for non-standard iteration patterns like quadrant-stride access). Both use the same composable operation set.

## API Reference

### `alloc_logical(logical_shape, pdim_size, dtype, name=None, sbm=None, collapse_trivial_p_tile=False)`

Allocates an SBUF tensor with container layout from a logical shape.

```python
# Allocates container shape (128, 4, 64) in SBUF
buf = tile_stream.alloc_logical((512, 64), pdim_size=128, dtype=nl.float16, name="buf")
```

**Parameters:**
- `logical_shape`: `(P, *F)` — must have at least 2 dimensions
- `pdim_size`: Hardware partition dimension size
- `dtype`: Data type for the allocation
- `name`: Optional name for debugging
- `sbm`: Optional `BufferManager` for stack allocation
- `collapse_trivial_p_tile`: When `True` and `n_p_tiles == 1`, skips the p_tile dimension

**Returns:** `nl.NkiTensor` view of the allocated buffer.

### `tile(tensor, tile_shape, tile_dims=None, tile_view=None, virtual_grid=None, iter_order=None)`

Creates a `TileStream` from an SBUF or HBM tensor. Automatically detects the buffer type and returns the appropriate stream.

```python
ts = tile_stream.tile(buf, tile_shape=(128, 32), iter_order=RowMajor())
```

**Parameters:**
- `tensor`: SBUF tensor (from `alloc_logical` or `nl.NkiTensor`), HBM tensor, or any `nl.NkiTensor` view. Returns `None` if `tensor` is `None`.
- `tile_shape`: `(p_tile, *f_tiles)` — the size of each tile in logical dimensions. First element is the partition tile size. Must be `<= pdim_size` or an exact multiple of it.
- `tile_dims`: Which logical dimensions to tile. Defaults to all dims (trailing). Remaining dims become grid-only.
- `tile_view`: Optional `ViewSpec` to transform each extracted tile (reshape, broadcast, permute, etc.). **Note:** `tile_view` may be removed in a future version. All current use cases (broadcast, reshape+broadcast) can be expressed by applying view methods on the tensor before tiling and using `tile_dims` to make the original dimensions grid-only. See the bias broadcast example in Common Patterns.
- `virtual_grid`: Extra outer grid dimensions prepended to `tile_grid` for iteration count matching without extra allocation. Used for data reuse (e.g., weights consumed by multiple output tiles).
- `iter_order`: `RowMajor()` (default), `ColMajor()`, `DimOrder(order)`, or `ViewOrder(spec)`.

**Returns:** `TileStream` instance (or `None` if `tensor` is `None`).

### `TileStream`

The core class. Provides tile-by-tile iteration over a tiled tensor.

**Key methods:**

| Method | Returns | Description |
|---|---|---|
| `get_tile()` | `nl.NkiTensor` | Get current tile and advance the iterator |
| `get_tile_at_index(grid_pos)` | `nl.NkiTensor` | Get tile at a specific grid position |
| `reset_cur_tile()` | `None` | Reset iteration back to the first tile |
| `get_tile_shape()` | `tuple` | Effective tile shape (after `tile_view` transforms) |
| `get_base_tile_shape()` | `tuple` | Original tile shape (before transforms) |
| `get_tile_grid()` | `tuple` | Number of tiles in each dimension |
| `get_num_tiles()` | `int` | Total tile count (product of grid) |
| `get_logical_shape()` | `tuple` | Logical shape `(P, *F)` |
| `get_pdim_size()` | `int` | Partition dimension size |
| `get_n_p_tiles()` | `int` | Number of p_tiles this tile spans |
| `get_container()` | `nl.NkiTensor` | Underlying container tensor |
| `get_iter_order()` | `IterOrder` | Iteration order strategy |
| `get_tile_dims()` | `tuple` | Which logical dims are tiled |
| `get_grid_dims()` | `tuple` | Which logical dims are grid-only |

## Iteration Orders

Iteration orders control the order in which tiles are visited.

### RowMajor / ColMajor

Named presets for common iteration patterns:

| Order | Behavior |
|---|---|
| `RowMajor()` | Rightmost dimension changes fastest (default) |
| `ColMajor()` | Partition dim (0) changes fastest, then free dims right-to-left |

### DimOrder

Custom dimension ordering. `order` is a tuple listing dimensions from fastest-changing to slowest-changing. `RowMajor` and `ColMajor` are named presets of `DimOrder`.

```python
DimOrder(order=(2, 1, 0))  # dim 2 fastest, dim 0 slowest
```

### ViewOrder

Reshapes the tile grid before iteration using a `ViewSpec`. The ViewSpec transforms are applied to the grid shape (not the tile data), producing a new iteration space with potentially more dimensions, reordered dimensions, or selected slices. ViewOrder exists for iteration patterns that cannot be expressed by reordering tile dimensions alone — specifically when the grid itself needs restructuring. In most cases, the preferred approach is to apply view methods on the tensor before passing it to `tile()`, so that the tiling naturally produces the desired grid; ViewOrder is reserved for cases where the iteration order depends on grid structure rather than tile content.

```python
# Split grid dim 0 (size 32) into (4, 8), then iterate dim 1 (quadrant) slowest
ViewOrder(ViewSpec().reshape_dim(0, (4, 8)).permute(1, 0))
```

`ViewOrder` supports `reshape_dim` with non-divisible dimensions (the last group is automatically smaller to cover the remainder) and `select` (fixing a grid dimension to a constant). For example, `reshape_dim(0, (3, 4))` on a grid dim of size 10 produces groups of sizes `(4, 4, 2)` — the outer dim iterates 3 groups, and the last group has only 2 elements instead of 4.

### APOrder (Theoretical)

`APOrder` would express iteration orders via NKI access pattern (`.ap()`) semantics — stride/count pairs that define how to step through the flat tile index space. This is the lowest-level representation: every iteration order ultimately compiles down to a sequence of stride/count pairs.

```python
# Hypothetical: iterate 4 quadrants of 8 tiles, quadrant-stride access
APOrder(pattern=[[8, 4], [1, 8]])  # stride=8 count=4 (outer), stride=1 count=8 (inner)
```

**Relationship hierarchy:**

```
APOrder          — stride/count pairs on flat index space (most expressive, least readable)
  ↑
ViewOrder        — ViewSpec transforms on the grid (reshape_dim, permute, slice, select)
  ↑
DimOrder         — custom dimension ordering
  ↑
RowMajor/ColMajor — specific dimension ordering
```

# Primitives

Primitives are operations that consume TileStreams. They are split into two levels:

- **Level 0**: Pure loop removal. Each primitive maps to a single `nisa` instruction per tile. The primitive only handles iteration — no flags, no heuristics, no multi-instruction sequences.
- **Level 1**: Composite operations. Each primitive issues multiple `nisa` instructions per tile, and may use flags or heuristics to select between code paths (e.g., bias addition, quantization mode, engine selection).

## Level 0 — Single-Instruction Primitives

### DMA Primitives

#### `Load` / `load`

Loads data from HBM into an SBUF TileStream.

```python
# Class API (tiled)
dma.Load(dst=sbuf_ts, src=hbm_ts).execute()

# Compact API (whole tensor, no tiling)
dma.load(dst=sbuf_tensor, src=hbm_tensor)
```

#### `Store` / `store`

Stores data from SBUF TileStream back to HBM.

```python
# Class API (tiled)
dma.Store(dst=hbm_ts, src=sbuf_ts).execute()

# Compact API (whole tensor, no tiling)
dma.store(dst=hbm_tensor, src=sbuf_tensor)
```

#### `TensorCopy` / `tensor_copy`

Copies data between two SBUF TileStreams.

```python
# Class API (tiled)
dma.TensorCopy(dst=dst_ts, src=src_ts).execute()

# Compact API (whole tensor, no tiling)
dma.tensor_copy(dst=dst_tensor, src=src_tensor)
```

### BLAS Primitives

#### `TensorTensor`

Element-wise binary: `dst = op(src1, src2)`.

```python
blas.TensorTensor(dst=dst_ts, src1=a_ts, src2=b_ts, op=nl.multiply).execute()
```

#### `TensorScalar` / `tensor_scalar`

Element-wise tensor-scalar: `dst = op1(op0(src, operand0), operand1)`.

```python
# Class API
blas.TensorScalar(dst=dst_ts, src=src_ts, op0=nl.multiply, operand0=0.5).execute()

# Compact API (whole tensor)
blas.tensor_scalar(dst=buf, op0=nl.multiply, operand0=0.5)
```

#### `Activation` / `activation`

Applies activation functions (e.g., `nl.gelu`, `nl.silu`) with optional bias and scale. Maps to a single `nisa.activation` per tile.

#### `QuantizeMX`

MX-format quantization producing quantized data + scales. Maps to a single `nisa.quantize_mx` per tile.

#### `Broadcast` / `broadcast`

Broadcasts a tensor across dimensions. Maps to a single `nisa.tensor_copy` per tile (stream shuffle).

#### `Reciprocal` / `reciprocal`

Element-wise reciprocal. Maps to a single `nisa.activation` per tile.

## Level 1 — Composite Primitives

### `Matmul`

Matrix multiplication with optional MX quantization, bias, and dequantization. For each output tile, caches stationary and moving tiles, then loops over K (contraction dimension) accumulating into PSUM. After accumulation, evicts PSUM to SBUF with optional dequantization scaling, bias addition, and PSUM reshape.

```python
blas.Matmul(
    dst=out_ts,
    moving=moving_ts,
    stationary=stationary_ts,
    moving_scale=m_scale_ts,     # optional, required together with stationary_scale for MX
    stationary_scale=s_scale_ts, # optional, required together with moving_scale for MX
    bias=bias_ts,                # optional, added during PSUM eviction
    dequant_scale=dq_ts,         # optional, applied during PSUM eviction
    dequant_type=QuantizationType.ROW, # optional, default NONE
    psum_evict_view=view_spec,   # optional ViewSpec applied to PSUM before eviction
    perf_mode="double_row",      # optional, passes matmul_perf_mode.double_row to nisa.nc_matmul
    psum_buffer_degree=2,        # optional, explicit PSUM bank rotation (None = auto allocation)
    skip_evict=False,            # optional, when True accumulates directly into dst (must be nl.psum)
).execute()
```

**Tile shape contract:** `dst=(P, F_dst)`, `stationary=(K, P)`, `moving=(K, F_mov)` where `F_dst = F_mov * pack_factor`. The K dimension is the shared reduction axis.

**Packing:** When one operand has more grid entries than the output, multiple tiles are accumulated into one output tile. If `stationary_grid > dst_grid`, stationary tiles are packed (multiple K groups per output). If `moving_grid > dst_grid`, moving tiles are packed. The matmul auto-detects the packing direction.

### `Transpose` / `transpose`

Transposes P and F dimensions between TileStreams: `src (P, F) → dst (F, P)`. Issues `nisa.nc_transpose` (SBUF → PSUM), then `nisa.tensor_copy` (PSUM → SBUF) or `nisa.activation` (when source and destination dtypes differ, for implicit cast).

```python
# Class API (tiled)
blas.Transpose(dst=dst_ts, src=src_ts).execute()

# Compact API (whole tensor)
blas.transpose(dst=dst_tensor, src=src_tensor)
```

**Implicit broadcasting:** Supports broadcast-transpose patterns:
- `src (P, 1) → dst (B, P)`: broadcasts F=1 to B partitions in the output
- `src (1, F) → dst (F, B)`: broadcasts P=1 to B free elements in the output

**Tile constraint:** Only 2D tiles are supported.

## Compact (Lowercase) Primitives

Most primitives have a lowercase variant (e.g., `dma.load`, `blas.activation`, `blas.transpose`) that operates on entire tensors without explicit tiling. These are syntax sugar: they internally call `tile_stream.tile()` on the input tensors with a tile shape that covers the full logical extent, then execute the primitive on the resulting single-tile stream.

This is useful when a tensor fits within a single tile and no custom iteration order is needed — the common case for small buffers or post-processing steps:

```python
# Instead of:
dma.Load(
    dst=tile_stream.tile(sbuf_buf, (128, H)),
    src=tile_stream.tile(hbm_tensor, (128, H)),
).execute()

# Write:
dma.load(dst=sbuf_buf, src=hbm_tensor)
```

```python
# Instead of:
blas.Transpose(
    dst=tile_stream.tile(dst_buf, (H1_shard, H0)),
    src=tile_stream.tile(src_buf, (H0, H1_shard)),
).execute()

# Write:
blas.transpose(dst=dst_buf, src=src_buf)
```

## Common Patterns

### ColMajor for Multi-K Matmuls

When `k_grid > 1`, use `ColMajor()` on stationary and moving streams. This ensures K tiles are visited contiguously, preventing tile cache scrambling during accumulation.

```python
blas.Matmul(
    dst=tile_stream.tile(out_buf, (bxs, F_MAX)),
    moving=tile_stream.tile(wght_buf, (d, h_sharded), iter_order=ColMajor()),
    stationary=tile_stream.tile(attn_buf, (d, bxs), iter_order=ColMajor()),
).execute()
```

### Views Before Tiling

Apply `reshape_dim`, `permute`, `slice` on the tensor before `tile()` to match the desired layout.

```python
wght_view = (
    weight
    .slice(1, h_start, h_start + h_sharded)
    .reshape_dim(1, (h0, h1))
    .reshape_dim(0, (n_p_tiles, d_packed))
    .permute((1, 0, 2, 3))
)
wght_ts = tile_stream.tile(wght_view, (d_packed, _pmax), tile_dims=(0, 1), iter_order=ColMajor())
```

### tile_dims for Selective Tiling

Use `tile_dims` to tile only specific dimensions while iterating others as grid-only.

```python
# 3D weight: (d_packed, n_p_tiles, H) — tile d_packed and H, iterate n_p_tiles as grid-only
stat_ts = tile_stream.tile(weight_buf, (_pmax, I128_tile), tile_dims=(0, 2), iter_order=ColMajor())
```

### Bias Broadcast via a View (Preferred over tile_view)

Instead of using `tile_view` to broadcast per-tile, apply `expand` and `broadcast` on the tensor before tiling, then use `tile_dims` to make the original dimension grid-only.

```python
# bias_buf logical shape: (h0, h1)
# Goal: each tile should be (h0, bxs_tile), iterating over h1

# With tile_view (may be removed):
bias_ts = tile_stream.tile(bias_buf, (h0, 1), tile_view=view().broadcast(-1, bxs_tile))

# Preferred: build the view before tiling
bias_expanded = bias_buf.expand(2).broadcast(2, bxs_tile)  # (h0, h1, bxs_tile)
bias_ts = tile_stream.tile(bias_expanded, (h0, bxs_tile), tile_dims=(0, 2))
# h1 is grid-only → each tile is (h0, bxs_tile), iterating h1 times
```

### psum_evict_view for Output Reshaping

Reshape PSUM before eviction when the matmul output layout differs from the destination layout.

```python
blas.Matmul(
    dst=dst_ts, moving=mov_ts, stationary=stat_ts,
    psum_evict_view=view().reshape_dim(1, (_q_width, BxS_tile)).permute((0, 2, 1)),
).execute()
```

### Slice Before Tiling

When the allocated buffer is larger than needed (e.g., output P < allocated P), slice the view before `tile()`.

```python
out_sb = tile_stream.alloc_logical((_pmax * n_tiles, BxS * _q_width), _pmax, nl.bfloat16)
out_sb_sliced = out_sb.slice(dim=0, start=0, end=I128_tile)
dst_ts = tile_stream.tile(out_sb_sliced, (I128_tile, BxS * _q_width))
```

### Store with Reshape to HBM

Apply view reshaping on source and destination when storing back to HBM.

```python
dma.store(dst=output.select(1, lnc_id), src=out_buf.reshape_dim(1, (h1, bxs)))
```
