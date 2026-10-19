# Matmul Loop Nest Abstraction

## Overview

`matmul_loop_nest` replaces hand-written K×M×N matmul loops with a single function call. It handles PSUM allocation, bank management, operand indexing, edge tiles, and drain callbacks automatically.

## Quick Start

```python
from nkilib_src.nkilib.core.utils.tiled_tensor import TiledTensor
from nkilib_src.nkilib.experimental.primitives.matmul_loop_nest import matmul_loop_nest

# Allocate operands as TiledTensors
stat = TiledTensor.alloc(grid=(K_tiles, M_tiles), tile_size=(128, 128),
                         dtype=nl.bfloat16, buffer=nl.sbuf, sbm=sbm)
mov = TiledTensor.alloc(grid=(K_tiles, N_tiles), tile_size=(128, 512),
                        dtype=nl.bfloat16, buffer=nl.sbuf)
dst = TiledTensor.alloc(grid=(M_tiles, N_tiles), tile_size=(128, 512),
                        dtype=nl.bfloat16, buffer=nl.sbuf)

# One call replaces the entire loop nest
matmul_loop_nest(stationary=stat, moving=mov, dst_sbuf=dst)
```

## TiledTensor Construction

### Allocate fresh tiles
```python
# SBUF with BufferManager
weights = TiledTensor.alloc(grid=(n_heads, 1), tile_size=(128, 512),
                            dtype=nl.bfloat16, buffer=nl.sbuf, sbm=sbm)

# PSUM with bank management (one bank per tile, no reuse)
psum = TiledTensor.alloc(grid=(1, N_tiles), tile_size=(128, 512),
                         dtype=nl.float32, buffer=nl.psum, num_banks=N_tiles)

# Rotating buffer (double-buffering)
mov = TiledTensor.alloc(grid=(K_tiles, 1), tile_size=(128, 512),
                        dtype=nl.bfloat16, buffer=nl.sbuf, rotate=(0, 2))
```

### View over existing tensor (zero-cost grid ops)
```python
# Tile a contiguous buffer
attn_tiled = TiledTensor(attn_shuffled, tile_size=(128, bxs_size))

# Reshape to expose a dimension, then tile
view = buffer.reshape_dim(1, (n_heads, bxs_size))
tiled = TiledTensor(view, tile_size=(128, 1, block_size))
# tile_size=1 on a dim → iterate over it (becomes K or N axis)

# Grid ops (zero-cost, just stride math)
tiled = tiled.select(dim=2, index=block_idx)  # pick one block
tiled = tiled.squeeze_dim(0)                   # remove size-1 dim
tiled = tiled.slice(dim=1, start=0, end=cur_size)  # narrow range
```

### Wrap pre-existing tiles
```python
tiles = [tensor_a, tensor_b, tensor_c]
tiled = TiledTensor._make_tile_list(tiles, grid=(3, 1), tile_size=tiles[0].shape)
```

## matmul_loop_nest Parameters

```python
matmul_loop_nest(
    stationary,          # TiledTensor [K, M] — stationary operand
    moving,              # TiledTensor [K, N] — moving operand
    dst_psum=None,       # TiledTensor [M, N] in PSUM (auto-allocated if None)
    dst_sbuf=None,       # TiledTensor [M, N] in SBUF — drain target
    col_factor=1,        # PE column tiling (M-axis packing)
    col_dim=128,         # Column tile size
    n_packing=1,         # N-axis PSUM packing (multiple N tiles per bank)
    num_psum_banks=8,    # Banks for auto-allocation
    loop_order="NM",     # "NM" or "MN"
    stationary_dims=None,  # {"K": dim_idx, "N": dim_idx} for nD grids
    moving_dims=None,      # {"K": dim_idx, "M": dim_idx} for nD grids
    load_weights=None,   # fn(k, n, buf) — DMA callback per K tile
    on_post_matmul=None, # fn(psum, k, m, n) — after each nc_matmul
    on_drain=None,       # fn(psum, sbuf, m, n) — after all K tiles
    on_k_group_end=None, # fn(psum, m, n, col_factor) — reduce columns
    should_compute=None, # fn(k, m, n) → bool — skip tiles
    matmul_kwargs=None,  # fn(k, col_idx) → dict — extra nc_matmul args
)
```

## Key Patterns

### Basic GEMM
```python
matmul_loop_nest(stationary=stat, moving=mov, dst_sbuf=dst)
```

### K accumulation with weight loading
```python
def load_wt(k, n, buf):
    nisa.dma_copy(dst=mov[k, n], src=weights_hbm[k*128:(k+1)*128, :])

matmul_loop_nest(stationary=stat, moving=mov, dst_sbuf=dst, load_weights=load_wt)
```

### Custom drain (quant + bias)
```python
def drain(psum_tile, sbuf_tile, m, n):
    nisa.tensor_tensor(dst=sbuf_tile, data1=psum_tile, data2=scale, op=nl.multiply)
    nisa.tensor_tensor(dst=sbuf_tile, data1=sbuf_tile, data2=bias, op=nl.add)

matmul_loop_nest(stationary=stat, moving=mov, dst_sbuf=dst, on_drain=drain)
```

### Factored-N (stat and mov have independent N dimensions)
```python
# stat varies with (K=heads, N0=h2_indices)
# mov varies with (K=heads, N1=bxs_tiles)
# Loop iterates N0 × N1 automatically
matmul_loop_nest(
    stationary=w_tiled,   # grid (n_heads, cur_h2)
    moving=attn_tiled,    # grid (n_heads, num_bxs_tiles)
    dst_psum=psum,
    stationary_dims={"K": 0, "N": 1},
    moving_dims={"K": 0},
    n_packing=NUM_BS_PER_PSUM_BANK,
    on_drain=drain,
)
```

### N-axis PSUM packing
```python
# When N_tile < 512, pack multiple N tiles per PSUM bank
matmul_loop_nest(
    ...,
    n_packing=4,  # 4 tiles of 128 packed into one 512-wide bank
    on_drain=drain,  # receives full (M, 512) bank
)
```

## PSUM Rules

1. **num_banks must equal N_tiles** — never reuse banks across outer iterations (nc_matmul accumulates)
2. **PSUM auto-slice** — matmul_loop_nest slices PSUM to match actual tile sizes
3. **n_packing** — groups N tiles into one bank at different column offsets; drain receives full bank

## Parser Mode Compatibility

For `--nki-compilation-mode=parser`, callbacks must be bound methods of NKIObject classes (no lambdas, no inner functions):

```python
@dataclass
class MyDrain(nl.NKIObject):
    scale: object
    def run(self, psum_tile, sbuf_tile, m, n):
        nisa.tensor_tensor(dst=sbuf_tile, data1=psum_tile, data2=self.scale, op=nl.multiply)

cb = MyDrain(scale=scale_sb)
matmul_loop_nest(..., on_drain=cb.run)
```
