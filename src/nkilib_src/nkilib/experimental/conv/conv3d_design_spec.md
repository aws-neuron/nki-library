# Conv3D Kernel Design Spec

## Overview

The `conv3d` kernel implements 3D convolution optimized for Trainium2 (trn2) and Trainium3 (trn3). A 3D convolution slides a filter kernel across three spatial dimensions (depth, height, width) of an input tensor, computing a weighted sum at each position to produce an output tensor. The kernel supports configurable stride, asymmetric zero-padding, dilation, bias, optional fused activation, LNC sharding, an optional fused batch normalization (equivalent to `torch.nn.BatchNorm2d`, see [Fused Batch Normalization](#fused-batch-normalization)), and an optional fused residual add on either side of the activation (see [Fused Residual Add](#fused-residual-add)). It also supports Conv2D (by setting `D=1, K_d=1`) and Conv1D (by setting `D=1, H=1, K_d=1, K_h=1`).

The kernel's strategy is to use the **tensor engine** (`nisa.nc_matmul`) to perform the convolution as a series of matrix multiplications. For each filter position `(k_d, k_h, k_w)`, the convolution reduces along the `C_in` (input channel) dimension with a stationary tensor of the filter weights for a given filter position, with `C_in` on the partition dimension and `C_out` on the free dimension, and a moving tensor corresponding to input values with `C_in` on the partition dimension and and different spatial output positions packed along the free dimension. Each column in the free dimension of the moving tensor corresponds to the input position that the filter at `(k_d, k_h, k_w)` would multiply with for a given location.

By contracting along `C_in` and accumulating across all filter positions into PSUM, we accumulate complete output values using the tensor engine, avoiding the need to use the vector engine for reduction. Once full output values are accumulated in PSUM, we fuse bias addition and activation on the copy out to SBUF then to HBM.

When batch normalization is fused in training mode, the convolution output is written to a separate raw HBM scratch buffer (`conv_out`) and the per-channel statistics are accumulated on the copy-out; a subsequent finalize pass then normalizes `conv_out` into the final `y_out`. In eval mode (`batch_norm_mode=BatchNormMode.EVAL`) the normalization uses the given running statistics, so it is applied directly on the PSUM copy-out and no scratch buffer or finalize pass is needed. See [Fused Batch Normalization](#fused-batch-normalization) for the full data flow, which changes the return value and the meaning of the activation fusion.

### Public interface

```python
class BatchNormMode(Enum):
    NONE = 0      # no fused batchnorm
    EVAL = 1      # normalize with the given running stats; no stats / momentum update
    TRAINING = 2  # compute batch stats, normalize with them, momentum-update running stats

class ResidualAddLoc(Enum):
    NONE = 0      # no fused residual add
    PRE_ACT = 1   # y = act(post_norm_conv + residual)
    POST_ACT = 2  # y = act(post_norm_conv) + residual

def conv3d(
    x_in, filters, bias=None,
    stride=(1, 1, 1), padding=(0, 0, 0, 0, 0, 0), dilation=(1, 1, 1),
    activation_fn=None, lnc_shard=False,
    # Fused batch normalization (gamma / beta / momentum / running stats are all required
    # together whenever batch_norm_mode is EVAL or TRAINING):
    batch_norm_mode=BatchNormMode.NONE, output_pre_norm=False,
    batch_norm_eps=1e-5, gamma=None, beta=None, momentum=0.1,
    running_means=None, running_variances=None,
    # Fused residual add (residuals_in is required whenever residual_add_loc is not NONE):
    residual_add_loc=ResidualAddLoc.NONE, residuals_in=None,
) -> y_out  # or a 5- / 6-tuple in BatchNormMode.TRAINING (see below)
```

The `batch_norm_mode` enum makes the three batchnorm behaviors mutually exclusive by construction; only the orthogonal `output_pre_norm` flag remains, and it is valid in `TRAINING` mode alone.

Without batchnorm (`BatchNormMode.NONE`) the kernel returns `y_out` `[B, C_out, D_out, H_out, W_out]`. With `BatchNormMode.TRAINING` it returns the tuple `(y_out, means, variances, updated_running_means, updated_running_variances)`; `gamma`, `beta`, `momentum`, `running_means`, and `running_variances` must all be supplied. With `BatchNormMode.EVAL` it returns only `y_out`. With `output_pre_norm=True` the raw pre-batchnorm convolution output is inserted after `y_out`, giving the 6-tuple `(y_out, conv_out, means, variances, updated_running_means, updated_running_variances)` — see [Returning the pre-batchnorm output](#returning-the-pre-batchnorm-output-output_pre_normtrue). See [Fused Batch Normalization](#fused-batch-normalization) for what each output holds and [Eval mode](#eval-mode-batchnormmodeeval) for the eval-mode path. `output_pre_norm` is a don't-care in `BatchNormMode.NONE` and is rejected in `BatchNormMode.EVAL`.

**Output dimensions:**
```
D_out = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
H_out = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
W_out = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1
```

### Tensor Layout

| Tensor  | Shape                              | Notes |
|---------|------------------------------------|-------|
| Input   | `[B, C_in, D, H, W]`              | Standard channels-first, unchanged from PyTorch |
| Filters | `[K_d, K_h, K_w, C_in, C_out]`    | Reshaped from PyTorch's `[C_out, C_in, K_d, K_h, K_w]` |
| Bias    | `[C_out]`                          | Standard 1D, unchanged from PyTorch |
| Residual| `[B, C_out, D_out, H_out, W_out]` | Same layout as the output (only with a fused residual add) |
| Output  | `[B, C_out, D_out, H_out, W_out]` | Standard channels-first, unchanged from PyTorch |

The input, output, and bias tensors use the same layout as standard PyTorch convolution.

The filter tensor is reshaped to `[K_d, K_h, K_w, C_in, C_out]`. This is done because for a given filter position `(k_d, k_h, k_w)`, we need to load a `[C_in, C_out]` slice as the stationary tensor for the matmul. With this layout, the `C_in` and `C_out` dimensions are contiguous in memory for each filter position, making the DMA copies to load filters into SBUF more efficent. As filters are a weight this reshape can be done once on host at a negligible amortized cost.

### Tiling Strategy

The outermost loop is over batch. Within each batch, we keep a set of C_out tiles (filter weights) resident in SBUF, these are the filters needed to fully compute a portion of the output channels across all `C_in` tiles.

For the spatial dimensions, we have three axes: depth (D), height (H), and width (W). Since W values are contiguous in memory, we process multiple output positions along W simultaneously. We also know that for a given `(d, h)` pair, the W output positions are independent and contiguous, so we stack multiple D-H positions along the free dimension of the moving tensor to try to hit the maximum free dimension size (512 elements) for the tensor engine. This is the `num_dh_stacked` parameter.

If `W > 512`, we tile along W as well (`W_tile`).

Within each spatial tile, we iterate over **C_in tiles** for the input. For each C_in tile, we perform a large contiguous DMA copy of the input window from HBM into an SBUF buffer. We then use the vector and scalar engines to scatter the input window into the stacked layout expected by the matmul, where each column in the moving tensor corresponds to the next spatial position that a particular filter position would multiply with. We allocate PSUM banks for the output tiles and accumulate matmul results across all C_in tiles and filter positions. Once we have accumulated complete output values in PSUM, we copy the results out.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW PIPELINE                                        │
│                                                                                  │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐              │
│  │  HBM         │     │  SBUF            │     │  SBUF            │              │
│  │  x_in[C_in,  │ DMA │  Input Window    │ tc  │  Stacked Input   │              │
│  │   D,H,W]     │────►│  [C_in,d_win,    │────►│  [K_REP*C_in,    │              │
│  │              │     │   h_win,w_win]   │     │   num_dh*W_tile] │              │
│  └──────────────┘     └──────────────────┘     └────────┬─────────┘              │
│                        (DMA engine)             (vector/scalar engine)           │
│                                                         │                        │
│                                                         │ nc_matmul              │
│                                                         ▼                        │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐              │
│  │  HBM         │     │  SBUF            │     │  PSUM            │              │
│  │  y_out[C_out,│ DMA │  Result          │ tc  │  Accumulated     │              │
│  │   D_out,     │◄────│  [C_out,         │◄────│  [C_out,         │              │
│  │   H_out,W_out│     │   num_dh*W_tile] │     │   num_dh*W_tile] │              │
│  └──────────────┘     └──────────────────┘     └──────────────────┘              │
│                        (DMA engine)        (vector/scalar engine)                │
│                                            + optional bias/activation            │
│                                                                                  │
│  tc = tensor_copy    DMA = dma_copy    nc_matmul = tensor engine matmul          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Pseudocode:

```
for batch in range(B):
    for c_out_group in range(0, C_out, c_out_interleave * P_MAX):

        load filters + bias for this C_out group

        for dh_group in range(dh_start, dh_end, num_dh_stacked):
            for w_tile in range(0, W_out, W_tile):

                allocate PSUM banks for output tiles

                DMA load this tile's residual slice from HBM → SBUF (if fused)

                for c_in_tile in range(0, C_in, P_MAX):

                    DMA load input window from HBM → SBUF

                    scatter input window → stacked input layout

                    nc_matmul: accumulate into PSUM

                apply bias + activation + residual add, copy PSUM → result SBUF

                DMA copy result SBUF → HBM
```

### Memory Strategy

We use `SbufManager`'s heap allocator to allocate all buffers at kernel start, calculating the total memory needed upfront. Based on the workload characteristics, we determine several interleaving factors:

- **`c_out_interleave`**: How many C_out tiles we process at once. This is prioritized because having more C_out tiles (or ideally all of them) resident in SBUF means we don't have to reload filter data leading to high temporal locality and reuse for the filter weights.

- **`input_window_interleave`**: How many input window buffers we allocate in SBUF. The input window is the contiguous region of `x_in` copied from HBM to SBUF via DMA. Having at least two buffers enables double-buffering (loading the next window while the current one is being scattered). More buffers help when the workload is memory-intensive (large spatial dimensions or large `C_in`).

- **`stacked_input_interleave`**: How many stacked moving tensor buffers we allocate for the actual matmul input. More of these means the vector and scalar engines don't have to wait for a matmul to finish before writing the next scattered input, avoiding memory antidependencies.

- **`w_out_interleave`**: How many result SBUF buffers we allocate for W tiles. More buffers allow overlapping DMA stores of previous results with compute of the current tile.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          SBUF MEMORY LAYOUT                                │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  BN stats buffer    (C_OUT_REP x col_size,                       │      │
│  │                      c_out_tiles x flushes x 6) fp32             │      │
│  │  (training batchnorm only; one nisa.bn_stats group per flush)     │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Bias buffers       [c_out_interleave] x (P_MAX, 1) fp32         │      │
│  │  (loaded once per C_out  group)                                  │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Filter buffers     [c_in_tiles x K_outer_tiles]                 │      │
│  │                     x (stacked_filter_dim, c_out_wide)           │      │
│  │  (loaded once per C_out group)                                   │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Result SBUF        [w_out_interleave x c_out_interleave]        │      │
│  │                     x (P_MAX, num_dh_stacked x W_tile)           │      │
│  │  (rotated across W tiles for store/compute overlap)              │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Residual SBUF      same shape / rotation as Result SBUF          │      │
│  │  (only with a fused residual add, non-training batchnorm modes)   │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Input windows      [input_window_interleave]                    │      │
│  │                     x (P_MAX, d_window, h_window, w_window)      │      │
│  │  (multi-buffered for DMA load / scatter overlap)                 │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │  Stacked inputs     [stacked_input_interleave x K_outer]         │      │
│  │                     x (stacked_filter_dim, effective_free)       │      │
│  │  (multi-buffered for scatter / matmul overlap)                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

The memory configuration calculation algorithm starts with maximum interleaving and iteratively reduces until the total SBUF budget is met.

> **Note:** These memory calculations could be improved with a cost model that considers the relative costs of DMA, scatter, and matmul for a given configuration.

### Sharding Strategy

When LNC=2, we either:

- **Shard on D-H** (`shard_on_dh=True`): Each core processes half of the `D_out * H_out` positions and loads all the filters. This is beneficial when the filter data is small relative to the spatial work.

- **Shard on C_out** (`shard_on_dh=False`): Each core processes only half the output channels but must process the entire input window. This is beneficial when the filter data is large.

We determine which strategy to use by comparing the "waste" (load imbalance) of splitting each dimension across two cores: `waste = 2 * ceil(work / 2) - work`. The dimension with less waste is chosen.

> **Note:** This calculation can be improved with a cost model and more analysis. For example, it might be better to shard on C_out in some cases so that both cores avoid reloading filters entirely, even if the D-H split would be more balanced.

**Cross-core note (For batchnorm).** At LNC=2 both cores execute the whole kernel and need to synchronize the cross-core statistics reduction with `nisa.sendrecv` (SBUF↔SBUF).

### Engine Balancing

The `tensor_copy` operation can run on either the scalar engine or the vector engine. Balancing between these is important for good performance, so we use a configurable modulo factor: every N-th call is sent to one engine vs. the other (default: 1 in 4 calls go to scalar, rest to vector).

Similarly, `memset` can run on the vector engine or GPSIMD engine, and we balance between them with a separate modulo factor (default: 1 in 2 calls go to GPSIMD).

> **Note:** These ratios are currently fixed constants. They could be improved using a cost model that considers whether operations like bias (which requires `tensor_scalar` on the vector engine) or activation (which requires `activation` on the scalar engine) are being used, as those would change the load on each engine.

> **Note:** For `memset`, it may be worth considering whether DMAs use HWDGE (sync engine) vs. SWDGE (GPSIMD) to determine how much work can be offloaded to the GPSIMD.

### K-Replication

When `C_in` is not a multiple of the partition dimension (128), some matmuls will have underutilized tensor engine lanes. To mitigate this when `C_in < 128`, we replicate multiple filter positions along the partition dimension within a single matmul call, packing more useful work into each tensor engine invocation.

The number of positions we can replicate depends on partition access rules:

| C_in range | Partition stride | Max K_REP | Explanation |
|------------|-----------------|-----------|-------------|
| 64 < C_in ≤ 128 | C_in | 1 | Can only access partitions 0–127 from position 0; no room to replicate |
| 32 < C_in ≤ 64 | 64 | 2 | Can access 0–63 from position 0, 64–127 from partition 64 |
| C_in ≤ 32 | 32 | 4 | Can access 0–31 from 0, 32–63 from 32, 64–95 from 64, 96–127 from 96 |

This enables better tensor engine utilization but requires `memset` operations in some cases to zero-fill the empty spaces in the partition dimension not occupied by real data.

### Padding

We only support asymmetric zero-padding currently.

Padding is handled through two mechanisms:

1. **Memset**: Zero-initialize the portions of the stacked input buffer that correspond to padded positions.

2. **Variable free-dimension matmul** (to avoid memsets where possible): Instead of memset-ing padding positions, we adjust the free dimension of the `nc_matmul`. For example, if `W = 128` and we stack 4 D-H groups, the full free dimension would be `4 × 128 = 512`. But if some positions on the edges have padding, we can decrease the free dimension (e.g., `128 → 127`), start accumulating at an offset in PSUM, and use a strided access pattern for the `nc_matmul`. This avoids wasted memset operations and wasted matmul compute on zero-padded positions.

### Fusing Bias and Activation

We support fusing bias addition and activation function on the copy from PSUM to SBUF:

- **Neither bias nor activation**: We do a `tensor_copy` from PSUM to SBUF, which allows us to balance across engines (vector/scalar).
- **Bias only**: We use `tensor_scalar` (vector engine) to add the bias in-flight during the PSUM → SBUF copy.
- **Activation only**: We use `activation` (scalar engine) to apply the activation in-flight during the PSUM → SBUF copy.
- **Both bias and activation**: We first apply bias using `tensor_scalar` (vector engine) from PSUM → SBUF, then apply activation using `activation` (scalar engine) from SBUF → SBUF.

> **Note:** In `BatchNormMode.TRAINING`, the activation is **not** applied here. Batchnorm statistics must be computed on the raw (pre-activation) convolution output, so the activation is deferred to the rescale phase and applied to the normalized values instead (see [Fused Batch Normalization](#fused-batch-normalization)). Bias, if present, is still fused in on the copy-out.

### Fused Residual Add

When `residual_add_loc` is `PRE_ACT` or `POST_ACT`, a residual tensor `residuals_in` of the output's shape `[B, C_out, D_out, H_out, W_out]` is added to the fully post-processed convolution output — always **after** the fused batchnorm, when one is present, and on the requested side of the activation:

```
PRE_ACT :  y = act(bn(conv + bias) + residual)
POST_ACT:  y = act(bn(conv + bias)) + residual
```

When `activation_fn` is `None` the two locations are algebraically identical, so both collapse onto the same (post-activation) code path. The residual add is available in **every** `BatchNormMode`; it never affects the batchnorm statistics, which are always computed on the raw convolution output (so `conv_out` under `output_pre_norm=True` also stays residual-free).

#### `BatchNormMode.NONE` / `EVAL`: on the PSUM eviction

Both non-training modes produce the final output on the PSUM eviction, so the residual is added there too, in `_apply_bias_activation_and_copy`. The residual tile is staged in SBUF with the **same layout as the result tile** it is added into (same partition bands, same band-local free coordinates), so the add is a single `tensor_tensor` in place on the result tile:

```
result = result + residual        # tensor_tensor, before or after the eviction's activation
```

`residual_add_loc` only changes the placement relative to the `activation`. One interaction on the `BatchNormMode.NONE` path: `PRE_ACT` with **both** bias and activation has to un-fuse the usual single `activation(psum, bias=...)` into `tensor_scalar(bias)` → `tensor_tensor(residual)` → `activation`, since the residual must land between the two. Eval mode is unaffected, because the bias is already folded into its affine coefficients. Never-written (fully padded) positions are filled with `f(0 + bias)` / `bn(0 + bias)` *before* the add, so they pick the residual up like every other position.

#### Memory

Because the residual is result-shaped, it reuses the result buffers' allocation strategy exactly: one residual slot per `(w_out, c_out)` interleave slot, same shape, same rotation. `_build_memory_config` accounts for this with `result_bufs_per_slot` (2 with a residual, 1 without), which doubles the result-tile term in the fit search, the alignment accounting and the total — so the interleave factors shrink automatically if the extra buffers do not fit. The load itself mirrors the store: one DMA per `(C_out tile, column-tiling band)`, issued in the inner W-tile loop **before** the tile's compute so it overlaps the matmuls (`_dma_result_bands` with `to_hbm=False` — the same helper the result store uses, so the load and store cannot disagree on the band layout).

**In-place staging.** The second buffer is only needed when the eviction writes the result tile *before* it reads the residual. When instead the residual is consumed by the very first op — the one that reads PSUM, i.e. `tensor_tensor(psum, residual)` or `scalar_tensor_tensor(psum, bias, residual)` — that op's own write is what overwrites the staged value, so the residual can be staged **into the result tile itself** and `result_bufs_per_slot` stays 1. `Conv3dConfig.residual_in_place()` gates this: `BatchNormMode.NONE` with either no activation or a `PRE_ACT` residual. The batchnorm affine and `POST_ACT`-with-activation both evict into the tile first, so they keep a separate staging tile.

This halves the hot-path result footprint on the eligible forms, which matters at the memory-fit boundary: on `C_in=C_out=1024, 1×60×104` the fit search picks `store_pipe=7` instead of `3`, and on `C_in=2048, C_out=1024, 1×30×52` it keeps double-buffering (`store_pipe=2`) rather than falling to `1`. Measured on trn2, the recovered depth did **not** change wall-clock on those shapes (−0.03% / +0.21% / +0.12%, within a ±0.34% run-to-run band) because they run at 93–96% MFU and are compute-bound, not eviction-pipeline bound. The value is the freed SBUF, not latency.

**Merged-store path.** Under store-batch merging the residual load is issued **once per batch group**, as one batch-strided DMA per `(C_out tile, band)` — the merged store and the merged load are the same helper, `_dma_merged_bands(..., to_hbm=)`, so they cannot disagree on the layout (the same way `_dma_result_bands` serves both directions on the per-batch path). Issuing it inside the per-batch compute loop instead would cost `store_batch_merge`× as many DMA triggers as the store that path exists to shrink. Measured on trn2 (`B=4, C_in=C_out=32, 1×32×32`, 6 runs each, medians): **26.5 µs → 23.6 µs (−10.8%)** for `POST_ACT` and **27.1 µs → 24.1 µs (−10.9%)** for `PRE_ACT`, with disjoint ranges (the slowest merged run still beats the fastest per-batch run by 4–6%).

#### `BatchNormMode.TRAINING`: in the rescale phase

Training mode does not produce the final output in the main loop (the eviction writes the raw `conv_out`), so it adds the residual in the rescale phase instead, where the normalization happens. A second reload buffer (`bn_residual_reload`, identical in shape to `bn_reload`) is loaded band-by-band alongside `conv_out`, halving the reload interleave the free SBUF supports.

With **no activation** the residual is just one `tensor_tensor` after the existing rescale `tensor_scalar`. With an activation, the add uses the additive coefficient `true_beta = -neg_true_beta` (computed by negating `neg_true_beta` in place once per `C_out` tile, so no extra buffer):

- **`PRE_ACT`**: `activation(op=copy, scale=true_gamma, bias=true_beta)` applies the *whole* affine in one instruction, then `tensor_tensor` adds the residual, then `activation` applies `ActFn`. Three ops — deliberately not two.

  A two-op form is available (`scalar_tensor_tensor` for `compute_tile * true_gamma + residual`, then `activation(bias=true_beta)`) but is **numerically wrong to use here**: it splits the affine so the value written back to `reload_buf` still carries the *un-shifted* mean magnitude, and that intermediate is rounded to the buffer's dtype. The error scales with `mean / std`; in bf16 with a per-channel mean of ~10 std it puts ~18% of elements outside the kernel's own `rtol=atol=1e-2` tolerance (0% for the three-op form). Zero-mean test inputs hide this entirely, which is why the cheaper form looks correct in the suite.

  This ordering **is** enforced by a test, in every mode: the `TRAINING` cases of `CONV3D_RESIDUAL_LARGE_MEAN_PARAMS` run at `x_in_mean=10.0` (a few rows at 4.0 — see below). That only became possible once the statistics moved to `nisa.bn_stats` / `nisa.bn_aggr` ([Design choice 1](#design-choice-1--statistics-via-nisabn_stats--nisabn_aggr)) — the former `E[Y²] − E[Y]²` identity lost the variance output's tolerance at a *lower* `mean / std` than this fold needs, so reaching the regime broke the statistics before it could exercise the rescale, and only `NONE` / `EVAL` could be covered.

  Most `TRAINING` large-mean cases run in **float32** rather than bfloat16, which is load-bearing: `TRAINING` stages the raw convolution output in an `x_in.dtype` HBM tensor (`conv_out`) and `bn_stats` reads it back from there, so in bfloat16 the statistics are taken over values already rounded to 8 mantissa bits. That rounding enters the variance in proportion to `(mean/std)²` even with an arithmetically exact aggregation, versus 0.000% in float32, so float32 is what puts the algorithm itself under test. `NONE` / `EVAL` stage nothing and stay bfloat16.

  The exact `x_in_mean` and dtype are per-row rather than uniform, and the test table is the authority — each row carries the measured reason. Two competing limits set them: raising the mean amplifies the *statistics* error the test is trying to catch, but it equally amplifies the error in the `out` output, because normalizing subtracts the channel mean and so measures the convolution's own accumulation error against `std` instead of against `|conv|`. That second effect is inherent to normalizing a large-mean tensor and independent of how the statistics are computed, so shapes squeezed from both sides run bfloat16 (whose looser tolerance admits the `out` error) or a lower mean. `x_in_mean=20.0` in float32 is past the `out` tolerance for the larger-`C_in` shapes.
- **`POST_ACT`**: `activation` computes `ActFn(compute_tile * true_gamma + true_beta)` with its `scale` / `bias` operands, then one `tensor_tensor` adds the residual.

`POST_ACT` therefore costs two instructions per reloaded tile and `PRE_ACT` three; only the latter pays for the mean shift landing on the wrong side of a dtype rounding. Column tiling and LNC sharding need no special handling: the coefficients are already replicated across bands, and each core adds the residual only on the output region it produced itself.

### Stride and Dilation

Stride and dilation are handled by the `tensor_copy` operations that scatter data from the input window buffer into the stacked moving tensor layout. When copying input data for a given filter position and output position, the source indices into the input window account for stride (which output position maps to which input position) and dilation (which filter position maps to which input offset). The scatter logic computes the correct source coordinates using:

```
d_in = d_out * stride_d + k_d * dilation_d - pad_d
h_in = h_out * stride_h + k_h * dilation_h - pad_h
w_in = w_out * stride_w + k_w * dilation_w - pad_w
```

and uses strided `tensor_copy` operations to gather the correct input elements with the appropriate step sizes.

### Fused Batch Normalization

When `batch_norm_mode` is `EVAL` or `TRAINING`, the kernel applies a fused batch normalization equivalent to `torch.nn.BatchNorm2d`. There are two modes:

- **Training mode** (`batch_norm_mode=BatchNormMode.TRAINING`) — described in the rest of this section: the per-`C_out`-channel mean and variance of the convolution output are computed from the batch and the running statistics are momentum-updated.
- **Eval mode** (`batch_norm_mode=BatchNormMode.EVAL`) — the supplied `running_means` / `running_variances` are used for the normalization and are *not* updated, equivalent to `BatchNorm2d.eval()` / `track_running_stats=False`. See [Eval mode](#eval-mode-batchnormmodeeval).

In training mode the kernel computes the per-`C_out`-channel mean and variance of the convolution output (reduced over all of `B, D_out, H_out, W_out`), plus a momentum update of the running statistics. The normalized output is

```
y_out = gamma * (conv_out - mean) / sqrt(var + eps) + beta
```

**Return value.** `(y_out, means, variances, updated_running_means, updated_running_variances)`. `means` / `variances` are `[C_out]` fp32 holding the raw output statistics (variance with correction 0); `updated_running_*` are `[C_out, 1]` fp32 holding the momentum-updated running statistics (variance with correction 1). `gamma`, `beta`, `running_means`, `running_variances` are required inputs.

#### Design choice 1 — statistics via `nisa.bn_stats` / `nisa.bn_aggr`

The mean and variance are produced by the hardware's dedicated batchnorm-statistics instructions.

`nisa.bn_stats` reduces a tile to **6 values per partition** — a `(count, mean, variance * count)` triple for the tile's even elements and another for its odd elements — computing them in fp32 on the Vector Engine. `nisa.bn_aggr` consumes any number of such triples and emits the pooled `(mean, variance)` per partition. Because the merge is **count-weighted and pooled** rather than a difference of two large numbers, the result is exact regardless of how the elements were split across invocations, and no `1/N` scaling is applied in the kernel at all.

**Two-level merge.** A tile's groups are aggregated in levels rather than by one wide `bn_aggr`, because the number of groups grows with the flush count (thousands of columns at large `B`) and a single flat aggregation would need a correspondingly wide gather buffer:

1. one `bn_aggr` per column-tiling band, reading that band's groups **in place** in `bn_stats_bufs`;
2. one `bn_aggr` over a rebuilt `(count, mean, variance * count)` triple per band, weighted by that band's element count.

Rebuilding a triple from a band's `(mean, variance)` is exact: a `bn_aggr` triple is precisely what the first level produced, scaled by the band's count. The per-band counts are static and derived by `_bn_band_element_counts`, which mirrors the main loop's D-H grouping — every output position belongs to exactly one `(flush, band)` pair.

Under D-H sharding a third level is added: each core pools its own half to a single triple per `C_out` tile, the cores exchange those compact triples with `nisa.sendrecv`, and both halves are pooled. Pooling triples rather than averaging two finished means keeps the result exact when the halves hold different element counts (odd `D_out * H_out`).

#### Design choice 2 — precomputed affine coefficients (`true_gamma` / `neg_true_beta`)

The rescale is algebraically `gamma * (conv_out − mean) / sqrt(var + eps) + beta`. Applying it directly would require multiple passes through the data. To avoid this, this phase **precomputes two per-channel coefficients once per `C_out` tile** to fold the whole expression into a single fused `tensor_scalar` per reloaded tile:

```
true_gamma    = gamma * rsqrt(var + eps)
neg_true_beta = mean * true_gamma - beta
y_out         = conv_out * true_gamma - neg_true_beta
```

This alternative formula is algebraically equivalent while on requiring one pass through `conv_out` and introducing small operations to produce `true_gamma` and `neg_true_beta`.

#### Streaming statistics accumulation (main loop)

The statistics are taken as the output is produced, so no extra pass over the data is needed. On every output tile's PSUM → SBUF eviction:

1. the tile is evicted with an **activation copy** (`op=nl.copy`), folding any bias into the instruction's `bias` operand. No activation *function* is applied here — `TRAINING` defers it to the rescale phase so the statistics see the raw convolution output;
2. **`nisa.bn_stats`** runs on the evicted result tile, once per column-tiling band, writing that band's 6 values into the flush's own slot of `bn_stats_bufs` (partition = channel within a `C_out` tile, with band `g` at partitions `[g * col_size, …)`; free = one 6-wide group per `(C_out tile, flush)`).

Nothing is read-modify-written per flush, so the hot path carries no accumulator dependency. Each band passes **its own real free width** to `bn_stats`: a ragged tail band's never-written columns are neither counted nor read. This matters more than it did for a plain sum — a `bn_stats` count *weights* the aggregation, so zero-filling those columns would dilute the mean rather than add nothing. Unwritten group slots (a tail flush can activate fewer than `C_OUT_REP` bands) are pre-zeroed once at allocation, giving them a zero count that `bn_aggr` ignores.

#### Finalize pass: three phases

The finalize pass runs three phases, each its own function:

1. **`_finalize_stats_phase`** — aggregate the `bn_stats` groups with `nisa.bn_aggr` (per band, then pooled across bands, then across cores if sharded on D-H), then write the final `mean` / `variance` per channel to HBM **and** leave them resident in persistent SBUF columns (`means_sbuf` / `variances_sbuf`).
2. **`_rescale_phase`** — normalize `conv_out` → `y_out` using the SBUF-resident statistics, applying the optional activation afterward.
3. **`_momentum_update_phase`** — update the running statistics.

Reading the statistics straight from the SBUF columns in the rescale and momentum phases avoids reloading them from HBM.

#### Reload multibuffering

The rescale phase reloads `conv_out` from HBM one spatial tile at a time into a reload buffer, rescales it, and stores it to `y_out`. The reload buffer is multibuffered (`bn_reload_interleave` slots, round-robined) so a slot's DMA store overlaps the next slot's DMA load and compute. The interleave is capped at the number of reload iterations that actually exist and then sized to fit the SBUF left free after the other finalize buffers are allocated (queried live from the `SbufManager`, so there is no duplicated per-buffer byte accounting).

#### Column tiling (low `C_out`)

When `C_out < 128` the matmul's stationary free dim (`M = C_out`) uses only `M` of the 128 PE columns. **Column tiling** (`C_OUT_REP > 1`, chosen by `_get_c_out_replication_params`: `C_OUT_REP ∈ {4, 2, 1}` for `C_out ≤ {32, 64, 128}`) packs `C_OUT_REP` D-H bands into a single output flush, each band running its matmuls on a different PE column tile (start column `g * col_size`) and landing in a different partition band (`[g * col_size, g * col_size + M)`) of one wide PSUM bank. This fills otherwise-idle PE columns and yields one wide eviction instead of `C_OUT_REP` separate flushes.

Batchnorm interacts with column tiling in a few places:

- The statistics buffer is `C_OUT_REP * col_size` partitions tall (each band holds the `bn_stats` groups for a disjoint subset of D-H positions); `_finalize_stats_phase` aggregates each band on its own partitions and moves the resulting 2-element `(mean, variance)` back to the canonical `[0, M)` layout with `nc_stream_shuffle` before pooling the bands.
- The rescale phase replicates the affine coefficients across the `C_OUT_REP` partition bands (via `nc_stream_shuffle`) so a single `tensor_scalar` rescales `C_OUT_REP` spatial regions at once, using all `C_OUT_REP * M` partitions.
- **Ragged D-H tail flushes** (a group whose D-H count is not a multiple of the per-band `dh_head_size`, e.g. one truncated at a `d_out` boundary) are still column-tiled, with the **last band partial**, rather than falling back to a single rep-1 flush. A rep-1 fallback would evict `num_dh_positions * W_tile` columns into a result buffer sized for only one band's `dh_head_size * W_tile`, overflowing it. Keeping uniform `dh_head_size`-wide bands (last one partial) keeps every eviction within the per-band width. On the batchnorm path each band passes its **own real free width** to `nisa.bn_stats`, so a partial band's unwritten tail columns are never read — they must not merely be zeroed, since a `bn_stats` count *weights* the aggregation and zeros would pull the mean toward 0 rather than contribute nothing.

#### Supported shapes (TRAINING only)

`bn_stats_bufs` is the one buffer in the kernel whose size is set by the *problem* rather than by a tile: it holds `ceil(C_out / P_MAX) * bn_partial_iters` six-element fp32 groups, where `bn_partial_iters = B * (D-H groups) * (W tiles)`, and it is live for the whole kernel because `nisa.bn_aggr` combines every group only in the finalize pass. It therefore grows with batch **and** output spatial size, and it is reserved *before* the interleave fit search runs — so past a point it starves the convolution's own tiles.

The **interleave fit search is the capacity authority**: it already degrades the interleave factors and falls back to `min_buf_count=1`, so if it finds nothing, nothing fits. When it fails in TRAINING mode, `_build_memory_config` appends the statistics buffer's size and scaling to the failure — it is reserved up front, grows with the problem rather than with a tile, and is the only term the caller can act on by changing the shape or the mode. On trn2, `B=512` at `C_out=1024, D_out=1, 64×64` reserves 192 KB/partition of the 208 KB available and is refused; the same shape builds in `EVAL`.

A fraction-of-SBUF pre-check was tried here first and **removed**: any such constant is not an invariant in either direction. At a quarter of SBUF it rejected `B=139` at `C_out=256, 1×64×64` where the search actually accepts up to `B=518` — 73% of the supported range lost — while still admitting shapes that then failed the fit search anyway, which is the message the pre-check existed to avoid. The bound is therefore not a documented constant: it depends on the target's SBUF and on the rest of the shape, and is reported at build time.

The 6-fp32-per-flush cost is inherent to `nisa.bn_stats`, which emits `(count, mean, variance*count)` for the even and odd input elements. The earlier `sum(Y)`/`sum(Y²)` accumulators used one fp32 column per flush and so fit shapes this cap rejects, but they computed the variance by the `E[Y²] - E[Y]²` identity, which loses catastrophically at a large per-channel mean — the accuracy reason for the change. `BatchNormMode.EVAL` and `NONE` accumulate no statistics and have no such limit; `EVAL` is the mode for large-batch inference at these shapes.

#### LNC=2 sharding in the finalize pass

Each core only ever reads back the region it produced itself, so no cross-core HBM read (and no `core_barrier`) is needed:

- **Shard on D-H:** each core saw only its D-H subset, so its statistics cover only part of each channel. Each core pools its own half down to a single `(count, mean, variance * count)` triple per `C_out` tile, the two cores exchange those compact triples with `nisa.sendrecv`, and both halves are pooled by count — so every core computes identical, complete statistics for all channels. Exchanging *triples* rather than finished `(mean, variance)` pairs is what keeps this exact when the halves hold different element counts (odd `D_out * H_out`); the `sendrecv` also spans every partition band, since under column tiling a tile's groups are not confined to `[0, M)`. The rescale then has each core normalize only its own D-H spatial slice of `conv_out`.
- **Shard on C_out:** each core already owns every group for its own `C_out` slice and finalizes only that slice end-to-end (stats → rescale → momentum), with no cross-core communication.

#### Momentum update of running statistics

`_momentum_update_phase` updates the running statistics to match `torch.nn.BatchNorm2d`:

```
updated_running_means     = (1 - momentum) * running_means     + momentum * mean
updated_running_variances = (1 - momentum) * running_variances + momentum * (N / (N - 1)) * var
```

Note the running **variance** uses the **correction-1 (unbiased)** variance `(N / (N - 1)) * var`, while the `variances` output and the rescale use the correction-0 variance — again matching PyTorch. Both updates read the freshly computed statistics from the SBUF columns (no HBM reload) and are sharded by `C_out` slice so each core only touches the statistics it produced.

#### Eval mode (`BatchNormMode.EVAL`)

Eval mode normalizes with the **given** running statistics rather than the batch's own, and never updates them — matching `torch.nn.BatchNorm2d` in `eval()` (equivalently `track_running_stats=False`):

```
y_out = gamma * (conv_out - running_mean) / sqrt(running_var + eps) + beta
```

Because the normalization no longer depends on a reduction over the whole output, only `y_out` is returned.

The affine coefficients are the same `true_gamma` / `neg_true_beta` pair as the rescale phase (see [Design choice 2](#design-choice-2--precomputed-affine-coefficients-true_gamma--neg_true_beta)), now computed from the running statistics by the shared `_compute_true_bn_scales` helper. They are derived **once per `C_out` interleave group** (alongside the group's filter / bias load) and then applied on **every PSUM eviction of that group** inside `_apply_bias_activation_and_copy`:

```
result = psum * true_gamma - neg_true_beta      # one tensor_scalar, replaces the plain copy
result = act(result)                            # in place, if activation_fn is set
```

So the whole conv + bias + batchnorm + activation chain costs one `tensor_scalar` plus (optionally) one `activation` on the eviction — the same instruction count as the non-batchnorm bias+activation path, and the output written to HBM is already final.

**Bias folding.** A conv bias is folded into the coefficients instead of being added separately, since
```
gamma * ((y + b) - mean) / sqrt(var + eps) + beta  ==  y * true_gamma - (neg_true_beta - b * true_gamma)
```
This keeps the eviction at a single `tensor_scalar`. The bias is read from the already-loaded, band-replicated SBUF bias buffer.

**Padding.** Never-written (fully padded) free positions are zeroed in PSUM before the `tensor_scalar`, so the same fused op gives them their correct value — the batchnorm (and activation) of the bias — exactly as the non-batchnorm path fills them with `f(0 + bias)`.

**Column tiling and LNC=2.** The coefficient columns are `C_OUT_REP * col_size` partitions tall and replicated across bands with `nc_stream_shuffle` (the same layout as the bias), so the single eviction `tensor_scalar` covers every band. Under LNC sharding no cross-core communication is needed at all: there are no statistics to reduce, and each core writes only the output region it computed.

**Memory.** Eval mode's SBUF cost is a handful of single-column fp32 buffers (5 shared scratch columns plus 2 persistent coefficient columns per `C_out` interleave slot), reserved in `_build_memory_config` in place of the (much larger) training-mode accumulators and partials buffer.

#### Returning the pre-batchnorm output (`output_pre_norm=True`)

Training-mode batchnorm already writes the raw (un-normalized) convolution output to its own HBM buffer, `conv_out`, because the finalize pass has to reload it to normalize into `y_out` (an in-place HBM read-modify-write would be unsafe under LNC=2 redundant execution — see [LNC=2 sharding in the finalize pass](#lnc2-sharding-in-the-finalize-pass)). `output_pre_norm=True` simply **promotes that existing buffer to a returned output**:

```
(y_out, conv_out, means, variances, updated_running_means, updated_running_variances)
```

`conv_out` has the same shape and dtype as `y_out` and holds the convolution result *with* the bias but *before* the batchnorm and before the activation (the activation is applied after the normalization, so it never touches `conv_out`). Fully-padded output positions hold `0 + bias`, matching the non-batchnorm path's fill.

Because the tensor is produced either way, this costs nothing: no extra HBM allocation, no extra DMA, no extra compute — only the return signature changes. Note that `conv_out` was already `nl.shared_hbm`, so it is directly returnable.

**Mutual exclusion with eval mode.** `output_pre_norm=True` is rejected (via `kernel_assert` in `_validate_conv3d_inputs`) in `BatchNormMode.EVAL`: eval mode normalizes in place on the PSUM eviction and never materializes a pre-batchnorm tensor at all, so honoring the request would require re-introducing the staging buffer and a second pass — exactly the cost eval mode exists to avoid. `output_pre_norm` remains a don't-care in `BatchNormMode.NONE`, so the assertion only applies when batchnorm is actually fused.
