# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D Convolution kernel for NeuronCore with K-replication strategy."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.logging import get_logger
from .conv3d_helpers import (
    BatchNormMode,
    Conv3dConfig,
    ResidualAddLoc,
    _build_conv3d_config,
    _Conv3dComputeResult,
    _identity_shuffle_mask,
    _validate_conv3d_inputs,
    apply_bn_rescale,
    compute_bn_momentum_update,
)
from .conv3d_kernels import (
    _LOW_CHAN_MAX,
    _banded_wins,
    _conv3d_banded_compute,
    _conv3d_fallback_compute,
    _conv3d_grouped_compute,
    _conv3d_low_chan_compute,
    _validate_grouped_inputs,
)


@nki.jit
def conv3d(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
    batch_norm_mode: BatchNormMode = BatchNormMode.NONE,
    output_pre_norm: Optional[bool] = False,
    batch_norm_eps: Optional[float] = 1e-5,
    gamma: Optional[nl.NkiTensor] = None,
    beta: Optional[nl.NkiTensor] = None,
    momentum: Optional[float] = 0.1,
    running_means: Optional[nl.NkiTensor] = None,
    running_variances: Optional[nl.NkiTensor] = None,
    residual_add_loc: ResidualAddLoc = ResidualAddLoc.NONE,
    residuals_in: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    groups: int = 1,
) -> nl.NkiTensor:
    """
    3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.

    Implements a 3D convolution operation (x_in * filters + bias) optimized for NeuronCore
    using a K-replication strategy for filter loading and W-contiguous tiling for output
    computation. Supports configurable stride, padding, dilation, optional bias, and
    optional activation function.
    On trn2, a 2D call (D = 1, K_d = 1) of a thin layer (at most 64 channels) with wide rows runs on the row-banded
    path of _conv3d_banded_compute (image rows stacked on the tensor engine, no per-tap input copies) when the cost model in
    _banded_wins expects that to be faster. The result is equal up to rounding: 16-bit operands and fp32 PSUM
    accumulation as here, in a different summation order, so a few elements differ by under one ulp. A caller-provided
    sbm keeps the original path.
    Grouped and depthwise layers (groups > 1) run on the block-diagonal banded path of _conv3d_grouped_compute, which
    supports 2D calls (D = 1, K_d = 1, no depth padding) on trn2 without fused batchnorm, residual add, SquaredReLU or
    a caller-provided sbm; such calls are rejected while tracing.

    Intended Usage Range:
        B: 1-128
        C_in: 3-1280, C_out: 3-2048
        D: 1-1024, H: 1-1024, W: 1-1024
        K_d: 1-64, K_h: 1-64, K_w: 1-64
        Stride: 1-64 per dimension
        Dilation: 1-64 per dimension

    Dimensions:
        B: Batch size
        C_in: Number of input channels
        C_out: Number of output channels
        D: Input depth
        H: Input height
        W: Input width
        K_d: Filter depth
        K_h: Filter height
        K_w: Filter width
        D_out: Output depth = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
        H_out: Output height = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
        W_out: Output width = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1

    Args:
        x_in (nl.NkiTensor): [B, C_in, D, H, W], Input tensor on HBM.
        filters (nl.NkiTensor): [K_d, K_h, K_w, C_in, C_out], Filter weights on HBM;
            [K_d, K_h, K_w, C_in / groups, C_out] with groups > 1.
        bias (Optional[nl.NkiTensor]): [C_out], Optional bias tensor on HBM.
        stride (tuple[int, int, int]): (stride_d, stride_h, stride_w), Convolution strides.
        padding (tuple[int, int, int, int, int, int]): (pad_d_left, pad_d_right, pad_h_top,
            pad_h_bottom, pad_w_left, pad_w_right), Padding for each spatial dimension.
        dilation (tuple[int, int, int]): (dilation_d, dilation_h, dilation_w), Dilation factors.
        activation_fn (Optional[ActFnType]): Optional activation function to apply after conv.
            When a batchnorm is fused, the activation is instead applied after the fused
            batchnorm (in TRAINING mode the statistics are still computed on the raw
            convolution output).
        lnc_shard (bool): Does nothing. Will be deprecated in next release.
        batch_norm_mode (BatchNormMode): Fused-batchnorm mode. Default BatchNormMode.NONE.
            NONE: no fused batchnorm.
            TRAINING: compute the per-C_out-channel mean and variance of the convolution
                output (reduced over B, D_out, H_out, W_out), normalize with them
                (equivalent to torch.nn.BatchNorm2d) and momentum-update the running
                statistics.
            EVAL: normalize with the supplied running_means / running_variances and do NOT
                update them, equivalent to torch.nn.BatchNorm2d in module.eval(). Applied
                on the PSUM eviction, so no statistics, staging buffer or finalize pass.
            Either mode composes with activation_fn (applied to the normalized output) and
            with lnc_shard (TRAINING reduces the statistics across cores via nisa.sendrecv
            when sharded on D-H, else each core owns a disjoint C_out slice).
            Supported shapes (TRAINING only): the statistics buffer holds
            ceil(C_out / 128) * B * (D-H groups) * (W tiles) six-element fp32 groups for the
            whole kernel, so it scales with the problem rather than with a tile. Large B and
            a large output spatial size together can leave too little SBUF for the
            convolution's tiles, and the build is then rejected naming this buffer. The
            limit depends on the target's SBUF and the rest of the shape, so it is reported
            at build time. EVAL and NONE accumulate no statistics and are unaffected.
        output_pre_norm (Optional[bool]): When True, additionally return the raw
            pre-batchnorm convolution output as conv_out. Only valid in
            BatchNormMode.TRAINING, which already stages that tensor in HBM; rejected in
            EVAL (which never materializes it) and ignored in NONE. Default False.
        batch_norm_eps (Optional[float]): Epsilon added to the variance for numerical
            stability when a batchnorm is fused. Default 1e-5.
        gamma (Optional[nl.NkiTensor]): [C_out, 1], per-channel batchnorm scale. Required
            when a batchnorm is fused.
        beta (Optional[nl.NkiTensor]): [C_out, 1], per-channel batchnorm shift. Required
            when a batchnorm is fused.
        momentum (Optional[float]): Momentum for the running-statistics update in
            BatchNormMode.TRAINING, matching torch.nn.BatchNorm2d:
            x_running = (1 - momentum) * x_running + momentum * x_new. Default 0.1.
        running_means (Optional[nl.NkiTensor]): [C_out, 1], float32 running mean carried
            across steps. Required when a batchnorm is fused. In BatchNormMode.EVAL this is
            the mean used for the normalization.
        running_variances (Optional[nl.NkiTensor]): [C_out, 1], float32 running variance
            carried across steps. Required when a batchnorm is fused. In BatchNormMode.EVAL
            this is the variance used for the normalization.
        residual_add_loc (ResidualAddLoc): Insertion point of the fused residual add.
            Default ResidualAddLoc.NONE (no residual add).
            PRE_ACT: y = activation_fn(post_norm_conv + residuals_in).
            POST_ACT: y = activation_fn(post_norm_conv) + residuals_in.
            Valid in every batch_norm_mode; the residual is always added after the fused
            batchnorm (when one is present). Requires residuals_in.
        residuals_in (Optional[nl.NkiTensor]): [B, C_out, D_out, H_out, W_out], residual
            tensor added to the output. Required when residual_add_loc is not NONE, ignored
            otherwise. Same dtype as x_in.
        sbm (Optional[SbufManager]): Optional caller-provided SBUF manager for allocation. When
            provided, its allocation mode must match use_auto_allocation.
        use_auto_allocation (bool): Whether to use auto allocation for SBUF buffers.
        groups (int): Number of channel groups; C_in and C_out must be multiples of it. Output channel o reads input
            channels (o // (C_out / groups)) * (C_in / groups) + i, as in torch.nn.Conv3d(groups=groups). groups > 1
            runs on _conv3d_grouped_compute (see above). Default 1.

    Returns:
        y_out (nl.NkiTensor): [B, C_out, D_out, H_out, W_out], Output tensor on HBM. NONE and
        EVAL return only y_out (EVAL computes no statistics and leaves the running ones
        untouched). TRAINING returns
        (y_out, means, variances, updated_running_means, updated_running_variances): means
        and variances are [C_out] float32 raw (pre-normalization) statistics with correction
        0; the updated_* pair is [C_out, 1] float32 on shared HBM and uses the correction-1
        (unbiased) variance, matching torch.nn.BatchNorm2d. With output_pre_norm=True,
        conv_out (same shape and dtype as y_out) is inserted right after y_out.

    Pseudocode:
        cfg = build_config(x_in, filters, bias, stride, padding, dilation)
        tile_cfg = build_tile_config(cfg)
        mem_cfg = build_memory_config(cfg, tile_cfg)
        allocate SBUF buffers (filters, bias, results, input windows, stacked inputs)

        for c_out_group in range(0, C_out, c_out_interleave * P_MAX):
            load filters + bias for c_out_group into SBUF

            for batch in range(B):
                for dh_group in range(dh_start, dh_end, num_dh_stacked):
                    for w_tile in range(0, W_out, W_tile):
                        allocate PSUM banks for output tiles
                        DMA load this tile's residual slice from HBM to SBUF (if fused)

                        for c_in_tile in range(0, C_in, P_MAX):
                            DMA load input window [C_in_tile, D_win, H_win, W_win] from HBM to SBUF
                            for k_outer_tile in range(K_outer_tile_count):
                                scatter input window to stacked layout via tensor_copy
                                nc_matmul: stationary=filters, moving=stacked_input, dst=PSUM

                        apply bias + activation + residual add, copy PSUM to result SBUF
                        DMA store result SBUF to HBM

        free SBUF buffers

    Tiling Strategy:
        Loop nest: C_out groups -> Batch -> D-H groups -> W tiles -> (C_in tiles in _conv3d_output_tile)

        SBUF layout (per-buffer shapes in _build_memory_config):
        - Filters / bias: [c_out_interleave] (x [c_in_tile_count] x [K_outer_tile_count] for filters),
          loaded once per C_out group.
        - Result: [store_pipe] x [c_out_interleave], rotated across W tiles to pipeline stores with
          compute. A fused residual outside TRAINING stages into the same shape and rotation, sharing
          the result tile when the eviction consumes it in place (Conv3dConfig.residual_in_place).
        - Input windows / stacked inputs: multi-buffered to overlap HBM loads with the SBUF scatter,
          and the scatter with the matmul.

        Engine pipelining: tensor_copy and memset alternate engines to run concurrently.

        Tile size selection (all reduced iteratively in _build_tile_config until SBUF fits):
        - W_tile: min(W_out, F_MAX)
        - num_dh_stacked: packs multiple (d_out, h_out) on the free dim when W_out < F_MAX
        - c_out/c_in tile: min(P_MAX, channel_count)
        - K_REP: replicates filter positions within the partition stride for small C_in tiles
        - All sizes iteratively reduced in _build_tile_config until SBUF budget is met
    """
    if lnc_shard:
        get_logger("conv3d").warn(
            "lnc_shard does nothing and will be deprecated in the next release. "
            "Please stop passing lnc_shard to conv3d."
        )

    use_low_chan = x_in.shape[1] <= _LOW_CHAN_MAX
    cfg = _build_conv3d_config(
        x_in=x_in,
        filters=filters,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        activation_fn=activation_fn,
        batch_norm_mode=batch_norm_mode,
        output_pre_norm=output_pre_norm,
        residual_add_loc=residual_add_loc,
        transpose_taps=False,
    )
    if groups != 1:
        _validate_grouped_inputs(
            x_in=x_in, filters=filters, bias=bias, cfg=cfg, groups=groups, has_caller_sbm=sbm != None
        )
        return _conv3d_grouped_compute(x_in=x_in, filters=filters, bias=bias, cfg=cfg, groups=groups)
    _validate_conv3d_inputs(
        x_in=x_in,
        filters=filters,
        bias=bias,
        cfg=cfg,
        gamma=gamma,
        beta=beta,
        running_means=running_means,
        running_variances=running_variances,
        residuals_in=residuals_in,
        transpose_taps=False,
    )
    _, n_prgs, _ = get_verified_program_sharding_info("conv3d", (0, 1))
    if nisa.get_nc_version() == nisa.nc_version.gen3 and _banded_wins(cfg, x_in.dtype, sbm != None, n_prgs):
        return _conv3d_banded_compute(x_in, filters, bias, cfg)

    conv_out = nl.ndarray(
        shape=(cfg.B, cfg.C_out, cfg.D_out, cfg.H_out, cfg.W_out),
        dtype=x_in.dtype,
        buffer=nl.shared_hbm,
    )
    means = None
    variances = None
    updated_running_means = None
    updated_running_variances = None
    if cfg.is_batch_norm_training():
        y_out = nl.ndarray(
            shape=(cfg.B, cfg.C_out, cfg.D_out, cfg.H_out, cfg.W_out),
            dtype=x_in.dtype,
            buffer=nl.shared_hbm,
        )
        means = nl.ndarray(shape=(cfg.C_out,), dtype=nl.float32, buffer=nl.shared_hbm)
        variances = nl.ndarray(shape=(cfg.C_out,), dtype=nl.float32, buffer=nl.shared_hbm)
        updated_running_means = nl.ndarray(shape=(cfg.C_out, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        updated_running_variances = nl.ndarray(shape=(cfg.C_out, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    compute_result = None
    if use_low_chan:
        compute_result = _conv3d_low_chan_compute(
            x_in=x_in,
            filters=filters,
            bias=bias,
            gamma=gamma,
            beta=beta,
            batch_norm_eps=batch_norm_eps,
            running_means=running_means,
            running_variances=running_variances,
            residuals_in=residuals_in,
            cfg=cfg,
            conv_out=conv_out,
            means=means,
            variances=variances,
            sbm=sbm,
            use_auto_allocation=use_auto_allocation,
            transpose_taps=False,
        )
    else:
        compute_result = _conv3d_fallback_compute(
            x_in=x_in,
            filters=filters,
            bias=bias,
            gamma=gamma,
            beta=beta,
            batch_norm_eps=batch_norm_eps,
            running_means=running_means,
            running_variances=running_variances,
            residuals_in=residuals_in,
            cfg=cfg,
            conv_out=conv_out,
            means=means,
            variances=variances,
            sbm=sbm,
            use_auto_allocation=use_auto_allocation,
        )

    if cfg.is_batch_norm_training():
        _run_training_post_processing(
            cfg=cfg,
            compute_result=compute_result,
            conv_out=conv_out,
            y_out=y_out,
            gamma=gamma,
            beta=beta,
            batch_norm_eps=batch_norm_eps,
            momentum=momentum,
            running_means=running_means,
            running_variances=running_variances,
            updated_running_means=updated_running_means,
            updated_running_variances=updated_running_variances,
            residuals_in=residuals_in,
        )
        if cfg.output_pre_norm:
            return y_out, conv_out, means, variances, updated_running_means, updated_running_variances
        return y_out, means, variances, updated_running_means, updated_running_variances
    return conv_out


def _run_training_post_processing(
    cfg: Conv3dConfig,
    compute_result: _Conv3dComputeResult,
    conv_out: nl.NkiTensor,
    y_out: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    batch_norm_eps: float,
    momentum: float,
    running_means: nl.NkiTensor,
    running_variances: nl.NkiTensor,
    updated_running_means: nl.NkiTensor,
    updated_running_variances: nl.NkiTensor,
    residuals_in: Optional[nl.NkiTensor],
) -> None:
    post_config = compute_result.post_config
    post_buffers = compute_result.post_buffers
    apply_bn_rescale(
        cfg=cfg,
        conv_out=conv_out,
        y_out=y_out,
        gamma=gamma,
        beta=beta,
        batch_norm_eps=batch_norm_eps,
        means_sbuf=compute_result.means_sbuf,
        variances_sbuf=compute_result.variances_sbuf,
        post_config=post_config,
        post_buffers=post_buffers,
        identity_mask=_identity_shuffle_mask(),
        residuals_in=residuals_in,
    )

    if post_config.write_momentum:
        compute_bn_momentum_update(
            cfg=cfg,
            means_sbuf=compute_result.means_sbuf,
            variances_sbuf=compute_result.variances_sbuf,
            running_means=running_means,
            running_variances=running_variances,
            updated_running_means=updated_running_means,
            updated_running_variances=updated_running_variances,
            run_running_mean_buf=post_buffers.run_running_mean_buf,
            run_new_mean_buf=post_buffers.run_new_mean_buf,
            run_running_var_buf=post_buffers.run_running_var_buf,
            run_new_var_buf=post_buffers.run_new_var_buf,
            momentum=momentum,
            p_max=post_config.p_max,
            c_out_start=post_config.c_out_start,
            c_out_end=post_config.c_out_end,
        )

    while len(compute_result.sbm.heap) > compute_result.heap_depth:
        compute_result.sbm.pop_heap()
