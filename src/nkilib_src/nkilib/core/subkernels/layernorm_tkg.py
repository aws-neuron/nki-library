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

"""LayerNorm subkernel optimized for token generation (TKG) inference with LNC sharding support."""

from typing import Optional, Tuple, Union

import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import SbufManager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import get_logger
from ..utils.tensor_view import TensorView
from ..utils.tiled_range import TiledRange
from .norm_tkg_utils import (
    load_gamma_to_sbuf,
    load_input_to_sbuf,
    validate_shapes,
    validate_shapes_shard_on_h,
)

# Heuristic threshold for sharding on BxS to halve computation at the cost of extra local collective
SHARDING_THRESHOLD = 10

# Tile size for BxS dimension processing
BxS_FULL_TILE_SIZE = 512


def layernorm_tkg(
    input: Union[TensorView, nl.ndarray],
    gamma: Union[TensorView, nl.ndarray],
    output: Union[TensorView, nl.ndarray],
    beta: Optional[Union[TensorView, nl.ndarray]] = None,
    eps: float = 1e-6,
    shard_on_h: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    LayerNorm implementation optimized for inference token generation (decoding) phase.

    The output layout is specifically chosen to make the subsequent sharded matmul
    efficient for LNC > 1 case.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        H0: Partition dimension (128)
        H1: H // H0

    Args:
        input (Union[TensorView, nl.ndarray]): [B, S, H] when in HBM or [H0, BxS, H1] when in SBUF, Input tensor.
        gamma (Union[TensorView, nl.ndarray]): [1, H], Gamma tensor used in normalization, in HBM.
        output (Union[TensorView, nl.ndarray]): [H0, BxS, H1], Output tensor buffer.
        beta (Optional[Union[TensorView, nl.ndarray]]): [1, H], Beta tensor used in normalization, in HBM.
        eps (float): Epsilon to maintain numerical stability.
        shard_on_h (bool): If True, shard computation along H dimension instead of BxS.
        use_heap_memory (bool): Indicates whether to allocate memory on the heap instead of the stack.
        sbm (Optional[SbufManager]): Instance of SbufManager responsible for handling SBUF allocation.

    Returns:
        output (Union[TensorView, nl.ndarray]): [H0, BxS, H1], Normalized output tensor.

    Notes:
        - H must be divisible by 128 (partition dimension).
        - When LNC=2 and BxS > SHARDING_THRESHOLD, computation is sharded across cores.
        - Output layout is transposed for efficient downstream sharded matmul.
        - shard_on_h=True requires LNC=2 and shards the hidden dimension instead of BxS.

    Pseudocode:
        result = LayerNorm(hidden, gamma, beta, eps)
        result = result.reshape((BxS, H))
        t0 = result[:, 0:H//LNC_SIZE]
        t1 = result[:, H//LNC_SIZE:]
        t0 = t0.reshape((BxS, 128, H//128//LNC_SIZE)).transpose((1, 0, 2))
        t1 = t1.reshape((BxS, 128, H//128//LNC_SIZE)).transpose((1, 0, 2))
        result = np.concatenate([t0, t1], axis=2)
    """

    input_view = TensorView(input) if not isinstance(input, TensorView) else input
    gamma_view = TensorView(gamma) if not isinstance(gamma, TensorView) else gamma
    output_view = TensorView(output) if not isinstance(output, TensorView) else output
    beta_view = None
    if beta is not None:
        beta_view = TensorView(beta) if not isinstance(beta, TensorView) else beta

    if not sbm:
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=nl.tile_size.total_available_sbuf_size,
            logger=get_logger("layernorm_tkg"),
            use_auto_alloc=True,
        )

    sbm.open_scope(name="layernorm_tkg")

    if shard_on_h:
        _layernorm_tkg_shard_on_h(
            input_view=input_view,
            gamma_view=gamma_view,
            output_view=output_view,
            beta_view=beta_view,
            eps=eps,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )
    else:
        _layernorm_tkg_shard_on_bxs(
            input_view=input_view,
            gamma_view=gamma_view,
            output_view=output_view,
            beta_view=beta_view,
            eps=eps,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )

    sbm.close_scope()

    if isinstance(output, TensorView):
        return output_view
    else:
        return output


def _layernorm_tkg_shard_on_bxs(
    input_view: TensorView,
    gamma_view: TensorView,
    output_view: TensorView,
    beta_view: Optional[TensorView],
    eps: float = 1e-6,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    LayerNorm with sharding on the BxS dimension.

    Distributes work across LNC cores by splitting the batch-sequence dimension.
    Each core processes its shard independently, then results are exchanged if
    the output is in SBUF.

    Args:
        input_view (TensorView): [B, S, H] when in HBM or [H0, BxS, H1] when in SBUF, Input tensor view.
        gamma_view (TensorView): [1, H], Gamma tensor view.
        output_view (TensorView): [H0, BxS, H1], Output tensor view.
        beta_view (Optional[TensorView]): [1, H], Beta tensor view.
        eps (float): Epsilon for numerical stability.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (Optional[SbufManager]): SBUF memory manager instance.

    Returns:
        None: Results written to output_view.
    """
    BxS, H, H0, H1 = validate_shapes(input_view, gamma_view, output_view)

    alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack

    if output_view.is_sbuf():
        output_sb_view = output_view
    else:
        output_sb = alloc_tensor((H0, BxS, H1), dtype=input_view.dtype, buffer=nl.sbuf, name="layernorm_output_sb")
        output_sb_view = TensorView(output_sb)

    _, lnc, shard_id = get_verified_program_sharding_info("layernorm_tkg", (0, 1))

    num_shards = lnc
    do_shard = num_shards == 2 and BxS > SHARDING_THRESHOLD and BxS % lnc == 0
    if not do_shard:
        num_shards, shard_id = 1, 0

    shard_size = BxS // num_shards

    if not input_view.is_sbuf():
        input_view_flat = input_view.flatten_dims(start_dim=0, end_dim=1)
        input_view_sharded = input_view_flat.slice(dim=0, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)
    else:
        input_view_sharded = input_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)

    output_view_sharded = output_sb_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)

    _layernorm_tkg_impl(
        input=input_view_sharded,
        gamma=gamma_view,
        beta=beta_view,
        output=output_view_sharded,
        num_H_shards=lnc,
        hidden_actual=H,
        eps=eps,
        shard_on_h=False,
        use_heap_memory=use_heap_memory,
        sbm=sbm,
    )

    if output_view.is_sbuf():
        if do_shard:
            output_view_sharded_other_core = output_sb_view.slice(
                dim=1, start=(1 - shard_id) * shard_size, end=(2 - shard_id) * shard_size
            )
            nisa.sendrecv(
                dst=output_view_sharded_other_core.get_view(),
                src=output_view_sharded.get_view(),
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
            )
    else:
        output_hbm_view_sharded = output_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)
        nisa.dma_copy(dst=output_hbm_view_sharded.get_view(), src=output_view_sharded.get_view())
        if use_heap_memory:
            sbm.pop_heap()  # dealloc output_sb


def _layernorm_tkg_shard_on_h(
    input_view: TensorView,
    gamma_view: TensorView,
    output_view: TensorView,
    beta_view: Optional[TensorView],
    eps: float = 1e-6,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    LayerNorm with sharding on the H (hidden) dimension.

    Each LNC core processes a shard of the hidden dimension. Partial sums for
    mean and variance are exchanged between cores via sendrecv to compute the
    full statistics before normalization.

    Args:
        input_view (TensorView): [B, S, H] when in HBM or [H0, BxS, sharded_H1] when in SBUF, Input tensor view.
        gamma_view (TensorView): [1, H], Gamma tensor view.
        output_view (TensorView): [H0, BxS, H1] or [H0, BxS, sharded_H1] if in SBUF, Output tensor view.
        beta_view (Optional[TensorView]): [1, H], Beta tensor view.
        eps (float): Epsilon for numerical stability.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (Optional[SbufManager]): SBUF memory manager instance.

    Returns:
        None: Results written to output_view.
    """
    BxS, H, H0, H1, shard_H, shard_H1 = validate_shapes_shard_on_h(input_view, gamma_view, output_view)

    if output_view.is_sbuf():
        output_sb_view = output_view
    else:
        alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack
        output_sb = alloc_tensor(
            (H0, BxS, shard_H1),
            dtype=input_view.dtype,
            buffer=nl.sbuf,
            name="layernorm_output_sb",
        )
        output_sb_view = TensorView(output_sb)

    _, lnc, shard_id = get_verified_program_sharding_info("layernorm_tkg", (0, 1))

    # Shard input and gamma along H dimension
    input_view_sharded = input_view
    if not input_view.is_sbuf():
        input_view_flat = input_view.flatten_dims(start_dim=0, end_dim=1)
        input_view_sharded = input_view_flat.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    gamma_view_sharded = gamma_view.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    beta_view_sharded = None
    if beta_view is not None:
        beta_view_sharded = beta_view.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    _layernorm_tkg_impl(
        input=input_view_sharded,
        gamma=gamma_view_sharded,
        beta=beta_view_sharded,
        output=output_sb_view,
        num_H_shards=lnc,
        hidden_actual=H,
        eps=eps,
        shard_on_h=True,
        use_heap_memory=use_heap_memory,
        sbm=sbm,
    )

    if output_view.is_sbuf():
        output_view = output_sb_view
    else:
        output_hbm_view_sharded = output_view.slice(dim=2, start=shard_id * shard_H1, end=(shard_id + 1) * shard_H1)
        nisa.dma_copy(dst=output_hbm_view_sharded.get_view(), src=output_sb_view.get_view())


def _process_layernorm_tile(
    input_sb_view: TensorView,
    gamma_sb_view: TensorView,
    output_sb_view: TensorView,
    beta_sb_view: Optional[TensorView],
    zero_bias_view: TensorView,
    eps_view: TensorView,
    matmul_reduction_const_view: TensorView,
    bxs_tile: Tuple,
    hidden_actual: int,
    shard_on_h: bool,
    use_heap_memory: bool = False,
    sbm: SbufManager = None,
):
    """
    Process a single BxS tile of LayerNorm computation.

    Computes LayerNorm: output = gamma * (input - mean) / sqrt(var + eps) + beta
    where var = E[X^2] - E[X]^2.

    Args:
        input_sb_view (TensorView): [H0, BxS_tile, H1], Input tensor tile in SBUF.
        gamma_sb_view (TensorView): [H0, BxS_tile, H1], Gamma tensor view in SBUF (broadcasted).
        output_sb_view (TensorView): [H0, BxS_tile, H1], Output tensor tile in SBUF.
        beta_sb_view (Optional[TensorView]): [H0, BxS_tile, H1], Beta tensor view in SBUF (broadcasted).
        zero_bias_view (TensorView): [H0, 1], Zero bias for activation ops.
        eps_view (TensorView): [H0, 1], Epsilon value for numerical stability.
        matmul_reduction_const_view (TensorView): [H0, H0], Constant matrix for reduction (1/H).
        bxs_tile (Tuple): Tile information containing index and size.
        hidden_actual (int): Actual hidden dimension size for mean calculation.
        shard_on_h (bool): If True, exchange partial sums between cores.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (SbufManager): SBUF memory manager instance.

    Returns:
        None: Results written directly to output_sb_view.

    Notes:
        - Uses Var(X) = E[X^2] - E[X]^2 for variance computation.
        - All intermediates use FP32 for numerical precision.
    """
    alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack

    sbm.open_scope()

    num_allocated_tensor = 0
    inter_dtype = nl.float32

    kernel_assert(
        input_sb_view.shape == output_sb_view.shape,
        "Input and output tensor shapes must match for LayerNorm processing",
    )
    H0, BxS, H1 = input_sb_view.shape

    # Step 1: Compute mean(x^2) for variance
    input_squared_sb = alloc_tensor(
        shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf, name=f"layernorm_square_{bxs_tile.index}"
    )
    num_allocated_tensor += 1
    nisa.activation(
        dst=input_squared_sb[...],
        op=nl.square,
        data=input_sb_view.get_view(),
        bias=zero_bias_view.get_view(),
    )

    # Pack sum(x) and sum(x^2) into [H0, BxS*2] for a single sendrecv
    packed_reduced_sum_tensor = alloc_tensor(
        shape=(H0, BxS * 2),
        dtype=inter_dtype,
        buffer=nl.sbuf,
        name=f"layernorm_packed_reduced_sum_{bxs_tile.index}",
    )
    num_allocated_tensor += 1

    # Reduce squares along H1 dimension
    reduced_input_squared_view = TensorView(packed_reduced_sum_tensor).slice(dim=1, start=0, end=BxS)
    nisa.tensor_reduce(dst=reduced_input_squared_view.get_view(), op=nl.add, data=input_squared_sb[...], axis=2)

    # Step 2: Compute mean(x)
    reduced_input_view = TensorView(packed_reduced_sum_tensor).slice(dim=1, start=BxS, end=BxS * 2)
    nisa.tensor_reduce(dst=reduced_input_view.get_view(), op=nl.add, data=input_sb_view.get_view(), axis=2)

    if shard_on_h:
        _, _, shard_id = get_verified_program_sharding_info("layernorm_tkg", (0, 1))

        packed_remote = alloc_tensor(
            shape=(H0, BxS * 2),
            dtype=inter_dtype,
            buffer=nl.sbuf,
            name=f"layernorm_packed_remote_{bxs_tile.index}",
        )
        num_allocated_tensor += 1
        nisa.sendrecv(
            dst=packed_remote[...],
            src=packed_reduced_sum_tensor[...],
            send_to_rank=1 - shard_id,
            recv_from_rank=1 - shard_id,
            pipe_id=0,
        )

        # Unpack and combine: full = local + remote
        nisa.tensor_tensor(
            packed_reduced_sum_tensor,
            packed_reduced_sum_tensor,
            packed_remote,
            nl.add,
        )

    if sbm.is_auto_alloc():
        input_squared_mean = nl.ndarray((H0, BxS), dtype=inter_dtype, buffer=nl.psum)
        input_mean = nl.ndarray((H0, BxS), dtype=inter_dtype, buffer=nl.psum)
    else:
        input_squared_mean = nl.ndarray((H0, BxS), dtype=inter_dtype, buffer=nl.psum, address=(0, 0))
        # Offset by one PSUM bank: psum_fmax elements * 4 bytes per float32
        input_mean = nl.ndarray(
            (H0, BxS),
            dtype=inter_dtype,
            buffer=nl.psum,
            address=(0, 1 * nl.tile_size.psum_fmax * 4),
        )

    # Complete mean(x^2) via matmul with 1/H constant
    nisa.nc_matmul(
        dst=input_squared_mean,
        stationary=matmul_reduction_const_view.get_view(),
        moving=reduced_input_squared_view.get_view(),
    )

    # Complete mean(x) via matmul with 1/H constant
    nisa.nc_matmul(
        dst=input_mean,
        stationary=matmul_reduction_const_view.get_view(),
        moving=reduced_input_view.get_view(),
    )

    # Step 3: Center input by subtracting mean = X - E[x]
    input_mean_broadcast = TensorView(input_mean).expand_dim(dim=2).broadcast(dim=2, size=H1)
    nisa.tensor_tensor(
        output_sb_view.get_view(),
        input_sb_view.get_view(),
        input_mean_broadcast.get_view(),
        nl.subtract,
    )

    # Step 4: Compute variance = E[X^2] - E[X]^2
    # reuse reduced_input_view, reduced_input_squared_view
    nisa.activation(
        dst=reduced_input_squared_view.get_view(),
        op=nl.square,
        data=input_mean[...],
        bias=zero_bias_view.get_view(),
    )
    nisa.tensor_tensor(
        reduced_input_view.get_view(),
        input_squared_mean[...],
        reduced_input_squared_view.get_view(),
        nl.subtract,
    )
    var_view = reduced_input_view

    # Step 5: Compute 1/sqrt(var + eps)
    rsqrt_view = reduced_input_squared_view
    nisa.activation(dst=rsqrt_view.get_view(), op=nl.rsqrt, data=var_view.get_view(), bias=eps_view.get_view())

    # Step 6: Apply gamma scaling
    nisa.tensor_tensor(
        output_sb_view.get_view(),
        output_sb_view.get_view(),
        gamma_sb_view.get_view(),
        nl.multiply,
    )

    # Step 7: Normalize centered input
    rsqrt_broadcast = rsqrt_view.expand_dim(dim=2).broadcast(dim=2, size=H1)
    nisa.tensor_tensor(
        output_sb_view.get_view(),
        output_sb_view.get_view(),
        rsqrt_broadcast.get_view(),
        nl.multiply,
    )

    # Step 8: Apply beta bias if present
    if beta_sb_view is not None:
        nisa.tensor_tensor(
            output_sb_view.get_view(),
            output_sb_view.get_view(),
            beta_sb_view.get_view(),
            nl.add,
        )

    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()

    sbm.close_scope()


def _layernorm_tkg_impl(
    input: TensorView,
    gamma: TensorView,
    beta: Optional[TensorView],
    output: TensorView,
    num_H_shards: int,
    hidden_actual: int,
    eps: float,
    shard_on_h: bool = False,
    use_heap_memory: bool = False,
    sbm: SbufManager = None,
):
    """
    Perform LayerNorm on input tensor with sharding support.

    The input is loaded from HBM (or reused from SBUF), gamma/beta are loaded,
    and LayerNorm is computed tile-by-tile along the BxS dimension.

    Args:
        input (TensorView): [BxS, H] when in HBM or [H0, BxS, H1] when in SBUF, Input tensor view.
        gamma (TensorView): [1, H], Gamma tensor view.
        beta (Optional[TensorView]): [1, H], Beta tensor view.
        output (TensorView): [H0, BxS, H1], Output tensor view in SBUF.
        num_H_shards (int): Number of shards along H dimension.
        hidden_actual (int): Actual hidden dimension size for mean calculation.
        eps (float): Epsilon for numerical stability.
        shard_on_h (bool): If True, exchange partial sums between cores.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (SbufManager): SBUF memory manager instance.

    Returns:
        None: Results written directly to output tensor view.

    Notes:
        - Uses Static DMA for input data reads (superior performance vs DGE).
        - For LNC sharding: input [B, S, lnc, H//lnc] reshaped to [BxS, lnc, H0, H1//lnc].
        - After transpose: [H0, BxS, lnc, H1//lnc] reshaped back to [H0, BxS, H1].
        - LayerNorm performed on combined [H0, lnc, H1//lnc] dimension.
    """

    if input.is_sbuf():
        H0, BxS, H1 = input.shape
        H = H0 * H1
    else:
        BxS, H = input.shape
        H0 = nl.tile_size.pmax
        H1 = H // H0

    inter_dtype = nl.float32

    alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack

    num_allocated_tensor = 0

    # Load input, reuse output buffer
    if input.is_sbuf():
        input_sb_view = input
    else:
        input_sb_view = load_input_to_sbuf(
            input_hbm=input,
            input_sb=output,
            num_H_shards=num_H_shards,
            hidden_dim_tp=False,
            shard_on_h=shard_on_h,
            sbm=sbm,
        )

    # Load gamma
    gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, name="layernorm_gamma")
    num_allocated_tensor += 1
    gamma_sb_view = load_gamma_to_sbuf(
        gamma_hbm=gamma,
        gamma_sb=TensorView(gamma_sb),
        num_H_shards=num_H_shards,
        hidden_dim_tp=False,
        shard_on_h=shard_on_h,
    )

    # Load beta if present
    beta_sb_view = None
    if beta is not None:
        beta_sb = alloc_tensor(shape=(H0, H1), dtype=beta.dtype, name="layernorm_beta")
        num_allocated_tensor += 1
        beta_sb_view = load_gamma_to_sbuf(
            gamma_hbm=beta,
            gamma_sb=TensorView(beta_sb),
            num_H_shards=num_H_shards,
            hidden_dim_tp=False,
            shard_on_h=shard_on_h,
        )

    # Allocate shared constants
    zero_bias = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_zero_bias")
    nisa.memset(zero_bias, value=0.0)
    num_allocated_tensor += 1

    eps_sb = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_eps")
    nisa.memset(eps_sb, value=eps)
    num_allocated_tensor += 1

    matmul_reduction_const = alloc_tensor(
        shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_mm_reduced_const"
    )
    nisa.memset(dst=matmul_reduction_const, value=(1.0 / hidden_actual))
    num_allocated_tensor += 1

    for bxs_tile in TiledRange(BxS, BxS_FULL_TILE_SIZE):
        input_sb_view_tile = input_sb_view.slice(
            dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size
        )
        gamma_sb_view_tile = gamma_sb_view.expand_dim(dim=1).broadcast(dim=1, size=bxs_tile.size)
        beta_sb_view_tile = None
        if beta_sb_view is not None:
            beta_sb_view_tile = beta_sb_view.expand_dim(dim=1).broadcast(dim=1, size=bxs_tile.size)
        output_sb_view_tile = output.slice(
            dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size
        )

        _process_layernorm_tile(
            input_sb_view=input_sb_view_tile,
            gamma_sb_view=gamma_sb_view_tile,
            output_sb_view=output_sb_view_tile,
            beta_sb_view=beta_sb_view_tile,
            zero_bias_view=TensorView(zero_bias),
            eps_view=TensorView(eps_sb),
            matmul_reduction_const_view=TensorView(matmul_reduction_const),
            bxs_tile=bxs_tile,
            hidden_actual=hidden_actual,
            shard_on_h=shard_on_h,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )

    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()
