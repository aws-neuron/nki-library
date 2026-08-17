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

"""RMSNorm kernel optimized for token generation (decoding) phase with efficient sharding and memory management."""

from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.common_types import MoEBlockIOLayout
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..utils.logging import get_logger
from ..utils.tiled_range import TiledRange
from .norm_tkg_utils import (
    _DGE_MODE_NONE,
    load_gamma_to_sbuf,
    load_input_to_sbuf,
    pe_transpose,
    validate_shapes,
    validate_shapes_shard_on_h,
)

# Minimum BxS size to enable sharding (balances computation vs communication overhead)
SHARDING_THRESHOLD = 18

# Tile size for BxS dimension processing
BxS_FULL_TILE_SIZE = 512


def rmsnorm_tkg(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    hidden_dim_tp: bool = False,
    single_core_forced: bool = False,
    shard_on_h: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
    inp_layout: MoEBlockIOLayout = MoEBlockIOLayout.B_S_H,
):
    """
    RMSNorm implementation optimized for inference token generation (decoding) phase.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size

    Args:
        input (nl.NkiTensor): [B, S, H] when in HBM or [128, B×S, H//128] when in SBUF, Input tensor.
            When inp_layout=_128_Nprgs_Hfree_T, expects [128, n_prgs, H//128//n_prgs, B×S] in HBM.
        gamma (nl.NkiTensor): [1, H], Gamma tensor used in normalization on HBM
        output (nl.NkiTensor): [128, B×S, H//128], Output tensor
        eps (float): Epsilon to maintain numerical stability. Default is 1e-6
        hidden_actual (Optional[int]): Actual hidden dimension size for mean calculation
            when input is padded. Default is None
        hidden_dim_tp (bool): If True, input H dimension view is (H/128, 128); if False, (128, H/128). Default is False
        single_core_forced (bool): If True, force single-core execution. Default is False
        shard_on_h (bool): If True, shard computation along H dimension instead of BxS. Default is False
        use_heap_memory (bool): Indicates whether to allocate memory on heap instead of stack. Default is False
        sbm (Optional[SbufManager]): Instance of SbufManager responsible for handling sbuf allocation. Default is None
        inp_layout (MoEBlockIOLayout): Input tensor layout. When _128_Nprgs_Hfree_T, input is
            [128, n_prgs, H//128//n_prgs, T] in HBM; loads and permutes to [128, T, H//128]
            in SBUF before normalization. Default is B_S_H

    Returns:
        output (nl.NkiTensor): [128, BxS, H//128], Output tensor with RMSNorm applied

    Notes:
        - H must be divisible by 128 (partition dimension).
        - When LNC=2 and BxS > SHARDING_THRESHOLD, computation is sharded across cores.
        - Output layout is transposed for efficient downstream sharded matmul.
        - shard_on_h=True requires LNC=2 and is incompatible with hidden_dim_tp and single_core_forced.

    Pseudocode:
        result = RMSNorm(hidden, gamma, eps)
        result = result.reshape((BxS, -1))
        t0 = result[:, 0:H//LNC_SIZE]
        t1 = result[:, H//LNC_SIZE:]
        t0 = t0.reshape((BxS, 128, H//128//LNC_SIZE)).transpose((1, 0, 2))
        t1 = t1.reshape((BxS, 128, H//128//LNC_SIZE)).transpose((1, 0, 2))
        result = np.concatenate([t0, t1], axis=2)
    """

    input_view = input
    gamma_view = gamma
    output_view = output

    # Handle transposed input: load [H0, n_prgs, H1_shard, T] -> permute to [H0, T, H_free] in SBUF
    if inp_layout == MoEBlockIOLayout._128_Nprgs_Hfree_T:
        _pmax = nl.tile_size.pmax
        H_free = input.shape[1] * input.shape[2]  # n_prgs * H1_shard
        T = input.shape[3]
        input_raw_sb = nl.ndarray((_pmax, H_free, T), dtype=input.dtype, buffer=nl.sbuf, name="tin_raw")
        nisa.dma_copy(
            dst=input_raw_sb,
            src=input.reshape((_pmax, H_free * T))[:, : H_free * T].reshape((_pmax, H_free, T)),
        )
        input_perm_sb = nl.ndarray((_pmax, T, H_free), dtype=input.dtype, buffer=nl.sbuf, name="tin_perm")
        nisa.tensor_copy(
            dst=input_perm_sb,
            src=input_raw_sb.permute(dims=[0, 2, 1]),
        )
        input_view = input_perm_sb

    if not sbm:
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=nl.tile_size.total_available_sbuf_size,
            logger=get_logger("rmsnorm_tkg"),
            use_auto_alloc=True,
        )

    sbm.open_scope(name="rmsnorm_tkg")

    if shard_on_h:
        kernel_assert(not hidden_dim_tp, "hidden_dim_tp is not supported in shard on H implementation")
        kernel_assert(not single_core_forced, "single_core_forced is not supported in shard on H implementation")
        _rmsnorm_tkg_shard_on_h(
            input_view=input_view,
            gamma_view=gamma_view,
            output_view=output_view,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=False,
            single_core_forced=False,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )
    else:
        _rmsnorm_tkg_shard_on_bxs(
            input_view=input_view,
            gamma_view=gamma_view,
            output_view=output_view,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=hidden_dim_tp,
            single_core_forced=single_core_forced,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )

    sbm.close_scope()

    return output_view


def _rmsnorm_tkg_shard_on_bxs(
    input_view: nl.NkiTensor,
    gamma_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    hidden_dim_tp: bool = False,
    single_core_forced: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm with sharding on the BxS dimension.

    Distributes work across LNC cores by splitting the batch-sequence dimension.
    Each core processes its shard independently, then results are exchanged if
    the output is in SBUF.

    Args:
        input_view (NkiTensor): [B, S, H] when in HBM or [H0, BxS, H1] when in SBUF, Input tensor view.
        gamma_view (NkiTensor): [1, H], Gamma tensor view.
        output_view (NkiTensor): [H0, BxS, H1], Output tensor view.
        eps (float): Epsilon for numerical stability.
        hidden_actual (Optional[int]): Actual hidden dimension size for padded inputs.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        single_core_forced (bool): If True, force single-core execution.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (Optional[SbufManager]): SBUF memory manager instance.

    Returns:
        None: Results written to output_view.
    """
    BxS, H, H0, H1 = validate_shapes(input_view, gamma_view, output_view)

    if output_view.buffer == nl.sbuf:
        output_sb_view = output_view
    else:
        alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack
        output_sb = alloc_tensor((H0, BxS, H1), dtype=input_view.dtype, buffer=nl.sbuf)
        output_sb_view = output_sb

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    num_shards = lnc if not single_core_forced else 1
    do_shard = num_shards == 2 and BxS > SHARDING_THRESHOLD and BxS % lnc == 0
    if not do_shard:
        num_shards, shard_id = 1, 0

    # When single_core_forced=True, use num_shards (1) for H sharding
    # When single_core_forced=False, use lnc for H sharding (original behavior)
    num_H_shards_in_output_H_dim = 1 if hidden_dim_tp or single_core_forced else lnc

    shard_size = BxS // num_shards

    if not (input_view.buffer == nl.sbuf):
        input_view_flat = input_view.flatten_dims(start_dim=0, end_dim=1)
        input_view_sharded = input_view_flat.slice(dim=0, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)
    else:
        input_view_sharded = input_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)

    output_view_sharded = output_sb_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)

    _rmsnorm_tkg_llama_impl(
        input=input_view_sharded,
        gamma=gamma_view,
        output=output_view_sharded,
        num_H_shards=num_H_shards_in_output_H_dim,
        hidden_actual=H if not hidden_actual else hidden_actual,
        eps=eps,
        hidden_dim_tp=hidden_dim_tp,
        shard_on_h=False,
        use_heap_memory=use_heap_memory,
        sbm=sbm,
    )

    if output_view.buffer == nl.sbuf:
        if do_shard:
            output_view_sharded_other_core = output_sb_view.slice(
                dim=1, start=(1 - shard_id) * shard_size, end=(2 - shard_id) * shard_size
            )
            nisa.sendrecv(
                dst=output_view_sharded_other_core,
                src=output_view_sharded,
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
            )
    else:
        output_hbm_view_sharded = output_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)
        nisa.dma_copy(dst=output_hbm_view_sharded, src=output_view_sharded)
        if use_heap_memory:
            sbm.pop_heap()  # dealloc output_sb


def _rmsnorm_tkg_shard_on_h(
    input_view: nl.NkiTensor,
    gamma_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    hidden_dim_tp: bool = False,
    single_core_forced: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm with sharding on the H (hidden) dimension.

    Each LNC core processes a shard of the hidden dimension. Partial sums for
    RMS are exchanged between cores via sendrecv to compute the full statistics
    before normalization.

    Args:
        input_view (NkiTensor): [B, S, H] when in HBM or [H0, BxS, sharded_H1] when in SBUF, Input tensor view.
        gamma_view (NkiTensor): [1, H], Gamma tensor view.
        output_view (NkiTensor): [H0, BxS, H1] or [H0, BxS, sharded_H1] if in SBUF, Output tensor view.
        eps (float): Epsilon for numerical stability.
        hidden_actual (Optional[int]): Actual hidden dimension size for padded inputs.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        single_core_forced (bool): If True, force single-core execution.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (Optional[SbufManager]): SBUF memory manager instance.

    Returns:
        None: Results written to output_view.
    """
    BxS, H, H0, H1, shard_H, shard_H1 = validate_shapes_shard_on_h(input_view, gamma_view, output_view)

    if output_view.buffer == nl.sbuf:
        output_sb_view = output_view
    else:
        alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack
        output_sb = alloc_tensor((H0, BxS, shard_H1), dtype=input_view.dtype, buffer=nl.sbuf)
        output_sb_view = output_sb

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    input_view_sharded = input_view
    if not (input_view.buffer == nl.sbuf):
        input_view_flat = input_view.flatten_dims(start_dim=0, end_dim=1)
        input_view_sharded = input_view_flat.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    gamma_view = gamma_view.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    _rmsnorm_tkg_llama_impl(
        input=input_view_sharded,
        gamma=gamma_view,
        output=output_sb_view,
        num_H_shards=lnc,
        hidden_actual=H if not hidden_actual else hidden_actual,
        eps=eps,
        hidden_dim_tp=hidden_dim_tp,
        shard_on_h=True,
        use_heap_memory=use_heap_memory,
        sbm=sbm,
    )

    if output_view.buffer == nl.sbuf:
        output_view = output_sb_view
    else:
        output_hbm_view_sharded = output_view.slice(dim=2, start=shard_id * shard_H1, end=(shard_id + 1) * shard_H1)
        nisa.dma_copy(dst=output_hbm_view_sharded, src=output_sb_view)


def _process_rmsnorm_tile(
    input_sb_view: nl.NkiTensor,
    gamma_sb_view: nl.NkiTensor,
    output_sb_view: nl.NkiTensor,
    eps_view: nl.NkiTensor,
    matmul_reduction_const_view: nl.NkiTensor,
    bxs_tile: Tuple,
    hidden_actual: int,
    shard_on_h: bool,
    use_heap_memory: bool = False,
    sbm: SbufManager = None,
):
    """
    Process a single tile of RMSNorm computation.

    Args:
        input_sb_view (NkiTensor): [H0, BxS, H1], Input tensor view in SBUF
        gamma_sb_view (NkiTensor): [H0, BxS, H1], Gamma tensor view in SBUF (broadcasted)
        output_sb_view (NkiTensor): [H0, BxS, H1], Output tensor view in SBUF
        eps_view (NkiTensor): [H0, 1], Epsilon value for numerical stability
        matmul_reduction_const_view (NkiTensor): [H0, H0], Constant matrix for reduction
        bxs_tile (Tuple): Tile information containing index and size
        hidden_actual (int): Actual hidden dimension size
        shard_on_h (bool): If True, exchange partial sums between cores.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack
        sbm (SbufManager): SBUF memory manager instance

    Returns:
        None: Results written directly to output_sb_view

    Notes:
        - Computes RMSNorm: output = (input * gamma) / sqrt(mean(input^2) + eps)
        - Uses intermediate float32 precision for numerical stability
    """
    alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack

    sbm.open_scope()

    num_allocated_tensor = 0
    inter_dtype = nl.float32

    kernel_assert(
        input_sb_view.shape == output_sb_view.shape, "Input and output tensor shapes must match for RMSNorm processing"
    )
    H0, BxS, H1 = input_sb_view.shape

    # Compute x^2 for RMS calculation
    rmsnorm_square = alloc_tensor(shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf)
    num_allocated_tensor += 1
    nisa.activation(rmsnorm_square[...], op=nl.square, data=input_sb_view)

    # Reduce along H1 dimension to compute sum(x^2)
    rmsnorm_reduced_square = alloc_tensor(
        shape=(H0, BxS),
        dtype=inter_dtype,
        buffer=nl.sbuf,
    )
    num_allocated_tensor += 1
    nisa.tensor_reduce(rmsnorm_reduced_square[...], nl.add, rmsnorm_square[...], axis=2)

    if shard_on_h:
        _, _, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))
        # Exchange partial sums with the other core via sendrecv
        remote_reduced_square = alloc_tensor(
            shape=(H0, BxS),
            dtype=inter_dtype,
            buffer=nl.sbuf,
        )
        num_allocated_tensor += 1
        nisa.sendrecv(
            dst=remote_reduced_square[...],
            src=rmsnorm_reduced_square[...],
            send_to_rank=1 - shard_id,
            recv_from_rank=1 - shard_id,
            pipe_id=0,
        )

        # Combine partial sums: full_sq = local_sq + remote_sq
        nisa.tensor_tensor(
            rmsnorm_reduced_square[...],
            rmsnorm_reduced_square[...],
            remote_reduced_square[...],
            nl.add,
        )

    # Apply gamma scaling: input * gamma
    gamma_mult = alloc_tensor(shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf)
    num_allocated_tensor += 1
    gamma_mult_view = gamma_mult
    nisa.tensor_tensor(
        gamma_mult_view,
        input_sb_view,
        gamma_sb_view,
        nl.multiply,
    )

    # Complete reduction across H0 dimension using matmul
    if sbm.is_auto_alloc():
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum)
    else:
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
    nisa.nc_matmul(
        stationary=matmul_reduction_const_view,
        moving=rmsnorm_reduced_square,
        dst=final_reduced,
    )

    # Compute normalization factor: 1/sqrt(mean(x^2) + eps)
    hidden_scale = 1.0 / hidden_actual
    nisa.activation(
        rmsnorm_reduced_square[...],
        op=nl.rsqrt,
        data=final_reduced[...],
        scale=hidden_scale,
        bias=eps_view,
    )

    # Final RMSNorm: (input * gamma) * normalization_factor
    reduced_view = rmsnorm_reduced_square.expand_dim(dim=2).broadcast(dim=2, size=H1)
    nisa.tensor_tensor(
        output_sb_view,
        gamma_mult_view,
        reduced_view,
        nl.multiply,
    )

    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()

    sbm.close_scope()


def rmsnorm_tkg_th(
    input_hbm: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    num_H_shards: int,
    hidden_actual: int,
    eps: float,
    sbm: SbufManager = None,
    use_contiguous_x4: bool = False,
):
    """
    RMSNorm with contiguous DMA load and does rmsnorm in [T,H] layout.

    Computation is done entirely in [T, H] layout (T on partition dim, H on free dim).
    Only the final result is transposed to [H0, H1_shard, T] for downstream matmul.

    When use_contiguous_x4=True, the output layout is [H0, n_H512*4, T] instead of
    [H0, H2, T]. This produces contiguous-4 H packing
    where partition p' at free position (h512*4+q, t) holds H = 512*h512 + 4*p' + q.
    The layout change is done on-chip via tensor_copy permute + pe_transpose,
    avoiding the expensive HBM spill+reload that would otherwise be needed in the
    downstream MLP kernel. The caller permutes [H0, n_H512*4, T] → [H0, T, n_H512*4]
    to get [P0=128, BxS=T, F0=n_H512*4] for row_quantization.

    Args:
        input_hbm: [BxS, H] NkiTensor in HBM (already T-sliced by caller)
        gamma: [1, H] nl.NkiTensor in HBM
        output: [H0, H1_shard, T] NkiTensor in SBUF for gate/up projection,
                or [H0, n_H512*4, T] when use_contiguous_x4=True
        num_H_shards: number of H-shards (LNC count)
        hidden_actual: full H for mean calculation
        eps: epsilon
        sbm: SBUF memory manager
        use_contiguous_x4: if True, produce contiguous-4 H packing in output
    """
    T, H = input_hbm.shape
    T0 = nl.tile_size.pmax
    H0 = nl.tile_size.pmax
    _pmax = H0
    H1 = H // H0
    H2 = H1 // num_H_shards  # H1 tiles per shard
    H_per_shard = H0 * H2
    inter_dtype = nl.float32
    _q_width = 4  # x4 packing width

    if use_contiguous_x4:
        kernel_assert(H_per_shard % 512 == 0, "H_per_shard must be divisible by 512 for contiguous x4")
        n_H512 = H_per_shard // 512  # = H2 // _q_width

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))
    # When num_H_shards=1, both cores process full H — no H-shard slicing needed
    if num_H_shards == 1:
        shard_id = 0

    alloc_tensor = sbm.alloc_heap

    # Pre-allocate tensors at H0 size (reused across T-tiles)
    gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, buffer=nl.sbuf)
    sbuf_buffer = alloc_tensor(
        (T0, H),  # [128, H]
        dtype=input_hbm.dtype,
        buffer=nl.sbuf,
    )
    reduced_sq = alloc_tensor(shape=(T0, 1), dtype=inter_dtype, buffer=nl.sbuf)  # [128, 1]
    square_sb = alloc_tensor(shape=(H0, H), dtype=inter_dtype, buffer=nl.sbuf)  # [128, H]

    if not use_contiguous_x4:
        # Standard path: gamma in [H0, H2] layout for post-transpose application
        gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, buffer=nl.sbuf, name="rmsnorm_th_gamma")

        gamma_hbm_view = (
            gamma.flatten_dims(start_dim=0, end_dim=1)
            .reshape_dim(dim=0, shape=[num_H_shards, H0, H2])
            .permute(dims=[1, 0, 2])
        )
        gamma_sb_view = gamma_sb.reshape_dim(dim=1, shape=[num_H_shards, H2])
        nisa.dma_copy(dst=gamma_sb_view, src=gamma_hbm_view, dge_mode=_DGE_MODE_NONE)
        gamma_shard_view = (
            gamma_sb.reshape_dim(dim=1, shape=[num_H_shards, H2])
            .slice(dim=1, start=shard_id, end=shard_id + 1)
            .squeeze_dim(dim=1)
        )
    else:
        # Contiguous x4 path: gamma in [T0, H_per_shard] layout for pre-transpose application.
        # Load gamma[shard_start : shard_start + H_per_shard] as flat vector, broadcast on P dim.
        gamma_flat_sb = alloc_tensor(
            shape=(T0, H_per_shard), dtype=gamma.dtype, buffer=nl.sbuf, name="rmsnorm_th_gamma_flat"
        )
        H_shard_start_gamma = shard_id * H_per_shard
        gamma_hbm_flat = gamma.flatten_dims(start_dim=0, end_dim=1).slice(
            dim=0, start=H_shard_start_gamma, end=H_shard_start_gamma + H_per_shard
        )
        # Broadcast [1, H_per_shard] → [T0, H_per_shard]
        gamma_hbm_broadcast = gamma_hbm_flat.expand_dim(dim=0).broadcast(dim=0, size=T0)
        nisa.dma_copy(dst=gamma_flat_sb, src=gamma_hbm_broadcast, dge_mode=_DGE_MODE_NONE)

        # Temp buffer for tensor_copy permute before pe_transpose
        cx4_perm_buf = alloc_tensor(
            shape=(T0, H_per_shard), dtype=input_hbm.dtype, buffer=nl.sbuf, name="rmsnorm_th_cx4_perm"
        )

        # Temp buffer for pe_transpose output (reused across T-tiles)
        xpose_out_buf = alloc_tensor(
            shape=(_pmax, T0, n_H512 * _q_width), dtype=input_hbm.dtype, buffer=nl.sbuf, name="rmsnorm_th_xpose_out"
        )

    hidden_scale = 1.0 / hidden_actual
    H_shard_start = shard_id * H_per_shard

    # Compute PSUM banks per T-tile for bank rotation across tiles
    _psum_fmax = nl.tile_size.psum_fmax
    _dtype_size = sizeinbytes(input_hbm.dtype)
    _padded_tile_size = div_ceil(H0 * _dtype_size, 4) * 4 // _dtype_size  # 4 byte aligment
    _tiles_per_psum = _psum_fmax // _padded_tile_size
    _psum_banks_per_tile_rmsnorm = div_ceil(H2, _tiles_per_psum)

    for t_tile in TiledRange(T, T0):
        tile_T = t_tile.size

        # Slice per-tile views from pre-allocated buffers
        tile_buf = sbuf_buffer.slice(dim=0, start=0, end=tile_T)  # [T_tile, H]
        tile_sq = square_sb.slice(dim=0, start=0, end=tile_T)  # [T_tile , H]
        tile_red = reduced_sq.slice(dim=0, start=0, end=tile_T)  # [T_tile, 1]

        # --- Step 1: Contiguous load [tile_T, H] from HBM to SBUF ---
        input_hbm_tile = input_hbm.slice(dim=0, start=t_tile.start_offset, end=t_tile.end_offset)
        nisa.dma_copy(dst=tile_buf, src=input_hbm_tile, dge_mode=_DGE_MODE_NONE)

        # --- Step 2: RMSNorm in [tile_T, H] layout ---
        # Compute x^2 and reduce along H: [tile_T, H] -> [tile_T, 1]
        nisa.activation_reduce(
            tile_sq,
            op=nl.square,
            data=tile_buf,
            reduce_op=nl.add,
            reduce_res=tile_red,
        )

        # Normalization factor: 1/sqrt(mean(x^2) + eps)
        nisa.activation(tile_red, op=nl.rsqrt, data=tile_red, scale=hidden_scale, bias=eps)

        # input * normalization_factor for this shard's H slice
        input_shard = tile_buf.slice(dim=1, start=H_shard_start, end=H_shard_start + H_per_shard)  # [T_tile, H_shard]
        norm_result_shard = tile_buf.slice(dim=1, start=0, end=H_per_shard)  # [T_tile, H_shard]
        nisa.tensor_scalar(
            norm_result_shard,
            input_shard,
            op0=nl.multiply,
            operand0=tile_red,
        )

        if not use_contiguous_x4:
            # --- Standard path ---
            # Step 3: Transpose [tile_T, H_per_shard] → [H0, H2, tile_T]
            src = norm_result_shard.reshape_dim(dim=1, shape=[H0, H2])  # [tile_T, H0, H2]
            dst = output.slice(dim=2, start=t_tile.start_offset, end=t_tile.end_offset)  # [H0, H2, tile_T]
            dst_permuted = dst.permute(dims=[0, 2, 1])  # → [H0, tile_T, H2]
            pe_transpose(
                src=src,
                dst=dst_permuted,
                tile_size=H0,
                dtype=input_hbm.dtype,
                sbm=sbm,
                psum_bank_base=t_tile.index * _psum_banks_per_tile_rmsnorm,
            )

            # Step 4: Apply gamma in [H0, H2, tile_T] layout
            gamma_broadcast = gamma_shard_view.expand_dim(dim=2).broadcast(dim=2, size=tile_T)
            nisa.tensor_tensor(
                dst,
                dst,
                gamma_broadcast,
                nl.multiply,
            )
        else:
            # --- Contiguous x4 path ---
            # Produces output [H0, n_H512*4, T] where partition p' at free position
            # (h512*4+q, t) holds H = 512*h512 + 4*p' + q.

            # Step 3a: Apply gamma in [T, H_per_shard] layout (before transpose).
            # gamma_flat_sb is [T0, H_per_shard] with gamma values broadcast on P dim.
            gamma_tile = gamma_flat_sb.slice(dim=0, start=0, end=tile_T)
            nisa.tensor_tensor(
                norm_result_shard,
                norm_result_shard,
                gamma_tile,
                nl.multiply,
            )

            # Step 3b: Reshape [T, n_H512, 128, 4] and permute dims 1↔2 → [T, 128, n_H512, 4]
            # so 128 is at dim 1 for pe_transpose (tile_size=128).
            src_4d = norm_result_shard.reshape_dim(dim=1, shape=[n_H512, _pmax, _q_width])
            perm_tile = cx4_perm_buf.slice(dim=0, start=0, end=tile_T)
            perm_dst_4d = perm_tile.reshape_dim(dim=1, shape=[_pmax, n_H512, _q_width])
            nisa.tensor_copy(
                dst=perm_dst_4d,
                src=src_4d.permute(dims=[0, 2, 1, 3]),
            )

            # Step 3c: pe_transpose [T, 128, n_H512*4] → [128(P), tile_T, n_H512*4]
            # into a dedicated contiguous temp buffer, then tensor_copy permute
            # to output [128, n_H512*4, tile_T].
            perm_src_3d = perm_tile.reshape_dim(dim=1, shape=[_pmax, n_H512 * _q_width])
            xpose_out_tile = xpose_out_buf.slice(dim=1, start=0, end=tile_T)
            pe_transpose(
                src=perm_src_3d,
                dst=xpose_out_tile,
                tile_size=_pmax,
                dtype=input_hbm.dtype,
                sbm=sbm,
                psum_bank_base=t_tile.index * _psum_banks_per_tile_rmsnorm,
            )

            # Copy transposed tile [128, tile_T, n_H512*4] → output [128, n_H512*4, tile_T]
            # via tensor_copy with permute [0, 2, 1].
            out_tile = output.slice(dim=2, start=t_tile.start_offset, end=t_tile.end_offset)
            nisa.tensor_copy(
                dst=out_tile,
                src=xpose_out_tile.permute(dims=[0, 2, 1]),
            )

    # Cleanup (reverse allocation order)
    if use_contiguous_x4:
        sbm.pop_heap()  # deallocate xpose_out_buf
        sbm.pop_heap()  # deallocate cx4_perm_buf
        sbm.pop_heap()  # deallocate gamma_flat_sb
    else:
        sbm.pop_heap()  # deallocate gamma_sb
    sbm.pop_heap()  # deallocate square_sb
    sbm.pop_heap()  # deallocate reduced_sq
    sbm.pop_heap()  # deallocate sbuf_buffer


def _rmsnorm_tkg_llama_impl(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    num_H_shards: int,
    hidden_actual: Optional[int],
    eps: float,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
    use_heap_memory: bool = False,
    sbm: SbufManager = None,
):
    """
    Perform RMSNorm on input tensor with sharding support.

    Args:
        input (NkiTensor): [BxS, H] when in HBM or [H0, BxS, H1] when in SBUF, Input tensor view
        gamma (NkiTensor): [1, H], Gamma tensor view
        output (NkiTensor): [H0, BxS, H1], Output tensor view in SBUF
        num_H_shards (int): Number of shards along H dimension
        hidden_actual (Optional[int]): Actual hidden dimension size
        eps (float): Epsilon for numerical stability
        hidden_dim_tp (bool): If True, input H dimension view is (H/128, 128); if False, (128, H/128)
        shard_on_h (bool): If True, exchange partial sums between cores.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack
        sbm (SbufManager): SBUF memory manager instance

    Returns:
        None: Results written directly to output tensor view

    Notes:
        - Uses Static DMA for input data reads (superior performance vs DGE)
        - For LNC sharding: input [B, S, lnc, H//lnc] reshaped to [BxS, lnc, H0, H1//lnc]
        - After transpose: [H0, BxS, lnc, H1//lnc] reshaped back to [H0, BxS, H1]
        - RMSNorm performed on combined [H0, lnc, H1//lnc] dimension
    """

    if input.buffer == nl.sbuf:
        H0, BxS, H1 = input.shape
        H = H0 * H1
    else:
        BxS, H = input.shape
        H0 = nl.tile_size.pmax
        H1 = H // H0

    if hidden_dim_tp:
        kernel_assert(num_H_shards == 1, "When hidden_dim_tp is True, num_H_shards must be 1")

    inter_dtype = nl.float32

    alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack

    num_allocated_tensor = 0

    # Load input and reuse output_buffer
    if input.buffer == nl.sbuf:
        input_sb_view = input
    else:
        input_sb_view = load_input_to_sbuf(
            input_hbm=input,
            input_sb=output,
            num_H_shards=num_H_shards,
            hidden_dim_tp=hidden_dim_tp,
            shard_on_h=shard_on_h,
            sbm=sbm,
        )

    # Load gamma
    # if hidden_dim_tp is on, rmsnorm_gamma offset needs to be 32B aligned
    gamma_align = 32 if hidden_dim_tp else None
    gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, align=gamma_align)
    num_allocated_tensor += 1
    gamma_sb_view = load_gamma_to_sbuf(
        gamma_hbm=gamma,
        gamma_sb=gamma_sb,
        num_H_shards=num_H_shards,
        hidden_dim_tp=hidden_dim_tp,
        shard_on_h=shard_on_h,
    )

    # Load eps
    eps_sb = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf)
    num_allocated_tensor += 1
    nisa.memset(eps_sb, value=eps)

    # Load matmul reduction const
    matmul_reduction_const = alloc_tensor(shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf)
    num_allocated_tensor += 1
    nisa.memset(matmul_reduction_const, value=1.0)

    for bxs_tile in TiledRange(BxS, BxS_FULL_TILE_SIZE):
        input_sb_view_tile = input_sb_view.slice(
            dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size
        )
        gamma_sb_view_tile = gamma_sb_view.expand_dim(dim=1).broadcast(dim=1, size=bxs_tile.size)
        output_sb_view_tile = output.slice(
            dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size
        )
        _process_rmsnorm_tile(
            input_sb_view=input_sb_view_tile,
            gamma_sb_view=gamma_sb_view_tile,
            output_sb_view=output_sb_view_tile,
            eps_view=eps_sb,
            matmul_reduction_const_view=matmul_reduction_const,
            bxs_tile=bxs_tile,
            hidden_actual=hidden_actual,
            shard_on_h=shard_on_h,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )

    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()


# Tile size for partition dimension in DLoC rmsnorm
_DLOC_T_TILE_SIZE = 128


def _rmsnorm_tkg_dloc(
    input_hbm: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    output_sb: nl.NkiTensor,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    sync_output: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm for dynamic all-expert MoE (DLoC layout).

    Computes RMSNorm in [T, H] layout (T on partition dim) with T tiled into
    128-token chunks. Produces two outputs simultaneously per tile:
      1. HBM store: normalized [T, H] written to output_hbm
      2. On-chip transpose: written to output_sb at each core's shard

    Each LNC core processes T//2 tokens independently.

    Args:
        input_hbm: [T, H] in HBM
        gamma: [1, H] in HBM
        output_hbm: [T, H] in shared_hbm, both cores write their halves
        output_sb: [H0, T//2, H1] if sync_output=False, [H0, T, H1] if sync_output=True, in SBUF
        eps: epsilon for numerical stability
        hidden_actual: actual H for mean calculation (if input is padded)
        sync_output: if True, sendrecv to exchange shards so each core has full [H0, T, H1];
                     if False, each core only has its own [H0, T//2, H1] shard (other half undefined)
        sbm: SBUF memory manager
    """
    # Flatten [B, S, H] -> [T, H] if needed
    if len(input_hbm.shape) == 3:
        B, S, H = input_hbm.shape
        T = B * S
        input_hbm_view = input_hbm.flatten_dims(start_dim=0, end_dim=1)
    else:
        T, H = input_hbm.shape
        input_hbm_view = input_hbm
    H0 = nl.tile_size.pmax  # 128
    H1 = H // H0
    inter_dtype = nl.float32
    tile_size = _DLOC_T_TILE_SIZE

    if hidden_actual is None:
        hidden_actual = H

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg_dloc", (0, 1))
    kernel_assert(lnc == 2, "rmsnorm_tkg_dloc requires LNC=2")
    kernel_assert(T % 2 == 0, "T must be even for LNC=2 sharding")

    T_shard = T // 2
    kernel_assert(T_shard % tile_size == 0, f"T//2 must be divisible by {tile_size}")

    if not sbm:
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=nl.tile_size.total_available_sbuf_size,
            logger=get_logger("rmsnorm_tkg_dloc"),
            use_auto_alloc=True,
        )

    sbm.open_scope(name="rmsnorm_tkg_dloc")

    num_tiles = T_shard // tile_size
    hidden_scale = 1.0 / hidden_actual
    sb_shard_offset = shard_id * T_shard if sync_output else 0

    # Pre-allocate per-tile scratch buffers
    tile_buf = sbm.alloc_heap((tile_size, H), dtype=input_hbm.dtype, buffer=nl.sbuf)
    reduced_sq = sbm.alloc_heap((tile_size, 1), dtype=inter_dtype, buffer=nl.sbuf)
    square_sb = sbm.alloc_heap((tile_size, H), dtype=inter_dtype, buffer=nl.sbuf)

    # Load gamma [1, H] -> [tile_size, H] replicated across partition dim
    gamma_sb = sbm.alloc_heap((tile_size, H), dtype=gamma.dtype, buffer=nl.sbuf)
    gamma_hbm_view = gamma.broadcast(dim=0, size=tile_size)
    nisa.dma_copy(dst=gamma_sb, src=gamma_hbm_view, dge_mode=_DGE_MODE_NONE)

    output_hbm_view = output_hbm
    output_sb_view = output_sb

    for tile_idx in range(num_tiles):
        t_start = shard_id * T_shard + tile_idx * tile_size  # HBM offset
        sb_t_start = sb_shard_offset + tile_idx * tile_size  # SBUF offset (includes shard offset)

        # --- Load [tile_size, H] from HBM ---
        input_tile = input_hbm_view.slice(dim=0, start=t_start, end=t_start + tile_size)
        nisa.dma_copy(dst=tile_buf, src=input_tile, dge_mode=_DGE_MODE_NONE)

        # --- RMSNorm compute in [tile_size, H] layout ---
        # Square + reduce along H: [tile_size, H] -> [tile_size, 1]
        nisa.activation_reduce(
            square_sb,
            op=nl.square,
            data=tile_buf,
            reduce_op=nl.add,
            reduce_res=reduced_sq,
        )

        # rsqrt(mean(x^2) + eps): [tile_size, 1]
        nisa.activation(reduced_sq[...], op=nl.rsqrt, data=reduced_sq[...], scale=hidden_scale, bias=eps)

        # x * (1/rms) * gamma: [tile_size, H]
        nisa.tensor_scalar(
            tile_buf[...],
            tile_buf[...],
            op0=nl.multiply,
            operand0=reduced_sq,
        )
        nisa.tensor_tensor(
            tile_buf[...],
            tile_buf[...],
            gamma_sb[...],
            nl.multiply,
        )

        # --- Output 1: HBM store [tile_size, H] (no layout change needed) ---
        output_tile = output_hbm_view.slice(dim=0, start=t_start, end=t_start + tile_size)
        nisa.dma_copy(dst=output_tile, src=tile_buf, dge_mode=_DGE_MODE_NONE)

        # --- Output 2: Transpose [tile_size, H] -> [H0, tile_size, H1] into output_sb ---
        src_3d = tile_buf.reshape_dim(dim=1, shape=[H0, H1])  # [tile_size, H0, H1]
        dst_3d = output_sb_view.slice(dim=1, start=sb_t_start, end=sb_t_start + tile_size)
        pe_transpose(src=src_3d, dst=dst_3d, tile_size=H0, dtype=input_hbm.dtype, sbm=sbm)

    sbm.pop_heap()  # gamma_sb
    sbm.pop_heap()  # square_sb
    sbm.pop_heap()  # reduced_sq
    sbm.pop_heap()  # tile_buf

    # --- HBM barrier: both cores must finish writing before downstream reads ---
    nisa.core_barrier(output_hbm, (0, 1))

    # --- Optional: sync SBUF output across cores ---
    if sync_output:
        other_offset = (1 - shard_id) * T_shard
        local_src = output_sb_view.slice(dim=1, start=sb_shard_offset, end=sb_shard_offset + T_shard)
        remote_dst = output_sb_view.slice(dim=1, start=other_offset, end=other_offset + T_shard)
        nisa.sendrecv(
            dst=remote_dst,
            src=local_src,
            send_to_rank=1 - shard_id,
            recv_from_rank=1 - shard_id,
            pipe_id=0,
        )

    sbm.close_scope()
