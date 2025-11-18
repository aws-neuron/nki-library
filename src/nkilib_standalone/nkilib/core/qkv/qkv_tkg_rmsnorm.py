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

"""
RMSNorm TKG Kernel

This kernel implements the Root Mean Square Normalization (RMSNorm) operation
commonly used in transformer models. The kernel is specifically optimized for
Token Generation (TKG, also known as Decoding) scenarios where batch_size * seqlen
is small.

This kernel is designed with LNC support. When LNC=2 and B*S exceeds the sharding
threshold, the computation is sharded across cores.

"""

from typing import Optional
import nki.language as nl
import nki.isa as nisa

from ..utils.allocator import SbufManager, create_auto_alloc_manager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import Logger


# This is a heuristic to decide whether to shard on B*S to halve the computation
# at the cost of extra local collective
SHARDING_THRESHOLD = 18


def rmsnorm_tkg(
    inp: nl.ndarray,
    gamma: nl.ndarray,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    output_in_sbuf: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
) -> nl.ndarray:
    """
    RMSNorm Kernel for Token Generation

    This kernel computes the RMSNorm operation:
        output = inp * gamma / sqrt(mean(inp^2) + eps)
    typically used for normalization in transformer models.

    This kernel is optimized for Token Generation (aka Decoding) use cases where
    batch_size * seqlen is small. Using this kernel with B*S > 128 may result in
    degraded performance - use the CTE variant for large sequence lengths.

    The output layout is specifically chosen to make subsequent sharded matmul efficient.
    The result is transposed from [B, S, H] to [H0, B*S, H1] where H0=128 and H1=H/128.

    Data Types:
        This kernel supports nl.float32, nl.float16, and nl.bfloat16 data types.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size

    Args:
        inp (nl.ndarray):
            Input tensor in HBM.
            Shape:    [B, S, H]
        gamma (nl.ndarray):
            Gamma weight tensor in HBM used for normalization scaling.
            Shape:    [1, H]
        eps (float):
            Epsilon value to maintain numerical stability. Default: 1e-6.
        hidden_actual (int, optional):
            Actual hidden dimension for padded input tensors. If specified, normalization
            uses this value (1/hidden_actual) instead of 1/H for mean calculation.
        output_in_sbuf (bool):
            If True, output is kept in SBUF; otherwise stored to HBM. Default: False.
        use_heap_memory (bool):
            If True, allocates memory on the heap instead of the stack. Default: False.
        sbm (SbufManager, optional):
            Instance of SbufManager responsible for handling SBUF allocation.
            If None, auto-allocation manager is created.

    Returns:
        output (nl.ndarray):
            Output tensor. The tensor can reside in either SBUF or HBM.
            Shape:    [H0, B * S, H1] = [128, B * S, H/128]

    Restrictions:
        H must be divisible by 128 (nl.tile_size.pmax).
    """

    B, S, H = inp.shape
    H0 = nl.tile_size.pmax
    kernel_assert(H % H0 == 0, f"H must be divisible by {H0}, got {H} % {H0} == {H % H0}")
    H1 = H // H0
    kernel_assert(
        gamma.shape == (1, H),
        f"Malformed shape of gamma {gamma.shape}, expected {(1, H)}",
    )

    if not hidden_actual:
        hidden_actual = H

    if not sbm:
        sbm = create_auto_alloc_manager()

    # open sbm scope
    sbm.open_scope()

    # Allocate output buffers
    if not output_in_sbuf:
        output = nl.ndarray((H0, B * S, H1), dtype=inp.dtype, buffer=nl.shared_hbm)

    if use_heap_memory:
        sharded_sbuf_result = sbm.alloc_heap(
            (H0, B * S, H1), dtype=inp.dtype, buffer=nl.sbuf, name="rmsnorm_shared_sbuf"
        )
    else:
        sharded_sbuf_result = sbm.alloc_stack(
            (H0, B * S, H1), dtype=inp.dtype, buffer=nl.sbuf, name="rmsnorm_shared_sbuf"
        )

    # Check program dimensionality
    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    # We don't always do LNC2 sharding if BxS is less than threshold.
    num_shards = lnc
    do_shard = num_shards == 2 and (B * S) > SHARDING_THRESHOLD and (B * S) % lnc == 0
    if not do_shard:
        num_shards, shard_id = 1, 0  # Otherwise don't shard the compute.

    shard_size = (B * S) // num_shards

    # Process RMSNorm
    _rmsnorm_tkg_llama_impl(
        inp=inp,
        gamma=gamma,
        result=sharded_sbuf_result,
        bs_lb=shard_id * shard_size,
        bs_count=shard_size,
        lnc=lnc,
        hidden_actual=hidden_actual,
        eps=eps,
        use_heap_memory=use_heap_memory,
        sbm=sbm,
    )

    if output_in_sbuf:
        if do_shard:
            nisa.sendrecv(
                dst=sharded_sbuf_result[:, nl.ds((1 - shard_id) * shard_size, shard_size), :],
                src=sharded_sbuf_result[:, nl.ds(shard_id * shard_size, shard_size), :],
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
            )

        return sharded_sbuf_result.reshape((H0, B * S, H1))

    # save output to HBM
    sharded_sbuf_result = sharded_sbuf_result.reshape((H0, B * S, H1))
    output = output.reshape(sharded_sbuf_result.shape)

    nisa.dma_copy(
        dst=output[:, nl.ds(shard_id * shard_size, shard_size), :],
        src=sharded_sbuf_result[:, nl.ds(shard_id * shard_size, shard_size), :],
    )

    # deallocate sharded_sbuf_result
    if use_heap_memory:
        sbm.pop_heap()

    # close sbm scope
    sbm.close_scope()

    return output


def _rmsnorm_tkg_llama_impl(
    inp: nl.ndarray,
    gamma: nl.ndarray,
    result: nl.ndarray,
    bs_lb: int,
    bs_count: int,
    lnc: int,
    hidden_actual: int,
    eps: float,
    use_heap_memory: bool,
    sbm: SbufManager,
):
    """
    Internal implementation of RMSNorm computation.

    This function handles the core RMSNorm computation with support for LNC sharding.

    Layout Handling:
        - LNC 1: Input [B, S, H] is loaded from HBM to [H0, BxS, H1] SBUF
        - LNC 2: Input [B, S, H] is loaded from HBM to [H0, bs_lb:bs_lb+bs_count, H1] SBUF

    Args:
        inp (nl.ndarray): Input tensor of shape [B, S, H], this tensor is in HBM.
                          H must be divisible by nl.tile_size.pmax (128).
                          B*S*(H//128) must fit in SBUF.
        gamma (nl.ndarray): Gamma weight tensor of shape [1, H], this tensor is in HBM.
        result (nl.ndarray): Result SBUF tensor of shape (H0, B*S, H1) to write the result to.
        bs_lb (int): Inclusive lower bound of where to start processing on B*S dimension.
        bs_count (int): Number of batches to process on B*S dimension.
        lnc (int): Logical neuron core count, determines output sharding layout.
        hidden_actual (int): Actual hidden dimension for mean calculation (handles padding).
        eps (float): Epsilon value to maintain numerical stability.
        use_heap_memory (bool): If True, allocates memory on the heap instead of the stack.
        sbm (SbufManager): Instance of SbufManager responsible for handling SBUF allocation.

    Returns:
        None. Result is written directly to the provided result tensor.
    """

    # Already validated, skip validation
    B, S, H = inp.shape
    BxS = bs_count
    # All intermediates need to happen in FP32
    inter_dtype = nl.float32
    BxS = bs_count
    H0 = nl.tile_size.pmax
    H1 = H // H0  # number of element in each partition
    H2 = H1 // 2

    # open sbm scope
    sbm.open_scope()

    # Check if the kernel uses auto or manual allocation
    is_auto_alloc = sbm.is_auto_alloc()

    # Define alloc function to use: stack or heap
    if use_heap_memory:
        alloc_tensor = sbm.alloc_heap
    else:
        alloc_tensor = sbm.alloc_stack

    num_allocated_tensor = 0

    # Allocate input tensors
    if bs_count == B * S:
        input_sb = result
    else:
        input_sb = alloc_tensor(shape=(H0, bs_count, H1), dtype=inp.dtype, name="rmsnorm_input")
        num_allocated_tensor += 1

    gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, name="rmsnorm_gamma")
    num_allocated_tensor += 1

    #####################################################################
    ############### Load input, gamma and optionally beta ###############
    #####################################################################
    if lnc == 1:  # LNC 1
        # Reshape input (B, S, H) to (BxS, H0=128, H1=H//H0)
        # Load input shape of (BxS, H0, H1) to (H0, BxS, H1)
        inp = inp.reshape((BxS, H0, H1))
        nisa.dma_copy(
            dst=input_sb.ap([[BxS * H1, H0], [H1, BxS], [1, H1]]),
            src=inp.ap([[H1, H0], [H0 * H1, BxS], [1, H1]]),
        )

        # Reshape gamma (1, H) to (H0=128, H1=H//H0)
        # Load gamma shape of (H0, H1) to (H0, H1)
        gamma = gamma.reshape((H0, H1))
        nisa.dma_copy(dst=gamma_sb[:H0, :H1], src=gamma[:H0, :H1])
    else:  # LNC 2
        # Reshape input (B, S, H) to (BxS, 2, H0=128, H2=H//H0//2)
        # Load input shape of (BxS, 2, H0, H2) to (H0, BxS, 2 * H2)
        inp = inp.reshape((B * S, 2, H0, H2))
        for i_2 in nl.affine_range(2):
            src_pattern = [[H2, H0], [2 * H0 * H2, BxS], [1, H2]]
            src_offset = (bs_lb * 2 * H0 * H2) + (i_2 * H0 * H2)

            dst_pattern = [[BxS * 2 * H2, H0], [2 * H2, BxS], [1, H2]]
            dst_offset = i_2 * H2

            nisa.dma_copy(
                src=inp.ap(pattern=src_pattern, offset=src_offset),
                dst=input_sb.ap(pattern=dst_pattern, offset=dst_offset),
            )

        # Reshape gamma (1, H) to (2, H0=128, H2=H//H0//2)
        # Load gamma shape of (2, H0, H2) to (H0, 2 * H2)
        gamma = gamma.reshape((2, H0, H2))
        for i_2 in nl.affine_range(2):
            src_pattern = [[H2, H0], [1, H2]]
            src_offset = i_2 * H0 * H2

            dst_pattern = [[2 * H2, H0], [1, H2]]
            dst_offset = i_2 * H2

            nisa.dma_copy(
                src=gamma.ap(pattern=src_pattern, offset=src_offset),
                dst=gamma_sb.ap(pattern=dst_pattern, offset=dst_offset),
            )

    square = alloc_tensor(shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_square")
    num_allocated_tensor += 1
    nisa.activation(square[...], op=nl.square, data=input_sb[...])

    reduced = alloc_tensor(shape=(H0, BxS), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_reduced")
    num_allocated_tensor += 1
    nisa.tensor_reduce(reduced[...], nl.add, square[...], axis=1)

    # input sb * gamma
    nisa.tensor_tensor(
        input_sb.ap([[H1 * BxS, H0], [1, 1], [1, H1 * BxS]]),
        input_sb.ap([[H1 * BxS, H0], [1, 1], [1, H1 * BxS]]),
        gamma_sb.ap([[H1, H0], [0, BxS], [1, H1]]),
        nl.multiply,
    )

    rmsnorm_reduction_const = alloc_tensor(
        shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_reduced_const"
    )
    num_allocated_tensor += 1
    nisa.memset(rmsnorm_reduction_const, value=1.0)
    if is_auto_alloc:
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum)
    else:  # psum bank = 0
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
    nisa.nc_matmul(stationary=rmsnorm_reduction_const, moving=reduced, dst=final_reduced)

    hidden_scale = 1.0 / hidden_actual
    eps_bias = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="eps_bias")
    num_allocated_tensor += 1
    nisa.memset(eps_bias, value=eps)
    # Due to the lack of live variable analysis we are reusing reduce as a sqrt tensor
    nisa.activation(
        reduced[...],
        op=nl.rsqrt,
        data=final_reduced[...],
        scale=hidden_scale,
        bias=eps_bias,
    )

    nisa.tensor_tensor(
        result[0:H0, nl.ds(bs_lb, bs_count), 0:H1],
        input_sb.ap([[H1 * BxS, H0], [1, 1], [1, H1 * BxS]]),
        reduced.ap([[BxS, H0], [1, BxS], [0, H1]]),
        nl.multiply,
    )

    # deallocate heap memory
    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()

    # close sbm scope
    sbm.close_scope()
