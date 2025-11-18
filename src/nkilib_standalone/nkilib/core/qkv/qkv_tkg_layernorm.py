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
LayerNorm TKG Kernel

This kernel implements the Layer Normalization (LayerNorm) operation commonly used
in transformer models. The kernel is specifically optimized for Token Generation
(TKG, also known as Decoding) scenarios where batch_size * seqlen is small.

This kernel is designed with LNC support. When LNC=2 and B*S exceeds the sharding
threshold, the computation is sharded across cores.

"""

from typing import Optional

import nki.language as nl
import nki.isa as nisa
from nki.stdlib import mgrid, mgrid_wrap

from ..utils.allocator import SbufManager, create_auto_alloc_manager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import Logger


# This is a heuristic to decide whether to shard on B*S to halve the computation
# at the cost of extra local collective
SHARDING_THRESHOLD = 10


def layernorm_tkg(
    inp: nl.ndarray,
    gamma: nl.ndarray,
    beta: Optional[nl.ndarray] = None,
    eps: float = 1e-6,
    output_in_sbuf: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
) -> nl.ndarray:
    """
    LayerNorm Kernel for Token Generation

    This kernel computes the LayerNorm operation:
        output = (inp - mean(inp)) * gamma / sqrt(var(inp) + eps) + beta
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
        beta (nl.ndarray, optional):
            Beta bias tensor in HBM used for normalization shift.
            Shape:    [1, H]
            Default: None (no bias applied).
        eps (float):
            Epsilon value to maintain numerical stability. Default: 1e-6.
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

    if not sbm:
        sbm = create_auto_alloc_manager()

    # open sbm scope
    sbm.open_scope()

    # Allocate output buffers
    if not output_in_sbuf:
        output = nl.ndarray((H0, B * S, H1), dtype=inp.dtype, buffer=nl.shared_hbm)

    if use_heap_memory:
        sharded_sbuf_result = sbm.alloc_heap(
            (H0, B * S, H1),
            dtype=inp.dtype,
            buffer=nl.sbuf,
            name="layernorm_shared_sbuf",
        )
    else:
        sharded_sbuf_result = sbm.alloc_stack(
            (H0, B * S, H1),
            dtype=inp.dtype,
            buffer=nl.sbuf,
            name="layernorm_shared_sbuf",
        )

    # Check program dimensionality
    _, lnc, shard_id = get_verified_program_sharding_info("layernorm_tkg", (0, 1))

    # We don't always do LNC2 sharding if BxS is less than threshold.
    num_shards = lnc
    do_shard = num_shards == 2 and (B * S) > SHARDING_THRESHOLD and (B * S) % lnc == 0
    if not do_shard:
        num_shards, shard_id = 1, 0  # Otherwise don't shard the compute.

    shard_size = (B * S) // num_shards

    is_beta = True if beta != None else False

    # Process LayerNorm
    _layernorm_tkg_llama_impl(
        inp=inp,
        gamma=gamma,
        beta=beta,
        result=sharded_sbuf_result,
        bs_lb=shard_id * shard_size,
        bs_count=shard_size,
        is_beta=is_beta,
        lnc=lnc,
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


def _layernorm_tkg_llama_impl(
    inp: nl.ndarray,
    gamma: nl.ndarray,
    beta: Optional[nl.ndarray],
    result: nl.ndarray,
    bs_lb: int,
    bs_count: int,
    is_beta: bool,
    lnc: int,
    eps: float,
    use_heap_memory: bool,
    sbm: SbufManager,
):
    """
    Internal implementation of LayerNorm computation.

    This function handles the core LayerNorm computation with support for LNC sharding.

    Layout Handling:
        - LNC 1: Input [B, S, H] is loaded from HBM to [H0, BxS, H1] SBUF
        - LNC 2: Input [B, S, H] is loaded from HBM to [H0, bs_lb:bs_lb+bs_count, H1] SBUF

    Args:
        inp (nl.ndarray): Input tensor of shape [B, S, H], this tensor is in HBM.
                          H must be divisible by nl.tile_size.pmax (128).
                          B*S*(H//128) must fit in SBUF.
        gamma (nl.ndarray): Gamma weight tensor of shape [1, H], this tensor is in HBM.
        beta (nl.ndarray): Beta bias tensor of shape [1, H], this tensor is in HBM.
                           Can be None if is_beta is False.
        result (nl.ndarray): Result SBUF tensor of shape (H0, B*S, H1) to write the result to.
        bs_lb (int): Inclusive lower bound of where to start processing on B*S dimension.
        bs_count (int): Number of batches to process on B*S dimension.
        is_beta (bool): Whether beta bias is provided and should be applied.
        lnc (int): Logical neuron core count, determines output sharding layout.
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
    input_sb = alloc_tensor(shape=(H0, BxS, H1), dtype=inp.dtype, name="layernorm_input")
    num_allocated_tensor += 1
    gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, name="layernorm_gamma")
    num_allocated_tensor += 1
    if is_beta:
        beta_sb = alloc_tensor(shape=(H0, H1), dtype=beta.dtype, name="layernorm_beta")
        num_allocated_tensor += 1

    #####################################################################
    ############### Load input, gamma and optionally beta ###############
    #####################################################################
    if lnc == 1:  # LNC 1
        # Reshape input (B, S, H) to (BxS, H0=128, H1=H//H0)
        # Load input shape of (BxS, H0, H1) to (H0, BxS, H1)
        inp = inp.reshape((BxS, H0, H1))
        iH0, iBS, iH1 = mgrid[0:H0, 0:BxS, 0:H1]
        nisa.dma_copy(dst=mgrid_wrap(input_sb)[iH0, iBS, iH1], src=mgrid_wrap(inp)[iBS, iH0, iH1])

        # Reshape gamma (1, H) to (H0=128, H1=H//H0)
        # Load gamma shape of (H0, H1) to (H0, H1)
        gamma = gamma.reshape((H0, H1))
        nisa.dma_copy(dst=gamma_sb[:H0, :H1], src=gamma[:H0, :H1])
        # Reshape beta (1, H) to (H0=128, H1=H//H0)
        # Load beta shape of (H0, H1) to (H0, H1)
        if is_beta:
            beta = beta.reshape((H0, H1))
            nisa.dma_copy(dst=beta_sb[:H0, :H1], src=beta[:H0, :H1])
    else:  # LNC 2
        # Reshape input (B, S, H) to (BxS, 2, H0=128, H2=H//H0//2)
        # Load input shape of (BxS, 2, H0, H2) to (H0, BxS, 2 * H2)
        inp = inp.reshape((B * S, 2, H0, H2))
        iH0, iBS, iH2 = mgrid[0:H0, 0:BxS, 0:H2]
        for i_2 in nl.affine_range(2):
            src_mgrid_access = mgrid_wrap(inp)[(bs_lb + iBS), i_2, iH0, iH2]
            dst_mgrid_access = mgrid_wrap(input_sb.reshape((H0, BxS, 2, H2)))[iH0, iBS, i_2, iH2]
            nisa.dma_copy(dst=dst_mgrid_access, src=src_mgrid_access)
        # Reshape gamma (1, H) to (2, H0=128, H2=H//H0//2)
        # Load gamma shape of (2, H0, H2) to (H0, 2 * H2)
        gamma = gamma.reshape((2, H0, H2))
        iH0, iH2 = mgrid[0:H0, 0:H2]
        for i_2 in nl.affine_range(2):
            src_mgrid_access = mgrid_wrap(gamma)[i_2, iH0, iH2]
            dst_mgrid_access = mgrid_wrap(gamma_sb.reshape((H0, 2, H2)))[iH0, i_2, iH2]
            nisa.dma_copy(dst=dst_mgrid_access, src=src_mgrid_access)
        # Reshape beta (1, H) to (2, H0=128, H2=H//H0//2)
        # Load beta shape of (2, H0, H2) to (H0, 2 * H2)
        if is_beta:
            beta = beta.reshape((2, H0, H2))
            iH0, iH2 = mgrid[0:H0, 0:H2]
            for i_2 in nl.affine_range(2):
                src_mgrid_access = mgrid_wrap(beta)[i_2, iH0, iH2]
                dst_mgrid_access = mgrid_wrap(beta_sb.reshape((H0, 2, H2)))[iH0, i_2, iH2]
                nisa.dma_copy(dst=dst_mgrid_access, src=src_mgrid_access)

    #####################################################################
    ### Compute the mean and variance of the input tensor along H dim ###
    #####################################################################

    # shared params
    reduction_const = alloc_tensor(shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf, name="reduction_const")
    num_allocated_tensor += 1
    nisa.memset(dst=reduction_const, value=(1.0 / H))

    # calculate mean(input_sb^2)
    input_squared_sb = alloc_tensor(shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf, name="input_squared")
    num_allocated_tensor += 1
    zero_bias = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="zero_bias")
    num_allocated_tensor += 1
    nisa.memset(zero_bias, value=0.0)
    nisa.activation(dst=input_squared_sb[...], op=nl.square, data=input_sb[...], bias=zero_bias)
    reduced_input_squared_sb = alloc_tensor(
        shape=(H0, BxS), dtype=inter_dtype, buffer=nl.sbuf, name="reduced_input_squared"
    )
    num_allocated_tensor += 1
    nisa.tensor_reduce(dst=reduced_input_squared_sb[...], op=nl.add, data=input_squared_sb[...], axis=1)
    if is_auto_alloc:
        input_squared_mean = nl.ndarray((H0, BxS), dtype=inter_dtype, buffer=nl.psum)
    else:  # psum bank = 0
        input_squared_mean = nl.ndarray((H0, BxS), dtype=inter_dtype, buffer=nl.psum, address=(0, 0))
    nisa.nc_matmul(
        dst=input_squared_mean,
        stationary=reduction_const,
        moving=reduced_input_squared_sb,
    )

    # calculate mean(input_sb)
    reduced_input_sb = alloc_tensor(shape=(H0, BxS), dtype=inter_dtype, buffer=nl.sbuf, name="reduced_input")
    num_allocated_tensor += 1
    nisa.tensor_reduce(dst=reduced_input_sb[...], op=nl.add, data=input_sb[...], axis=1)
    if is_auto_alloc:
        input_mean = nl.ndarray((H0, BxS), dtype=inter_dtype, buffer=nl.psum)
    else:  # psum bank = 1
        input_mean = nl.ndarray((H0, BxS), dtype=inter_dtype, buffer=nl.psum, address=(0, 1 * 512 * 4))
    nisa.nc_matmul(dst=input_mean, stationary=reduction_const, moving=reduced_input_sb)

    # calculate input_sb - mean
    iH0, iBS, iH1 = mgrid[0:H0, 0:BxS, 0:H1]
    nisa.tensor_tensor(
        mgrid_wrap(input_sb)[iH0, iBS, iH1],
        mgrid_wrap(input_sb)[iH0, iBS, iH1],
        mgrid_wrap(input_mean)[iH0, iBS],
        nl.subtract,
    )

    # calculate mean(input_sb)^2
    squared_input_mean = alloc_tensor(shape=(H0, BxS), dtype=inter_dtype, buffer=nl.sbuf, name="squared_input_mean")
    num_allocated_tensor += 1
    nisa.activation(dst=squared_input_mean[...], op=nl.square, data=input_mean[...], bias=zero_bias)

    # calculate var
    var = alloc_tensor(shape=(H0, BxS), dtype=inter_dtype, buffer=nl.sbuf, name="var")
    num_allocated_tensor += 1
    nisa.tensor_tensor(var[...], input_squared_mean[...], squared_input_mean[...], nl.subtract)

    #####################################################################
    ############### Apply the mean, variance, gamma, beta ###############
    #####################################################################

    # calculate rsqrt (var) + eps
    rsqrt = alloc_tensor(shape=(H0, BxS), dtype=inter_dtype, buffer=nl.sbuf, name="rsqrt")
    num_allocated_tensor += 1
    eps_bias = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="eps_bias")
    num_allocated_tensor += 1
    nisa.memset(eps_bias, value=eps)
    nisa.activation(dst=rsqrt[...], op=nl.rsqrt, data=var[...], bias=eps_bias)

    # calculate (input_sb - mean) * 1/sqrt
    nisa.tensor_tensor(
        mgrid_wrap(input_sb)[iH0, iBS, iH1],
        mgrid_wrap(input_sb)[iH0, iBS, iH1],
        mgrid_wrap(rsqrt)[iH0, iBS],
        nl.multiply,
    )

    # apply gamma
    nisa.tensor_tensor(
        mgrid_wrap(input_sb)[iH0, iBS, iH1],
        mgrid_wrap(input_sb)[iH0, iBS, iH1],
        mgrid_wrap(gamma_sb)[iH0, iH1],
        nl.multiply,
    )

    # apply beta if present
    if is_beta:
        nisa.tensor_tensor(
            mgrid_wrap(input_sb)[iH0, iBS, iH1],
            mgrid_wrap(input_sb)[iH0, iBS, iH1],
            mgrid_wrap(beta_sb)[iH0, iH1],
            nl.add,
        )

    # Copy result to output
    nisa.tensor_copy(dst=result[0:H0, nl.ds(bs_lb, bs_count), 0:H1], src=input_sb[0:H0, 0:BxS, 0:H1])

    # deallocate heap memory
    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()

    # close sbm scope
    sbm.close_scope()
