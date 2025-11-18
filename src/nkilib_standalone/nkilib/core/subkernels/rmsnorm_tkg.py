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


import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import SbufManager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import Logger
from ..utils.tensor_view import TensorView

# This is a heuristic to decide whether to shard on BxS to halve the computation
# at the cost of extra local collective
SHARDING_THRESHOLD = 18


def rmsnorm_tkg_llama_impl(
    inp,
    gamma,
    result,
    bs_lb: int,
    bs_count: int,
    lnc: int,
    hidden_actual: int,
    eps: float,
    use_heap_memory: bool,
    sbm: SbufManager,
):
    """
    Perform RMSNorm on input tensor.

    The inp is of shape [B, S, H].
    H0 = nl.tile_size.pmax (128)
    H1 = H // H0
    For LNC, inp is split up to [B, S, lnc, H//lnc], and reshaped to [BxS, lnc, H0, H1//lnc].
    After inp is transposed to [H0, BxS, lnc, H1//lnc], and reshaped back to [H0, BxS, H1].
    Then perform RMSNorm on the combination of the [H0, lnc, H1//lnc] dimension.

    Please note that this kernel utilizes Static DMA for input data reads.
    Experimental results indicate that Static DMA offers superior performance.
    We may revert to DGE in the event of HBM out-of-memory (OOM) issues.

    :param inp ndarray: Tensor to perform RMSNorm on, which has shape [B, S, H].
        H must be divisible by nl.tile_size.pmax(128). BxS*(H//128) must fit in SBUF.
    :param gamma ndarray: Gamma to apply on the rmsnorm, which has shape [1, H].
    :param result ndarray: result SBUF tensor of shape (H0, BxS, H1) to write the result to
    :param bs_lb int: the inclusive lower bound of where to start to process on BxS dimension
    :param bs_count int: number of batches to process on BxS
    :param lnc int: output sharding layout
    :param eps float: epsilon to maintain numerical stability.
    :param use_heap_memory bool: Indicates whether to allocate memory on the heap instead of the stack.
    :param sbm SbufManager: Instance of SbufManager responsible for handling sbuf allocation
    :return: The tensor with rmsnorm performed.

    """

    # Hardware partition dim constraint
    H0 = nl.tile_size.pmax

    # check if input tensor is in sbuf
    input_in_sbuf = inp.buffer == nl.sbuf

    # Extract input dimensions: Batch, Sequence, Hidden
    if input_in_sbuf:
        _H0, full_BxS, H1 = inp.shape
        kernel_assert(
            _H0 == H0,
            f"inp tensor in SBUF does not have partition dimension H0 of {H0}, got {_H0}",
        )
        H = _H0 * H1
    else:
        B, S, H = inp.shape
        full_BxS = B * S
        kernel_assert(H % H0 == 0, f"inp tensor H dimension must be divisible by {H0}, got {H}")
        H1 = H // H0

    BxS = bs_count
    H2 = H1 // lnc  # sharded partition size for LNC sharding

    # All intermediates need to happen in FP32 for numerical precision
    inter_dtype = nl.float32

    # Gamma shape check
    kernel_assert(
        gamma.shape == (1, H),
        f"Malformed shape of gamma expected (1, {H}), got {gamma.shape}",
    )

    # Open SBUF memory scope
    sbm.open_scope()

    # Check if the kernel uses auto or manual allocation
    is_auto_alloc = sbm.is_auto_alloc()

    # Define allocation function: heap or stack
    if use_heap_memory:
        alloc_tensor = sbm.alloc_heap
    else:
        alloc_tensor = sbm.alloc_stack

    # Track number of allocated tensors for cleanup
    num_allocated_tensor = 0

    # Allocate SBUF tensor for input data
    if input_in_sbuf:
        input_sb = inp
    elif bs_count == full_BxS:  # If processing full batch, reuse result buffer to save memory
        input_sb = result
    else:
        input_sb = alloc_tensor(shape=(H0, bs_count, H1), dtype=inp.dtype, name="rmsnorm_input")
        num_allocated_tensor += 1

    # Allocate SBUF tensor for gamma tensor
    gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, name="rmsnorm_gamma")
    num_allocated_tensor += 1

    # Load input, gamma and beta tensors from HBM to SBUF
    if not input_in_sbuf:
        # Transform input: (B,S,H) -> (B,S,lnc,H0,H2) -> (BxS,lnc,H0,H2) -> (BxS,lnc,H0,H2) -> (H0,BxS,lnc,H2)
        input_view = (
            TensorView(inp)
            .reshape_dim(dim=2, sizes=[lnc, H0, H2])
            .flatten_dims(start_dim=0, end_dim=1)
            .slice(dim=0, start=bs_lb, end=bs_lb + BxS)
            .permute(dims=[2, 0, 1, 3])
        )
        # input_sb (H0,BxS,H1) -> (H0,BxS,lnc,H2)
        input_sb_view = TensorView(input_sb).reshape_dim(dim=2, sizes=[lnc, H2])
        nisa.dma_copy(
            dst=input_sb_view.get_view(),
            src=input_view.get_view(),
            dge_mode=nisa.dge_mode.none,
        )
    elif input_in_sbuf and bs_count != full_BxS:
        # If input is already in SBUF and not processing full batch, slice the input
        input_sb_view = TensorView(input_sb).slice(dim=1, start=bs_lb, end=bs_lb + BxS)
    else:
        # If input is already in SBUF and processing full batch
        input_sb_view = TensorView(input_sb)

    # Transform gamma for sharded layout: (1,H) -> (1,lnc,H0,H2) -> (lnc,H0,H2) -> (H0,lnc,H2)
    gamma_view = (
        TensorView(gamma)
        .reshape_dim(dim=1, sizes=[lnc, H0, H2])
        .flatten_dims(start_dim=0, end_dim=1)
        .permute(dims=[1, 0, 2])
    )
    # Reshape gamma buffer to match sharded layout
    gamma_sb_view = TensorView(gamma_sb).reshape_dim(dim=1, sizes=[lnc, H2])
    nisa.dma_copy(
        dst=gamma_sb_view.get_view(),
        src=gamma_view.get_view(),
        dge_mode=nisa.dge_mode.none,
    )

    # Step 1: Compute element-wise squares for RMS calculation
    # RMSNorm formula: x / sqrt(mean(x^2) + eps) * gamma
    rmsnorm_square = alloc_tensor(shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_square")
    num_allocated_tensor += 1
    nisa.activation(rmsnorm_square[...], op=nl.square, data=input_sb_view.get_view())

    # Step 2: Reduce squares along H1 dimension to compute mean(x^2)
    rmsnorm_reduced_square = alloc_tensor(
        shape=(H0, BxS),
        dtype=inter_dtype,
        buffer=nl.sbuf,
        name="rmsnorm_reduced_square",
    )
    num_allocated_tensor += 1
    nisa.tensor_reduce(rmsnorm_reduced_square[...], nl.add, rmsnorm_square[...], axis=1)

    # Step 3: Apply gamma scaling to input (part of RMSNorm formula)
    # Broadcast gamma from (H0, H1) to (H0, BxS, H1) for element-wise multiplication
    gamma_sb_view = TensorView(gamma_sb).expand_dim(dim=1).broadcast(dim=1, size=BxS)
    nisa.tensor_tensor(
        input_sb_view.get_view(),
        input_sb_view.get_view(),
        gamma_sb_view.get_view(),
        nl.multiply,
    )

    # Step 4: Perform matrix multiplication to complete the reduction
    # Use constant matrix of 1s to sum across the H0 dimension
    rmsnorm_reduction_const = alloc_tensor(
        shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_reduced_const"
    )
    num_allocated_tensor += 1
    nisa.memset(rmsnorm_reduction_const, value=1.0)

    # Allocate result buffer in PSUM for matrix multiplication output
    if is_auto_alloc:
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum)
    else:  # Manual allocation: use PSUM bank 0
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
    nisa.nc_matmul(
        stationary=rmsnorm_reduction_const,
        moving=rmsnorm_reduced_square,
        dst=final_reduced,
    )

    # Step 5: Compute normalization factor: 1/sqrt(mean(x^2) + eps)
    hidden_scale = 1.0 / hidden_actual  # Scale factor for mean calculation
    eps_bias = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="eps_bias")
    num_allocated_tensor += 1
    nisa.memset(eps_bias, value=eps)  # Epsilon for numerical stability

    # Compute reciprocal square root: rsqrt(mean(x^2) + eps)
    # Note: Reusing rmsnorm_reduced_square variable due to lack of live variable analysis
    nisa.activation(
        rmsnorm_reduced_square[...],
        op=nl.rsqrt,
        data=final_reduced[...],
        scale=hidden_scale,
        bias=eps_bias,
    )

    # Step 6: Final RMSNorm computation: (input * gamma) * rsqrt_factor
    # Broadcast normalization factor from (H0, BxS) to (H0, BxS, H1)
    reduced_view = TensorView(rmsnorm_reduced_square).expand_dim(dim=2).broadcast(dim=2, size=H1)
    nisa.tensor_tensor(
        result[0:H0, nl.ds(bs_lb, bs_count), 0:H1],
        input_sb_view.get_view(),
        reduced_view.get_view(),
        nl.multiply,
    )

    # Cleanup: deallocate heap memory if used
    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()

    # Close SBUF memory scope
    sbm.close_scope()


def rmsnorm_tkg(
    inp,
    gamma,
    eps: float = 1e-6,
    hidden_actual: int = None,
    output_in_sbuf: bool = False,
    use_heap_memory: bool = False,
    sbm: SbufManager = None,
):
    """
    RMSNorm implementation optimized for inference token generation (decoding) phase.

    The output layout is specifically choosen to make the subsquent sharded matmul efficient.

    Mathematically speaking, the result looks like the following,

    result = norm_name2func[NormType.RMS_NORM](hidden, gamma, eps)
    result = result.reshape((BxS, -1))
    t0 = result[:, 0:H//2]
    t1 = result[:, H//2:]
    t0 = t0.reshape((BxS, H0, H1//2)).transpose((1, 0, 2))
    t1 = t1.reshape((BxS, H0, H1//2)).transpose((1, 0, 2))
    result = np.concatenate([t0, t1], axis=2)

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size

    Args:
        inp (nl.ndarray): input tensor of shape [B, S, H]
        gamma (nl.ndarray): gamma tensor of shape [1, H] used in normallization, this tensor is in HBM.
        output (nl.ndarray): output tensor. The tensor can reside in either SBUF or HBM.
        eps (float): epsilon to maintain numerical stability.
        hidden_scale (float): 1/H. if specified, use this value to calculate mean; otherwise, use hidden parsed from inp.
                      This is to handle the case where hidden dimension in input tensor is padded.
        output_in_sbuf (bool): Indicate whether the output buffer is stored in HBM or kept in sbuf.
        use_heap_memory(bool): Indicates whether to allocate memory on the heap instead of the stack.
        sbm (SbufManager): Instance of SbufManager responsible for handling sbuf allocation

    Returns:
        output (nl.ndarray): output tensor of shape [128, B * S, H/128].

    """

    # Hardware partition dim constraint
    H0 = nl.tile_size.pmax

    # check if input tensor is in sbuf
    input_in_sbuf = inp.buffer == nl.sbuf

    # Extract input tensor dimensions
    if input_in_sbuf:
        _H0, BxS, H1 = inp.shape
        kernel_assert(
            _H0 == H0,
            f"inp tensor in SBUF does not have partition dimension H0 of 128, got {_H0}",
        )
        H = _H0 * H1
    else:
        B, S, H = inp.shape
        BxS = B * S
        kernel_assert(H % H0 == 0, f"inp tensor H dimension must be divisible by {H0}, got {H}")
        H1 = H // H0

    # Use actual hidden size if provided
    if not hidden_actual:
        hidden_actual = H

    # Initialize SBUF manager if not provided
    if not sbm:
        # Calculate required SBUF size: 16*BxS*H1 for intermediates + H0 for constants
        # Factor of 4 accounts for float32 byte size
        sbm = SbufManager(0, (16 * BxS * H1 + H0) * 4, Logger("rmsnorm_tkg"), use_auto_alloc=True)

    # Open SBUF memory scope
    sbm.open_scope()

    # Allocate output buffer in HBM if result should not stay in SBUF
    if not output_in_sbuf:
        output = nl.ndarray((H0, BxS, H1), dtype=inp.dtype, buffer=nl.shared_hbm)

    # Allocate intermediate result buffer in SBUF for computation
    if use_heap_memory:
        sharded_sbuf_result = sbm.alloc_heap((H0, BxS, H1), dtype=inp.dtype, buffer=nl.sbuf, name="rmsnorm_shared_sbuf")
    else:
        sharded_sbuf_result = sbm.alloc_stack(
            (H0, BxS, H1), dtype=inp.dtype, buffer=nl.sbuf, name="rmsnorm_shared_sbuf"
        )

    # Determine sharding configuration for parallel processing
    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    # Apply sharding only if beneficial: LNC2 + sufficient batch size + divisible
    num_shards = lnc
    do_shard = num_shards == 2 and BxS > SHARDING_THRESHOLD and BxS % lnc == 0
    if not do_shard:
        num_shards, shard_id = 1, 0  # Fall back to single core processing

    # Calculate work distribution per shard
    shard_size = BxS // num_shards

    # Execute RMSNorm computation on assigned shard
    rmsnorm_tkg_llama_impl(
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

    # Handle output based on requested buffer location
    if output_in_sbuf:
        # If sharded, exchange results between cores to get complete output
        if do_shard:
            nisa.sendrecv(
                dst=sharded_sbuf_result[:, nl.ds((1 - shard_id) * shard_size, shard_size), :],
                src=sharded_sbuf_result[:, nl.ds(shard_id * shard_size, shard_size), :],
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
            )

        return sharded_sbuf_result.reshape((H0, BxS, H1))

    # Copy results from SBUF to HBM
    sharded_sbuf_result = sharded_sbuf_result.reshape((H0, BxS, H1))
    output = output.reshape(sharded_sbuf_result.shape)

    # Copy only this shard's portion to HBM
    nisa.dma_copy(
        dst=output[:, nl.ds(shard_id * shard_size, shard_size), :],
        src=sharded_sbuf_result[:, nl.ds(shard_id * shard_size, shard_size), :],
    )

    # Cleanup: deallocate SBUF result buffer
    if use_heap_memory:
        sbm.pop_heap()

    # Close SBUF memory scope
    sbm.close_scope()

    return output
