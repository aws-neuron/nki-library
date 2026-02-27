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
This module implements memory-efficient cross entropy kernels for large vocabularies using the online log-sum-exp algorithm with batched processing.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .validation import validate_cross_entropy_backward_inputs, validate_cross_entropy_forward_inputs


@nki.jit(platform_target="trn2")
def cross_entropy_forward(
    logits_hbm: nl.ndarray,
    targets_hbm: nl.ndarray,
    positions_per_batch: int = 32,
    chunk_size: int = 32768,
    dtype: nki.dtype = nl.bfloat16,
) -> tuple[nl.ndarray, nl.ndarray]:
    """
    Cross entropy forward pass using online log-sum-exp algorithm with batching.

    This kernel computes cross entropy loss for large vocabularies using a memory-efficient
    online log-sum-exp algorithm. Optimized for LNC2 (2 cores) with batched processing where
    each core processes multiple positions in batches with vectorized operations.

    Dimensions:
        B: Batch size
        T: Sequence length per batch
        V: Vocabulary size
        num_positions: Total positions (B * T)
        positions_per_batch: Number of positions processed together
        chunk_size: Size of vocabulary chunks

    Args:
        logits_hbm (nl.ndarray): [num_positions, V], Input logits tensor in HBM.
            Supported dtypes: nl.bfloat16, nl.float32. MUST be 2D (already flattened).
        targets_hbm (nl.ndarray): [num_positions], Target indices tensor in HBM.
            dtype: nl.int32. MUST be 1D (already flattened).
        positions_per_batch (int): Number of positions to process together. Default: 32.
            Larger batches improve HBM bandwidth and SBUF utilization.
            Candidate values (powers of 2): 8, 16, 32, 64, 128.
            Must satisfy: positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB.
        chunk_size (int): Size of vocabulary chunks. Default: 32768 (32K).
            Must not exceed vocabulary size V or hardware limit (65535).
            Candidate values:
                65535 - F_MAX, ideal for 128K-256K vocabs (bf16 only)
                49152 - 3/4 of F_MAX
                40960 - Good balance
                32768 - Standard, good for 32K-128K vocabs
                16384 - Half of 32K
                8192  - Quarter of 32K
                4096  - Small vocab fallback
                2048  - Minimum practical
            Selection guide:
                - V ≤ 32K: Use chunk_size = V (single chunk)
                - 32K < V ≤ 128K: Use chunk_size = 32768 or 40960
                - 128K < V ≤ 256K: Use chunk_size = 65535 (bf16) or 32768 (fp32)
                - Always verify: positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB
        dtype (nki.dtype): Data type for internal computations. Default: nl.bfloat16.
            Supported types: nl.bfloat16 (2 bytes), nl.float32 (4 bytes).
            Controls precision of intermediate calculations and memory usage.

    Returns:
        loss_hbm (nl.ndarray): [num_positions], Cross entropy loss per position in HBM.
            dtype matches dtype parameter. Buffer: nl.shared_hbm (allocated internally).
        lse_state_hbm (nl.ndarray): [num_positions], Log-sum-exp values per position in HBM.
            dtype matches dtype parameter. Buffer: nl.shared_hbm (allocated internally).
            Saved for backward pass.

    Pseudocode:
        for each batch of positions:
            # Initialize state
            m = -inf  # max value seen so far
            d = 0     # sum of exponentials

            # Process vocabulary in chunks
            for each vocabulary chunk:
                # Load chunk for all positions in batch
                chunk = load_logits_chunk(positions, chunk_range)

                # Update running maximum
                m_new = max(m, max(chunk))

                # Correct previous sum with new maximum
                d = d * exp(m - m_new) + sum(exp(chunk - m_new))
                m = m_new

            # Compute log-sum-exp
            lse = m + log(d)

            # Compute loss
            target_logit = logits[position, target[position]]
            loss = lse - target_logit

            # Save results
            save(lse, loss)
    """

    validate_cross_entropy_forward_inputs(logits_hbm, targets_hbm, positions_per_batch, chunk_size)
    grid_ndim, num_cores, core_id = get_verified_program_sharding_info("cross_entropy_forward", (0, 1))

    num_positions = logits_hbm.shape[0]
    vocab_size = logits_hbm.shape[1]
    num_chunks = div_ceil(vocab_size, chunk_size)

    loss_hbm = nl.ndarray((num_positions,), dtype=dtype, buffer=nl.shared_hbm)
    lse_state_hbm = nl.ndarray((num_positions,), dtype=dtype, buffer=nl.shared_hbm)

    # Distribute positions contiguously across cores, last core handles remainder
    nominal_positions_per_core = num_positions // num_cores
    shard_offset = core_id * nominal_positions_per_core

    if core_id == num_cores - 1:
        positions_per_core = num_positions - shard_offset
    else:
        positions_per_core = nominal_positions_per_core

    # Create sharded views - logits_hbm kept unsharded for .ap() with scalar_offset
    position_idx = nl.ds(shard_offset, positions_per_core)
    targets_shard_hbm = targets_hbm[position_idx]
    loss_shard_hbm = loss_hbm[position_idx]
    lse_state_shard_hbm = lse_state_hbm[position_idx]

    num_batches = div_ceil(positions_per_core, positions_per_batch)

    batch_targets = nl.ndarray((positions_per_batch, 1), dtype=nl.int32, buffer=nl.sbuf)
    batch_m = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_d = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_chunk = nl.ndarray((positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf)
    batch_chunk_max = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_m_new = nl.ndarray((positions_per_batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    batch_m_diff = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_correction = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_d_corrected = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_exp_chunk = nl.ndarray((positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf)
    batch_sum_exp = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_d_new = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_log_d = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_lse = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_target_logits = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_loss = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)

    for batch_idx in range(num_batches):
        batch_start_idx = batch_idx * positions_per_batch
        actual_batch_size = min(positions_per_batch, positions_per_core - batch_start_idx)

        if actual_batch_size <= 0:
            break

        nisa.memset(dst=batch_m, value=-float('inf'), name=f"init_m_batch_{batch_idx}")
        nisa.memset(dst=batch_d, value=0.0, name=f"init_d_batch_{batch_idx}")

        nisa.dma_copy(
            dst=batch_targets[0:actual_batch_size],
            src=targets_shard_hbm[batch_start_idx : batch_start_idx + actual_batch_size],
            name=f"load_targets_batch_{batch_idx}",
        )

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, vocab_size)
            actual_chunk_len = chunk_end - chunk_start

            if actual_chunk_len < chunk_size:
                nisa.memset(dst=batch_chunk, value=-float('inf'), name=f"pad_batch_chunk_{batch_idx}_{chunk_idx}")

            nisa.dma_copy(
                dst=batch_chunk[0:actual_batch_size, 0:actual_chunk_len],
                src=logits_hbm[
                    shard_offset + batch_start_idx : shard_offset + batch_start_idx + actual_batch_size,
                    chunk_start:chunk_end,
                ],
                name=f"load_chunk_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.tensor_reduce(
                op=nl.maximum,
                data=batch_chunk,
                dst=batch_chunk_max,
                axis=1,
                name=f"max_chunk_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.tensor_tensor(
                dst=batch_m_new,
                data1=batch_m,
                data2=batch_chunk_max,
                op=nl.maximum,
                name=f"m_new_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.tensor_tensor(
                dst=batch_m_diff,
                data1=batch_m,
                data2=batch_m_new,
                op=nl.subtract,
                name=f"m_diff_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.activation(
                op=nl.exp,
                data=batch_m_diff,
                dst=batch_correction,
                name=f"correction_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.tensor_tensor(
                dst=batch_d_corrected,
                data1=batch_d,
                data2=batch_correction,
                op=nl.multiply,
                name=f"d_mult_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.tensor_scalar(
                dst=batch_exp_chunk,
                data=batch_chunk,
                op0=nl.subtract,
                operand0=batch_m_new,
                name=f"sub_chunk_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.activation(
                op=nl.exp,
                data=batch_exp_chunk,
                dst=batch_exp_chunk,
                name=f"exp_chunk_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.tensor_reduce(
                op=nl.add,
                data=batch_exp_chunk,
                dst=batch_sum_exp,
                axis=1,
                name=f"sum_exp_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.tensor_tensor(
                dst=batch_d_new,
                data1=batch_d_corrected,
                data2=batch_sum_exp,
                op=nl.add,
                name=f"d_add_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Update m and d for next iteration using tensor copy operations
            nisa.tensor_copy(src=batch_m_new, dst=batch_m, name=f"update_m_batch_{batch_idx}_chunk_{chunk_idx}")
            nisa.tensor_copy(src=batch_d_new, dst=batch_d, name=f"update_d_batch_{batch_idx}_chunk_{chunk_idx}")

        # Compute LSE: lse = m + log(d)
        nisa.activation(op=nl.log, data=batch_d, dst=batch_log_d, name=f"log_d_batch_{batch_idx}")

        nisa.tensor_tensor(dst=batch_lse, data1=batch_m, data2=batch_log_d, op=nl.add, name=f"lse_batch_{batch_idx}")

        # Compute Loss: loss = lse - logits[target]
        for position_idx in range(actual_batch_size):
            absolute_position = shard_offset + batch_start_idx + position_idx
            nisa.dma_copy(
                dst=batch_target_logits[position_idx : position_idx + 1, :],
                src=logits_hbm.ap(
                    pattern=[[vocab_size, 1], [1, 1]],
                    offset=absolute_position * vocab_size,
                    scalar_offset=batch_targets.ap(
                        pattern=[[1, 1], [1, 1]],
                        offset=position_idx,
                    ),
                    indirect_dim=1,
                ),
                name=f"load_target_logit_batch_{batch_idx}_pos_{position_idx}",
            )

        nisa.tensor_tensor(
            dst=batch_loss, data1=batch_lse, data2=batch_target_logits, op=nl.subtract, name=f"loss_batch_{batch_idx}"
        )

        nisa.dma_copy(
            dst=lse_state_shard_hbm[batch_start_idx : batch_start_idx + actual_batch_size],
            src=batch_lse[0:actual_batch_size, 0],
            name=f"store_lse_batch_{batch_idx}",
        )

        nisa.dma_copy(
            dst=loss_shard_hbm[batch_start_idx : batch_start_idx + actual_batch_size],
            src=batch_loss[0:actual_batch_size, 0],
            name=f"store_loss_batch_{batch_idx}",
        )

    return loss_hbm, lse_state_hbm


@nki.jit(platform_target="trn2")
def cross_entropy_backward(
    logits_hbm: nl.ndarray,
    targets_hbm: nl.ndarray,
    lse_state_hbm: nl.ndarray,
    reduction: str = "mean",
    positions_per_batch: int = 32,
    chunk_size: int = 32768,
    dtype: nki.dtype = nl.bfloat16,
    inplace: bool = True,
) -> nl.ndarray:
    """
    Cross entropy backward pass computing gradients with respect to logits.

    This kernel computes the gradient of cross entropy loss with respect to input logits
    using the formula: grad_logits[i, j] = grad_scale * (softmax(logits[i, j]) - 1{j == target[i]})
    where softmax is computed using the saved LSE state from the forward pass, and grad_scale
    is determined by the reduction parameter.

    Optimized for LNC2 (2 cores) with batched processing where each core processes multiple
    positions in batches with vectorized operations.

    TODO: Specify intended usage range (e.g., vocabulary size, sequence length, batch size).

    Dimensions:
        B: Batch size
        T: Sequence length per batch
        V: Vocabulary size
        num_positions: Total positions (B * T)
        positions_per_batch: Number of positions processed together
        chunk_size: Size of vocabulary chunks

    Args:
        logits_hbm (nl.ndarray): [num_positions, V], Input logits tensor in HBM.
            Supported dtypes: nl.bfloat16, nl.float32. MUST be 2D (already flattened).
            Same tensor used in forward pass.
        targets_hbm (nl.ndarray): [num_positions], Target indices tensor in HBM.
            dtype: nl.int32. MUST be 1D (already flattened).
            Same tensor used in forward pass.
        lse_state_hbm (nl.ndarray): [num_positions], Log-sum-exp values from forward pass in HBM.
            dtype matches dtype parameter. Saved state from cross_entropy_forward.
        reduction (str): How to scale gradients. Default: 'mean'.
            - 'mean': Scale by 1/num_positions (most common, matches PyTorch default)
            - 'sum': Scale by 1.0
        positions_per_batch (int): Number of positions to process together. Default: 32.
            Larger batches improve HBM bandwidth and SBUF utilization.
            Candidate values (powers of 2): 8, 16, 32, 64, 128.
            Recommended: 128 (P_MAX) for maximum throughput.
            Must satisfy: positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB.
        chunk_size (int): Size of vocabulary chunks. Default: 32768.
            SBUF per-partition limit: chunk_size × dtype_bytes × 2 ≤ 229,376.
            Recommended: 32768 for bf16, 16384 for fp32.
        dtype (nki.dtype): Data type for internal computations. Default: nl.bfloat16.
            Supported types: nl.bfloat16 (2 bytes), nl.float32 (4 bytes).
            Controls precision of intermediate calculations and memory usage.
        inplace (bool): If True, write gradients directly over logits_hbm to save HBM memory.
            Default: True. When True, logits_hbm is overwritten and cannot be used after.
            Saves num_positions × vocab_size × dtype_bytes of HBM memory.

    Returns:
        grad_logits_hbm (nl.ndarray): [num_positions, V], Gradient with respect to logits in HBM.
            dtype matches dtype parameter. If inplace=True, this is the same tensor as logits_hbm.
            Buffer: nl.shared_hbm (allocated internally if inplace=False).

    Pseudocode:
        # Determine gradient scaling
        if reduction == 'mean':
            grad_scale = 1.0 / num_positions
        elif reduction == 'sum':
            grad_scale = 1.0

        for each batch of positions:
            # Load saved state from forward pass
            lse = load_lse_state(positions)
            targets = load_targets(positions)

            # Process vocabulary in chunks
            for each vocabulary chunk:
                # Load logits chunk for all positions in batch
                chunk = load_logits_chunk(positions, chunk_range)

                # Compute softmax using saved LSE: softmax = exp(logits - lse)
                softmax_chunk = exp(chunk - lse)

                # Multiply by gradient scale
                grad_chunk = grad_scale * softmax_chunk

                # Subtract gradient at target position
                for each position in batch:
                    if target[position] in chunk_range:
                        grad_chunk[position, target[position] - chunk_start] -= grad_scale

                # Store gradient chunk
                save_grad_chunk(grad_chunk)
    """
    num_positions = logits_hbm.shape[0]
    vocab_size = logits_hbm.shape[1]
    num_chunks = div_ceil(vocab_size, chunk_size)

    # Validate all inputs (including reduction parameter)
    validate_cross_entropy_backward_inputs(
        logits_hbm, targets_hbm, lse_state_hbm, positions_per_batch, chunk_size, reduction=reduction
    )

    # Determine gradient scale based on reduction
    if reduction == "mean":
        grad_scale = 1.0 / num_positions
    else:  # reduction == "sum"
        grad_scale = 1.0

    grid_ndim, num_cores, core_id = get_verified_program_sharding_info("cross_entropy_backward", (0, 1))

    # Use inplace mode to save HBM memory by writing gradients over logits
    if inplace:
        grad_logits_hbm = logits_hbm
    else:
        grad_logits_hbm = nl.ndarray((num_positions, vocab_size), dtype=dtype, buffer=nl.shared_hbm)

    # Distribute positions contiguously across cores, last core handles remainder
    nominal_positions_per_core = num_positions // num_cores
    shard_offset = core_id * nominal_positions_per_core

    if core_id == num_cores - 1:
        positions_per_core = num_positions - shard_offset
    else:
        positions_per_core = nominal_positions_per_core

    # Create sharded views - logits_hbm kept unsharded for .ap() with scalar_offset
    position_idx = nl.ds(shard_offset, positions_per_core)
    targets_shard_hbm = targets_hbm[position_idx]
    lse_state_shard_hbm = lse_state_hbm[position_idx]
    grad_logits_shard_hbm = grad_logits_hbm[position_idx, :]

    num_batches = div_ceil(positions_per_core, positions_per_batch)

    # SBUF allocations for batch processing
    batch_targets = nl.ndarray((positions_per_batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    batch_lse = nl.ndarray((positions_per_batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    batch_logits_chunk = nl.ndarray((positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf)
    batch_softmax_chunk = nl.ndarray((positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf)
    batch_grad_chunk = nl.ndarray((positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf)

    # SBUF allocation for mask-based target correction
    # Use fp32 for mask computation to avoid precision loss with large vocab indices (>256 in bf16)
    # This buffer is reused: first holds chunk indices, then holds the mask after comparison
    batch_chunk_indices_and_mask = nl.ndarray((positions_per_batch, chunk_size), dtype=nl.float32, buffer=nl.sbuf)
    batch_correction = nl.ndarray((positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf)

    for batch_idx in range(num_batches):
        batch_start_idx = batch_idx * positions_per_batch
        actual_batch_size = min(positions_per_batch, positions_per_core - batch_start_idx)

        if actual_batch_size <= 0:
            break

        nisa.dma_copy(
            dst=batch_targets[0:actual_batch_size],
            src=targets_shard_hbm[batch_start_idx : batch_start_idx + actual_batch_size],
            name=f"load_targets_batch_{batch_idx}",
        )

        nisa.dma_copy(
            dst=batch_lse[0:actual_batch_size],
            src=lse_state_shard_hbm[batch_start_idx : batch_start_idx + actual_batch_size],
            name=f"load_lse_batch_{batch_idx}",
        )

        # Process vocabulary in chunks
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, vocab_size)
            actual_chunk_len = chunk_end - chunk_start

            # Initialize chunk with -inf for padding (exp(-inf) = 0, so padded values contribute zero gradient)
            if actual_chunk_len < chunk_size:
                nisa.memset(
                    dst=batch_logits_chunk, value=-float('inf'), name=f"pad_logits_chunk_batch_{batch_idx}_{chunk_idx}"
                )

            # Load logits chunk
            nisa.dma_copy(
                dst=batch_logits_chunk[0:actual_batch_size, 0:actual_chunk_len],
                src=logits_hbm[
                    shard_offset + batch_start_idx : shard_offset + batch_start_idx + actual_batch_size,
                    chunk_start:chunk_end,
                ],
                name=f"load_logits_chunk_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Compute softmax: exp(logits - lse)
            nisa.tensor_scalar(
                dst=batch_softmax_chunk,
                data=batch_logits_chunk,
                op0=nl.subtract,
                operand0=batch_lse,
                name=f"sub_lse_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            nisa.activation(
                op=nl.exp,
                data=batch_softmax_chunk,
                dst=batch_softmax_chunk,
                name=f"exp_softmax_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Multiply by gradient scale to get initial gradient
            nisa.tensor_scalar(
                dst=batch_grad_chunk,
                data=batch_softmax_chunk,
                op0=nl.multiply,
                operand0=grad_scale,
                name=f"mult_grad_scale_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Create a mask to identify where each position's target is in this chunk
            # Create vocab indices replicated across all positions
            # iota generates: value[p, f] = offset + f * step + p * channel_multiplier
            # With pattern=[[1, chunk_size]], offset=chunk_start, channel_multiplier=0:
            #   Each row: [chunk_start, chunk_start+1, ..., chunk_start+chunk_size-1]
            nisa.iota(
                dst=batch_chunk_indices_and_mask,
                pattern=[[1, chunk_size]],
                offset=chunk_start,
                channel_multiplier=0,
                name=f"create_chunk_indices_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Direct comparison: targets == indices (tensor_scalar broadcasts targets across free dim)
            # mask[i,j] = 1 if batch_targets[i] == batch_chunk_indices_and_mask[j], else 0
            # Reuse the same buffer for output since indices are no longer needed after comparison
            nisa.tensor_scalar(
                dst=batch_chunk_indices_and_mask,
                data=batch_chunk_indices_and_mask,
                op0=nl.equal,
                operand0=batch_targets,  # [P, 1] broadcasts across chunk_size
                name=f"create_target_mask_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Scale mask by grad_scale: correction = mask * grad_scale
            nisa.tensor_scalar(
                dst=batch_correction,
                data=batch_chunk_indices_and_mask,
                op0=nl.multiply,
                operand0=grad_scale,
                name=f"scale_correction_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Apply correction: grad = grad - (mask * grad_scale)
            nisa.tensor_tensor(
                dst=batch_grad_chunk,
                data1=batch_grad_chunk,
                data2=batch_correction,
                op=nl.subtract,
                name=f"apply_target_correction_batch_{batch_idx}_chunk_{chunk_idx}",
            )

            # Store the gradient chunk to HBM (with target corrections already applied)
            nisa.dma_copy(
                dst=grad_logits_shard_hbm[batch_start_idx : batch_start_idx + actual_batch_size, chunk_start:chunk_end],
                src=batch_grad_chunk[0:actual_batch_size, 0:actual_chunk_len],
                name=f"store_grad_chunk_batch_{batch_idx}_chunk_{chunk_idx}",
            )

    return grad_logits_hbm
