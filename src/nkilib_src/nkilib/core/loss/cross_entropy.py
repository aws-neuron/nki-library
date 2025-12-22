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
This kernel implements memory-efficient cross entropy for large vocabularies using the online log-sum-exp algorithm with batched processing.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .validation import validate_cross_entropy_forward_inputs


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

    Notes:
        - Batched version for LNC2 (2 cores): Each core processes multiple positions in batches
        - Positions assigned in strided pattern (core_id, core_id + 2, core_id + 4, ...)
        - Vectorized operations across batch dimension for efficiency
        - chunk_size must not exceed vocabulary size V
        - positions_per_batch must be in range (0, 128]
        - Per-allocation size constraint: positions_per_batch × chunk_size × dtype_bytes ≤ 24 MiB
        - Performance tuning: Increase positions_per_batch for better throughput (up to memory limit)
        - Performance tuning: Use larger chunk_size to reduce loop iterations (up to V and memory limit)

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

    # Load balancing: nominal_positions_per_core for consistent offsets, positions_per_core for actual work
    # Last core handles remainder when num_positions doesn't divide evenly
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
