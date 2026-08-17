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

"""NKI kernel for build_all2all_dispatch_metadata.

Computes dispatch metadata (send_counts, send_displs) from expert_index for
all-to-all-v collectives. Replaces the PyTorch scatter_-based implementation
to avoid XLA shape inference conflicts when co-compiled with moe_tkg.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

P_MAX = 128


@nki.jit
def build_all2all_dispatch_metadata(
    expert_index,
    num_experts,
    num_elements_per_token,
    replica_group_size,
):
    """Build metadata for all2all dispatch using NKI.

    Computes per-rank send_counts (with token deduplication) and send_displs
    from expert_index. Equivalent to the scatter_-based PyTorch implementation
    but avoids XLA tracing issues.

    Args:
        expert_index: [T, K] int32 tensor of expert indices per token.
        num_experts: Total number of experts.
        num_elements_per_token: Elements per token (e.g. H_CONCAT).
        replica_group_size: Number of destination ranks.

    Returns:
        [4, replica_group_size] float32 tensor (values are integers stored as float).
            Row 0: send counts, Row 1: send displacements,
            Row 2: recv counts (zeros), Row 3: recv displacements (zeros).
    """
    T = expert_index.shape[0]
    K = expert_index.shape[1]
    R = replica_group_size
    n_local_experts = num_experts // replica_group_size

    kernel_assert(T <= P_MAX, f"T must be <= {P_MAX}")
    kernel_assert(R <= P_MAX, f"replica_group_size must be <= {P_MAX}")

    # Load expert_index [T, K] into SBUF as float32
    expert_idx_sb = nl.ndarray((P_MAX, K), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=expert_idx_sb[0:T, 0:K], src=expert_index[0:T, 0:K])

    # Convert to float32 for arithmetic
    expert_idx_f32 = nl.ndarray((P_MAX, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=expert_idx_f32[0:T, 0:K], src=expert_idx_sb[0:T, 0:K])

    # For each rank r, check: expert_index >= r*n_local AND expert_index < (r+1)*n_local
    # This avoids float division rounding issues.
    # match[t,k] = relu(expert[t,k] - r*n_local + 0.5) > 0 AND
    #              relu((r+1)*n_local - 0.5 - expert[t,k]) > 0
    # Simplified: since expert values are integers:
    #   expert >= lo  iff  expert - lo + 0.5 > 0  (relu gives positive)
    #   expert < hi   iff  hi - expert - 0.5 > 0  (relu gives positive)
    # Then multiply both masks to get AND.

    sc_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sc_row, value=0.0)

    for r in range(R):
        lo = r * n_local_experts  # inclusive lower bound
        hi = (r + 1) * n_local_experts  # exclusive upper bound

        # ge_mask = relu(expert - lo + 0.5): positive iff expert >= lo
        ge_check = nl.ndarray((P_MAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=ge_check[0:T, 0:K],
            data=expert_idx_f32[0:T, 0:K],
            op0=nl.add,
            operand0=float(-lo + 0.5),
        )
        ge_mask = nl.ndarray((P_MAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ge_mask[0:T, 0:K], data=ge_check[0:T, 0:K], op=nl.relu)

        # lt_mask = relu(hi - 0.5 - expert): positive iff expert < hi
        # = relu(-(expert - hi + 0.5))
        lt_check = nl.ndarray((P_MAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=lt_check[0:T, 0:K],
            data=expert_idx_f32[0:T, 0:K],
            op0=nl.add,
            operand0=float(-hi + 0.5),
        )
        # Negate: we want relu(hi - 0.5 - expert) = relu(-(expert - hi + 0.5))
        neg_lt = nl.ndarray((P_MAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=neg_lt[0:T, 0:K],
            data=lt_check[0:T, 0:K],
            op0=nl.multiply,
            operand0=-1.0,
        )
        lt_mask = nl.ndarray((P_MAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=lt_mask[0:T, 0:K], data=neg_lt[0:T, 0:K], op=nl.relu)

        # AND: match_mask = min(ge_mask, lt_mask) — both positive means in range
        # Actually use multiply since both are >= 0, product > 0 iff both > 0
        match_mask = nl.ndarray((P_MAX, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=match_mask[0:T, 0:K],
            data1=ge_mask[0:T, 0:K],
            data2=lt_mask[0:T, 0:K],
            op=nl.multiply,
        )

        # Clamp to [0, 1] — values could be > 1 from relu outputs
        # Use min(match_mask, 1.0): any positive -> treat as 1
        # Actually for dedup we just need max across K, so > 0 is fine
        # Step 2: Reduce across K (axis=1) with max -> [T, 1]
        token_sends = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=token_sends[0:T, 0:1],
            data=match_mask[0:T, 0:K],
            op=nl.maximum,
            axis=(1,),
        )

        # Binarize: any positive -> 1.0, zero stays 0.0
        # Multiply by large constant then clamp to [0, 1]
        nisa.tensor_scalar(
            dst=token_sends[0:T, 0:1],
            data=token_sends[0:T, 0:1],
            op0=nl.multiply,
            operand0=1e6,
        )
        ones_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)
        nisa.tensor_tensor(
            dst=token_sends[0:T, 0:1],
            data1=token_sends[0:T, 0:1],
            data2=ones_col[0:T, 0:1],
            op=nl.minimum,
        )

        # Step 3: Sum across T (partition dim) -> transpose then reduce
        token_sends_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_transpose(dst=token_sends_row[0:1, 0:T], data=token_sends[0:T, 0:1])

        count_r = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=count_r[0:1, 0:1],
            data=token_sends_row[0:1, 0:T],
            op=nl.add,
            axis=(1,),
        )

        # Multiply by num_elements_per_token
        nisa.tensor_scalar(
            dst=count_r[0:1, 0:1],
            data=count_r[0:1, 0:1],
            op0=nl.multiply,
            operand0=float(num_elements_per_token),
        )

        # Store into sc_row at position r
        nisa.tensor_copy(dst=sc_row[0:1, r : r + 1], src=count_r[0:1, 0:1])

    # Compute send_displs: exclusive prefix sum
    ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)

    init_zero = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=init_zero, value=0.0)

    inclusive_sum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=inclusive_sum[0:1, 0:R],
        data0=ones_row[0:1, 0:R],
        data1=sc_row[0:1, 0:R],
        initial=init_zero[0:1, 0:1],
        op0=nl.multiply,
        op1=nl.add,
    )

    # exclusive = inclusive - counts
    sd_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=sd_row[0:1, 0:R],
        data1=inclusive_sum[0:1, 0:R],
        data2=sc_row[0:1, 0:R],
        op=nl.subtract,
    )

    # Output [4, R] as float32 (framework stores golden as float32)
    output = nl.ndarray((4, R), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output[0:1, 0:R], src=sc_row[0:1, 0:R])
    nisa.dma_copy(dst=output[1:2, 0:R], src=sd_row[0:1, 0:R])

    zeros_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros_row, value=0.0)
    nisa.dma_copy(dst=output[2:3, 0:R], src=zeros_row[0:1, 0:R])
    nisa.dma_copy(dst=output[3:4, 0:R], src=zeros_row[0:1, 0:R])

    return output
