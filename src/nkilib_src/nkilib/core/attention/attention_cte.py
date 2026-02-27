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
Attention Kernel For Context Encoding (Prefill)

ALGORITHM OVERVIEW:

Base Attention Algorithm:
The kernel computes: Output = softmax(scale * Q @ K^T) @ V

Breaking this down into steps:
  1. Compute attention scores: S = scale * Q @ K^T  (matmul, called MM1)
  2. Apply masking to S (causal, sliding window, etc.)
  3. Compute row-wise softmax: P = softmax(S) = exp(S - max(S)) / sum(exp(S - max(S)))
  4. Compute final output: O = P @ V  (matmul, called MM2)

The transpose flags (tp_q, tp_k, tp_out) control input/output tensor layouts without
changing the mathematical computation.

For a simple PyTorch reference implementation, see attention_cte_torch.py

FEATURES:

1. Causal Masking (causal_mask=True):
   - Masks upper triangle of attention scores: S[i,j] = -inf when i < j
   - Enables compute skipping: skip MM1/MM2 for upper triangle tiles

2. Sliding Window Attention (SWA, when sliding_window > 0):
   - Local attention: each query only attends to nearby keys within a window
   - Masks attention scores: S[i,j] = -inf when |i - j| > sliding_window
   - Currently only works with causal: masks both upper triangle AND positions outside window
   - When used with CP: loads only required KV slice to save memory

3. Context Parallelism (CP, global_cp_deg > 1, cp_offset is not None):
   - Distributes long sequence computation across multiple devices/ranks
   - Each rank (kernel call) processes a slice of Q sequence with full K/V
   - cp_offset indicates which Q slice this rank handles (runtime value)
   - Optionally supports strided Q slicing for better load balancing across CP ranks
   - Requires dynamic masking since offset unknown at compile time
   - Currently only supports causal attention

4. Prefix Caching (k_prior/v_prior provided):
   - K/V split into two parts: prior (cached) and active (current)
   - prior_used_len specifies how much of prior to use (dynamic mask)
   - Causal mask not required for prior portion (although we still apply SWA if applicable)

5. Sink Tokens (sink provided):
   - Add additional sink token to softmax denominator

6. Grouped Query Attention (GQA, batch_size_kv < batch_size):
   - Kernel handles GQA natively without explicit K/V replication

7. Support for training:
   - Kernel can optionally return maximum attention score and softmax denominator (per row) for backpropagation.

IMPLEMENTATION DETAILS AND LOOP STRUCTURE:

Level 1: LNC2 Sharding (on Trn2+)
  - Shards computation across 2 NeuronCores (LNC=2)
  - Primary sharding: Divides batch dimension evenly
  - Secondary sharding (for odd batch): Last batch item sharded on seqlen_q
    * Uses unequal split (65%/35%) for causal attention to balance load
    * Falls back to single core for short sequences (< 1024 tokens)

Level 2: Batch Loop
  - Iterates over batch items assigned to this core
  - Each batch item processes independently
  - For GQA: maps Q batch_id to correct KV batch_id

Level 3: Section Loop (Flash Attention for long sequences)
  - For K/V length > 10K tokens: divide into 8K-token sections
  - Process one section at a time to fit in SBUF memory
  - Maintains running statistics (max, sum) across sections
  - Final output computed using flash attention rescaling
  - For short sequences: single section contains all K/V
  - Check https://arxiv.org/abs/2205.14135 and https://arxiv.org/abs/2307.08691
    for more details about flash attention.

Level 4: Group Loop (Q sequence processing)
  - Q sequence divided into groups of 128 tokens (_Q_GRP_SZ)
  - Each group processes independently within a section
  - Software pipelining: overlaps operations across groups (i, i+1, i+2)
    * Group i:   PV computation, writeback
    * Group i+1: Exp computation
    * Group i+2: Q load, QK computation
  - Uses modular allocation for efficient buffer reuse

INTENDED USAGE:

The kernel supports sequence lengths up to 36864 and is optimized for q sequence
length larger than ~256 (i.e., prefill/context encoding workloads). The head dimension (d)
can be up to 128. Batch size up to 16 has been tested.

Input dtypes can be bfloat16, float16 or float32. The kernel uses float32 for softmax and
bfloat16 for other operations.

"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import align_to
from ..utils.kernel_assert import assert_shape, kernel_assert
from ..utils.kernel_helpers import PSUM_BANK_SIZE, div_ceil, get_verified_program_sharding_info
from ..utils.modular_allocator import ModularAllocator
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_FLOAT32_MIN = -3.4028235e38  # used for initialization and masking

"""
Kernel constraints (based on tested range, values outside range might work in practice)
"""
_MAX_BS = 32  # max tested batch size
_MAX_SEQLEN = 36864  # max tested seqlen
_MAX_BS_TIMES_SEQLEN_QK = 32.0 * 36864 * 36864  # max tested bs*seqlen_q*seqlen_k
_MAX_HEAD_DIM = 128  # max supported head dim (d)
_MIN_GLOBAL_CP_DEGREE = 1  # minimum context parallel degree
_MAX_GLOBAL_CP_DEGREE = 32  # minimum context parallel degree


"""
Sharding, tile size and threshold related constants
"""
_MIN_SEQLEN_FOR_LNC2_SHARDING = 1024  # if odd batch, then shard on sequence if len above this
_SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT = 0.5  # When sharding LNC2 on seqlen, pass 50% of seqlen to each shard
_SEQLEN_SHARDING_SPLIT_FACTOR_CAUSAL = (
    0.65  # When sharding LNC2 on seqlen, split 65%-35% in causal case to balance compute
)
_Q_GRP_SZ = 128
_V_TILE_SZ = 128  # V tile size for loading and MM2 operations
_K_TILE_SZ = 512  # K tile size for loading and MM1+masking operations
_EXP_TILE_SZ = 512  # Tile size for exp instructions (must equal _K_TILE_SZ)
_LARGE_TILE_SZ = 2048  # Larger tile size for allocations/pipelining (4 x 512 tiles)
_FLASH_ATTENTION_THRESHOLD = 10 * 1024  # Use flash attention above this K/V length
_FLASH_ATTENTION_SECTION_LENGTH = 8 * 1024  # Section size when using flash attention
_SWA_ALLOCATION_STRATEGY_THRESHOLD = (
    128  # for SWA, threshold above which allocate more q tiles and use range_select masking
)


@nki.jit
def attention_cte(
    q: nl.ndarray,
    k: nl.ndarray,
    v: nl.ndarray,
    scale: float = 1.0,
    causal_mask: bool = True,
    k_prior: Optional[nl.ndarray] = None,
    v_prior: Optional[nl.ndarray] = None,
    prior_used_len: Optional[nl.ndarray] = None,
    sink: Optional[nl.ndarray] = None,
    sliding_window: Optional[int] = None,
    tp_q: bool = True,
    tp_k: bool = False,
    tp_out: bool = False,
    cache_softmax: bool = False,
    softmax_dtype=nl.float32,
    cp_offset: Optional[nl.ndarray] = None,
    global_cp_deg: int = None,
    cp_strided_q_slicing: bool = False,
):
    """Entrypoint NKI kernel that supports multiple attention variants.

    The kernel can be invoked with 1D SPMD grid for LNC2 or without grid.

    Dimensions:
        batch_size: Number of query sequences
        batch_size_kv: Number of key/value sequences (for GQA)
        seqlen_q: Query sequence length
        seqlen_kv: Key/value sequence length
        seqlen_prior: Prior key/value sequence length (prefix caching)
        d: Head dimension size

    Args:
      q (nt.tensor): Query tensor with layout dependent on tp_q parameter
      k (nt.tensor): Key tensor with layout dependent on tp_k parameter
      v (nt.tensor): Value tensor with shape (batch_size_kv, seqlen, d)
      scale (float, optional): Scaling factor for attention scores. It must be set to 1.0 (default value) when
                     using sliding window, context parallel, or prefix caching. In these cases, q
                     can be scaled before calling the kernel.
      causal_mask (bool, optional): whether to use causal mask (default True)
      k_prior (nt.tensor, optional): (Prefix caching) Prior key tensor with layout dependent on tp_k parameter
      v_prior (nt.tensor, optional): (Prefix caching) Prior value tensor with shape (batch_size_kv, seqlen_prior, d)
      prior_used_len (nt.tensor, optional): (Prefix caching) Actual used length in prior with shape (1,)
      sink (nt.tensor, optional): Sink token tensor with shape (batch_size, 1)
      sliding_window (int, optional): Sliding window size for attention, None or 0 denotes no sliding window mask
      tp_q (bool): Query tensor transpose flag (default True)
      tp_k (bool): Key tensor transpose flag (default False)
      tp_out (bool): Output tensor transpose flag (default False)
      cache_softmax (bool): Whether to cache softmax intermediate values (default False)
      softmax_dtype (nl.dtype): Data type for softmax computations (current implementation tested with float32)
      cp_offset (nt.tensor, optional): Context parallel offset tensor with shape (1, 1)
      global_cp_deg (int, optional): Global context parallel degree
      cp_strided_q_slicing (bool, optional): Whether Q is strided for load balancing (default False)

    Returns:
      Output tensor with attention results. Shape depends on tp_out parameter.
      If cache_softmax is True, returns tuple of (output, out_neg_max, out_sum_recip).

    IO Shapes:
      - q:
        (batch_size, seqlen_q, d) when tp_q is True
        (batch_size, d, seqlen_q) when tp_q is False
      - k:
        (batch_size_kv, seqlen_kv, d) when tp_k is True
        (batch_size_kv, d, seqlen_kv) when tp_k is False
      - v: (batch_size_kv, seqlen_kv, d)
      - returns output with shape:
        (batch_size, d, seqlen_q) if tp_out is True
        (batch_size, seqlen_q, d) if tp_out is False

      - The math performed is softmax(q @ k) @ v (details described in top-level documentation)

    Prefix Caching:
      If k_prior, v_prior and prior_used_len are specified for prefix caching, the
      computation is equivalent to prepending the prior k/v (up to prior_used_len)
      behind the active k/v and shifting the mask accordingly. The shapes of k_prior
      and v_prior must match the shapes of k and v respectively on the non-seqlen
      dimensions, while the shape of prior_used_len is (1,). By setting prior_used_len,
      the actual prefix length can be chosen dynamically at runtime (up to seqlen_prior).

    MHA and Native GQA Support:
      For MHA attention, the heads can be included as part of the batch dimension.
      For GQA, when batch_size_kv < batch_size, we expect that batch_size % batch_size_kv == 0,
      and the computation is equivalent to first applying torch_interleave on K and V.
      Note that GQA typically applies to nheads dimension but the kernel combines
      batch and nheads dimensions.

    Softmax Caching (useful during training):
      When cache_softmax is True and out_neg_max/out_sum_recip are provided, returns
      the negative max and reciprocal sum in the softmax with shapes:
        padded_seq_grps = ceil(seqlen_q / 128) (_Q_GRP_SZ = 128)
        neg_max: (batch_size, 128, padded_seq_grps)
        recip: (batch_size, 128, padded_seq_grps)

    Context Parallel Support:
      Enabled when global_cp_deg is set. Since the Q seqlen offset is usually not
      known at compile time (based on rank ID), it is expected to be a (1, 1) HBM
      input to the kernel. global_cp_deg (a compile time constant int) denotes total
      number of ranks / CP degree.

      When cp_strided_q_slicing is False, this indicates FAL has sharded Q into contiguous chunks.
      In this case, cp_offset should be passed as rank_id * partial_q_seqlen

      When cp_strided_q_slicing is True, this indicates Q has been sharded in row-strided manner
      where stride is global_cp_deg. In this case, cp_offset should be passed as rank_id. Note that
      K & V are still assumed to be contiguous.

      As an example for seqlen_q=4, seqlen_kv=12, global_cp_deg=3, rank_id = 1:
        cp_strided_q_slicing False => q seqlen slice is [4, 5, 6, 7] and cp_offset is 4.
        cp_strided_q_slicing True  => q seqlen slice is [1, 4, 7, 10] and cp_offset is 1.
        In both cases, KV token order is simply [0, 1, 2, ..., 10, 11]

    Pseudocode:
      ```
      # High-level algorithm (see module docstring for detailed implementation)
      for each batch in batch_size:
        for each section in K/V (flash attention sectioning):
          for each Q group (128 tokens):
            # MM1: Compute attention scores
            scores = Q @ K^T * scale

            # Apply masking (causal, sliding window, CP, prefix caching)
            scores = apply_masks(scores)

            # Softmax with running statistics for flash attention
            max_score = max(scores)
            exp_scores = exp(scores - max_score)
            sum_exp = sum(exp_scores)

            # Update running statistics across sections
            update_flash_attention_stats(max_score, sum_exp)

            # MM2: Compute output
            output += exp_scores @ V

          # Normalize output using flash attention correction
          if last_section:
            output = output / sum_exp
      ```

    """
    if sliding_window is None:
        sliding_window = 0

    if k_prior is not None:
        is_prefix_caching = True
        kernel_assert(v_prior is not None, "k_prior is not None but v_prior is None for prefix caching")
        kernel_assert(
            prior_used_len is not None,
            "k_prior is not None but prior_used_len is None for prefix caching",
        )
    else:
        is_prefix_caching = False
        kernel_assert(v_prior is None, "k_prior is None but v_prior is not None.")
        kernel_assert(prior_used_len is None, "k_prior is None but prior_used_len is not None.")

    seqlen_q, seqlen_k_active, seqlen_k_prior, d, out_shape, softmax_shape = _check_input_and_return_shape(
        q,
        k,
        v,
        is_prefix_caching,
        k_prior,
        v_prior,
        prior_used_len,
        tp_q,
        tp_k,
        tp_out,
        cache_softmax,
    )
    result = nl.ndarray(shape=out_shape, dtype=q.dtype, buffer=nl.shared_hbm)

    out_sum_recip, out_neg_max = None, None
    if cache_softmax:
        out_neg_max = nl.ndarray(shape=softmax_shape, dtype=softmax_dtype, buffer=nl.shared_hbm)
        out_sum_recip = nl.ndarray(shape=softmax_shape, dtype=softmax_dtype, buffer=nl.shared_hbm)

    bs = q.shape[0]
    bs_kv = k.shape[0]

    # Batch size checks
    kernel_assert(bs > 0, f"Batch size must be positive, got {bs}")
    kernel_assert(
        bs <= _MAX_BS,
        f"attention_cte kernel is not tested for batch size above {_MAX_BS}, got {bs}.",
    )
    kernel_assert(bs_kv > 0, f"Batch size must be positive, got {bs_kv}")
    kernel_assert(
        bs % bs_kv == 0,
        f"Q batch size must be a multiple of KV batch size, got {bs=}, {bs_kv=}",
    )

    # Sequence length checks
    seqlen_k_total = seqlen_k_active + seqlen_k_prior if seqlen_k_prior else seqlen_k_active
    kernel_assert(
        seqlen_q <= _MAX_SEQLEN,
        f"attention_cte kernel is not tested for seqlen above {_MAX_SEQLEN}, got {seqlen_q=}.",
    )
    kernel_assert(
        seqlen_k_total <= _MAX_SEQLEN,
        f"attention_cte kernel is not tested for seqlen above {_MAX_SEQLEN}, got {seqlen_k_total=}.",
    )
    bs_seqlen_qk_product = float(bs * seqlen_q) * seqlen_k_total  # use float to avoid overflow
    kernel_assert(
        bs_seqlen_qk_product <= _MAX_BS_TIMES_SEQLEN_QK,
        f"attention_cte kernel is not tested for batch size x seqlen_q x seqlen_k above {_MAX_BS_TIMES_SEQLEN_QK}, "
        f"got {bs_seqlen_qk_product=}.",
    )
    kernel_assert(
        sliding_window <= _MAX_SEQLEN,
        f"attention_cte kernel is not tested for sliding window above {_MAX_SEQLEN}, got {sliding_window=}.",
    )
    kernel_assert(sliding_window >= 0, f"sliding_window must be >= 0, got {sliding_window=}.")

    # head dim
    kernel_assert(d > 0, f"d must be > 0, got {d=}.")
    kernel_assert(
        d <= _MAX_HEAD_DIM,
        f"we do not support head_dim > {_MAX_HEAD_DIM}, got head dim {d}",
    )

    # Context parallel
    if global_cp_deg:
        kernel_assert(
            _MIN_GLOBAL_CP_DEGREE <= global_cp_deg <= _MAX_GLOBAL_CP_DEGREE,
            f"attention_cte kernel is not tested for global_cp_deg outside [{_MIN_GLOBAL_CP_DEGREE}, {_MAX_GLOBAL_CP_DEGREE}], "
            f"got {global_cp_deg=}.",
        )

    # Create AttnConfig with high-level configuration
    ac = AttnConfig(
        seqlen_q=seqlen_q,
        seqlen_k_active=seqlen_k_active,
        seqlen_k_prior=seqlen_k_prior,
        d=d,
        tp_q=tp_q,
        tp_k=tp_k,
        tp_out=tp_out,
        is_prefix_caching=is_prefix_caching,
        causal_mask=causal_mask,
        use_swa=sliding_window > 0,
        sliding_window=sliding_window,
        use_cp=global_cp_deg is not None,
        global_cp_deg=global_cp_deg,
        cp_strided_q_slicing=cp_strided_q_slicing,
        scale=scale,
        cache_softmax=cache_softmax,
        dtype=q.dtype,
        softmax_dtype=softmax_dtype,
    )

    grid_ndim, num_shard, shard_id = get_verified_program_sharding_info("attention_cte", max_sharding=2)
    # Shard on batch size while it is divisible (if not sharded, num_shard = 1, shard_id = 0)
    num_bs_per_shard = bs // num_shard
    bs_offset = shard_id * num_bs_per_shard

    for batch_idx in range(num_bs_per_shard):
        kv_batch_id = _q_to_kv_batch_id(batch_idx + bs_offset, bs, bs_kv)
        _attention_cte_impl(
            q,
            k,
            v,
            k_prior,
            v_prior,
            prior_used_len,
            result,
            batch_idx + bs_offset,
            kv_batch_id,
            ac,
            sink=sink,
            out_neg_max=out_neg_max,
            out_sum_recip=out_sum_recip,
            cp_offset=cp_offset,
        )

    has_remainder = (bs % num_shard) != 0
    last_batch = bs - 1

    # shard on seqlen_q for the remainder bs
    if has_remainder:
        last_batch_id = _q_to_kv_batch_id(last_batch, bs, bs_kv)
        if seqlen_q >= _MIN_SEQLEN_FOR_LNC2_SHARDING:
            # shard unequally on seqlen_q when causal mask and not sliding window/CP
            # For CP we shard unequally when we have strided Q slicing.
            use_causal_divide_factor = (
                causal_mask and (cp_offset is None or cp_strided_q_slicing) and (sliding_window == 0)
            )
            divide_factor = (
                _SEQLEN_SHARDING_SPLIT_FACTOR_CAUSAL
                if use_causal_divide_factor
                else _SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT
            )
            if is_prefix_caching and use_causal_divide_factor:
                s_active, s_prior = v.shape[1], v_prior.shape[1]
                divide_factor = (
                    _SEQLEN_SHARDING_SPLIT_FACTOR_CAUSAL * s_active + _SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT * s_prior
                ) / (s_active + s_prior)

            total_grps = div_ceil(seqlen_q, _Q_GRP_SZ)
            batch_0_grp = int(total_grps * divide_factor)
            batch_1_grp = total_grps - batch_0_grp

            batch_length = shard_id * batch_1_grp + (1 - shard_id) * batch_0_grp  # shard_id is 0 or 1
            _attention_cte_impl(
                q,
                k,
                v,
                k_prior,
                v_prior,
                prior_used_len,
                result,
                last_batch,
                last_batch_id,
                ac,
                sink=sink,
                out_neg_max=out_neg_max,
                out_sum_recip=out_sum_recip,
                shard_seqlen_q_start=shard_id * batch_0_grp,
                shard_seqlen_q_length=batch_length,
                cp_offset=cp_offset,
            )
        else:
            # Have core 0 do all the work
            if shard_id == 0:
                _attention_cte_impl(
                    q,
                    k,
                    v,
                    k_prior,
                    v_prior,
                    prior_used_len,
                    result,
                    last_batch,
                    last_batch_id,
                    ac,
                    sink=sink,
                    out_neg_max=out_neg_max,
                    out_sum_recip=out_sum_recip,
                    cp_offset=cp_offset,
                )

    if cache_softmax:
        return result, out_neg_max, out_sum_recip
    else:
        return result


@dataclass
class AttnConfig(nl.NKIObject):
    """High-level attention configuration set at kernel entry point.

    Contains user-facing parameters and computed configuration flags.
    """

    # Sequence dimensions
    seqlen_q: int = None
    seqlen_k_active: int = None
    seqlen_k_prior: int = None
    d: int = None

    # Transpose flags
    tp_q: bool = None
    tp_k: bool = None
    tp_out: bool = None

    # Masking configuration
    is_prefix_caching: bool = None
    causal_mask: bool = None

    # Sliding window attention
    use_swa: bool = None
    sliding_window: int = None

    # Context parallelism
    use_cp: bool = None
    global_cp_deg: int = None
    cp_strided_q_slicing: bool = None

    # Other
    scale: float = None
    cache_softmax: bool = None
    dtype: Any = None
    softmax_dtype: Any = None


def _attention_cte_impl(
    q,
    k_active,
    v_active,
    k_prior,
    v_prior,
    prior_used_len,
    o,
    batch_id,
    batch_id_kv,
    ac: AttnConfig,
    sink=None,
    out_neg_max=None,
    out_sum_recip=None,
    shard_seqlen_q_start=-1,
    shard_seqlen_q_length=-1,
    cp_offset: Any = None,
):
    """
    Internal implementation function for attention computation.

    This function processes a single batch and handles the core attention computation
    with flash attention optimization, sectioning, and various masking strategies.

    Args:
      q, k_active, v_active: Input tensors
      k_prior, v_prior, prior_used_len: Prefix caching tensors
      o: Output tensor
      batch_id, batch_id_kv: Batch indices
      ac: High-level attention configuration
      sink: Optional sink token tensor
      out_neg_max, out_sum_recip: Optional softmax cache outputs
      shard_seqlen_q_start, shard_seqlen_q_length: Seqlen sharding parameters
      cp_offset: Context parallel offset tensor

    High-level logic:
    For large enough K/V length, we divide the K/V into sections of 8k.

    For each section:
      a. Load K and V to SBUF
      b. Loop over Q (groups) - each group has seqlen 128 (_Q_GRP_SZ)
      c. Within each group:
        i.   Load Q
        ii.  Compute QK^T (MM1) and max
        iii. Compute exponential and transpose
        iv.  Compute PV (MM2)
        v.   Write to output

    Handling multiple sections:
      We keep running max, sum, etc. buffers that we keep updating as we go through
      sections and use these to update the output using flash attention.

    Pipelining:
      To maximize utilization of the hw engines, we use software pipelining to
      inform scheduler decisions. This means we manually interleave iterations of
      the group loop to make sure certain operations of group i+2 should start even
      as group i is being processed. In addition we use modulo allocation for tensors
      to enable pipelining across Q groups and K/V tiles within a section.
    """
    is_seqlen_sharded = shard_seqlen_q_start >= 0

    # Compute all tile parameters including section length and number of sections
    atp = _compute_tile_parameters(ac, is_seqlen_sharded)

    # Update shard length if not sharded
    if not is_seqlen_sharded:
        shard_seqlen_q_start = 0
        shard_seqlen_q_length = atp.num_grps

    # Initialize allocator and buffer container
    allocator = ModularAllocator(initial_address=0)
    bufs = AttnInternalBuffers()

    # Allocate shared utilities (zero bias, sink)
    bufs.zero_bias_tensor = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)
    nisa.memset(bufs.zero_bias_tensor, 0.0)

    if sink is not None:
        bufs.sink_sb = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)
        # Load sink from HBM to SBUF
        nisa.dma_copy(dst=bufs.sink_sb[0, 0], src=sink[batch_id, 0])
        stream_shuffle_broadcast(src=bufs.sink_sb, dst=bufs.sink_sb)

    # Setup range select bounds for dynamic masking (used in CP/SWA/Prefix caching)
    _setup_range_select_bounds(ac, atp, bufs, allocator, cp_offset, prior_used_len)

    # Allocate running statistics (persistent across sections)
    bufs.mm1_running_max = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
    bufs.exp_running_sum = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
    bufs.exp_sum_reciprocal = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)

    # Mark allocator checkpoint for section-level reset
    sbuf_addr_outer = allocator.get_current_address()

    # Loop over k/v sections
    for section_idx in range(atp.num_sections):
        # Compute some quantities for this section such as offset and whether section contains active/prior KV
        section_offset = atp.section_len * section_idx
        section_offset_active, section_contains_prefix = _compute_section_offset_active(
            section_offset, ac.is_prefix_caching, atp.seqlen_k_prior_padded
        )
        next_section_offset = atp.section_len * (section_idx + 1)
        next_section_offset_active, next_section_contains_prefix = _compute_section_offset_active(
            next_section_offset, ac.is_prefix_caching, atp.seqlen_k_prior_padded
        )
        sp = SectionParams(
            section_idx=section_idx,
            section_offset=section_offset,
            section_offset_active=section_offset_active,
            next_section_offset_active=next_section_offset_active,
            section_contains_prefix=section_contains_prefix,
            next_section_contains_prefix=next_section_contains_prefix,
        )

        # Allocate the internal buffers
        allocator.set_current_address(sbuf_addr_outer)
        _allocate_attention_buffers(allocator, ac, atp, bufs, sink)
        sbuf_addr = allocator.get_current_address()

        # Load K and V for the section
        sbuf_addr = _load_k_tile(
            k_active,
            k_prior,
            bufs.k_sb,
            batch_id_kv,
            sp,
            nl.bfloat16,
            ac.tp_k,
            atp.num_k_tiles_per_section,
            sbuf_addr,
            load_offset_active=bufs.k_offset_sb_u32,
        )
        sbuf_addr = _load_v_tile(
            v_active,
            v_prior,
            bufs.v_sb,
            batch_id_kv,
            sp,
            nl.bfloat16,
            atp.num_v_tiles_per_section,
            sbuf_addr,
            load_offset_active=bufs.k_offset_sb_u32,
        )

        # Start with Q load and actual compute
        if shard_seqlen_q_length <= 1:
            # no pipelining when there's only 1 group
            _load_q_impl(shard_seqlen_q_start, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
            _qk_and_max_impl(shard_seqlen_q_start, ac, atp, sp, bufs)
            _update_max_impl(shard_seqlen_q_start, ac, atp, sp, bufs, sink)
            _exp_impl(shard_seqlen_q_start, ac, atp, sp, bufs, sink)
            _pv_impl(shard_seqlen_q_start, ac, atp, sp, bufs)
            _write_back_impl(shard_seqlen_q_start, ac, atp, sp, bufs, o, batch_id)

        else:
            # Do software pipelining. We have a group loop and some initial/final calls
            # outside the loop.
            _load_q_impl(shard_seqlen_q_start, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
            _qk_and_max_impl(shard_seqlen_q_start, ac, atp, sp, bufs)
            _update_max_impl(shard_seqlen_q_start, ac, atp, sp, bufs, sink)
            _exp_impl(shard_seqlen_q_start, ac, atp, sp, bufs, sink)

            _load_q_impl(shard_seqlen_q_start + 1, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
            _qk_and_max_impl(shard_seqlen_q_start + 1, ac, atp, sp, bufs)
            _update_max_impl(shard_seqlen_q_start + 1, ac, atp, sp, bufs, sink)
            shard_seqlen_q_end = shard_seqlen_q_start + shard_seqlen_q_length

            for grp_i in range(shard_seqlen_q_start, shard_seqlen_q_end - 2):  # for each block of seq_q
                if ac.use_swa and atp.is_causal:
                    nisa.memset(bufs.mm2_sb[grp_i][...], value=0.0)  # when use_swa, mm_i == 0 is not the initial tile

                # We try to perform software pipelining where the following operations are overlapped:
                # grp_i   :  PV, write_back
                # grp_i+1 :  EXP
                # grp_i+2 :  Load Q, QK+Max
                _load_q_impl(grp_i + 2, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
                _exp_impl(grp_i + 1, ac, atp, sp, bufs, sink)
                _fused_qkmax_and_pv_impl(grp_i, ac, atp, sp, bufs)
                _write_back_impl(grp_i, ac, atp, sp, bufs, o, batch_id)
                _update_max_impl(grp_i + 2, ac, atp, sp, bufs, sink)

            _pv_impl(shard_seqlen_q_end - 2, ac, atp, sp, bufs)

            _write_back_impl(shard_seqlen_q_end - 2, ac, atp, sp, bufs, o, batch_id)
            _exp_impl(shard_seqlen_q_end - 1, ac, atp, sp, bufs, sink)
            _pv_impl(shard_seqlen_q_end - 1, ac, atp, sp, bufs)
            _write_back_impl(shard_seqlen_q_end - 1, ac, atp, sp, bufs, o, batch_id)

        # If used with training, we need to also return the softmax intermediates
        # num_grps is total number of groups, shard_seqlen_q_length is current shard
        # write from [0:128, shard_seqlen_q_start:shard_seqlen_q_start+shard_seqlen_q_length]
        # to [batch_id, 0:128, shard_seqlen_q_start:shard_seqlen_q_start+shard_seqlen_q_length]
        dst_ap = [[atp.num_grps, atp.sb_p], [1, shard_seqlen_q_length]]
        dst_offset = batch_id * atp.sb_p * atp.num_grps + shard_seqlen_q_start
        src_ap = [[atp.num_grps, atp.sb_p], [1, shard_seqlen_q_length]]
        src_offset = shard_seqlen_q_start
        if out_neg_max is not None:
            nisa.dma_copy(
                out_neg_max.ap(pattern=dst_ap, offset=dst_offset),
                src=bufs.mm1_running_max.ap(pattern=src_ap, offset=src_offset),
            )
        if out_sum_recip is not None:
            nisa.dma_copy(
                out_sum_recip.ap(pattern=dst_ap, offset=dst_offset),
                src=bufs.exp_sum_reciprocal.ap(pattern=src_ap, offset=src_offset),
            )


@dataclass
class AttnInternalBuffers(nl.NKIObject):
    """Container for all SBUF and PSUM tensor buffers used in attention computation."""

    # SBUF tensors to load q/k/v into
    q_sb = None
    k_sb = None
    v_sb = None

    # SBUF/PSUM tensors for computation

    # QK and max
    mm1_psum = None  # output of MM1 on PSUM
    mm1_copy_sb = None  # Copy mm1_psum to SBUF for using affine select (Pool engine input needs to be SBUF)
    mm1_affine_select_output = None  # Output of affine select, goes via TSCR to produce mm1_masked
    mm1_masked = None  # Masked and scaled (if scale != 1.0) output after MM1 (in SBUF) - produced via affine_select+TSCR or range select
    mm1_partial_max = None  # tile-wise max after MM1
    mm1_section_max = None  # max for section after MM1
    mm1_running_max = None  # (persistent across sections) Running max across sections for output after MM1
    prev_mm1_running_max = None  # Previous running max across sections for output after MM1 (used to hold value temporarily before section update)
    flash_attn_correction_factor = None  # Correction factor for flash attn (exp(prev_max-curr_max))

    # Exp
    exp_sb = None  # Output of exp
    exp_partial_sum = None  # Exp-sum per tile
    exp_section_sum = None  # Exp-sum for section
    exp_tp_sb = None  # Transposed output of Exp (input to MM2)
    exp_running_sum = None  # (persistent across sections) Running sum across sections after exp
    prev_exp_running_sum = (
        None  # Previous running max across sections after exp (used to hold value temporarily before section update)
    )
    exp_sum_reciprocal = None  # (persistent across sections) Reciprocal of exp-sum, calculated in the last section

    # PV
    mm2_psum = None  # output of MM2 on PSUM
    mm2_sb = None  # accumulate output of MM2 (mm2_psum) into SBUF
    mm2_prev_output = None  # output from previous section, loaded from HBM to SBUF
    mm2_accum_flash_attn = None  # Accumulated and scaled by flash_attn_correction_factor output of MM2 across sections
    mm2_final = None  # Output in final section, scaled by exp_sum_reciprocal

    # Optional buffers (for tp_out=True)
    tp_flash_attn_correction_factor_psum = None  # nc_transpose of flash_attn_correction_factor on PSUM
    tp_flash_attn_correction_factor_sb = None  # transpose of flash_attn_correction_factor copied to SBUF
    tp_exp_sum_reciprocal_psum = None  # nc_transpose of exp_sum_reciprocal on PSUM
    tp_exp_sum_reciprocal_sb = None  # transpose of exp_sum_reciprocal copied to SBUF
    mm2_prev_output_scaled = None  # Scaled version of prev_output before accumulating

    # Shared/utility tensors
    zero_bias_tensor = None  # zeros, used for initialization/fallback in multiple places
    sink_sb = None  # sink loaded to SBUF
    range_sel_lbs = None  # lower bound for range_select for CP/SWA/Prefix caching
    range_sel_ubs = None  # upper bound for range_select for CP/SWA/Prefix caching
    range_sel_lbs_prior = None  # lower bound for range_select for Prefix caching
    range_sel_ubs_prior = None  # upper bound for range_select for Prefix caching
    k_offset_sb_u32 = None  # used for dynamic load for only required k/v for CP+SWA


@dataclass
class SectionParams(nl.NKIObject):
    section_idx = None  # Index of section
    section_offset = None  # Offset of section
    section_offset_active = None  # Offset of active K (adjusted by subtracting prior)
    next_section_offset_active = None  # Offset of active K for next section
    section_contains_prefix = None  # Whether current section contains prefix
    next_section_contains_prefix = None  # Whether next section contains prefix


@dataclass
class AttnTileParams(nl.NKIObject):
    """Tile and buffer sizing parameters computed during implementation.

    Contains derived parameters specific to buffer allocation and tiling.
    """

    seqlen_k_active_updated: int = None  # use updated value based on CP/SWA
    seqlen_k_prior_padded: int = None  # k_prior len padded to multiple of _K_TILE_SZ (512)
    is_causal: bool = None  # generally same as causal_mask, but can be modified for CP.
    # Used to determine whether compute is eliminated.

    # Partition/tile sizes
    sb_p: int = None  # SBUF partition size (128)

    # Group parameters
    num_grps: int = None
    num_q_grps_per_load: int = None  # load multiple q groups for better DMA efficiency
    can_pack_q_load: bool = None  # whether Q loads can be packed into num_q_grps_per_load

    # Tile counts per section
    num_large_tiles_per_section: int = None
    num_k_tiles_per_section: int = None
    num_v_tiles_per_section: int = None

    # Exp parameters
    exp_inst_elems: int = None  # exp tile size
    num_exp_insts_per_large_tile: int = None

    # Transpose and MM2 parameters
    # After transpose the scores are laid out as (128,4,128) which effectively stores 4 KxQ tiles
    # of 128x128 (recall K tile size is 512)
    num_tps_in_mm2_grp: int = None  # Number of transpose/MM2 per MM2 group (4)
    mm2_grp_sz: int = None  # Total free dim for MM2 group (4*128 = 512)

    # Use optimized allocation for SWA where more Q groups are allocated since each group
    # only handles relatively small few K tiles
    use_swa_optimized_allocation: bool = None

    # Dynamic masking - whether to use range select for masking instead of affine select.
    # Required when we need runtime-determined masking (e.g., CP/prefix caching),
    # or for performance reasons (e.g., SWA to avoid multiple copies with affine select)
    dynamic_sel_mask: bool = None

    # Section
    section_len: int = None  # Length of section, typically 8k if multiple sections, else same as k seqlen
    num_sections: int = None  # Number of sections


def _compute_tile_parameters(
    ac: AttnConfig,
    is_seqlen_sharded: bool,
) -> AttnTileParams:
    """
    Compute all tile and partition parameters for attention computation.

    Args:
      ac: High-level attention configuration
      is_seqlen_sharded: Whether Q sequence is sharded

    Returns:
      AttnTileParams: Complete tile parameter configuration
    """
    atp = AttnTileParams()

    # Validate scale parameter for special modes
    if ac.use_swa or ac.is_prefix_caching or ac.use_cp:
        # Only scale = 1.0 supported in these cases due to use of range select instead of TSCR
        kernel_assert(
            ac.scale == 1.0,
            f"SWA/Prefix Caching/CP only support scale=1.0, but got {ac.scale=}",
        )

    # When we use CP, tiles are dynamically masked (mask unknown at compile time), so we turn off causal
    # to disable compute-skipping. For strided Q slicing, we do not turn off causal masking since
    # compute can be eliminated from the region that is masked in all ranks.
    atp.is_causal = ac.causal_mask
    kernel_assert(ac.causal_mask or not ac.use_cp, "CP currently only supports causal attn")
    kernel_assert(ac.causal_mask or not ac.use_swa, "SWA currently only supports causal attn")
    atp.dynamic_sel_mask = False
    if ac.use_cp:
        if not ac.cp_strided_q_slicing:
            atp.is_causal = False
        atp.dynamic_sel_mask = True

    atp.seqlen_k_active_updated = ac.seqlen_k_active
    atp.use_swa_optimized_allocation = (
        False  # whether to allocate more q groups and fewer k tiles for exp and transpose
    )

    # Handle sliding window attention, in which case only at most (seqlen_q + sliding_window - 1) KV slice is loaded (when CP)
    if ac.use_swa:
        # When using SWA+CP (dynamic sbuf CP offsets), we (1) do dynamic masking with range_selects and (2) load reduced KV
        # When not using CP, we apply both upper (causal) and lower (sliding window) triangular compute skipping;
        # Note that the reduced KV load for CP+SWA only applies to (active) K not to K_prior.
        # For K_prior, caller can choose to pass only the required KV since it not always possible
        # to determine the required seqlen a priori on due to dynamic prior_used_len.
        # When using strided Q slicing, we need to load entire KV due to masking pattern.
        if ac.use_cp and not ac.cp_strided_q_slicing:
            atp.seqlen_k_active_updated = min(ac.seqlen_k_active, ac.seqlen_q + ac.sliding_window - 1)
            atp.seqlen_k_active_updated = min(ac.seqlen_k_active, align_to(atp.seqlen_k_active_updated, 512))
        else:
            if ac.sliding_window <= _SWA_ALLOCATION_STRATEGY_THRESHOLD:
                # use range select to save on excess copy instructions on DVE
                atp.dynamic_sel_mask = True
                # We only use 1 or 2 tile per group so want to overlap the groups more.
                if not ac.is_prefix_caching:
                    # When prefix caching is enabled, we use static_range which means
                    # that cannot reduce number of 2048 tiles too much without causing
                    # data race. So we only use this optimization without prefix caching
                    atp.use_swa_optimized_allocation = True

    # Partition size
    atp.sb_p = nl.tile_size.pmax
    # assert that _Q_GRP_SZ = _V_TILE_SZ = atp.sb_p (= 128) since that is an implict assumption in the code
    # and updating it requires careful updates.
    kernel_assert(
        _Q_GRP_SZ == atp.sb_p,
        f"Internal error: expect Q group size to match SBUF partition dimension, got {_Q_GRP_SZ=}, {atp.sb_p=}",
    )
    kernel_assert(
        _V_TILE_SZ == atp.sb_p,
        f"Internal error: expect V tile size to match SBUF partition dimension, got {_V_TILE_SZ=}, {atp.sb_p=}",
    )

    # Group configuration
    atp.num_grps = div_ceil(ac.seqlen_q, atp.sb_p)
    atp.can_pack_q_load = not is_seqlen_sharded

    num_q_grps_per_load_dtype = 4 if ac.dtype == nl.float32 else 8  # fewer groups for float32 for SBUF memory
    atp.num_q_grps_per_load = min(num_q_grps_per_load_dtype if atp.can_pack_q_load else 1, atp.num_grps)
    kernel_assert(
        atp.num_q_grps_per_load > 0,
        f"num_q_grps_per_load must be positive, got {atp.num_q_grps_per_load}. "
        f"This occurs when num_grps={atp.num_grps} is 0 or negative. "
        f"Please check that batch_size, num_heads, and sequence length parameters are positive integers.",
    )

    atp.seqlen_k_prior_padded = None
    if ac.is_prefix_caching:
        # Pad k_prior length to 512 because that is the loading and masking tile size
        # and we don't want to mix prior/active into a single tile.
        atp.seqlen_k_prior_padded = align_to(ac.seqlen_k_prior, _K_TILE_SZ)

    if ac.is_prefix_caching:
        # With prefix caching we ensure every _K_TILE_SZ (512) tile is either full prior or
        # fully active. Note that a section can still contain a mix of prior and
        # active. The different lengths are as shown below:
        #
        # +------------------+---------+----+-------------------------------------+
        # |                  |         |    |                                     |
        # +------------------+---------+----+-------------------------------------+
        #  <----------------->
        #   prior_used_len
        #   (dynamic mask)
        #  <--------------------------->
        #         seqlen_k_prior
        #  <--------------------------------><------------------------------------>
        #        seqlen_k_prior_padded                  seqlen_k_active
        #        (multiple of 512)
        total_seqlen_k = atp.seqlen_k_prior_padded + atp.seqlen_k_active_updated
    else:
        total_seqlen_k = atp.seqlen_k_active_updated

    use_flash_attn = total_seqlen_k > _FLASH_ATTENTION_THRESHOLD
    if use_flash_attn:
        atp.section_len = min(total_seqlen_k, _FLASH_ATTENTION_SECTION_LENGTH)
    else:
        atp.section_len = total_seqlen_k

    kernel_assert(atp.section_len > 0, f"section_len must be positive, got {atp.section_len}")
    atp.num_sections = div_ceil(total_seqlen_k, atp.section_len)

    if not use_flash_attn:
        kernel_assert(
            atp.num_sections == 1,
            "Logic fault, must only have 1 section if not using flash_attn",
        )

    # Tile counts per section
    atp.num_large_tiles_per_section = div_ceil(atp.section_len, _LARGE_TILE_SZ)
    atp.num_k_tiles_per_section = div_ceil(atp.section_len, _K_TILE_SZ)
    atp.num_v_tiles_per_section = div_ceil(atp.section_len, _V_TILE_SZ)

    # K/V tile sizes for exp and transpose/MM2
    atp.exp_inst_elems = _EXP_TILE_SZ
    atp.num_exp_insts_per_large_tile = _LARGE_TILE_SZ // atp.exp_inst_elems
    atp.num_tps_in_mm2_grp = _K_TILE_SZ // atp.sb_p  # 512 // 128 = 4
    atp.mm2_grp_sz = _K_TILE_SZ

    return atp


def _setup_range_select_bounds(
    ac: AttnConfig,
    atp: AttnTileParams,
    bufs: AttnInternalBuffers,
    allocator: ModularAllocator,
    cp_offset: Any,
    prior_used_len: Any,
) -> tuple:
    """
    Set up range select bounds for dynamic masking (CP/SWA/prefix caching).
    """
    # Populate range select lower and/or upper bounds. NOTE: both bounds are inclusive
    if atp.dynamic_sel_mask:
        # Populate CP offset if needed
        cp_offset_sb = None
        if ac.use_cp:
            # Check and load CP offset, then broadcast onto all partitions
            # Note that range_select only supports fp32 bounds, thus all compute for bounds here use fp32
            kernel_assert((cp_offset is not None), "cp_offset missing but global_cp_deg is provided")
            kernel_assert(
                (cp_offset.shape == (1, 1)),
                "cp_offset shape must be (1, 1) for CP attn",
            )
            cp_offset_sb = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)

            nisa.dma_copy(
                dst=cp_offset_sb[0, 0],
                src=cp_offset.ap(pattern=[[1, 1], [1, 1]], offset=0),
            )
            stream_shuffle_broadcast(src=cp_offset_sb, dst=cp_offset_sb)

        # Create range select upper bounds with IOTA + CP offset (if exists)
        bufs.range_sel_ubs = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
        # fill in q positions 0 ... num_grps*_Q_GRP_SZ
        # Important: need to set channel_multiplier
        # Note in case of cp_strided_q_slicing we fill in q positions as 0, global_cp_deg, 2*global_cp_deg ...
        if ac.use_cp and ac.cp_strided_q_slicing:
            nisa.iota(
                bufs.range_sel_ubs[...],
                pattern=[[ac.global_cp_deg * atp.sb_p, atp.num_grps]],
                channel_multiplier=ac.global_cp_deg,
            )
        else:
            nisa.iota(
                bufs.range_sel_ubs[...],
                pattern=[[atp.sb_p, atp.num_grps]],
                channel_multiplier=1,
            )

        if ac.use_cp:
            nisa.tensor_scalar(
                bufs.range_sel_ubs[...],
                bufs.range_sel_ubs,
                op0=nl.add,
                operand0=cp_offset_sb,
            )

        # Create range select lower bounds for sliding window
        if ac.use_swa:
            bufs.range_sel_lbs = allocator.alloc_sbuf_tensor(
                shape=bufs.range_sel_ubs.shape, dtype=bufs.range_sel_ubs.dtype
            )
            nisa.tensor_scalar(
                bufs.range_sel_lbs,
                bufs.range_sel_ubs,
                op0=nl.add,
                operand0=-(ac.sliding_window - 1.0),
            )

    # Setup prefix caching bounds
    if ac.is_prefix_caching:
        # with prefix caching, during the prior part:
        # - the ubs are wrt prior_used_len [note we don't need causal mask and/or CP offset here].
        # - the lbs are used when SWA is enabled (in this case also we need to offset the bounds by prior_used_len)
        #   where the cp offset (if any) is already included.
        # Note that we do not need the k_offset subtraction because the prior KV is loaded fully
        prior_used_len_sb = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)
        nisa.dma_copy(
            dst=prior_used_len_sb[0, 0],
            src=prior_used_len.ap(pattern=[[1, 1], [1, 1]], offset=0),
        )

        stream_shuffle_broadcast(src=prior_used_len_sb, dst=prior_used_len_sb)
        bufs.range_sel_ubs_prior = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
        # explicit broadcast ap needed
        nisa.tensor_scalar(
            bufs.range_sel_ubs_prior[...],
            bufs.zero_bias_tensor.ap(pattern=[[1, atp.sb_p], [0, atp.num_grps]], offset=0),
            op0=nl.add,
            operand0=prior_used_len_sb,
        )

        if ac.use_swa:
            bufs.range_sel_lbs_prior = allocator.alloc_sbuf_tensor(
                shape=bufs.range_sel_ubs_prior.shape,
                dtype=bufs.range_sel_ubs_prior.dtype,
            )
            if bufs.range_sel_lbs is not None:
                nisa.tensor_scalar(
                    bufs.range_sel_lbs_prior[...],
                    bufs.range_sel_lbs,
                    op0=nl.add,
                    operand0=prior_used_len_sb,
                )
            else:
                kernel_assert(not ac.use_cp, "CP+SWA should have dynamic_sel_mask")
                # in this case the SWA mask != yet incorporated via range_sel_lbs so we
                # add iota and -(sliding_window - 1.) here. This case will happen when
                # sliding window is large and hence we prefer to use affine_select rather
                # than range_select in active region (but still need dynamic mask for prior).
                nisa.iota(
                    bufs.range_sel_lbs_prior[...],
                    pattern=[[atp.sb_p, atp.num_grps]],
                    channel_multiplier=1,
                )
                nisa.tensor_scalar(
                    bufs.range_sel_lbs_prior[...],
                    bufs.range_sel_lbs_prior,
                    op0=nl.add,
                    operand0=prior_used_len_sb,
                    op1=nl.add,
                    operand1=-(ac.sliding_window - 1.0),
                )

    # If using SWA and CP, compute K load offset = max(0, cp_offset - sliding_window + 1)
    # Also adjust range select bounds because K seqlen now does not start from 0
    if ac.use_swa and ac.use_cp and not ac.cp_strided_q_slicing:
        # Find K load offset to fp32 (required dtype as tensor scalar operand)
        k_offset_sb = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)
        nisa.tensor_scalar(
            k_offset_sb[...],
            cp_offset_sb,
            op0=nl.add,
            operand0=-(atp.seqlen_k_active_updated - ac.seqlen_q),
            op1=nl.maximum,
            operand1=0.0,
        )
        bufs.k_offset_sb_u32 = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
        nisa.tensor_copy(bufs.k_offset_sb_u32[0, 0], k_offset_sb[0, 0])

        # Adjust range select bounds
        nisa.tensor_scalar(
            bufs.range_sel_lbs[...],
            bufs.range_sel_lbs,
            op0=nl.subtract,
            operand0=k_offset_sb,
        )
        nisa.tensor_scalar(
            bufs.range_sel_ubs[...],
            bufs.range_sel_ubs,
            op0=nl.subtract,
            operand0=k_offset_sb,
        )


def _allocate_attention_buffers(
    allocator: ModularAllocator,
    ac: AttnConfig,
    atp: AttnTileParams,
    bufs: AttnInternalBuffers,
    sink: Any,
):
    """
    Allocate all SBUF and PSUM buffers needed for attention computation.

    Modifies bufs in-place by allocating and assigning all computation buffers.

    We use the modular allocator with num_free_tiles chosen in order to achieve
    multi-buffering and avoid anti-dependencies. The degree of multi-buffering
    along the Q group/KV tile axis is chosen based on experimentation.
    """

    # Define the partition and free dimension for the two matmuls
    mm1_p, mm1_n = atp.sb_p, nl.tile_size.psum_fmax
    mm2_p, mm2_n = atp.sb_p, ac.d

    p_k, n_k = ac.d, _K_TILE_SZ  # d is reduction dim for MM1
    bufs.k_sb = allocator.alloc_sbuf_tensor(
        shape=(p_k, n_k),
        dtype=nl.bfloat16,
        block_dim=[atp.num_k_tiles_per_section],
        num_free_tiles=[atp.num_k_tiles_per_section],
        align_to=32,  # align for dma transpose
    )

    p_v, n_v = atp.sb_p, ac.d  # d is free dim for MM2
    bufs.v_sb = allocator.alloc_sbuf_tensor(
        shape=(p_v, n_v),
        dtype=nl.bfloat16,
        block_dim=[atp.num_v_tiles_per_section],
        num_free_tiles=[atp.num_v_tiles_per_section],
    )

    bufs.q_sb = allocator.alloc_sbuf_tensor(
        shape=(ac.d, atp.sb_p * atp.num_q_grps_per_load),
        dtype=nl.bfloat16,
        block_dim=[div_ceil(atp.num_grps, atp.num_q_grps_per_load)],
        num_free_tiles=[2],
        align_to=32,  # align for dma transpose
    )

    bufs.flash_attn_correction_factor = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    # buffer to hold the partial row-wise maximum from mm1, if we have sink, need one more elt from sink tensor
    mm1_partial_max_n_elts = atp.num_k_tiles_per_section + (sink is not None)
    bufs.mm1_partial_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, mm1_partial_max_n_elts),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
        align_to=4,
    )

    bufs.mm1_section_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    n_final_reduce_sum_elts = div_ceil(atp.section_len, atp.exp_inst_elems) + (sink is not None)
    bufs.exp_partial_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, n_final_reduce_sum_elts),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.exp_section_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.prev_mm1_running_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.prev_exp_running_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    if ac.tp_out:
        bufs.mm2_prev_output = allocator.alloc_sbuf_tensor(
            shape=(ac.d, atp.sb_p),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )
    else:
        bufs.mm2_prev_output = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, ac.d),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )

    if ac.tp_out:
        bufs.mm2_accum_flash_attn = allocator.alloc_sbuf_tensor(
            shape=(ac.d, atp.sb_p),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )
    else:
        bufs.mm2_accum_flash_attn = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, ac.d),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )

    if ac.tp_out:
        bufs.mm2_prev_output_scaled = allocator.alloc_sbuf_tensor(
            shape=(ac.d, atp.sb_p),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )

        # PSUM allocations for tp_flash_attn_correction_factor_psum and tp_exp_sum_reciprocal_psum
        bufs.tp_flash_attn_correction_factor_psum = []
        bufs.tp_exp_sum_reciprocal_psum = []
        for grp_idx in range(atp.num_grps):
            tp_flash_attn_correction_factor_psum_tile = nl.ndarray(
                (ac.d, atp.sb_p),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, ((grp_idx % 2) * 4 + 3) * PSUM_BANK_SIZE),
            )
            bufs.tp_flash_attn_correction_factor_psum.append(tp_flash_attn_correction_factor_psum_tile)
            tp_exp_sum_reciprocal_psum_tile = nl.ndarray(
                (ac.d, atp.sb_p),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, ((grp_idx % 2) * 4 + 3) * PSUM_BANK_SIZE),
            )
            bufs.tp_exp_sum_reciprocal_psum.append(tp_exp_sum_reciprocal_psum_tile)

        bufs.tp_flash_attn_correction_factor_sb = allocator.alloc_sbuf_tensor(
            shape=(ac.d, atp.sb_p),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[4],
        )

        bufs.tp_exp_sum_reciprocal_sb = allocator.alloc_sbuf_tensor(
            shape=(ac.d, atp.sb_p),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[4],
        )

    if ac.tp_out:
        bufs.mm2_final = allocator.alloc_sbuf_tensor(
            shape=(ac.d, atp.sb_p),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )
    else:
        bufs.mm2_final = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, ac.d),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )

    if ac.tp_out:
        bufs.mm2_sb = allocator.alloc_sbuf_tensor(
            shape=(mm2_n, mm2_p),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )
    else:
        bufs.mm2_sb = allocator.alloc_sbuf_tensor(
            shape=(mm2_p, mm2_n),
            dtype=nl.float32,
            block_dim=[atp.num_grps],
            num_free_tiles=[2],
        )

    bufs.mm1_masked = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ),
        dtype=nl.float32,
        block_dim=[atp.num_grps, atp.num_large_tiles_per_section],
        num_free_tiles=[2, atp.num_large_tiles_per_section],
    )

    bufs.exp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ),
        dtype=nl.bfloat16,
        block_dim=[atp.num_grps, atp.num_large_tiles_per_section],
        num_free_tiles=([4, 2] if atp.use_swa_optimized_allocation else [1, atp.num_large_tiles_per_section]),
    )

    # mm1_psum PSUM allocation
    bufs.mm1_psum = []
    for grp_idx in range(atp.num_grps):
        grp_row = []
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            tile_row = []
            for k_tile_idx in range(4):
                mm1_psum_tile = nl.ndarray(
                    (mm1_p, mm1_n),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, (k_tile_idx % 4) * PSUM_BANK_SIZE),
                )
                tile_row.append(mm1_psum_tile)
            grp_row.append(tile_row)
        bufs.mm1_psum.append(grp_row)

    if not atp.dynamic_sel_mask:
        bufs.mm1_copy_sb = allocator.alloc_sbuf_tensor(
            shape=(mm1_p, mm1_n),
            dtype=nl.float32,
            block_dim=[atp.num_grps, atp.num_large_tiles_per_section, 4],
            num_free_tiles=[1, 1, 2],
        )

        bufs.mm1_affine_select_output = allocator.alloc_sbuf_tensor(
            shape=(mm1_p, mm1_n),
            dtype=nl.float32,
            block_dim=[atp.num_grps, atp.num_large_tiles_per_section, 4],
            num_free_tiles=[1, 1, 2],
        )

    bufs.exp_tp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.mm2_grp_sz),
        dtype=nl.bfloat16,
        block_dim=[
            atp.num_grps,
            atp.num_large_tiles_per_section,
            atp.num_tps_in_mm2_grp,
        ],
        num_free_tiles=(
            [4, 2, atp.num_tps_in_mm2_grp]
            if atp.use_swa_optimized_allocation
            else [2, atp.num_large_tiles_per_section, atp.num_tps_in_mm2_grp]
        ),
        align_to=32,  # align for dma transpose
    )

    # mm2_psum allocation
    bufs.mm2_psum = []
    for grp_idx in range(atp.num_grps):
        grp_row = []
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            if ac.tp_out:
                mm2_psum_tile = nl.ndarray(
                    (mm2_n, mm2_p),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, ((4 + (large_tile_idx % 4)) * PSUM_BANK_SIZE)),
                )
            else:
                mm2_psum_tile = nl.ndarray(
                    (mm2_p, mm2_n),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, ((4 + (large_tile_idx % 4)) * PSUM_BANK_SIZE)),
                )
            grp_row.append(mm2_psum_tile)
        bufs.mm2_psum.append(grp_row)


def _q_to_kv_batch_id(batch_id: int, bs: int, bs_kv: int) -> int:
    """Map Q batch id to KV batch id for native GQA support.

    Currently we implement native GQA support by simply using the correct KV batch id
    corresponding to the Q batch id but not attempting to optimize the KV loads themselves.
    We still get the benefit of not needing to replicate the KV before calling the kernel.

    Example: bs=6, bs_kv=2: mapping from batch_id_q -> batch_id_kv:
             {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    """
    return batch_id // (bs // bs_kv)


def _check_input_and_return_shape(
    q,
    k,
    v,
    is_prefix_caching,
    k_prior,
    v_prior,
    prior_used_len,
    tp_q,
    tp_k,
    tp_out,
    cache_softmax,
) -> tuple:
    """Validate input tensor shapes and compute output shapes.

    Check the shape of inputs base on kernel_name, and return the tuple,
    (seqlen_q, seqlen_k, seqlen_k_prior, d, out_shape, cache_softmax_shape)
    """
    if is_prefix_caching:
        kernel_assert(
            prior_used_len.shape == (1,),
            "Received unexpected shape for prior_used_len. "
            f"Expected (1,), received {prior_used_len.shape}. "
            "User note: prefix caching expects a single "
            "prior_used_len meaning it cannot be used "
            "if multiple requests (batch) are used with different "
            "prior_used_len values.",
        )
        assert_shape(
            prior_used_len,
            (1,),
            "prior_used_len",
            error_text="User note: prefix caching expects a single "
            "prior_used_len meaning it cannot be used "
            "if multiple requests (batch) are used with different "
            "prior_used_len values.",
        )

    if tp_q:
        batch_size, seqlen_q, d = q.shape
    else:
        batch_size, d, seqlen_q = q.shape
    seqlen_k_dim = 1 if tp_k else 2
    seqlen_k = k.shape[seqlen_k_dim]
    batch_size_kv = k.shape[0]
    if tp_k:
        assert_shape(k, (batch_size_kv, seqlen_k, d), "k")
    else:
        assert_shape(k, (batch_size_kv, d, seqlen_k), "k")
    assert_shape(v, (batch_size_kv, seqlen_k, d), "v")
    if is_prefix_caching:
        seqlen_k_prior = k_prior.shape[seqlen_k_dim]
        if tp_k:
            assert_shape(k_prior, (batch_size_kv, seqlen_k_prior, d), "k_prior")
        else:
            assert_shape(k_prior, (batch_size_kv, d, seqlen_k_prior), "k_prior")
        assert_shape(v_prior, (batch_size_kv, seqlen_k_prior, d), "v_prior")
    else:
        seqlen_k_prior = None

    out_seqlen = seqlen_q

    if tp_out:
        out_shape = (batch_size, d, out_seqlen)
    else:
        out_shape = (batch_size, out_seqlen, d)

    # compute the shape for cached softmax tensors, negative max and recipriocal sum
    cache_softmax_tile_size = 128
    if cache_softmax:
        # Current testing/golden does not account for the padded portion properly
        kernel_assert(
            seqlen_q % cache_softmax_tile_size == 0,
            f"For cache softmax, attention_cte currently expects seqlen_q multiple of {cache_softmax_tile_size}, got {seqlen_q=}",
        )
    padded_seq_grps = div_ceil(out_seqlen, cache_softmax_tile_size)
    cache_softmax_shape = [batch_size, cache_softmax_tile_size, padded_seq_grps]

    return (seqlen_q, seqlen_k, seqlen_k_prior, d, out_shape, cache_softmax_shape)


def _load_q_tile(q, out, tp_q, batch_id, grp_i, seqlen_offset, load_dtype, sbuf_addr, grps_per_load=1) -> int:
    """Load Q tile from HBM to SBUF, handling transpose based on tp_q flag.

    When tp_q is True, assumes q = (bs, seqlen, d).

    When tp_q is False, assumes q = (bs, d, seqlen), will perform transpose then save into out.

    out[grp_i // grps_per_load] has shape (d, 128 * grps_per_load)
    """
    kernel_assert(str(load_dtype) == str(out[0].dtype), "Conflicting dtype")
    local_allocator = ModularAllocator(initial_address=sbuf_addr)
    if tp_q:
        _, seqlen, d = q.shape
        if grps_per_load > 1:
            # Use DMA transpose
            num_p = min(seqlen - seqlen_offset, _Q_GRP_SZ * grps_per_load)

            # TODO: fix below check once proper type is available for tensor dtype
            if str(q.dtype) == str(nl.bfloat16):
                nisa.dma_transpose(
                    dst=out[grp_i // grps_per_load].ap([[_Q_GRP_SZ * grps_per_load, d], [1, 1], [1, 1], [1, num_p]]),
                    src=q.ap(
                        [[d, num_p], [1, 1], [1, 1], [1, d]],
                        offset=batch_id * seqlen * d + seqlen_offset * d,
                    ),
                )
            else:
                # Need a buffer with same dtype as q as dma_transpose requires same I/O dtype
                buffer = local_allocator.alloc_sbuf_tensor(
                    shape=(d, _Q_GRP_SZ * grps_per_load),
                    dtype=q.dtype,
                    align_to=32,  # align for dma transpose
                )

                nisa.dma_transpose(
                    dst=buffer.ap([[_Q_GRP_SZ * grps_per_load, d], [1, 1], [1, 1], [1, num_p]]),
                    src=q.ap(
                        [[d, num_p], [1, 1], [1, 1], [1, d]],
                        offset=batch_id * seqlen * d + seqlen_offset * d,
                    ),
                )

                nisa.tensor_copy(out[grp_i // grps_per_load][:, :num_p], buffer[:, :num_p])
        else:
            # Use NC transpose
            kernel_assert(
                grps_per_load == 1,
                "tp Q on Trn1/shard on seqlen does not yet support packed load",
            )
            loaded = local_allocator.alloc_sbuf_tensor(shape=(_Q_GRP_SZ, d), dtype=load_dtype)
            tp_dt = load_dtype
            psum_buf = nl.ndarray((d, _Q_GRP_SZ), dtype=tp_dt, buffer=nl.psum, address=(0, 0))

            num_p = min(seqlen - seqlen_offset, _Q_GRP_SZ)
            # Convert load() to access pattern
            # Original: load(dst=loaded[nl.ds(0, num_p), :], src=q[batch_id, nl.ds(seqlen_offset, _Q_GRP_SZ), 0:d], dtype=load_dtype)
            # q shape: (bs, seqlen, d), accessing q[batch_id, seqlen_offset:seqlen_offset+_Q_GRP_SZ, 0:d]
            # Pattern: [[d, num_p], [1, d]]
            # Offset: batch_id*seqlen*d + seqlen_offset*d
            loaded_dst_pat = loaded.ap(pattern=[[d, num_p], [1, d]], offset=0)
            q_src_pat = q.ap(
                pattern=[[d, num_p], [1, d]],
                offset=batch_id * seqlen * d + seqlen_offset * d,
            )
            nisa.dma_copy(dst=loaded_dst_pat, src=q_src_pat)

            nisa.nc_transpose(psum_buf[:d, :num_p], loaded[:num_p, :d])
            num_f = min(seqlen - seqlen_offset, _Q_GRP_SZ)
            nisa.tensor_copy(out[grp_i][:d, :num_f], psum_buf[:d, :num_f])
    else:
        _, d, seqlen = q.shape

        num_f = min(seqlen - seqlen_offset, _Q_GRP_SZ * grps_per_load)
        # Convert load() to access pattern
        # Original: load(dst=out[grp_i // grps_per_load][nl.ds(0, d), nl.ds(0, num_f)], src=q[batch_id, nl.ds(0, d), nl.ds(seqlen_offset, num_f)], dtype=load_dtype)
        # q shape: (bs, d, seqlen), accessing q[batch_id, 0:d, seqlen_offset:seqlen_offset+num_f]
        # Pattern: [[seqlen, d], [1, num_f]]
        # Offset: batch_id*d*seqlen + seqlen_offset
        out_dst_pat = out[grp_i // grps_per_load].ap(pattern=[[_Q_GRP_SZ * grps_per_load, d], [1, num_f]], offset=0)
        q_src_pat = q.ap(
            pattern=[[seqlen, d], [1, num_f]],
            offset=batch_id * d * seqlen + seqlen_offset,
        )
        nisa.dma_copy(dst=out_dst_pat, src=q_src_pat)


def _get_kv_tile_apc(
    is_prefix_caching,
    k_active,
    k_prior,
    seqlen_active,
    seqlen_prior,
    seqlen_offset,
    load_offset_active,
) -> tuple:
    """Determine which KV tensor (active or prior) to use based on sequence offset.

    Get information about KV tile (used for Prefix Caching)
    """
    if not is_prefix_caching:
        return k_active, seqlen_active, seqlen_offset, load_offset_active
    else:
        seqlen_prior_padded = align_to(seqlen_prior, _K_TILE_SZ)
        if seqlen_offset >= seqlen_prior_padded:
            return (
                k_active,
                seqlen_active,
                seqlen_offset - seqlen_prior_padded,
                load_offset_active,
            )
        else:
            return (
                k_prior,
                seqlen_prior,
                seqlen_offset,
                None,  # no load_offset used for prior
            )


def _load_k_tile(
    k_active,
    k_prior,
    out,
    batch_id,
    sp: SectionParams,
    load_dtype,
    tp_k,
    num_tiles,
    sbuf_addr,
    load_offset_active=None,
) -> int:
    """Load K tiles from HBM to SBUF in _K_TILE_SZ (512)-element chunks, handling transpose and prefix caching.

    k has shape
     (bs, d, seqlen) when tp_k=False
     (bs, seqlen, d) when tp_k=True, i.e. a transpose is performed
    k_prior (if passed) has shape identical to k except for the seqlen.
    Return shape of out[i] is (d, _K_TILE_SIZE) where i = 0..num_k_tiles_per_section
    """
    if tp_k:
        _, seqlen_active, _ = k_active.shape
    else:
        _, _, seqlen_active = k_active.shape
    seqlen_prior = None
    is_prefix_caching = k_prior is not None
    if is_prefix_caching:
        if tp_k:
            _, seqlen_prior, _ = k_prior.shape
        else:
            _, _, seqlen_prior = k_prior.shape
    if num_tiles > 0:
        d, n = out[0].shape
    sb_p = nl.tile_size.pmax
    stride_f = _K_TILE_SZ

    kernel_assert(n == _K_TILE_SZ, f"expect to load in tile of size {_K_TILE_SZ=}")
    kernel_assert(str(load_dtype) == str(out[0].dtype), "load dtype mismatch")
    local_allocator = ModularAllocator(initial_address=sbuf_addr)
    sbuf_addr_max = sbuf_addr
    if tp_k:
        sbuf_addr_outer = local_allocator.get_current_address()
        for tile in range(num_tiles):
            local_allocator.set_current_address(sbuf_addr_outer)

            # for APC we need to use either k or k_prior and appropriately adjust sequence length and other quantities
            k, seqlen, seqlen_offset, load_offset = _get_kv_tile_apc(
                is_prefix_caching,
                k_active,
                k_prior,
                seqlen_active,
                seqlen_prior,
                sp.section_offset + tile * n,
                load_offset_active,
            )

            if seqlen_offset >= seqlen:
                # since we always use section_len/512 tiles even for last section
                # we might exit early
                return sbuf_addr_max

            use_dma_tp = load_offset is None  # cannot use dma tp when using dynamic offset
            if use_dma_tp:
                num_p = min(seqlen - seqlen_offset, n)
                # TODO: fix below check once proper type is available for tensor dtype
                if str(k.dtype) == str(nl.bfloat16):
                    nisa.dma_transpose(
                        dst=out[tile].ap([[n, d], [1, 1], [1, 1], [1, num_p]]),
                        src=k.ap(
                            [[d, num_p], [1, 1], [1, 1], [1, d]],
                            offset=batch_id * seqlen * d + seqlen_offset * d,
                        ),
                    )
                else:
                    # Need a buffer with same dtype as k as dma_transpose requires same I/O dtype
                    buffer = local_allocator.alloc_sbuf_tensor(
                        shape=(d, n),
                        dtype=k.dtype,
                        align_to=32,  # align for dma transpose
                    )
                    sbuf_addr_max = max(sbuf_addr_max, local_allocator.get_current_address())
                    nisa.dma_transpose(
                        dst=buffer.ap([[n, d], [1, 1], [1, 1], [1, num_p]]),
                        src=k.ap(
                            [[d, num_p], [1, 1], [1, 1], [1, d]],
                            offset=batch_id * seqlen * d + seqlen_offset * d,
                        ),
                    )

                    nisa.tensor_copy(out[tile][:, :num_p], buffer[:, :num_p])
            else:  # not use_dma_tp
                num_pe_tps = _K_TILE_SZ // sb_p  # number of transposes (4)
                loaded = local_allocator.alloc_sbuf_tensor(shape=(sb_p, num_pe_tps, d), dtype=load_dtype)
                sbuf_addr_max = max(sbuf_addr_max, local_allocator.get_current_address())
                tp_dt = load_dtype
                psum_buf = nl.ndarray((d, num_pe_tps, sb_p), dtype=tp_dt, buffer=nl.psum, address=(0, 0))

                if load_offset is not None:
                    # NOTE: NKI is incapable of handling both dynamic and constant offsets in IndirectLoad, also it cannot handle
                    # dynamic offset used with more than one axis, so we must use four DMAs
                    for tp_idx in range(num_pe_tps):
                        num_p = min(seqlen - seqlen_offset - tp_idx * sb_p, sb_p)
                        if num_p > 0:
                            ind_offset = local_allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
                            sbuf_addr_max = max(sbuf_addr_max, local_allocator.get_current_address())
                            nisa.tensor_scalar(
                                ind_offset,
                                load_offset,
                                nl.add,
                                seqlen_offset + tp_idx * sb_p,
                            )
                            loaded_dst_pat = loaded.ap(
                                pattern=[[num_pe_tps * d, num_p], [1, d]],
                                offset=tp_idx * d,
                            )
                            k_src_pat = k.ap(
                                pattern=[[d, num_p], [1, d]],
                                scalar_offset=ind_offset,
                                offset=batch_id * seqlen * d,
                                indirect_dim=1,
                            )
                            nisa.dma_copy(dst=loaded_dst_pat, src=k_src_pat)
                else:  # not load_offset
                    # Use strided load to load four tiles of [128, d]
                    num_inner_f = min(div_ceil(seqlen - seqlen_offset, sb_p), num_pe_tps)
                    num_p = min(seqlen - seqlen_offset - num_inner_f * num_pe_tps, sb_p)

                    # Convert load() to access pattern with 2D mask
                    # Original: load(dst=loaded[...], src=k[batch_id, seqlen_offset + i_b*128 + i_p, i_f], mask=i_b*128+i_p < seqlen-seqlen_offset)
                    if seqlen_offset < seqlen:
                        # case 1: handle rectangular
                        # Offset: batch_id*seqlen*d + seqlen_offset*d
                        num_inner_f = min(num_pe_tps, (seqlen - seqlen_offset) // sb_p)
                        num_p = sb_p
                        loaded_dst_pat = loaded.ap(
                            pattern=[[num_pe_tps * d, num_p], [d, num_inner_f], [1, d]],
                            offset=0,
                        )

                        k_src_pat = k.ap(
                            pattern=[[d, num_p], [d * sb_p, num_inner_f], [1, d]],
                            offset=batch_id * seqlen * d + seqlen_offset * d,
                        )
                        nisa.dma_copy(dst=loaded_dst_pat, src=k_src_pat)
                        # case 2: handle last row
                        if num_inner_f < num_pe_tps and (seqlen - seqlen_offset) % sb_p != 0:
                            num_p = min(
                                sb_p,
                                seqlen - seqlen_offset - (seqlen - seqlen_offset) // sb_p * sb_p,
                            )
                            offset = batch_id * seqlen * d + seqlen_offset * d + num_inner_f * sb_p * d
                            loaded_dst_pat = loaded.ap(
                                pattern=[[num_pe_tps * d, num_p], [1, d]],
                                offset=num_inner_f * d,
                            )
                            k_src_pat = k.ap(pattern=[[d, num_p], [1, d]], offset=offset)
                            nisa.dma_copy(dst=loaded_dst_pat, src=k_src_pat)

                if seqlen_offset < seqlen:
                    # Transpose loaded[128, 4, d] with four PE transposes
                    for tp_idx in range(num_pe_tps):
                        num_p = min(seqlen - seqlen_offset - tp_idx * sb_p, sb_p)
                        if num_p > 0:
                            nisa.nc_transpose(
                                psum_buf[:d, tp_idx, :num_p],
                                loaded[:num_p, tp_idx, :d],
                            )

                    # Copy out transposed results
                    num_f = min(seqlen - seqlen_offset, n)
                    nisa.tensor_copy(
                        out[tile][:d, :num_f],
                        psum_buf.reshape((d, _K_TILE_SZ))[:d, :num_f],
                    )
    else:  # no tp k
        for tile in range(num_tiles):
            # for APC we need to use either k or k_prior and appropriately adjust sequence length and other quantities
            k, seqlen, seqlen_offset, load_offset = _get_kv_tile_apc(
                is_prefix_caching,
                k_active,
                k_prior,
                seqlen_active,
                seqlen_prior,
                sp.section_offset + tile * n,
                load_offset_active,
            )

            num_f = min(seqlen - seqlen_offset, n)
            if num_f > 0:
                if load_offset is not None:
                    # NOTE: NKI is incapable of handling both dynamic and constant offsets in IndirectLoad
                    ind_offset = local_allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
                    sbuf_addr_max = max(sbuf_addr_max, local_allocator.get_current_address())
                    nisa.tensor_scalar(ind_offset, load_offset, nl.add, seqlen_offset)

                    out_dst_pat = out[tile].ap(pattern=[[stride_f, d], [1, num_f]], offset=0)
                    k_src_pat = k.ap(
                        pattern=[[seqlen, d], [1, num_f]],
                        scalar_offset=ind_offset,
                        offset=batch_id * d * seqlen,
                        indirect_dim=2,
                    )
                    nisa.dma_copy(dst=out_dst_pat, src=k_src_pat)
                else:
                    # Convert load() to access pattern
                    # Original: load(dst=out[i][nl.ds(0, d), nl.ds(0, num_f)], src=k[batch_id, nl.ds(0, d), nl.ds(seqlen_offset, num_f)], dtype=load_dtype)
                    # k shape: (bs, d, seqlen), accessing k[batch_id, 0:d, seqlen_offset:seqlen_offset+num_f]
                    # Pattern: [[seqlen, d], [1, num_f]]
                    # Offset: batch_id*d*seqlen + seqlen_offset
                    out_dst_pat = out[tile].ap(pattern=[[stride_f, d], [1, num_f]], offset=0)
                    k_src_pat = k.ap(
                        pattern=[[seqlen, d], [1, num_f]],
                        offset=batch_id * d * seqlen + seqlen_offset,
                    )

                    nisa.dma_copy(dst=out_dst_pat, src=k_src_pat)

    return sbuf_addr_max


def _load_v_tile(
    v_active,
    v_prior,
    out,
    batch_id,
    sp: SectionParams,
    load_dtype,
    num_tiles,
    sbuf_addr,
    load_offset_active=None,
) -> int:
    """Load V tiles from HBM to SBUF in _V_TILE_SZ (128)-element chunks, handling prefix caching.

    - v of shape (bs, seqlen, d).
    - out[i] has shape (_V_TILE_SZ, d) where i = 0..num_v_tiles_per_section
    - v_prior (if passed) has shape identical to v except for the seqlen.
    """
    local_allocator = ModularAllocator(initial_address=sbuf_addr)
    _, seqlen_active, _ = v_active.shape
    seqlen_prior = None
    is_prefix_caching = v_prior is not None
    if is_prefix_caching:
        _, seqlen_prior, _ = v_prior.shape
    if num_tiles > 0:
        p, n = out[0].shape

    d = n
    kernel_assert(str(load_dtype) == str(out[0].dtype), "load dtype mismatch")

    for tile in range(num_tiles):
        v, seqlen, seqlen_offset, load_offset = _get_kv_tile_apc(
            is_prefix_caching,
            v_active,
            v_prior,
            seqlen_active,
            seqlen_prior,
            sp.section_offset + p * tile,
            load_offset_active,
        )
        num_p = min(seqlen - seqlen_offset, p)
        if num_p > 0:
            if load_offset is not None:
                # NOTE: NKI is incapable of handling both dynamic and constant offsets in IndirectLoad
                ind_offset = local_allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
                nisa.tensor_scalar(ind_offset, load_offset, nl.add, seqlen_offset)
                out_dst_pat = out[tile].ap(pattern=[[n, num_p], [1, n]], offset=0)
                v_src_pat = v.ap(
                    pattern=[[d, num_p], [1, n]],
                    scalar_offset=ind_offset,
                    offset=batch_id * seqlen * d,
                    indirect_dim=1,
                )
                nisa.dma_copy(dst=out_dst_pat, src=v_src_pat)
            else:
                # Convert load() to access pattern
                # Original: load(dst=out[i][nl.ds(0, num_p), nl.ds(0, n)], src=v[batch_id, nl.ds(seqlen_offset, num_p), nl.ds(0, n)], dtype=load_dtype)
                # v shape: (bs, seqlen, d), accessing v[batch_id, seqlen_offset:seqlen_offset+num_p, 0:n]
                # Pattern: [[d, num_p], [1, n]]
                # Offset: batch_id*seqlen*d + seqlen_offset*d
                out_dst_pat = out[tile].ap(pattern=[[n, num_p], [1, n]], offset=0)
                v_src_pat = v.ap(
                    pattern=[[d, num_p], [1, n]],
                    offset=batch_id * seqlen * d + seqlen_offset * d,
                )
                nisa.dma_copy(dst=out_dst_pat, src=v_src_pat)

    return local_allocator.get_current_address()


def _compute_section_offset_active(k_section_offset, is_prefix_caching, seqlen_k_prior_padded) -> tuple:
    """Compute active K offset and determine if section contains prefix data for proper masking."""
    section_contains_prefix = False
    k_section_offset_active = k_section_offset
    if is_prefix_caching:
        # in prefix caching case, for masking we fall back to causal=False case
        # unless we are in prior portion. Even for active portion, we need to
        # adjust the offset.
        if k_section_offset >= seqlen_k_prior_padded:
            k_section_offset_active = k_section_offset - seqlen_k_prior_padded
        else:
            section_contains_prefix = True
    return k_section_offset_active, section_contains_prefix


def _load_q_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
    q,
    batch_id,
    sbuf_addr,
):
    """Load Q group from HBM to SBUF if compute is needed for this group."""
    # Only load every num_q_grps_per_load grps
    if grp_i % atp.num_q_grps_per_load == 0:
        has_any_compute_pred = (
            _has_any_compute_causal(grp_i, sp.section_offset_active, ac, atp.num_q_grps_per_load)
            if (atp.is_causal and not sp.section_contains_prefix)
            else True
        )
        if has_any_compute_pred:
            q_seqlen_offset = grp_i * _Q_GRP_SZ
            _load_q_tile(
                q,
                bufs.q_sb,
                ac.tp_q,
                batch_id,
                grp_i,
                q_seqlen_offset,
                bufs.q_sb[0].dtype,
                sbuf_addr,
                grps_per_load=atp.num_q_grps_per_load,
            )


def _qk_and_max_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Compute QK^T matmul (MM1) and find row-wise maximum for this Q group.
    Also apply masking if relevant.
    """
    has_any_compute_pred = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    if has_any_compute_pred:
        nisa.memset(bufs.mm1_partial_max[grp_i], value=_FLOAT32_MIN)

        for large_tile_idx in range(atp.num_large_tiles_per_section):
            _qk_and_max_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)


def _update_max_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
    sink,
):
    """Update running maximum across sections and compute flash attention correction factor."""
    has_any_compute_pred = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    if not has_any_compute_pred:
        return

    # Step 1: Compute section max
    # If we have sink, need to include it in final max compute
    if (sink is not None) and (sp.section_idx == 0):
        nisa.tensor_copy(bufs.mm1_partial_max[grp_i][:, atp.num_k_tiles_per_section], bufs.sink_sb)

    nisa.tensor_reduce(
        bufs.mm1_section_max[grp_i][:, 0],
        nl.maximum,
        bufs.mm1_partial_max[grp_i],
        1,
        negate=True,
    )

    # Step 2: compute and store running max, and flash attention correction factor
    if atp.num_sections != 1:
        if sp.section_idx == 0:
            nisa.tensor_copy(bufs.mm1_running_max[:, grp_i], bufs.mm1_section_max[grp_i])
            nisa.memset(bufs.flash_attn_correction_factor[grp_i][...], value=0.0)
        if sp.section_idx > 0:
            nisa.activation(
                bufs.prev_mm1_running_max[grp_i][...],
                nl.copy,
                bufs.mm1_running_max[:, grp_i],
                scale=-1.0,
                bias=bufs.zero_bias_tensor,
            )
            nisa.tensor_tensor(
                bufs.mm1_running_max[:, grp_i],
                bufs.mm1_running_max[:, grp_i],
                bufs.mm1_section_max[grp_i],
                op=nl.minimum,
            )
            nisa.activation(
                bufs.flash_attn_correction_factor[grp_i][:, 0],
                nl.exp,
                bufs.prev_mm1_running_max[grp_i],
                bias=bufs.mm1_running_max[:, grp_i],
                scale=1.0,
            )
    else:
        nisa.tensor_copy(bufs.mm1_running_max[:, grp_i], bufs.mm1_section_max[grp_i])


def _exp_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
    sink,
):
    """Compute exponential of masked QK scores, accumulate sum, and perform transpose (required for MM2)."""
    has_any_compute_pred = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    if not has_any_compute_pred:
        return

    q_seqlen_offset = grp_i * atp.sb_p
    nisa.memset(bufs.exp_partial_sum[grp_i][...], value=0.0)

    for large_tile_idx in range(atp.num_large_tiles_per_section):
        kernel_assert(
            atp.exp_inst_elems == 512, "Internal validation failed."
        )  # prefix caching code assumes this currently, if we increase tile size to 2048, we will need to update logic

        for exp_tile_idx in range(atp.num_exp_insts_per_large_tile):
            is_prior_tile, seqlen_k, k_start_pos, _ = _get_kv_tile_apc(
                ac.is_prefix_caching,
                False,
                True,
                atp.seqlen_k_active_updated,
                ac.seqlen_k_prior,
                sp.section_offset + large_tile_idx * _LARGE_TILE_SZ + exp_tile_idx * atp.exp_inst_elems,
                None,
            )
            num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            num_f = min(seqlen_k - k_start_pos, atp.exp_inst_elems)

            q_start_pos = grp_i * _Q_GRP_SZ
            # Only produce matmul if the tile is in the lower triangle, use tile bot-left corner so adjust q
            if atp.is_causal and not is_prior_tile:
                exp_sel_mask = _has_any_compute_causal(grp_i, k_start_pos, ac)
            else:
                exp_sel_mask = True

            # If using SWA, also skip bot-left lower triangle.
            if ac.use_swa and atp.is_causal and not is_prior_tile:
                # Use tile top-right corner so adjust k; also adjust q for sliding window
                exp_sel_mask = exp_sel_mask and _has_any_compute_swa(grp_i, k_start_pos, atp.exp_inst_elems, ac)

            if exp_sel_mask and seqlen_k > k_start_pos:
                # Step 1: Compute exponential
                nisa.activation_reduce(
                    bufs.exp_sb[grp_i][large_tile_idx][:num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)],
                    op=nl.exp,
                    data=bufs.mm1_masked[grp_i][large_tile_idx][
                        :num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)
                    ],
                    reduce_op=nl.add,
                    reduce_res=bufs.exp_partial_sum[grp_i][
                        :num_p,
                        large_tile_idx * atp.num_exp_insts_per_large_tile + exp_tile_idx,
                    ],
                    bias=bufs.mm1_running_max[:num_p, grp_i],
                )

                # Step 2: Perform DMA transpose
                num_f_outer = num_f // atp.sb_p
                num_f_inner = num_f % atp.sb_p
                # split dma_transpose into two parts to satisfy API since we have both Q and K sequence masking
                # Focusing on exp_tp_sb which is arranged as [128, 4, 128] where each of the 4 [128, 128] blocks
                # share the same Q seqlen (on free dim) and cover 4 tiles of K seqlen (partition dim)
                # First region, we have num_f_outer [128, 128] blocks each having full partition dim (K) and each
                # accessing num_p (<128) on the free dim (Q).
                # Second region, we handle the remaining K (num_f_inner) - here we have the (num_f_outer+1)th [128,128]
                # block being utilized with num_f_inner access on partition dim and num_p on the free dim.

                # Example: num_f_outer = 3, num_f_inner = 33, num_p = 100
                # Region 1: AP: [[512, 128], [128, 3], [1, 100]] => a, b, c = np.mgrid[0:128, 0:3, 0:100]
                # Region 2: AP: [[512, 33], [128, 1], [1, 100]]  => a, b, c = np.mgrid[0:33, 0:1, 0:100] with offset 128 * 3

                # NOTE: we add the [1,1] because we need 4 dims for dma_transpose

                # Case 1: handle 0:128x
                if num_f_outer >= 1:
                    nisa.dma_transpose(
                        dst=bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                            [
                                [atp.mm2_grp_sz, atp.sb_p],
                                [1, 1],
                                [atp.sb_p, num_f_outer],
                                [1, num_p],
                            ]
                        ),
                        src=bufs.exp_sb[grp_i][large_tile_idx].ap(
                            [
                                [_LARGE_TILE_SZ, num_p],
                                [1, 1],
                                [atp.sb_p, num_f_outer],
                                [1, atp.sb_p],
                            ],
                            offset=exp_tile_idx * atp.mm2_grp_sz,
                        ),
                    )

                # Case 2: handle num_f - 128x
                if num_f_inner > 0:
                    nisa.dma_transpose(
                        dst=bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                            [
                                [atp.mm2_grp_sz, num_f_inner],
                                [1, 1],
                                [atp.sb_p, 1],
                                [1, num_p],
                            ],
                            offset=num_f_outer * atp.sb_p,
                        ),
                        src=bufs.exp_sb[grp_i][large_tile_idx].ap(
                            [
                                [_LARGE_TILE_SZ, num_p],
                                [1, 1],
                                [atp.sb_p, 1],
                                [1, num_f_inner],
                            ],
                            offset=exp_tile_idx * atp.mm2_grp_sz + num_f_outer * atp.sb_p,
                        ),
                    )

    # If there is sink, subtract max from it, then take its exp, then append it to sums
    if (sink is not None) and (sp.section_idx == 0):
        frs_sink_idx = bufs.exp_partial_sum[grp_i].shape[-1] - 1
        nisa.activation(
            bufs.exp_partial_sum[grp_i][:, frs_sink_idx],
            op=nl.exp,
            data=bufs.sink_sb,
            bias=bufs.mm1_running_max[:, grp_i],
        )


def _pv_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Compute score@value matmul (P@V, MM2) for this Q group."""
    has_any_compute_pred = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    if has_any_compute_pred:
        nisa.memset(bufs.mm2_sb[grp_i][...], value=0.0)

        for large_tile_idx in range(atp.num_large_tiles_per_section):
            _pv_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)


def _fused_qkmax_and_pv_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Fused implementation computing QK+max for group i+2 while computing PV for group i (software pipelining)."""
    qkmax_grp = grp_i + 2

    has_any_compute_pred_pv = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    has_any_compute_pred_qkmax = (
        _has_any_compute_causal(qkmax_grp, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    if has_any_compute_pred_qkmax:
        nisa.memset(bufs.mm1_partial_max[qkmax_grp][...], value=_FLOAT32_MIN)

    for large_tile_idx in range(atp.num_large_tiles_per_section):
        if has_any_compute_pred_pv:
            _pv_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)

        if has_any_compute_pred_qkmax:
            _qk_and_max_large_tile_impl(qkmax_grp, large_tile_idx, ac, atp, sp, bufs)


def _write_back_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
    o,
    batch_id,
):
    """Finalize output using flash attention: accumulate across sections, apply softmax normalization, write to HBM."""
    has_any_compute_pred = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    # if we have compute for this section but not the next section,
    # then this is the last section to have compute, and we need to
    # execute the final section logic to write final results
    next_has_compute_pred = (
        _has_any_compute_causal(grp_i, sp.next_section_offset_active, ac)
        if (atp.is_causal and not sp.next_section_contains_prefix)
        else True
    )
    is_last_section_with_compute = has_any_compute_pred and (not next_has_compute_pred)

    if not has_any_compute_pred:
        return

    # Step 1: Compute/update exp-sum and its reciprocal
    q_seqlen_offset = grp_i * atp.sb_p
    nisa.tensor_reduce(bufs.exp_section_sum[grp_i][...], nl.add, bufs.exp_partial_sum[grp_i], axis=1)
    if atp.num_sections != 1:
        if sp.section_idx == 0:
            nisa.tensor_copy(bufs.exp_running_sum[:, grp_i], bufs.exp_section_sum[grp_i])
        if sp.section_idx > 0:
            nisa.tensor_copy(
                bufs.prev_exp_running_sum[grp_i][...],
                bufs.exp_running_sum[:, grp_i],
            )
            nisa.tensor_scalar(
                bufs.exp_running_sum[:, grp_i],
                bufs.prev_exp_running_sum[grp_i][:, 0],
                nl.multiply,
                bufs.flash_attn_correction_factor[grp_i],
                op1=nl.add,
                operand1=bufs.exp_section_sum[grp_i],
            )
        if (sp.section_idx == atp.num_sections - 1) or is_last_section_with_compute:
            nisa.reciprocal(
                bufs.exp_sum_reciprocal[:, grp_i],
                bufs.exp_running_sum[:, grp_i],
            )
    else:
        nisa.reciprocal(bufs.exp_sum_reciprocal[:, grp_i], bufs.exp_section_sum[grp_i])

    num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
    num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

    """
    Step 2:
    - if first section:
      - if last section with compute:
        - write to HBM after scaling by reciprocal
      - else:
        - write to HBM
    - else:
      - Load previous output and apply flash attention correction and accumulate
      - if last section with compute:
        - write to HBM after scaling by reciprocal
      - write to HBM
    """
    if atp.num_sections != 1:
        if sp.section_idx == 0:
            if is_last_section_with_compute:
                # Last section, so scale by reciprocal and write current output to HBM
                _scale_reciprocal_write_back_impl(bufs.mm2_sb[grp_i], grp_i, ac, atp, bufs, o, batch_id, num_p, num_f)
            else:
                # Not last section, so just write current output to HBM
                _write_back_o_impl(bufs.mm2_sb[grp_i], grp_i, ac, atp, o, batch_id, num_p, num_f)

        if sp.section_idx > 0:
            # Load previous output scale by flash_attn_correction_factor and accumulate
            if ac.tp_out:
                # Original: load(dst=mm2_prev_output[grp_i], src=o[batch_id, nl.ds(0, d), nl.ds(grp_i*sb_p, 128)], mask=mm2_sbuf_f_mask)
                # o shape: (batch_size, d, seqlen), accessing o[batch_id, 0:d, grp_i*sb_p:grp_i*sb_p+num_f]
                # Offset: batch_id*d*seqlen_q + grp_i*sb_p
                prev_dst_pat = bufs.mm2_prev_output[grp_i].ap(pattern=[[atp.sb_p, ac.d], [1, num_f]], offset=0)
                o_src_pat = o.ap(
                    pattern=[[ac.seqlen_q, ac.d], [1, num_f]],
                    offset=batch_id * ac.d * ac.seqlen_q + grp_i * atp.sb_p,
                )
                nisa.dma_copy(dst=prev_dst_pat, src=o_src_pat)

                nisa.nc_transpose(
                    bufs.tp_flash_attn_correction_factor_psum[grp_i].ap(
                        pattern=[[atp.sb_p, ac.d], [1, atp.sb_p]], offset=0
                    ),
                    bufs.flash_attn_correction_factor[grp_i].ap(pattern=[[1, atp.sb_p], [0, ac.d]], offset=0),
                )
                nisa.tensor_copy(
                    bufs.tp_flash_attn_correction_factor_sb[grp_i][:, :num_f],
                    bufs.tp_flash_attn_correction_factor_psum[grp_i][:, :num_f],
                )
                nisa.tensor_tensor(
                    bufs.mm2_prev_output_scaled[grp_i][:, :num_f],
                    bufs.mm2_prev_output[grp_i][:, :num_f],
                    bufs.tp_flash_attn_correction_factor_sb[grp_i][:, :num_f],
                    nl.multiply,
                )
                nisa.tensor_tensor(
                    bufs.mm2_accum_flash_attn[grp_i][:, :num_f],
                    bufs.mm2_prev_output_scaled[grp_i][:, :num_f],
                    bufs.mm2_sb[grp_i][:, :num_f],
                    nl.add,
                )
            else:
                # Original: load(dst=mm2_prev_output[grp_i], src=o[batch_id, grp_i*sb_p+ip_o, if_o], mask=mm2_sbuf_p_mask)
                # o shape: (batch_size, seqlen, d), accessing o[batch_id, grp_i*sb_p:grp_i*sb_p+num_p, 0:num_f]
                # Offset: batch_id*seqlen_q*d + grp_i*sb_p*d
                prev_dst_pat = bufs.mm2_prev_output[grp_i].ap(pattern=[[ac.d, num_p], [1, ac.d]], offset=0)
                o_src_pat = o.ap(
                    pattern=[[ac.d, num_p], [1, ac.d]],
                    offset=batch_id * ac.seqlen_q * ac.d + grp_i * atp.sb_p * ac.d,
                )
                nisa.dma_copy(dst=prev_dst_pat, src=o_src_pat)
                nisa.scalar_tensor_tensor(
                    bufs.mm2_accum_flash_attn[grp_i][:num_p, : ac.d],
                    data=bufs.mm2_prev_output[grp_i][:num_p, : ac.d],
                    op0=nl.multiply,
                    operand0=bufs.flash_attn_correction_factor[grp_i][:num_p, 0],
                    op1=nl.add,
                    operand1=bufs.mm2_sb[grp_i][:num_p, : ac.d],
                )
            if sp.section_idx == atp.num_sections - 1 or is_last_section_with_compute:
                # Last section, so scale by reciprocal and write accumulated output to HBM
                _scale_reciprocal_write_back_impl(
                    bufs.mm2_accum_flash_attn[grp_i],
                    grp_i,
                    ac,
                    atp,
                    bufs,
                    o,
                    batch_id,
                    num_p,
                    num_f,
                )
            else:
                # Not last section, just write accumulated output to HBM
                _write_back_o_impl(
                    bufs.mm2_accum_flash_attn[grp_i],
                    grp_i,
                    ac,
                    atp,
                    o,
                    batch_id,
                    num_p,
                    num_f,
                )
    else:
        # Only one section, so scale by reciprocal and write current output to HBM
        _scale_reciprocal_write_back_impl(bufs.mm2_sb[grp_i], grp_i, ac, atp, bufs, o, batch_id, num_p, num_f)


def _scale_reciprocal_write_back_impl(
    src_buf,
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    bufs: AttnInternalBuffers,
    o,
    batch_id,
    num_p,
    num_f,
):
    """
    Write back o for the final section after multiplication by reciprocal. Transposes reciprocal if tp_out.
    """
    if ac.tp_out:
        # Original: tp_exp_sum_reciprocal_psum[grp_i] = nisa.nc_transpose(exp_sum_reciprocal[ip_broadcast, grp_i])
        nisa.nc_transpose(
            bufs.tp_exp_sum_reciprocal_psum[grp_i].ap(pattern=[[atp.sb_p, ac.d], [1, atp.sb_p]], offset=0),
            bufs.exp_sum_reciprocal.ap(pattern=[[atp.num_grps, atp.sb_p], [0, ac.d]], offset=grp_i),
        )
        nisa.tensor_copy(
            bufs.tp_exp_sum_reciprocal_sb[grp_i][: ac.d, :num_f],
            bufs.tp_exp_sum_reciprocal_psum[grp_i][: ac.d, :num_f],
        )
        nisa.tensor_tensor(
            bufs.mm2_final[grp_i][: ac.d, :num_f],
            src_buf[: ac.d, :num_f],
            bufs.tp_exp_sum_reciprocal_sb[grp_i][: ac.d, :num_f],
            nl.multiply,
        )
    else:
        nisa.activation(
            bufs.mm2_final[grp_i][:num_p, : ac.d],
            nl.copy,
            src_buf[:num_p, : ac.d],
            scale=bufs.exp_sum_reciprocal[:num_p, grp_i],
            bias=bufs.zero_bias_tensor[:num_p],
        )

    _write_back_o_impl(bufs.mm2_final[grp_i], grp_i, ac, atp, o, batch_id, num_p, num_f)


def _write_back_o_impl(src_buf, grp_i, ac: AttnConfig, atp: AttnTileParams, o, batch_id, num_p, num_f):
    """Helper function to write a source buffer to HBM output (o) with proper transpose handling.

    Args:
      src_buf: Source buffer in SBUF to copy from
      grp_i: Q group index
      ac: Attention configuration
      atp: Tile parameters
      o: Output HBM tensor
      batch_id: Batch index
      num_p: Number of partition elements
      num_f: Number of free elements
    """
    if ac.tp_out:
        # o shape: (batch_size, d, seqlen), accessing o[batch_id, 0:d, grp_i*sb_p:grp_i*sb_p+num_f]
        # Offset: batch_id*d*seqlen_q + grp_i*sb_p
        o_dst_pat = o.ap(
            pattern=[[ac.seqlen_q, ac.d], [1, num_f]],
            offset=batch_id * ac.d * ac.seqlen_q + grp_i * atp.sb_p,
        )
        src_pat = src_buf.ap(pattern=[[atp.sb_p, ac.d], [1, num_f]], offset=0)
    else:
        # o shape: (batch_size, seqlen, d), accessing o[batch_id, grp_i*sb_p:grp_i*sb_p+num_p, 0:d]
        # Offset: batch_id*seqlen_q*d + grp_i*sb_p*d
        o_dst_pat = o.ap(
            pattern=[[ac.d, num_p], [1, ac.d]],
            offset=batch_id * ac.seqlen_q * ac.d + grp_i * atp.sb_p * ac.d,
        )
        src_pat = src_buf.ap(pattern=[[ac.d, num_p], [1, ac.d]], offset=0)
    nisa.dma_copy(dst=o_dst_pat, src=src_pat)


def _qk_and_max_large_tile_impl(
    qkmax_grp,
    large_tile_idx,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Compute QK^T matmul (MM1) and find row-wise maximum for this Q group and large (2048) K tile.
    Also apply masking if relevant.
    """

    q_seqlen_offset = qkmax_grp * atp.sb_p

    # perform matmul and masking in 512 (_K_TILE_SZ) tile increment on the seqlen dimension
    num_k_tiles_in_large_tile = _LARGE_TILE_SZ // _K_TILE_SZ
    for k_tile_idx in range(num_k_tiles_in_large_tile):
        # Extract relevant tensor tiles for convenience
        mm1_psum_tile = bufs.mm1_psum[qkmax_grp][large_tile_idx][k_tile_idx]
        if not atp.dynamic_sel_mask:
            mm1_copy_sb_tile = bufs.mm1_copy_sb[qkmax_grp][large_tile_idx][k_tile_idx]
            mm1_affine_select_output_tile = bufs.mm1_affine_select_output[qkmax_grp][large_tile_idx][k_tile_idx]
        mm1_masked_tile = bufs.mm1_masked[qkmax_grp][large_tile_idx]
        mm1_partial_max_tile = bufs.mm1_partial_max[qkmax_grp]

        k_tile_idx_in_section = large_tile_idx * num_k_tiles_in_large_tile + k_tile_idx
        k_tile_idx_global = atp.num_k_tiles_per_section * sp.section_idx + k_tile_idx_in_section
        is_prior_tile, seqlen_k, k_start_pos, _ = _get_kv_tile_apc(
            ac.is_prefix_caching,
            False,
            True,
            atp.seqlen_k_active_updated,
            ac.seqlen_k_prior,
            k_tile_idx_global * _K_TILE_SZ,
            None,
        )

        if atp.is_causal and not is_prior_tile:
            # Only produce matmul if the tile is in the lower triangle, use tile bot-left corner so adjust q
            matmul_selection = _has_any_compute_causal(qkmax_grp, k_start_pos, ac)
            # If using SWA, also skip bot-left lower triangle.
            if ac.use_swa:
                # Use tile top-right corner so adjust k; also adjust q for sliding window
                matmul_selection = matmul_selection and _has_any_compute_swa(qkmax_grp, k_start_pos, _K_TILE_SZ, ac)
        else:
            matmul_selection = True

        if q_seqlen_offset >= ac.seqlen_q or k_start_pos >= seqlen_k:  # make sure we don't extend bound
            matmul_selection = False

        if matmul_selection and k_tile_idx_in_section < atp.num_k_tiles_per_section:
            num_f = min(seqlen_k - k_start_pos, _K_TILE_SZ)
            num_q_free = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

            # Step 1: MM1 matmul
            nisa.nc_matmul(
                mm1_psum_tile[:num_q_free, :num_f],
                bufs.q_sb[qkmax_grp // atp.num_q_grps_per_load][
                    : ac.d,
                    nl.ds((qkmax_grp % atp.num_q_grps_per_load) * _Q_GRP_SZ, num_q_free),
                ],
                bufs.k_sb[k_tile_idx_in_section][:, :num_f],
            )

            # Step 2: Masking
            num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            num_f = min(seqlen_k - k_start_pos, _K_TILE_SZ)

            diagonal_sel_mask = (
                matmul_selection and ((qkmax_grp * _Q_GRP_SZ) < (k_start_pos + _K_TILE_SZ))
                if (atp.is_causal and not is_prior_tile and not atp.dynamic_sel_mask)
                else False
            )
            if ac.use_swa and atp.is_causal and not is_prior_tile:
                # when we are using SWA, above condition for diagonal_sel_mask
                # might miss some conditions where masking needs to be applied
                # since it only checks for causal condition. Therefore we either
                # need dynamic mask or affine select mask
                diagonal_sel_mask = not atp.dynamic_sel_mask

            if diagonal_sel_mask:  # static diagonal mask
                # q_pos = qkmax_grp*sb_p + nl.arange(num_p)[:, None]
                # k_pos = k_start_pos + nl.arange(num_f)[None, :]

                # Mask off upper-triangle with affine_select
                # causal_pred = (q_pos >= k_pos)  # causal predicate preventing q tokens to look beyond
                # qkmax_grp*sb_p - k_start_pos
                nisa.tensor_copy(
                    mm1_copy_sb_tile[:num_p, :num_f],
                    mm1_psum_tile[:num_p, :num_f],
                )
                nisa.affine_select(
                    mm1_affine_select_output_tile[:num_p, :num_f],
                    pattern=[[-1, num_f]],
                    offset=qkmax_grp * atp.sb_p - k_start_pos,
                    channel_multiplier=1,
                    cmp_op=nl.greater_equal,
                    on_true_tile=mm1_copy_sb_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                )

                # Need extra affine_sel for smaller lower-triangle if use_swa (affine_sel cannot combination of masks)
                if ac.use_swa:
                    # swa_pred = (q_pos < k_pos + sliding_window) # k_pos + sliding_window - 1 >= q_pos
                    nisa.affine_select(
                        mm1_affine_select_output_tile[:num_p, :num_f],
                        pattern=[[1, num_f]],
                        offset=(k_start_pos + ac.sliding_window - 1 - qkmax_grp * atp.sb_p),
                        channel_multiplier=-1,
                        cmp_op=nl.greater_equal,
                        on_true_tile=mm1_affine_select_output_tile[:num_p, :num_f],
                        on_false_value=_FLOAT32_MIN,
                    )

                nisa.tensor_scalar_reduce(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    data=mm1_affine_select_output_tile[:num_p, :num_f],
                    op0=nl.multiply,
                    operand0=ac.scale,
                    reduce_op=nl.maximum,
                    reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx_in_section],
                )

            elif atp.dynamic_sel_mask or is_prior_tile:  # dynamic (compile-time unknown) mask
                if is_prior_tile:
                    bound0 = bufs.range_sel_lbs_prior[:num_p, qkmax_grp] if ac.use_swa else bufs.zero_bias_tensor
                    bound1 = bufs.range_sel_ubs_prior[:num_p, qkmax_grp]
                    comp_op1 = nl.less  # k < prior_used_len
                else:
                    bound0 = bufs.range_sel_lbs[:num_p, qkmax_grp] if ac.use_swa else bufs.zero_bias_tensor
                    bound1 = bufs.range_sel_ubs[:num_p, qkmax_grp]
                    comp_op1 = nl.less_equal  # k <= q + cp_offset

                kernel_assert(ac.scale == 1.0, "range_select path doesn't support scale != 1.0")
                nisa.range_select(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    on_true_tile=mm1_psum_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                    comp_op0=nl.greater_equal,
                    comp_op1=comp_op1,
                    bound0=bound0,
                    bound1=bound1,
                    reduce_op=nl.maximum,
                    reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx_in_section],
                    range_start=k_start_pos,
                )

            else:  # no masking
                nisa.tensor_scalar_reduce(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    data=mm1_psum_tile[:num_p, :num_f],
                    op0=nl.multiply,
                    operand0=ac.scale,
                    reduce_op=nl.maximum,
                    reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx_in_section],
                )


def _pv_large_tile_impl(
    pv_grp,
    large_tile_idx,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Perform MM2 (P@V) matmul for the Q grp and large (2048) V tile."""

    q_seqlen_offset = pv_grp * atp.sb_p
    num_mm2_grps_in_large_tile = _LARGE_TILE_SZ // atp.mm2_grp_sz
    mm2_psum_set = False  # track if matmul happens so we can skip later step

    mm2_psum_tile = bufs.mm2_psum[pv_grp][large_tile_idx]

    # Step 1: Perform matmul and accumulate in PSUM for each group
    for mm2_grp_i in range(num_mm2_grps_in_large_tile):
        num_mm2_per_grp = atp.mm2_grp_sz // _V_TILE_SZ
        num_mm2_per_large_tile = num_mm2_per_grp * num_mm2_grps_in_large_tile

        exp_tp_sb_tile = bufs.exp_tp_sb[pv_grp][large_tile_idx][mm2_grp_i]

        is_prior_tile, seqlen_k, k_start_pos_512_tile, _ = _get_kv_tile_apc(
            ac.is_prefix_caching,
            False,
            True,
            atp.seqlen_k_active_updated,
            ac.seqlen_k_prior,
            sp.section_offset + large_tile_idx * _LARGE_TILE_SZ + mm2_grp_i * atp.mm2_grp_sz,
            None,
        )

        for mm2_i in range(num_mm2_per_grp):
            v_tile_idx = large_tile_idx * num_mm2_per_large_tile + mm2_grp_i * num_mm2_per_grp + mm2_i
            k_start_pos = k_start_pos_512_tile + mm2_i * _V_TILE_SZ

            num_p = min(seqlen_k - k_start_pos, _V_TILE_SZ)
            num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

            # Only produce matmul if the tile is in the lower triangle, use tile bot-left corner so adjust q
            mm2_sel_mask = (
                _has_any_compute_causal(pv_grp, k_start_pos, ac) if (atp.is_causal and not is_prior_tile) else True
            )
            # If using SWA, also skip bot-left lower triangle.
            if ac.use_swa and atp.is_causal and not is_prior_tile:
                # Use tile top-right corner so adjust k; also adjust q for sliding window
                mm2_sel_mask = mm2_sel_mask and _has_any_compute_swa(pv_grp, k_start_pos, _V_TILE_SZ, ac)

            if mm2_sel_mask and v_tile_idx < atp.num_v_tiles_per_section and num_p > 0 and num_f > 0:
                mm2_psum_set = True
                # src partition mask: (k_start_pos+nl.arange(128)[:, None]<seqlen_k)
                # exp_tp_sb free mask: if_l_mm2 + q_seqlen_offset < seqlen_q
                if ac.tp_out:
                    nisa.nc_matmul(
                        mm2_psum_tile[: ac.d, :num_f],
                        bufs.v_sb[v_tile_idx][:num_p, : ac.d],
                        exp_tp_sb_tile[:num_p, nl.ds(mm2_i * _V_TILE_SZ, num_f)],
                    )
                else:
                    nisa.nc_matmul(
                        mm2_psum_tile[:num_f, : ac.d],
                        exp_tp_sb_tile[:num_p, nl.ds(mm2_i * _V_TILE_SZ, num_f)],
                        bufs.v_sb[v_tile_idx][:num_p, : ac.d],
                    )

    # Step 2: accumulate the MM2 groups (reduction dim 512) into the large tile reduction dim 2048

    # check if first 512 tile in the large tile (2048) is a prior tile, in that case we always need to
    # apply loop_reduce_sel.
    is_prior_tile, seqlen_k, k_start_pos, _ = _get_kv_tile_apc(
        ac.is_prefix_caching,
        False,
        True,
        atp.seqlen_k_active_updated,
        ac.seqlen_k_prior,
        sp.section_offset + large_tile_idx * _LARGE_TILE_SZ,
        None,
    )
    k_seqlen_mask = k_start_pos < seqlen_k
    if is_prior_tile:
        kernel_assert(
            k_seqlen_mask, "Internal validation failed."
        )  # if first 512 tile is prior, we must have k_start_pos<seqlen_k since prior is only padded to 512 max

    # Only produce matmul if the tile is in the lower triangle, use tile bot-left corner so adjust q
    loop_reduce_sel = (
        _has_any_compute_causal(pv_grp, k_start_pos, ac) if (atp.is_causal and not is_prior_tile) else True
    )
    # If using SWA, also skip bot-left lower triangle.
    if ac.use_swa and atp.is_causal and not is_prior_tile:
        # Use tile top-right corner so adjust k; also adjust q for sliding window
        loop_reduce_sel = loop_reduce_sel and _has_any_compute_swa(pv_grp, k_start_pos, _LARGE_TILE_SZ, ac)

    if loop_reduce_sel and k_seqlen_mask and mm2_psum_set:
        if ac.tp_out:
            num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            if large_tile_idx == 0:
                nisa.tensor_copy(bufs.mm2_sb[pv_grp][:, :num_f], mm2_psum_tile[:, :num_f])
            else:
                nisa.tensor_tensor(
                    bufs.mm2_sb[pv_grp][:, :num_f],
                    bufs.mm2_sb[pv_grp][:, :num_f],
                    mm2_psum_tile[:, :num_f],
                    nl.add,
                )
        else:
            num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            if (
                large_tile_idx == 0
            ):  # when using swa, this is going to be skipped, which is why we require mm2_sb to be zero-memset
                nisa.tensor_copy(bufs.mm2_sb[pv_grp][:num_p, :], mm2_psum_tile[:num_p, :])
            else:
                nisa.tensor_tensor(
                    bufs.mm2_sb[pv_grp][:num_p, :],
                    bufs.mm2_sb[pv_grp][:num_p, :],
                    mm2_psum_tile[:num_p, :],
                    nl.add,
                )


def _has_any_compute_causal(q_grp: int, k_start_pos: int, ac: AttnConfig, num_grps: int = 1):
    """
    Return true if the given q group has any compute (i.e., we cannot fully mask out)
    for the provided k start position, based on the causal mask.

    :param q_grp: q group index
    :param k_start_pos: start pos of k
    """
    # We can completely eliminate compute when even the largest q in the group is
    # less than the smallest k in tile
    max_q_in_grp = q_grp * _Q_GRP_SZ + _Q_GRP_SZ * num_grps - 1
    if ac.cp_strided_q_slicing:
        # multiply by stride (global_cp_deg) and add global_cp_deg - 1
        # since we need to account for worst case (rank_id = global_cp_deg - 1)
        max_q_in_grp = max_q_in_grp * ac.global_cp_deg + ac.global_cp_deg - 1
    min_k_in_tile = k_start_pos
    return max_q_in_grp >= min_k_in_tile


def _has_any_compute_swa(q_grp: int, k_start_pos: int, k_tile_size: int, ac: AttnConfig):
    """
    Return true if the given q group has any compute (i.e., we cannot fully mask out)
    for the provided k start position, based on the sliding window mask.

    :param q_grp: q group index
    :param k_start_pos: start pos of k
    :param k_tile_size: tile size of k
    :param ac: AttnConfig
    """
    # We can completely eliminate compute when when even the smallest q in tile is
    # >= largest k in tile + sw
    min_q_in_grp = q_grp * _Q_GRP_SZ
    if ac.cp_strided_q_slicing:
        # multiply by stride (global_cp_deg). For min q we can assume rank_id = 0
        min_q_in_grp = min_q_in_grp * ac.global_cp_deg
    max_k_in_tile = k_start_pos + k_tile_size - 1
    return min_q_in_grp < max_k_in_tile + ac.sliding_window
