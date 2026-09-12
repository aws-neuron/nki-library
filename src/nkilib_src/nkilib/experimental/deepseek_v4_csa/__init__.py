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

"""DeepSeek-V4 Compressed Sparse Attention (CSA) kernels for Trainium3.

CSA replaces attention's full comparison against every past token with a
selection. The model keeps a compressed KV cache, a small "lightning indexer"
scores every compressed position, and the attention reads only the
``index_topk`` highest-scoring positions plus a local sliding window. The compute
cost of the attention body is therefore O(``window_size`` + ``index_topk``) and
does not grow with the context length -- at a 32K context one decode block takes
0.337 ms on Trainium3 in BF16.

Layout
------
``csa_common``
    Config dataclasses and the host-side tables (RoPE, window bias) the kernels
    take as inputs.
``csa_decode_attention``
    Decode kernels, headlined by ``nki_indexer_score_topk_gather_2core`` -- the
    fused megakernel that runs the indexer score, the GpSimd top-k and the O(k)
    sparse attention in a single ``[2]``-grid launch.
``csa_prefill_attention``
    Prefill kernels: the fused RMS+RoPE projection tail, the compressor, the
    indexer's bisection top-k mask, and the two sparse-attention variants.
``csa_tp_all_reduce``
    The 2-LNC ``ncc.all_reduce`` that sums the head-parallel output partials
    across tensor-parallel ranks.
``csa_block``
    The composition layer: complete prefill and decode attention blocks, plus a
    runnable driver that grades them against ``csa_block_torch``.

These kernels use ``priority=`` DMA class-of-service hints, which are
NeuronCore-v4 only, so they target trn3.

Each module's own docstring carries the design rationale for what it holds: why the
sequence rather than the head axis is split below the rank boundary, why
``name=`` on a ``shared_hbm`` allocation is load-bearing on a ``[2]``-grid kernel,
and how the snake layout that ``nisa.topk`` requires is assembled.
"""

from .csa_common import (
    CSAConfig,
    CSAConfigFull,
    precompute_freqs_cos_sin,
    precompute_win_bias_parts,
    shard_for_tp,
)
from .csa_decode_attention import (
    nisa_topk_snake_kernel,
    nki_decode_gather_ok_kernel,
    nki_indexer_qproj_gemv,
    nki_indexer_score_2core,
    nki_indexer_score_kernel,
    nki_indexer_score_topk_2core,
    nki_indexer_score_topk_gather_2core,
    nki_indexer_score_topk_kernel,
    nki_qkv_rms_rope_kernel,
)
from .csa_prefill_attention import (
    nki_compressor_core_kernel,
    nki_fused_csa_attn_kernel,
    nki_gather_csa_attn_kernel,
    nki_indexer_score_mask_kernel,
    nki_rms_rope_kernel,
)
from .csa_tp_all_reduce import TPAllReduceNKI, nki_tp_all_reduce_kernel, tp_all_reduce

__all__ = [
    "CSAConfig",
    "CSAConfigFull",
    "TPAllReduceNKI",
    "nisa_topk_snake_kernel",
    "nki_compressor_core_kernel",
    "nki_decode_gather_ok_kernel",
    "nki_fused_csa_attn_kernel",
    "nki_gather_csa_attn_kernel",
    "nki_indexer_qproj_gemv",
    "nki_indexer_score_2core",
    "nki_indexer_score_kernel",
    "nki_indexer_score_mask_kernel",
    "nki_indexer_score_topk_2core",
    "nki_indexer_score_topk_gather_2core",
    "nki_indexer_score_topk_kernel",
    "nki_qkv_rms_rope_kernel",
    "nki_rms_rope_kernel",
    "nki_tp_all_reduce_kernel",
    "precompute_freqs_cos_sin",
    "precompute_win_bias_parts",
    "shard_for_tp",
    "tp_all_reduce",
]
