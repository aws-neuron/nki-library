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
Tests for attention_bwd kernel implementation
with various sequence length and head configurations.
"""

from typing import Any, Optional, Tuple, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.attention.attention_bwd import attention_bwd
from nkilib_src.nkilib.core.attention.attention_bwd_torch import attention_bwd_torch_ref, compute_o_lse
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.integration.nkilib.utils.test_kernel_common import convert_to_torch
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

PMAX = 128


def generate_sequence_packing_bounds(cu_seqlens_q, cu_seqlens_k, total_q: int, batch_size: int):
    """Expand cu_seqlens into per-token bound_min/bound_max float32 arrays.

    Args:
        cu_seqlens_q: List of lists, one per batch, of cumulative sequence lengths for Q.
        cu_seqlens_k: List of lists, one per batch, of cumulative sequence lengths for K.
        total_q: Total query sequence length.
        batch_size: Number of batches.

    Returns:
        bound_min, bound_max: Arrays of shape (batch_size, total_q).
    """
    bound_min = np.zeros((batch_size, total_q), dtype=np.float32)
    bound_max = np.zeros((batch_size, total_q), dtype=np.float32)
    for b in range(batch_size):
        for i in range(len(cu_seqlens_q[b]) - 1):
            q_start, q_end = int(cu_seqlens_q[b][i]), int(cu_seqlens_q[b][i + 1])
            bound_min[b, q_start:q_end] = int(cu_seqlens_k[b][i])
            bound_max[b, q_start:q_end] = int(cu_seqlens_k[b][i + 1])
    return bound_min, bound_max


def generate_inputs(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: Any,
    causal: bool,
    mixed_precision: bool = True,
    sliding_window: Optional[int] = -1,
    num_sinks: int = 0,
    softmax_scale: Optional[float] = None,
    seqlens_list: Optional[list] = None,
    transpose_dv: bool = False,
    cp_offset: int = 0,
    seq_len_k: Optional[int] = None,
    head_dim_v: Optional[int] = None,
) -> dict:
    import torch

    generate_tensor = gaussian_tensor_generator()
    head_dim_v = head_dim_v if head_dim_v is not None else head_dim

    # Sequence packing: derive seq_len and bound vectors from seqlens_list
    bound_min, bound_max = None, None
    if seqlens_list is not None:
        seq_len = sum(seqlens_list)
        # Generate per-batch cu_seqlens by rotating seqlens_list for each batch
        cu_seqlens = []
        for b in range(batch_size):
            rotated = seqlens_list[b % len(seqlens_list) :] + seqlens_list[: b % len(seqlens_list)]
            cu_seqlens.append(np.concatenate([[0], np.cumsum(rotated)]).astype(np.int32))
        bound_min, bound_max = generate_sequence_packing_bounds(cu_seqlens, cu_seqlens, seq_len, batch_size)

    actual_seq_len_k = seq_len_k if seq_len_k is not None else seq_len

    q = generate_tensor(name="q", shape=(batch_size, num_q_heads, head_dim, seq_len), dtype=dtype)
    k = generate_tensor(name="k", shape=(batch_size, num_kv_heads, head_dim, actual_seq_len_k), dtype=dtype)
    v = generate_tensor(name="v", shape=(batch_size, num_kv_heads, head_dim_v, actual_seq_len_k), dtype=dtype)
    dy = generate_tensor(name="dy", shape=(batch_size, num_q_heads, head_dim_v, seq_len), dtype=dtype)

    sinks = None
    if num_sinks > 0:
        sinks_shape = (1, num_q_heads) if num_sinks == 1 else (1, num_q_heads, num_sinks)
        sinks = np.repeat(generate_tensor(name="sinks", shape=sinks_shape, dtype=dtype), batch_size, axis=0)

    o_proj, lse, _ = compute_o_lse(
        convert_to_torch(q),
        convert_to_torch(k),
        convert_to_torch(v),
        causal,
        mixed_precision,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        sinks=convert_to_torch(sinks) if num_sinks > 0 else None,
        bound_min=torch.from_numpy(bound_min) if bound_min is not None else None,
        bound_max=torch.from_numpy(bound_max) if bound_max is not None else None,
        cp_offset=cp_offset,
    )
    o_ref = dt.static_cast(o_proj.float().numpy(), dtype)
    lse_ref = lse.float().numpy()

    result = {
        "q_ref": q,
        "k_ref": k,
        "v_ref": v,
        "o_ref": o_ref,
        "dy_ref": dy,
        "lse_ref": lse_ref,
        "sinks_ref": sinks if num_sinks > 0 else None,
        "use_causal_mask": causal,
        "mixed_precision": mixed_precision,
        "softmax_scale": softmax_scale,
        "sliding_window": sliding_window,
        "bound_min": bound_min,
        "bound_max": bound_max,
        "transpose_dv": transpose_dv,
        "cp_offset": cp_offset,
    }
    return result


def is_negative_test_case(
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    sliding_window: Optional[int],
) -> Tuple[bool, Optional[str]]:
    """Check if a test case is a negative (expected to fail) test case."""
    if seq_len % 128 != 0:
        return True, f"seq_len ({seq_len}) should be multiple of 128"

    if num_q_heads % num_kv_heads != 0:
        return True, (f"num_q_heads ({num_q_heads}) should be divisible by num_kv_heads ({num_kv_heads})")

    min_tiles_needed = div_ceil(head_dim, PMAX)
    if head_dim % min_tiles_needed != 0:
        return True, (
            f"head_dim ({head_dim}) can't be tiled equally among minimum number of tiles ({min_tiles_needed})"
        )

    if sliding_window is not None and not causal:
        return True, "Sliding window is supported for causal attention only"

    return False, None


@final
@pytest_test_metadata(name="Attention backward")
@pytest_marks(["attention", "bwd"])
class TestAttentionBwdKernel:
    """Test class for attention bwd kernel"""

    _heavy_mark = pytest.mark.xdist_group("heavy_attn_bwd")

    # fmt: off
    attention_bwd_params = "batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, causal, sliding_window, num_sinks"
    _ABBREVS = {
        "batch_size": "b", "num_q_heads": "qh", "num_kv_heads": "kvh",
        "seq_len": "s", "head_dim": "dh", "dtype": "dt", "causal": "causal",
        "sliding_window": "sw", "num_sinks": "sinks",
    }

    # ===== 4K seq_len configs (11) =====
    test_cases_4K = [
        (1, 16, 8, 4096, 128, nl.bfloat16, True, None, 0),   # qwen3-1.7b
        (1, 32, 4, 4096, 128, nl.bfloat16, True, None, 0),   # qwen3-moe
        (1, 32, 8, 4096, 128, nl.bfloat16, True, None, 0),   # qwen3-8b
        (1, 64, 8, 4096, 128, nl.bfloat16, True, None, 0),   # qwen3-32b
        (1, 4, 2, 4096, 128, nl.bfloat16, True, None, 0),    # qwen3-1.7b TP=4
        (1, 8, 1, 4096, 128, nl.bfloat16, True, None, 0),    # qwen3-moe TP=4
        (1, 8, 2, 4096, 128, nl.bfloat16, True, None, 0),    # qwen3-8b TP=4
        (1, 16, 2, 4096, 128, nl.bfloat16, True, None, 0),   # qwen3-32b TP=4
        (1, 32, 32, 4096, (192, 128), nl.bfloat16, True, None, 0),  # deepseek-v3 MLA TP=4
        (1, 64, 8, 4096, 64, nl.bfloat16, True, None, 0),    # gpt-oss-20b
        (1, 64, 8, 4096, 64, nl.bfloat16, True, None, 1),    # gpt-oss-20b (sinks=1)
    ]

    # ===== 8K seq_len configs (7) =====
    test_cases_8K = [
        pytest.param(1, 32, 8, 8192, 128, nl.bfloat16, True, None, 0, marks=_heavy_mark),   # llama3.1-8b
        (1, 8, 2, 8192, 128, nl.bfloat16, True, None, 0),    # llama3.1-8b TP=4
        (1, 8, 4, 8192, 128, nl.bfloat16, True, None, 0),    # gemma2-27b TP=4
        (1, 16, 2, 8192, 128, nl.bfloat16, True, None, 0),   # llama3-70b TP=4
        (1, 16, 8, 8192, 256, nl.bfloat16, True, None, 0),   # gemma2-9b
        pytest.param(1, 32, 8, 8192, 64, nl.bfloat16, True, None, 0, marks=_heavy_mark),    # llama3.2-1b
        pytest.param(1, 32, 32, 8192, 128, nl.bfloat16, True, None, 0, marks=_heavy_mark),  # MHA baseline
    ]

    # ===== Misc configs (5) =====
    test_cases_misc = [
        (1, 12, 12, 1024, 64, nl.bfloat16, True, None, 0),    # gpt-2 (small seqlen)
        (1, 16, 16, 1024, 64, nl.bfloat16, False, None, 0),   # modernbert (non-causal)
        (1, 8, 1, 8192, 96, nl.bfloat16, True, None, 0),      # (head_dim=96, MQA)
        (1, 16, 8, 8192, 256, nl.float32, True, None, 0),     # gemma2-9b (dtype=float32)
        pytest.param(1, 8, 2, 16384, 128, nl.bfloat16, True, None, 0, marks=_heavy_mark),    # phi-4 (16K seqlen, reduced heads)
        pytest.param(1, 2, 2, 8704, 128, nl.bfloat16, True, None, 0, marks=_heavy_mark),     # seqlen_k not a multiple of k_seq_section_len (8192 + 512 tail): regression for double-buffered KV prefetch dma_copy size mismatch
    ]

    # ===== Sliding window configs (4) =====
    test_cases_sliding_window = [
        (1, 64, 8, 4096, 64, nl.bfloat16, True, 128, 0),      # gpt-oss-20b (window=128)
        (1, 64, 8, 4096, 64, nl.bfloat16, True, 128, 1),      # gpt-oss-20b (window=128, sinks=1)
        (1, 32, 16, 8192, 128, nl.bfloat16, True, 4096, 0),   # gemma2-27b (window=4096)
        (1, 16, 8, 8192, 256, nl.bfloat16, True, 4096, 0),    # gemma2-9b (window=4096)
    ]

    # ===== Ring attention baseline comparison (2) =====
    test_cases_ring_baseline = [
        pytest.param(1, 2, 2, 8192, 128, nl.bfloat16, True, None, 0, id="ring_baseline_causal"),
        pytest.param(1, 2, 2, 8192, 128, nl.bfloat16, False, None, 0, id="ring_baseline_nocausal"),
    ]

    attention_bwd_test_cases = test_cases_4K + test_cases_8K + test_cases_misc + test_cases_sliding_window + test_cases_ring_baseline

    base_test_cases = [
        pytest.param(
            batch_size,
            num_heads,
            num_heads,
            2048,
            128,
            dtype,
            causal,
            None,
            num_sinks,
            marks=(
                pytest.mark.fast
                if (batch_size, num_heads, dtype, causal, num_sinks) == (1, 1, nl.float32, True, 2)
                else ()
            ),
        )
        for batch_size in [1, 2]
        for num_heads in [1, 3]
        for dtype in [nl.bfloat16, nl.float32]
        for causal in [True, False]
        for num_sinks in [0, 2]
    ]
    negative_test_cases = [
        pytest.param(1, 2, 2, 2040, 128, nl.bfloat16, True, None, 0),  # seq_len should be multiple of 128
        pytest.param(1, 3, 2, 2048, 128, nl.bfloat16, True, None, 0),  # num_q_heads should be divisible by num_heads_kv
        pytest.param(1, 2, 2, 2048, 260, nl.bfloat16, True, None, 0, marks=pytest.mark.fast),  # head_dim can't be tiled equally among minimum number of tiles (here 3)
        pytest.param(1, 2, 2, 2048, 128, nl.bfloat16, False, 128, 0, marks=pytest.mark.fast),  # Sliding window is supported for causal attn only
    ]
    unit_test_cases = base_test_cases + negative_test_cases
    # fmt: on

    def _run_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: Any,
        causal: bool,
        sliding_window: Optional[int],
        num_sinks: int,
        softmax_scale: Optional[float],
        transpose_dv: bool = False,
    ) -> None:
        np.random.seed(0)

        # Support asymmetric head dims: head_dim can be (d_qk, d_v) tuple
        if isinstance(head_dim, tuple):
            head_dim, head_dim_v = head_dim
        else:
            head_dim_v = head_dim

        is_negative, _ = is_negative_test_case(num_q_heads, num_kv_heads, seq_len, head_dim, causal, sliding_window)

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(
                batch_size=batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                causal=causal,
                sliding_window=sliding_window,
                num_sinks=num_sinks,
                softmax_scale=softmax_scale,
                transpose_dv=transpose_dv,
                head_dim_v=head_dim_v,
            )

        def output_tensors(kernel_input):
            q = kernel_input["q_ref"]
            k = kernel_input["k_ref"]
            bs_k, nheads_kv_k, d_head_k, seqlen_k = k.shape
            if transpose_dv:
                dv_shape = (bs_k, seqlen_k, nheads_kv_k, d_head_k)
            else:
                dv_shape = k.shape
            result = {
                "out_dq_ref": np.zeros(q.shape, q.dtype),
                "out_dk_ref": np.zeros(k.shape, k.dtype),
                "out_dv_ref": np.zeros(dv_shape, k.dtype),
            }
            if kernel_input["sinks_ref"] is not None:
                sinks = kernel_input["sinks_ref"]
                result["out_dsinks_ref"] = np.zeros(sinks.shape, sinks.dtype)
            return result

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_bwd,
            torch_ref=torch_ref_wrapper(attention_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2,
            atol=1e-5,
            is_negative_test=is_negative,
        )

    @pytest_parametrize(attention_bwd_params, unit_test_cases, abbrevs=_ABBREVS)
    def test_attention_bwd_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: Any,
        causal: bool,
        sliding_window: Optional[int],
        num_sinks: int,
    ) -> None:
        """Test attention backward kernel with minimal configurations."""
        softmax_scale = 1.0 if causal else None
        lnc_count = 2 if batch_size * num_kv_heads % 2 == 0 else 1
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count, platform_target=platform_target)
        self._run_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            causal=causal,
            sliding_window=sliding_window,
            num_sinks=num_sinks,
            softmax_scale=softmax_scale,
        )

    @pytest_parametrize(attention_bwd_params, attention_bwd_test_cases, abbrevs=_ABBREVS)
    def test_attention_bwd(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: Any,
        causal: bool,
        sliding_window: Optional[int],
        num_sinks: int,
    ) -> None:
        """Test attention backward kernel with various configurations."""
        lnc_count = 2 if batch_size * num_kv_heads % 2 == 0 else 1
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count, platform_target=platform_target)
        self._run_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            causal=causal,
            sliding_window=sliding_window,
            num_sinks=num_sinks,
            softmax_scale=None,
        )

    # ===== Transpose dV =====
    # fmt: off
    transpose_dv_params = "batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, causal, sliding_window, num_sinks"
    _TRANSPOSE_DV_ABBREVS = {
        "batch_size": "b", "num_q_heads": "qh", "num_kv_heads": "kvh",
        "seq_len": "s", "head_dim": "dh", "dtype": "dt", "causal": "causal",
        "sliding_window": "sw", "num_sinks": "sinks",
    }

    transpose_dv_fast_test_cases = [
        pytest.param(1, 1, 1, 2048, 128, nl.float32, True, None, 0, marks=pytest.mark.fast),
    ]

    transpose_dv_test_cases = [
        (1, 32, 32, 4096, 128, nl.bfloat16, True, None, 0),    # MHA 32h 4k
        (1, 32, 8, 4096, 128, nl.bfloat16, True, None, 0),     # GQA qwen3-8b
        (1, 8, 2, 4096, 128, nl.bfloat16, True, None, 0),      # GQA qwen3-8b TP=4
        (1, 8, 2, 8192, 128, nl.bfloat16, True, None, 0),      # GQA llama3.1-8b TP=4
        (1, 12, 12, 1024, 64, nl.bfloat16, True, None, 0),     # gpt-2
        (1, 64, 8, 4096, 64, nl.bfloat16, True, 128, 0),       # sliding window=128
        (1, 32, 16, 8192, 128, nl.bfloat16, True, 4096, 0),    # sliding window=4096
        (1, 64, 8, 4096, 64, nl.bfloat16, True, None, 1),      # sinks=1
        (1, 64, 8, 4096, 64, nl.bfloat16, True, 128, 1),       # sliding window + sinks
        (1, 1, 1, 2048, 128, nl.bfloat16, True, None, 0),
        (1, 1, 1, 2048, 128, nl.bfloat16, False, None, 0),
        (2, 3, 3, 2048, 128, nl.bfloat16, True, None, 0),
    ]
    # fmt: on

    @pytest_parametrize(
        transpose_dv_params, transpose_dv_fast_test_cases + transpose_dv_test_cases, abbrevs=_TRANSPOSE_DV_ABBREVS
    )
    def test_attention_bwd_transpose_dv(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: Any,
        causal: bool,
        sliding_window: Optional[int],
        num_sinks: int,
    ) -> None:
        """Test attention backward kernel with transpose_dv=True (output in [bs, s, h, d] layout)."""
        lnc_count = 2 if batch_size * num_kv_heads % 2 == 0 else 1
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count, platform_target=platform_target)
        self._run_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            causal=causal,
            sliding_window=sliding_window,
            num_sinks=num_sinks,
            softmax_scale=None,
            transpose_dv=True,
        )

    # ===== Sequence Packing =====
    # fmt: off
    sequence_packing_params = "batch_size, num_q_heads, num_kv_heads, head_dim, dtype, seqlens_list, causal, num_sinks, sliding_window, neg_case"
    _SEQ_PACK_ABBREVS = {
        "batch_size": "b", "num_q_heads": "qh", "num_kv_heads": "kvh", "head_dim": "dh", "dtype": "dt",
        "seqlens_list": "seqs", "causal": "causal", "num_sinks": "sinks",
        "sliding_window": "sw", "neg_case": "neg",
    }

    sequence_packing_test_cases = [
        # Basic (bs=1)
        pytest.param(1, 1, 1, 128, nl.bfloat16, [128, 128],       True,  0, None, None),  # two equal seqs, causal
        pytest.param(1, 1, 1, 128, nl.bfloat16, [256, 256],       True,  0, None, None),  # larger equal seqs
        pytest.param(1, 1, 1, 64,  nl.bfloat16, [128, 256, 128],  True,  0, None, None),  # three seqs, varying lengths
        pytest.param(1, 1, 1, 128, nl.bfloat16, [512],            True,  0, None, None),  # single seq
        pytest.param(1, 1, 1, 128, nl.bfloat16, [128, 384],       True,  0, None, None),  # unequal lengths
        pytest.param(1, 1, 1, 128, nl.float32, [128, 128],       False, 0, None, None, marks=pytest.mark.fast),  # non-causal, float32
        pytest.param(1, 1, 1, 64,  nl.bfloat16, [128, 256, 128],  False, 0, None, None),  # non-causal, varying lengths
        # Sink
        (1, 2, 2, 64, nl.bfloat16, [2048, 2000, 48], True, 1, None, None),  # sink + sequence packing
        # GQA (bs=1)
        pytest.param(1, 8, 1, 128, nl.bfloat16, [256, 256], True,  0, None, None),   # GQA factor 8
        pytest.param(1, 4, 2, 128, nl.bfloat16, [512, 512], True,  0, None, None),   # GQA factor 2
        (1, 8, 8, 128, nl.bfloat16, [256, 256], True,  0, None, None),   # MHA
        # Seqlen stress (bs=1)
        (1, 1, 1, 128, nl.bfloat16, [128] * 16,           True, 0, None, None),   # many short seqs, total=2048
        (1, 1, 1, 128, nl.bfloat16, [4096, 512],          True, 0, None, None),   # one long + one short, total=4608
        (1, 1, 1, 128, nl.bfloat16, [512, 4096],          True, 0, None, None),   # one short + one long, total=4608
        (1, 1, 1, 128, nl.bfloat16, [2048] * 4,           True, 0, None, None),   # multi-section, total=8192
        (1, 1, 1, 128, nl.bfloat16, [127, 129, 255, 513], True, 0, None, None),   # non-power-of-2 lengths, total=1024
        # Larger tests (bs=1)
        (1, 16, 2, 128, nl.bfloat16, [1024, 1072]+[500]*4,   True, 0, None, None),   # qwen3-32b TP=4, 4096
        (1, 16, 8, 256, nl.float32,  [1000]*8+[192],         True, 0, None, None),   # gemma2-9b (dtype=float32), 8192
        # Sliding window + sequence packing (bs=1)
        pytest.param(1, 1, 1, 128, nl.bfloat16, [256, 256], True, 0, 128, None, marks=pytest.mark.fast),   # SWA=128, two seqs
        pytest.param(1, 1, 1, 128, nl.bfloat16, [128, 384], True, 0, 64,  None),   # SWA=64, unequal
        (1, 1, 1, 128, nl.bfloat16, [512, 512],             True, 0, 256, None),   # SWA=256, equal seqs
        (1, 1, 1, 128, nl.bfloat16, [2048] * 4,             True, 0, 512, None),   # SWA=512, multi-section
        # Multi-batch (bs>1) — each batch gets rotated seqlens for different masks
        pytest.param(2, 1, 1, 128, nl.bfloat16, [128, 384],       True,  0, None, None),  # bs=2, basic causal
        pytest.param(2, 1, 1, 128, nl.bfloat16, [128, 256, 128],  False, 0, None, None),  # bs=2, non-causal
        pytest.param(3, 1, 1, 128, nl.bfloat16, [768, 256],       True,  0, None, None),  # bs=3
        pytest.param(2, 8, 1, 128, nl.bfloat16, [256, 768],       True,  0, None, None),  # bs=2, GQA factor 8
        pytest.param(2, 4, 2, 128, nl.bfloat16, [512, 512],       True,  0, None, None, marks=pytest.mark.fast),  # bs=2, GQA factor 2
        pytest.param(2, 1, 1, 128, nl.bfloat16, [128, 896],       True, 0, 128, None),   # bs=2, SWA
        pytest.param(4, 1, 1, 128, nl.bfloat16, [128, 256, 128],  True, 0, None, None),  # bs=4, three seqs
        (4, 2, 2, 64,  nl.bfloat16, [2048, 2000, 48], True, 1, None, None),  # bs=4, sink + seq packing
        # Negative
        pytest.param(1, 1, 1, 128, nl.bfloat16, [128, 128], True, 0, None, "only_bound_min", marks=pytest.mark.fast, id="only_bound_min"),
        pytest.param(1, 1, 1, 128, nl.bfloat16, [128, 128], True, 0, None, "wrong_dtype",    marks=pytest.mark.fast, id="wrong_dtype"),
    ]    # fmt: on

    @pytest_parametrize(sequence_packing_params, sequence_packing_test_cases, abbrevs=_SEQ_PACK_ABBREVS)
    def test_attention_bwd_sequence_packing(self, test_manager: Orchestrator, platform_target: Platforms, batch_size, num_q_heads, num_kv_heads, head_dim, dtype, seqlens_list, causal, num_sinks, sliding_window, neg_case):
        """Test attention_bwd with sequence packing."""
        inputs = generate_inputs(batch_size=batch_size, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                                 seq_len=0, head_dim=head_dim, dtype=dtype, causal=causal,
                                 seqlens_list=seqlens_list, num_sinks=num_sinks,
                                 sliding_window=sliding_window if sliding_window else -1)

        if neg_case == "only_bound_min":
            inputs.pop("bound_max")
        elif neg_case == "wrong_dtype":
            inputs["bound_min"] = inputs["bound_min"].astype(np.int32)
            inputs["bound_max"] = inputs["bound_max"].astype(np.int32)

        def input_generator(test_config, input_tensor_def=None):
            return inputs

        def output_tensors(kernel_input):
            q, k = kernel_input["q_ref"], kernel_input["k_ref"]
            result = {"out_dq_ref": np.zeros(q.shape, q.dtype),
                      "out_dk_ref": np.zeros(k.shape, k.dtype),
                      "out_dv_ref": np.zeros(k.shape, k.dtype)}
            if kernel_input.get("sinks_ref") is not None:
                result["out_dsinks_ref"] = np.zeros(kernel_input["sinks_ref"].shape, kernel_input["sinks_ref"].dtype)
            return result

        UnitTestFramework(test_manager=test_manager, kernel_entry=attention_bwd,
                          torch_ref=torch_ref_wrapper(attention_bwd_torch_ref),
                          kernel_input_generator=input_generator,
                          output_tensor_descriptor=output_tensors).run_test(
            test_config=None, compiler_args=CompilerArgs(enable_birsim=False, logical_nc_config=2 if batch_size * num_kv_heads % 2 == 0 else 1, platform_target=platform_target),
            rtol=2e-2, atol=1e-5, is_negative_test=neg_case is not None)

    # ===== Context Parallelism (cp_offset) =====
    # fmt: off
    cp_offset_params = "batch_size, num_q_heads, num_kv_heads, seq_len_q, seq_len_k, head_dim, dtype, sliding_window, cp_offset"
    _CP_OFFSET_ABBREVS = {
        "batch_size": "b", "num_q_heads": "qh", "num_kv_heads": "kvh",
        "seq_len_q": "sq", "seq_len_k": "sk", "head_dim": "dh", "dtype": "dt",
        "sliding_window": "sw", "cp_offset": "cp",
    }

    cp_offset_test_cases = [
        # Simulates CP shard 1 in sliding window ring attention:
        # Q is local chunk (512 tokens), K is prev_window + local (1024 tokens), cp_offset=512
        pytest.param(1, 1, 1, 512, 1024, 128, nl.bfloat16, 512, 512),
        pytest.param(1, 2, 2, 512, 1024, 128, nl.bfloat16, 512, 512),
        pytest.param(1, 8, 2, 512, 1024, 128, nl.bfloat16, 512, 512),   # GQA
        # Larger sequence chunks
        (1, 4, 2, 1024, 2048, 128, nl.bfloat16, 1024, 1024),
        (1, 8, 2, 2048, 4096, 128, nl.bfloat16, 2048, 2048),
        # Different cp_offset values
        pytest.param(1, 1, 1, 512, 1024, 128, nl.bfloat16, 512, 256, marks=pytest.mark.fast),
        # Multi-batch
        pytest.param(2, 2, 2, 512, 1024, 128, nl.bfloat16, 512, 512),
        # Equal Q/K lengths with offset — use smaller cp_offset so sliding window still covers valid K range
        pytest.param(1, 1, 1, 512, 512, 128, nl.bfloat16, 512, 256),
        # seqlen_k (8704 = 8192 + 512 tail) not a multiple of k_seq_section_len: regression for
        # double-buffered KV prefetch dma_copy size mismatch in ring sliding-window backward.
        pytest.param(1, 1, 1, 512, 8704, 128, nl.bfloat16, 512, 512, id="prefetch_tail_8704"),
    ]
    # fmt: on

    @pytest_parametrize(cp_offset_params, cp_offset_test_cases, abbrevs=_CP_OFFSET_ABBREVS)
    def test_attention_bwd_cp_offset(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        head_dim: int,
        dtype: Any,
        sliding_window: int,
        cp_offset: int,
    ) -> None:
        """Test attention backward kernel with cp_offset for context parallelism sliding window ring attention."""
        np.random.seed(0)

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(
                batch_size=batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                seq_len=seq_len_q,
                head_dim=head_dim,
                dtype=dtype,
                causal=True,
                sliding_window=sliding_window,
                cp_offset=cp_offset,
                seq_len_k=seq_len_k,
            )

        def output_tensors(kernel_input):
            q = kernel_input["q_ref"]
            k = kernel_input["k_ref"]
            return {
                "out_dq_ref": np.zeros(q.shape, q.dtype),
                "out_dk_ref": np.zeros(k.shape, k.dtype),
                "out_dv_ref": np.zeros(k.shape, k.dtype),
            }

        lnc_count = 2 if batch_size * num_kv_heads % 2 == 0 else 1
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count, platform_target=platform_target)
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_bwd,
            torch_ref=torch_ref_wrapper(attention_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2,
            atol=1e-5,
            is_negative_test=False,
        )
