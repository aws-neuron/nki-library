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

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.integration.nkilib.utils.test_kernel_common import convert_to_torch
from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import Any, Optional, Tuple, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.attention.attention_bwd import attention_bwd
from nkilib_src.nkilib.core.attention.attention_bwd_torch import attention_bwd_torch_ref, compute_o_lse
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil

PMAX = 128


def generate_sequence_packing_bounds(cu_seqlens_q, cu_seqlens_k, total_q: int):
    """Expand cu_seqlens into per-token bound_min/bound_max float32 vectors."""
    bound_min = np.zeros(total_q, dtype=np.float32)
    bound_max = np.zeros(total_q, dtype=np.float32)
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        bound_min[q_start:q_end] = int(cu_seqlens_k[i])
        bound_max[q_start:q_end] = int(cu_seqlens_k[i + 1])
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
    sliding_window: int = -1,
    num_sinks: int = 0,
    softmax_scale: Optional[float] = None,
    seqlens_list: Optional[list] = None,
) -> dict:
    import torch

    generate_tensor = gaussian_tensor_generator()

    # Sequence packing: derive seq_len and bound vectors from seqlens_list
    bound_min, bound_max = None, None
    if seqlens_list is not None:
        seq_len = sum(seqlens_list)
        cu_seqlens = np.concatenate([[0], np.cumsum(seqlens_list)]).astype(np.int32)
        bound_min, bound_max = generate_sequence_packing_bounds(cu_seqlens, cu_seqlens, seq_len)

    q = generate_tensor(name="q", shape=(batch_size, num_q_heads, head_dim, seq_len), dtype=dtype)
    k = generate_tensor(name="k", shape=(batch_size, num_kv_heads, head_dim, seq_len), dtype=dtype)
    v = generate_tensor(name="v", shape=(batch_size, num_kv_heads, head_dim, seq_len), dtype=dtype)
    dy = generate_tensor(name="dy", shape=(batch_size, num_q_heads, head_dim, seq_len), dtype=dtype)

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
@pytest_test_metadata(
    name="Attention backward",
    pytest_marks=["attention", "bwd"],
)
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
        (1, 32, 32, 4096, 128, nl.bfloat16, True, None, 0),  # deepseek-v3.2 TP=4
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
    ]

    # ===== Sliding window configs (4) =====
    test_cases_sliding_window = [
        (1, 64, 8, 4096, 64, nl.bfloat16, True, 128, 0),      # gpt-oss-20b (window=128)
        (1, 64, 8, 4096, 64, nl.bfloat16, True, 128, 1),      # gpt-oss-20b (window=128, sinks=1)
        (1, 32, 16, 8192, 128, nl.bfloat16, True, 4096, 0),   # gemma2-27b (window=4096)
        (1, 16, 8, 8192, 256, nl.bfloat16, True, 4096, 0),    # gemma2-9b (window=4096)
    ]

    attention_bwd_test_cases = test_cases_4K + test_cases_8K + test_cases_misc + test_cases_sliding_window
    # fmt: on

    base_test_cases = [
        pytest.param(batch_size, num_heads, num_heads, 2048, 128, dtype, causal, None, num_sinks)
        for batch_size in [1, 2]
        for num_heads in [1, 3]
        for dtype in [nl.bfloat16, nl.float32]
        for causal in [True, False]
        for num_sinks in [0, 2]
    ]
    negative_test_cases = [
        pytest.param(1, 2, 2, 2040, 128, nl.bfloat16, True, None, 0),  # seq_len should be multiple of 128
        pytest.param(1, 3, 2, 2048, 128, nl.bfloat16, True, None, 0),  # num_q_heads should be divisible by num_heads_kv
        pytest.param(
            1, 2, 2, 2048, 260, nl.bfloat16, True, None, 0
        ),  # head_dim can't be tiled equally among minimum number of tiles (here 3)
        pytest.param(
            1, 2, 2, 2048, 128, nl.bfloat16, False, 128, 0
        ),  # Sliding window is supported for causal attn only
    ]
    fast_test_cases = base_test_cases + negative_test_cases

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
    ) -> None:
        np.random.seed(0)

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
            )

        def output_tensors(kernel_input):
            q = kernel_input["q_ref"]
            k = kernel_input["k_ref"]
            result = {
                "out_dq_ref": np.zeros(q.shape, q.dtype),
                "out_dk_ref": np.zeros(k.shape, k.dtype),
                "out_dv_ref": np.zeros(k.shape, k.dtype),
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

    @pytest.mark.fast
    @pytest_parametrize(attention_bwd_params, fast_test_cases, abbrevs=_ABBREVS)
    def test_attention_bwd_fast(
        self,
        test_manager: Orchestrator,
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
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count)
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
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count)
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

    # ===== Sequence Packing =====
    # fmt: off
    sequence_packing_params = "num_q_heads, num_kv_heads, head_dim, dtype, seqlens_list, causal, num_sinks, sliding_window, neg_case"
    _SEQ_PACK_ABBREVS = {
        "num_q_heads": "qh", "num_kv_heads": "kvh", "head_dim": "dh", "dtype": "dt",
        "seqlens_list": "seqs", "causal": "causal", "num_sinks": "sinks",
        "sliding_window": "sw", "neg_case": "neg",
    }

    sequence_packing_test_cases = [
        # Basic
        pytest.param(1, 1, 128, nl.bfloat16, [128, 128],       True,  0, None, None, marks=pytest.mark.fast),  # two equal seqs, causal
        pytest.param(1, 1, 128, nl.bfloat16, [256, 256],       True,  0, None, None, marks=pytest.mark.fast),  # larger equal seqs
        pytest.param(1, 1, 64,  nl.bfloat16, [128, 256, 128],  True,  0, None, None, marks=pytest.mark.fast),  # three seqs, varying lengths
        pytest.param(1, 1, 128, nl.bfloat16, [512],            True,  0, None, None, marks=pytest.mark.fast),  # single seq
        pytest.param(1, 1, 128, nl.bfloat16, [128, 384],       True,  0, None, None, marks=pytest.mark.fast),  # unequal lengths
        pytest.param(1, 1, 128, nl.float32, [128, 128],       False, 0, None, None, marks=pytest.mark.fast),  # non-causal, float32
        pytest.param(1, 1, 64,  nl.bfloat16, [128, 256, 128],  False, 0, None, None, marks=pytest.mark.fast),  # non-causal, varying lengths
        # Sink
        (2, 2, 64, nl.bfloat16, [2048, 2000, 48], True, 1, None, None),  # sink + sequence packing
        # GQA
        pytest.param(8, 1, 128, nl.bfloat16, [256, 256], True,  0, None, None, marks=pytest.mark.fast),   # GQA factor 8
        pytest.param(4, 2, 128, nl.bfloat16, [512, 512], True,  0, None, None, marks=pytest.mark.fast),   # GQA factor 2
        (8, 8, 128, nl.bfloat16, [256, 256], True,  0, None, None),   # MHA
        # Seqlen stress
        (1, 1, 128, nl.bfloat16, [128] * 16,           True, 0, None, None),   # many short seqs, total=2048
        (1, 1, 128, nl.bfloat16, [4096, 512],          True, 0, None, None),   # one long + one short, total=4608
        (1, 1, 128, nl.bfloat16, [512, 4096],          True, 0, None, None),   # one short + one long, total=4608
        (1, 1, 128, nl.bfloat16, [2048] * 4,           True, 0, None, None),   # multi-section, total=8192
        (1, 1, 128, nl.bfloat16, [127, 129, 255, 513], True, 0, None, None),   # non-power-of-2 lengths, total=1024
        # Larger tests
        (16, 2, 128, nl.bfloat16, [1024, 1072]+[500]*4,   True, 0, None, None),   # qwen3-32b TP=4, 4096
        (16, 8, 256, nl.float32,  [1000]*8+[192],         True, 0, None, None),   # gemma2-9b (dtype=float32), 8192
        # Sliding window + sequence packing
        pytest.param(1, 1, 128, nl.bfloat16, [256, 256], True, 0, 128, None, marks=pytest.mark.fast),   # SWA=128, two seqs
        pytest.param(1, 1, 128, nl.bfloat16, [128, 384], True, 0, 64,  None, marks=pytest.mark.fast),   # SWA=64, unequal
        (1, 1, 128, nl.bfloat16, [512, 512],             True, 0, 256, None),   # SWA=256, equal seqs
        (1, 1, 128, nl.bfloat16, [2048] * 4,             True, 0, 512, None),   # SWA=512, multi-section
        # Negative
        pytest.param(1, 1, 128, nl.bfloat16, [128, 128], True, 0, None, "only_bound_min", marks=pytest.mark.fast, id="only_bound_min"),
        pytest.param(1, 1, 128, nl.bfloat16, [128, 128], True, 0, None, "wrong_dtype",    marks=pytest.mark.fast, id="wrong_dtype"),
    ]    # fmt: on

    @pytest_parametrize(sequence_packing_params, sequence_packing_test_cases, abbrevs=_SEQ_PACK_ABBREVS)
    def test_attention_bwd_sequence_packing(self, test_manager, num_q_heads, num_kv_heads, head_dim, dtype, seqlens_list, causal, num_sinks, sliding_window, neg_case):
        """Test attention_bwd with sequence packing."""
        inputs = generate_inputs(batch_size=1, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
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
            test_config=None, compiler_args=CompilerArgs(enable_birsim=False, logical_nc_config=2 if num_kv_heads % 2 == 0 else 1),
            rtol=2e-2, atol=1e-5, is_negative_test=neg_case is not None)
