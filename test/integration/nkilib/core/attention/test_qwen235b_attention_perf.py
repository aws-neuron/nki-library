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

"""Qwen3-235B-A22B BF16 core-attention (flash attention) FORWARD perf.

Companion to the MXFP8 matmul suites (test_qwen235b_perf.py /
test_qwen235b_backward_perf.py), which measure the QKV/O projection matmuls and
explicitly scope core attention out. This suite measures the fused flash-attention
kernel (attention_cte) that computes softmax(scale * Q@K^T) @ V for the same
Qwen3-235B-A22B training config.

Operating point (matches the projection suites): Attention TP4, sequence length
4096, no context parallelism (CP1). After TP4 head sharding of the model dims
(n_q_heads=64, n_kv_heads=4, head_dim=128) each device sees:

    bs=16 Q heads, bs_kv=1 KV head (GQA factor 16), d=128,
    seqlen_q = seqlen_kv = 4096, causal mask, bf16, LNC2.

Heads are folded into the batch dim, so attention_cte takes no separate num_heads
arg. Because the kernel is BF16 there are no prequant / scale-packing arms (unlike
the MXFP8 matmul kernel) -- this is a single production forward arm. Perf
(MFU / MBU / latency / cycles) is recorded automatically by the metrics collector.

The dims come from get_attention_config() so this stays in sync with the shared
attention model-config source (test_attention_cte_model_config.py).
"""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.attention.attention_bwd import attention_bwd
from nkilib_src.nkilib.core.attention.attention_bwd_torch import attention_bwd_torch_ref

from test.integration.nkilib.core.attention.test_attention_bwd import (
    generate_inputs,
    is_negative_test_case,
)
from test.integration.nkilib.core.attention.test_attention_cte import (
    CAUSAL_MASK_DIM_NAME,
    SINK_DIM_NAME,
    TP_K_DIM_NAME,
    TP_OUT_DIM_NAME,
    TP_Q_DIM_NAME,
)
from test.integration.nkilib.core.attention.test_attention_cte import (
    # Aliased so pytest does not re-collect the whole attention-CTE suite into this
    # module: any module-level name matching `Test*` is collected, so importing the
    # class unaliased duplicates every CTE test here and the duplicate pairs collide
    # on a shared artifacts directory under xdist. We only need its helper method.
    TestRangedAttentionCTEKernels as _RangedAttentionCTEKernels,
)
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# ============================================================================
# Operating point: Attention TP4, CP1, sequence length 4096 (the training config
# the QKV/O projection suites are also pinned to). CP1 => tp_k=True, no CP slicing.
# ============================================================================

_MODEL = "qwen3_235b"
_TP = 4
_CP = 1
_SEQLEN = 4096

# get_attention_config lives in the model-config module, which raises ImportError under
# SKIP_MODEL_TESTS (set in some CI builds). Guard it so this module still imports there —
# the tests then collect with an empty parameter set (skipped) instead of erroring.
try:
    from test.integration.nkilib.core.attention.test_attention_cte_model_config import (
        get_attention_config,
    )

    _QWEN235B_ATTN_CONFIGS = [get_attention_config(_MODEL, tp=_TP, cp=_CP, seqlen=_SEQLEN)]
except ImportError:
    _QWEN235B_ATTN_CONFIGS = []

_QWEN235B_ATTN_IDS = [
    f"tp{_TP}_cp{_CP}_s{_SEQLEN}_bs{c['bs']}_bskv{c['bs'] // c['gqa_factor']}_d{c['d']}" for c in _QWEN235B_ATTN_CONFIGS
]


# ============================================================================
# Test class
# ============================================================================


# @IGNORE_FAST - hardware perf benchmark suite; both arms compile + run on TRN3 B1 and
# collect profiler metrics, so there is no meaningful fast/compile-only variant.
@pytest_test_metadata(name="Qwen3-235B Attention Perf Benchmarks", tags=["model"])
@pytest_marks(["attention", "cte", "model", "optimal", "mx"])
class TestQwen235BAttentionPerf:
    """Qwen3-235B BF16 core-attention (flash attention) forward perf at TP4/CP1/S=4096."""

    @pytest.mark.optimal
    @pytest.mark.parametrize("conf", _QWEN235B_ATTN_CONFIGS, ids=_QWEN235B_ATTN_IDS)
    def test_forward(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        conf,
    ):
        cp_degree = conf["cp_degree"]
        seqlen_kv = conf["seqlen_kv"]
        seqlen_q = seqlen_kv // cp_degree
        use_cp = cp_degree > 1
        cp_strided_q_slicing = cp_degree > 1
        gqa_factor = conf["gqa_factor"]

        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
        _RangedAttentionCTEKernels().run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: conf["causal_mask"],
                TP_Q_DIM_NAME: conf["tp_q"],
                TP_K_DIM_NAME: conf["tp_k"],
                TP_OUT_DIM_NAME: conf["tp_out"],
                SINK_DIM_NAME: conf["sink"],
            },
            bs=conf["bs"],
            bs_kv=conf["bs"] // gqa_factor,
            cache_softmax=False,
            cp_degree=cp_degree,
            d=conf["d"],
            dtype=nl.bfloat16,
            seqlen_kv=seqlen_kv,
            seqlen_kv_prior=conf["seqlen_kv_prior"],
            prior_used_len=conf["prior_used_len"],
            seqlen_q=seqlen_q,
            sliding_window=conf["sliding_window"],
            softmax_scale=1.0,
            cp_rank_id=conf["cp_rank_id"],
            use_cp=use_cp,
            cp_strided_q_slicing=cp_strided_q_slicing,
        )

    @pytest.mark.optimal
    @pytest.mark.parametrize("conf", _QWEN235B_ATTN_CONFIGS, ids=_QWEN235B_ATTN_IDS)
    def test_backward(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        conf,
    ):
        # attention_bwd keeps heads on their own axis (no batch-folding that the
        # forward does): one sequence => batch_size=1, num_q_heads=bs (16),
        # num_kv_heads=bs//gqa_factor (1). Same 16:1 GQA, different tensor rank.
        batch_size = 1
        num_q_heads = conf["bs"]
        num_kv_heads = conf["bs"] // conf["gqa_factor"]
        seq_len = conf["seqlen_kv"] // conf["cp_degree"]
        head_dim = conf["d"]
        causal = conf["causal_mask"]
        # Backward "no sliding window" sentinel is -1 (generate_inputs default),
        # not 0/None; causal=True so this is never a negative case.
        sliding_window = -1
        softmax_scale = 1.0

        # The backward kernel parallelizes across bs * nheads_kv, which is 1 for
        # this shape, so LNC2 would leave a core idle. Follow the backward suite's
        # rule (LNC2 only when bs*nheads_kv is even).
        lnc = 2 if (batch_size * num_kv_heads) % 2 == 0 else 1
        compiler_args = CompilerArgs(logical_nc_config=lnc, platform_target=platform_target)

        is_negative, _ = is_negative_test_case(num_q_heads, num_kv_heads, seq_len, head_dim, causal, sliding_window)

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(
                batch_size=batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=nl.bfloat16,
                causal=causal,
                sliding_window=sliding_window,
                num_sinks=0,
                softmax_scale=softmax_scale,
            )

        def output_tensors(kernel_input):
            q = kernel_input["q_ref"]
            k = kernel_input["k_ref"]
            return {
                "out_dq_ref": np.zeros(q.shape, q.dtype),
                "out_dk_ref": np.zeros(k.shape, k.dtype),
                "out_dv_ref": np.zeros(k.shape, k.dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_bwd,
            torch_ref=torch_ref_wrapper(attention_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2,
            atol=1e-5,
            is_negative_test=is_negative,
        )
