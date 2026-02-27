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
import math

try:
    from test.integration.nkilib.core.attention.test_attention_cte_model_config import attention_cte_model_configs
except ImportError:
    attention_cte_model_configs = []

from functools import lru_cache
from test.integration.nkilib.utils.tensor_generators import np_random_sample
from test.integration.nkilib.utils.test_kernel_common import convert_to_torch
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import (
    BoundedRange,
    FilterResult,
    assert_negative_test_case,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import Any, Optional, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.attention.attention_cte import (
    _MAX_BS,
    _MAX_BS_TIMES_SEQLEN_QK,
    _MAX_GLOBAL_CP_DEGREE,
    _MAX_HEAD_DIM,
    _MAX_SEQLEN,
    _MIN_GLOBAL_CP_DEGREE,
    attention_cte,
)
from nkilib_src.nkilib.core.attention.attention_cte_torch import (
    attention_cte_torch_ref,
)

# the shape of q, k, v are interlinked, therefore
# we pass in the config as one object
ATTN_CTE_OPT = "attn_cte"
BATCH_DIM_NAME = "bs"
GQA_FACTOR_DIM_NAME = "gqa"
SEQLEN_KV_DIM_NAME = "s_kv"
SEQLEN_KV_PRIOR_DIM_NAME = "s_kv_prior"
CP_DEGREE_DIM_NAME = "cp_deg"
CP_STRIDED_Q_DIM_NAME = "cp_strided"
SEQLEN_Q_DIM_NAME = "s_q"
D_DIM_NAME = "d"
SLIDING_WINDOW_DIM_NAME = "sw"
CAUSAL_MASK_DIM_NAME = "causal"
TP_Q_DIM_NAME = "tp_q"
TP_K_DIM_NAME = "tp_k"
TP_OUT_DIM_NAME = "tp_out"
SINK_DIM_NAME = "sink"
DTYPE_DIM_NAME = "dtype"
CP_RANK_ID_DIM_NAME = "cp_rank_id"
PRIOR_USED_LEN_DIM_NAME = "prior_used_len"

# Keys to exclude from metadata matching
_METADATA_EXCLUDE_KEYS = {"self", "test_manager", "collector"}

# max bs*seqlen_q*seqlen_k to run validation on (to avoid OOM during testing)
_MAX_BS_TIMES_SEQLEN_QK_VALIDATE = 4.0 * 16 * 1024 * 16 * 1024

dtype_mapping = {
    np.float32: "float32",
}


def build_attention_cte_input(
    bs,
    bs_kv,
    d,
    dtype,
    seqlen_q,
    seqlen_kv,
    is_prefix_caching,
    seqlen_kv_prior,
    prior_used_len,
    tp_q,
    tp_k,
    tp_out,
    sink,
    softmax_scale,
    causal_mask,
    sliding_window,
    use_cp,
    cp_strided_q_slicing,
    cp_degree,
    cp_rank_id,
    cache_softmax,
    softmax_dtype=np.float32,
):
    softmax_dtype = dtype_mapping[softmax_dtype]

    if use_cp:
        cp_offset_val = cp_rank_id if cp_strided_q_slicing else cp_rank_id * seqlen_q
        cp_offset = dt.static_cast(np.full(shape=(1, 1), fill_value=cp_offset_val, dtype=nl.int32), nl.int32)
    else:
        cp_offset = None

    random_gen = np_random_sample()
    q = random_gen(shape=(bs, seqlen_q, d) if tp_q else (bs, d, seqlen_q), dtype=dtype)
    k = random_gen(shape=(bs_kv, seqlen_kv, d) if tp_k else (bs_kv, d, seqlen_kv), dtype=dtype)
    v = random_gen(shape=(bs_kv, seqlen_kv, d), dtype=dtype)

    k_prior, v_prior, prior_used_len_t = None, None, None
    if is_prefix_caching:
        k_prior = random_gen(shape=(bs_kv, seqlen_kv_prior, d) if tp_k else (bs_kv, d, seqlen_kv_prior), dtype=dtype)
        v_prior = random_gen(shape=(bs_kv, seqlen_kv_prior, d), dtype=dtype)
        prior_used_len_t = dt.static_cast(np.full(shape=(1,), fill_value=prior_used_len, dtype=nl.int32), nl.int32)
    # Make sink a big value to be sensitive
    sink_t = dt.static_cast(np.random.uniform(low=15.0, high=25.0, size=(bs, 1)), nl.float32) if sink else None

    # Generate output tensors (to be used in place of golden generation for negative test cases)
    out = np.ndarray(shape=(bs, d, seqlen_q) if tp_out else (bs, seqlen_q, d), dtype=dtype)
    outputs = {"out": out}
    if cache_softmax:
        padded_seq_grps = math.ceil(seqlen_q / 128.0)
        neg_max = np.ndarray(shape=(bs, 128, padded_seq_grps), dtype=softmax_dtype)
        recip = np.ndarray(shape=(bs, 128, padded_seq_grps), dtype=softmax_dtype)
        outputs["out_cached_negative_max"] = neg_max
        outputs["out_cached_sum_reciprocal"] = recip

    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "scale": softmax_scale,
        "causal_mask": causal_mask,
        "k_prior": k_prior,
        "v_prior": v_prior,
        "prior_used_len": prior_used_len_t,
        "sink": sink_t,
        "sliding_window": sliding_window,
        "tp_q": tp_q,
        "tp_k": tp_k,
        "tp_out": tp_out,
        "cp_offset": cp_offset,
        "global_cp_deg": cp_degree,
        "cp_strided_q_slicing": cp_strided_q_slicing,
        "cache_softmax": cache_softmax,
        "softmax_dtype": softmax_dtype,
    }

    return inputs, outputs


def attention_cte_forward_golden(
    inps,
    dtype,
    softmax_dtype=np.float32,
):
    dtype_mapping_torch_np = {
        np.float32: torch.float32,
    }
    out = attention_cte_torch_ref(
        q=convert_to_torch(inps["q"]),
        k=convert_to_torch(inps["k"]),
        v=convert_to_torch(inps["v"]),
        scale=inps["scale"],
        causal_mask=inps["causal_mask"],
        k_prior=convert_to_torch(inps.get("k_prior", None)),
        v_prior=convert_to_torch(inps.get("v_prior", None)),
        prior_used_len=convert_to_torch(inps.get("prior_used_len", None)),
        sink=convert_to_torch(inps.get("sink", None)),
        sliding_window=inps["sliding_window"],
        tp_q=inps["tp_q"],
        tp_k=inps["tp_k"],
        tp_out=inps["tp_out"],
        cache_softmax=inps["cache_softmax"],
        softmax_dtype=dtype_mapping_torch_np[softmax_dtype],
        cp_offset=convert_to_torch(inps.get("cp_offset", None)),
        global_cp_deg=inps["global_cp_deg"],
        cp_strided_q_slicing=inps["cp_strided_q_slicing"],
    )
    if inps["cache_softmax"]:
        out_golden, neg_max, recip = out
        return {
            "out": dt.static_cast(out_golden.numpy(), dtype),
            "out_cached_negative_max": dt.static_cast(neg_max.numpy(), softmax_dtype),
            "out_cached_sum_reciprocal": dt.static_cast(recip.numpy(), softmax_dtype),
        }
    else:
        return {"out": dt.static_cast(out.numpy(), dtype)}


@lru_cache(maxsize=1)
def _get_attention_cte_metadata():
    return load_model_configs("test_attention_cte")


@pytest_test_metadata(name="Attention CTE", pytest_marks=["attention", "cte"], tag=["model"])
@final
class TestRangedAttentionCTEKernels:
    def run_range_attention_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options,
        collector: IMetricsCollector,
        lnc_degree,  # FIXME, pass to compiler/runtime once framework supports
        softmax_scale,
        cache_softmax=False,  # test cache softmax (training)
        softmax_dtype=np.float32,
        skip_no_causal=False,
        # skip causal=False tests (useful when this comes up as a boundary test for CP/SWA)
    ):
        np.random.seed(0)  # so we generate random tensors reproducibly

        attn_opt = test_options
        bs = attn_opt[BATCH_DIM_NAME]
        gqa_factor = attn_opt[GQA_FACTOR_DIM_NAME]
        if gqa_factor == 0 or gqa_factor > _MAX_BS:
            pytest.skip("gqa_factor out of bounds")
        if bs % gqa_factor == 0:
            bs_kv = bs // gqa_factor  # GQA case when gqa_factor != 1
        else:
            bs_kv = bs  # MHA

        seqlen_kv = attn_opt[SEQLEN_KV_DIM_NAME]

        use_cp = CP_DEGREE_DIM_NAME in attn_opt
        if use_cp:
            assert SEQLEN_Q_DIM_NAME not in attn_opt, f"both {SEQLEN_Q_DIM_NAME} and {CP_DEGREE_DIM_NAME} provided"
            cp_degree = attn_opt[CP_DEGREE_DIM_NAME]
            if cp_degree <= 0:
                pytest.skip("cp degree of 0 not allowed")
            seqlen_q = seqlen_kv // cp_degree
            cp_rank_id = np.random.randint(max(cp_degree, 1))
            if attn_opt[CP_STRIDED_Q_DIM_NAME] not in [0, 1]:
                pytest.skip("Got not binary value for bool parameter.")
            cp_strided_q_slicing = attn_opt[CP_STRIDED_Q_DIM_NAME] == 1
        else:
            cp_degree = None
            cp_rank_id = None
            cp_strided_q_slicing = False
            seqlen_q = attn_opt[SEQLEN_Q_DIM_NAME]

        seqlen_kv_prior = attn_opt.get(SEQLEN_KV_PRIOR_DIM_NAME, None)

        d = attn_opt[D_DIM_NAME]
        sliding_window = attn_opt.get(SLIDING_WINDOW_DIM_NAME, 0)
        if seqlen_kv == 0 or seqlen_q == 0:
            pytest.skip("Skip test due to tensor shape being 0 which is unsupported in NKI")
        prior_used_len = None
        if seqlen_kv_prior is not None:
            if seqlen_kv_prior == 0:
                pytest.skip("Skip test with seqlen_kv_prior since shape cannot be 0 in NKI")
            prior_used_len = np.random.randint(seqlen_kv_prior)

        # handle bool dims
        bool_dims = {}
        for dim_name in [
            CAUSAL_MASK_DIM_NAME,
            TP_Q_DIM_NAME,
            TP_K_DIM_NAME,
            TP_OUT_DIM_NAME,
            SINK_DIM_NAME,
        ]:
            if attn_opt[dim_name] not in [0, 1]:
                pytest.skip("Got not binary value for bool parameter.")
            bool_dims[dim_name] = attn_opt[dim_name] == 1

        if skip_no_causal and not bool_dims[CAUSAL_MASK_DIM_NAME]:
            pytest.skip("Test config requires causal mask, skipping.")

        # handle dtype dim
        if attn_opt[DTYPE_DIM_NAME] not in [0, 1, 2]:
            pytest.skip("Got invalid value for dtype parameter.")
        dtype_int_to_type = {0: nl.bfloat16, 1: nl.float16, 2: nl.float32}
        dtype = dtype_int_to_type[attn_opt[DTYPE_DIM_NAME]]

        is_negative_test_case = False
        # GQA condition (since bs is derived from batch kv and gqa factor)
        if bs > _MAX_BS:
            is_negative_test_case = True

        seqlen_kv_total = seqlen_kv + seqlen_kv_prior if seqlen_kv_prior else seqlen_kv

        # seqlen_kv_total can't be above _MAX_SEQLEN
        if seqlen_kv_total > _MAX_SEQLEN:
            is_negative_test_case = True

        # bs times seqlen can't be too large otherwise tracing too slow
        if bs * seqlen_q * seqlen_kv_total > _MAX_BS_TIMES_SEQLEN_QK:
            is_negative_test_case = True

        # softmax scale must be 1.0 for SWA/Prefix Caching/CP
        if (sliding_window > 0 or seqlen_kv_prior or use_cp) and softmax_scale != 1.0:
            is_negative_test_case = True

        # SWA/CP only works with causal mask
        if (sliding_window > 0 or use_cp) and not bool_dims[CAUSAL_MASK_DIM_NAME]:
            is_negative_test_case = True

        # Cache softmax currently requires seqlen_q multiple of 128
        if cache_softmax and seqlen_q % 128 != 0:
            is_negative_test_case = True

        if not use_cp and sliding_window > 0:
            if seqlen_q - sliding_window >= seqlen_kv:
                pytest.skip(
                    "Skipping test because some queries do not attend to any KV (due to SWA). "
                    "This causes NaN (0/0) which is not handled well by tests."
                )

        with assert_negative_test_case(is_negative_test_case):
            self.run_attention_cte_test(
                test_manager,
                compiler_args,
                collector,
                bool_dims,
                bs,
                bs_kv,
                cache_softmax,
                cp_degree,
                d,
                dtype,
                seqlen_kv,
                seqlen_kv_prior,
                seqlen_q,
                sliding_window,
                softmax_scale,
                cp_rank_id,
                use_cp,
                cp_strided_q_slicing,
                prior_used_len,
                softmax_dtype,
            )

    def run_attention_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        bool_dims: dict[Any, Any],
        bs: int,
        bs_kv: int,
        cache_softmax: bool,
        cp_degree: int,
        d: int,
        dtype: str,
        seqlen_kv: int,
        seqlen_kv_prior,
        seqlen_q: int,
        sliding_window: int,
        softmax_scale,
        cp_rank_id,
        use_cp: bool,
        cp_strided_q_slicing: bool = False,
        prior_used_len: Optional[int] = None,
        softmax_dtype=np.float32,
    ):
        skip_validation = False
        seqlen_kv_total = (seqlen_kv + seqlen_kv_prior) if seqlen_kv_prior else seqlen_kv
        if bs * seqlen_q * seqlen_kv_total > _MAX_BS_TIMES_SEQLEN_QK_VALIDATE:
            skip_validation = True

        kernel_input, placeholder_output = build_attention_cte_input(
            bs=bs,
            bs_kv=bs_kv,
            d=d,
            dtype=dtype,
            seqlen_kv=seqlen_kv,
            seqlen_q=seqlen_q,
            is_prefix_caching=seqlen_kv_prior is not None,
            seqlen_kv_prior=seqlen_kv_prior,
            prior_used_len=prior_used_len,
            tp_q=bool_dims[TP_Q_DIM_NAME],
            tp_k=bool_dims[TP_K_DIM_NAME],
            tp_out=bool_dims[TP_OUT_DIM_NAME],
            sink=bool_dims[SINK_DIM_NAME],
            softmax_scale=softmax_scale,
            causal_mask=bool_dims[CAUSAL_MASK_DIM_NAME],
            sliding_window=sliding_window,
            use_cp=use_cp,
            cp_strided_q_slicing=cp_strided_q_slicing,
            cp_degree=cp_degree,
            cp_rank_id=cp_rank_id,
            cache_softmax=cache_softmax,
            softmax_dtype=softmax_dtype,
        )

        # Create lazy golden generator - captures local variables via closure
        def create_attention_cte_golden():
            return attention_cte_forward_golden(
                inps=kernel_input,
                dtype=dtype,
                softmax_dtype=softmax_dtype,
            )

        test_manager.execute(
            KernelArgs(
                kernel_func=attention_cte,
                kernel_input=kernel_input,
                compiler_input=compiler_args,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_attention_cte_golden if not skip_validation else None,
                        output_ndarray=placeholder_output,
                    ),
                    relative_accuracy=2e-2,
                    absolute_accuracy=1e-5,
                ),
            )
        )

    # fmt: off
    attention_cte_no_cp_unit_params = "bs, seqlen_kv, seqlen_q, d, softmax_scale, dtype, causal_mask, tp_q, tp_out, tpbSgCyclesSum"
    attention_cte_no_cp_unit_perms = [
        # causal mask false
        [1, 9216, 4096, 64, 0.5, np.float32, False, True, False, 580e6],
        [1, 4096, 9216, 64, 0.5, np.float32, False, True, False, 567e6],
        [1, 9216, 9216, 64, 0.5, np.float32, False, True, False, 125e7],
        [1, 9216, 9216, 64, 0.134, nl.bfloat16, False, True, False, 125e7],

        # Causal attention
        # [1, 32*1024-1, 32*1024-1, 128, 1.000, np.float16, True, True, False, 950e7 * 1.02], # TODO: NKIFE-518
        # [1, 32*1024, 32*1024, 64, 1.000, np.float16, True, True, False, 930e7 * 1.02], # TODO: NKIFE-518
        # [1, 64*1024, 64*1024, 64, 1.000, np.float16, True, True, False, 3623e7 * 1.02], # TODO: wait for trace time fix
        [1, 16 * 1024, 16 * 1024, 64, 1.000, np.float16, True, True, False, 196e7],
        [1, 9216, 9216, 64, 1.000, nl.bfloat16, True, True, False, 694e6],

        # Layouts favourable to LLM cases - don't transpose Q inside kernel and transpose output
        [1, 16 * 1024, 16 * 1024, 128, 1.000, np.float16, True, False, True, 203e7],
        # [1, 32*1024, 32*1024, 128, 1.000, np.float16, True, False, True, 990e7*1.02], # TODO: NKIFE-518
        [1, 16 * 1024 + 16, 16 * 1024 + 16, 128, 1.000, np.float16, True, False, True, 207e7],
        # [1, 32*1024+221, 32*1024+221, 128, 1.000, np.float16, True, False, True, 1023e7*1.02], # TODO: NKIFE-518
        # [1, 64*1024, 64*1024, 128, 1.000, bfloat16, True, False, True, 3781e7*1.02], # TODO: wait for trace time fix

        # Unequal Q vs. KV seqlen
        [1, 16384, 2048, 128, 1.000, np.float32, False, True, False, 520e6],
        [1, 16384, 4096, 128, 1.000, nl.bfloat16, False, True, False, 981e6],
        [1, 16384, 8192, 128, 1.000, np.float32, False, True, False, 196e7],
        [1, 32768, 4096, 128, 1.000, nl.bfloat16, False, True, False, 188e7],

        # Configuration from TnX 32core parallel RMSNorm Llama testcase
        [1, 16384, 16384, 8, 1.000, np.float32, True, False, False, 188e7],

        [3, 2048, 2048, 128, 1.000, np.float16, True, True, False, 209e6],
        [3, 1024, 512, 128, 1.000, np.float16, False, False, True, None],

        [1, 128, 128, 128, 1.000, nl.bfloat16, True, False, True, None],

        # Llama4 + vision configs, have bs = image batches
        [1, 36864, 36864, 128, 1.000, nl.bfloat16, True, False, True, None],
        [512, 4096, 4096, 128, 1.000, nl.bfloat16, False, False, True, None],

    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_no_cp_unit_params, attention_cte_no_cp_unit_perms)
    def test_attention_cte_no_cp_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        seqlen_kv,
        seqlen_q,
        d,
        softmax_scale,
        dtype,
        causal_mask,
        tp_q,
        tp_out,
        tpbSgCyclesSum,  # FIXME: use qor once framework supports
    ):
        compiler_args = CompilerArgs()
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: causal_mask,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: False,
                SINK_DIM_NAME: False,
            },
            bs=bs,
            bs_kv=bs,
            cache_softmax=False,
            cp_degree=None,
            d=d,
            dtype=dtype,
            seqlen_kv=seqlen_kv,
            seqlen_kv_prior=None,
            seqlen_q=seqlen_q,
            sliding_window=0,
            softmax_scale=softmax_scale,
            cp_rank_id=None,
            use_cp=False,
        )

    # fmt: off
    attention_cte_no_cp_vnc_apc_klr_unit_params = \
        "vnc_degree, bs, seqlen_kv, seqlen_q, prior_len, prior_used_len, d, softmax_scale, dtype, causal_mask, tp_q, tp_k, tp_out, sink, sliding_window, tpbSgCyclesSum"
    attention_cte_no_cp_vnc_apc_klr_unit_perms = [
        # causal False (range of prior_used_len)
        [2, 2, 2048, 2048, 4096, 0, 64, 1.0, nl.bfloat16, False, False, False, True, True, 0, (232e6, None)],
        [2, 2, 2048, 2048, 4096, 4095, 64, 1.0, nl.bfloat16, False, False, False, True, True, 0, (232e6, None)],
        [2, 2, 2048, 2048, 4096, 4096, 64, 1.0, nl.bfloat16, False, False, False, True, True, 0, (232e6, None)],

        # causal True (range of prior_used_len)
        [2, 2, 2048, 2048, 4096, 0, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (232e6, None)],
        [2, 2, 2048, 2048, 4096, 1, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (232e6, None)],
        [2, 2, 2048, 2048, 4096, 2048, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (232e6, None)],
        [2, 2, 2048, 2048, 4096, 4095, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (232e6, None)],
        [2, 2, 2048, 2048, 4096, 4096, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (232e6, None)],

        # Test if single core works
        [1, 3, 2048, 2048, 512, 500, 128, 1.0, nl.bfloat16, False, False, False, False, False, 0, (297e6, None)],
        [1, 2, 2048, 2048, 512, 500, 128, 1.0, nl.bfloat16, True, False, False, False, False, 0, (165e6, None)],

        # combinations of tp_out, tp_q and tp_k
        [2, 2, 2048, 2048, 4096, 2048, 64, 1.0, nl.bfloat16, True, False, False, False, True, 0, (219e6, None)],
        [2, 2, 2048, 2048, 4096, 2048, 64, 1.0, nl.bfloat16, True, True, True, False, True, 0, (220e6, None)],
        [2, 2, 2048, 2048, 4096, 2048, 64, 1.0, nl.bfloat16, True, True, True, True, True, 0, (227e6, None)],

        # SWA
        [2, 2, 2048, 2048, 4096, 4096, 64, 1.0, nl.bfloat16, True, False, False, True, True, 128, (194e6, None)],
        [2, 2, 2048, 2048, 4096, 2048, 64, 1.0, nl.bfloat16, True, False, False, True, True, 128, (194e6, None)],
        [2, 2, 2048, 2048, 512, 500, 64, 1.0, nl.bfloat16, True, False, False, True, True, 1024, (116e6, None)],

        # odd lengths and num_sections > 1
        [2, 1, 2001, 2001, 3555, 3000, 64, 1.0, nl.bfloat16, True, False, False, True, False, 0, (140e6, None)],
        [2, 3, 2001, 2001, 3555, 3555, 64, 1.0, nl.bfloat16, True, False, False, True, False, 0, (304e6, None)],
        [2, 1, 16384 + 1, 16384 + 1, 4096 + 1, 4096, 64, 1.0, nl.bfloat16, True, False, False, True, False, 0, (167e7, None)],
        [2, 3, 10240, 10240, 1, 1, 64, 1.0, nl.bfloat16, True, False, False, True, False, 0, (151e7, None)],

        # GPT-OSS configs (with sink. Use causal mask or SWA)
        [2, 2, 4096, 4096, 2048, 2048 - 45, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (324e6, None)],
        [2, 2, 6144, 6144, 2048, 2048 - 511, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (546e6, None)],
        [2, 2, 8192, 8192, 4096, 4096 - 512, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (114e7, None)],
        [2, 2, 8192, 8192, 4096, 4096 - 512, 64, 1.0, nl.bfloat16, True, False, True, True, True, 0, (115e7, None)],
        [2, 2, 8192, 8192, 4096, 4096 - 512, 64, 1.0, nl.bfloat16, True, True, True, True, True, 0, (117e7, None)],
        [2, 2, 4096, 4096, 2048, 2048 - 513, 64, 1.0, nl.bfloat16, True, False, False, True, True, 128, (219e6, None)],
        [2, 2, 6144, 6144, 512, 400, 64, 1.0, nl.bfloat16, True, False, False, True, True, 128, (224e6, None)],
        [2, 2, 8192, 8192, 128, 100, 64, 1.0, nl.bfloat16, True, False, False, True, True, 128, (275e6, None)],
        [2, 2, 8192, 8192, 128, 100, 64, 1.0, nl.bfloat16, True, False, True, True, True, 128, (290e6, None)],
        [2, 2, 8192, 8192, 128, 100, 64, 1.0, nl.bfloat16, True, True, True, True, True, 128, (295e6, None)],

        # Llama 70B configs
        [2, 1, 256, 256, 2048, 2048 - 128, 128, 1.0, nl.bfloat16, True, True, True, True, False, 0, (53e6, None)],
        [2, 1, 2048, 2048, 2048, 2048 - 511, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (111e6, None)],
        [2, 1, 4096, 4096, 2048, 1543, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (204e6, None)],
        [2, 1, 4096, 4096, 2048, 1543, 128, 1.0, nl.bfloat16, True, True, True, True, False, 0, (198e6, None)],
        [2, 1, 8192, 8192, 4096, 900, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (635e6, None)],
        [2, 1, 8192, 8192, 4096, 900, 128, 1.0, nl.bfloat16, True, False, True, True, False, 0, (644e6, None)],
        [2, 1, 8192, 8192, 4096, 900, 128, 1.0, nl.bfloat16, True, True, True, True, False, 0, (644e6, None)],
        [2, 1, 20480, 20480, 4096, 4096 - 512, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (228e7, None)],
        [2, 1, 20480, 20480, 4096, 4096 - 512, 128, 1.0, nl.bfloat16, True, False, True, True, False, 0, (233e7, None)],
        [2, 1, 20480, 20480, 4096, 4096 - 512, 128, 1.0, nl.bfloat16, True, True, True, True, False, 0, (236e7, None)],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_no_cp_vnc_apc_klr_unit_params, attention_cte_no_cp_vnc_apc_klr_unit_perms)
    def test_attention_cte_no_cp_vnc_apc_klr_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        vnc_degree,
        bs,
        seqlen_kv,
        seqlen_q,
        prior_len,
        prior_used_len,
        d,
        softmax_scale,
        dtype,
        causal_mask,
        tp_q,
        tp_k,
        tp_out,
        sink,
        sliding_window,
        tpbSgCyclesSum,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: causal_mask,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: tp_k,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs,
            cache_softmax=False,
            cp_degree=None,
            d=d,
            dtype=dtype,
            seqlen_kv=seqlen_kv,
            seqlen_kv_prior=prior_len,
            seqlen_q=seqlen_q,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=None,
            use_cp=False,
            prior_used_len=prior_used_len,
        )

    # fmt: off
    attention_cte_no_cp_vnc_param_unit_params = \
        "vnc_degree, bs, seqlen_kv, seqlen_q, d, softmax_scale, dtype, causal_mask, tp_q, tp_out, sink, sliding_window, tpbSgCyclesSum"
    attention_cte_no_cp_vnc_param_unit_perms = [
        [2, 1, 8192, 8192, 64, 0.125, np.float32, False, True, False, False, 0, (520e6, None)],
        [2, 1, 8192, 4096, 64, 0.125, np.float32, False, True, False, False, 0, (296e6, None)],
        [2, 1, 4096, 8192, 64, 0.125, np.float32, False, True, False, False, 0, (313e6, None)],
        [2, 1, 8192, 32768, 64, 0.125, np.float32, False, True, False, False, 0, (187e7, None)],
        [2, 1, 8192, 8192, 64, 1.0, np.float32, True, True, False, False, 0, (343e6, None)],
        # dhead=128 configs for real-world usecases

        [2, 1, 8192, 8192, 128, 1.0, np.float32, True, True, False, False, 0, (352e6, None)],
        [2, 1, 8192, 8192, 128, 1.0, np.float32, True, False, False, False, 0, (363e6, None)],
        [2, 3, 8192, 8192, 128, 1.0, np.float32, True, False, False, False, 0, (903e6, None)],
        [2, 3, 8192, 8192, 128, 1.0, np.float32, False, False, False, False, 0, (149e7, None)],

        # [2, 1, 32768, 32768, 64, 0.125, np.float32, False, True, False, False, 0, (735e7, None)], # TODO: NKIFE-518
        # Same case as above but BF16
        [2, 1, 32768, 32768, 64, 1.0, nl.bfloat16, True, True, False, False, 0, (427e7, None)],
        # Batch > 1
        [2, 4, 8192, 2048, 64, 0.125, np.float32, False, True, False, False, 0, (560e6, None)],
        [2, 3, 2048, 1024, 128, 0.125, nl.bfloat16, False, False, False, False, 0, (115e6, None)],
        # Test if single core works
        [1, 3, 2048, 1024, 128, 0.125, nl.bfloat16, False, False, False, False, 0, (155e6, None)],
        # Config from Llama3 405B - key difference is tp_out = True. Not tested in other configs.
        [2, 3, 8192, 8192, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (916e6, None)],
        [2, 2, 16*1024, 16*1024, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (211e7, None)],
        [2, 2, 1024, 1024, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (71e6, None)],
        [2, 2, 512, 512, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (55e6, None)],
        # Llama3 70B 64 Q head with TP==64 + LNC2. So each LNC processes 1 Q head.
        [2, 1, 1024, 1024, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (59e6, None)],
        [2, 1, 256, 256, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (43e6, None)],
        [2, 1, 512, 512, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (52e6, None)],
        # Smaller dhead
        [2, 2, 1024, 1024, 64, 1.0, nl.bfloat16, True, False, True, False, 0, (69e6, None)],
        # Long seqlen with TP out enabled
        # [2, 2, 32*1024, 32*1024, 128, 1.0, nl.bfloat16, True, False, True, False, 0, (765e7, None)], # TODO: NKIFE-518
        # Enable both TP settings
        [2, 2, 1024, 1024, 128, 1.0, nl.bfloat16, True, True, True, False, 0, (69e6, None)],
        # Vision encoder
        [2, 2, 577, 577, 88, 1.0, nl.bfloat16, False, True, False, False, 0, (56e6, None)],
        [2, 2, 577, 577, 88, 1.0, np.float32, False, True, False, False, 0, (56e6, None)],

        # Test TP-Q, packed load and q dtype =/!= load_dtype
        [2, 2, 729, 729, 128, 1.0, nl.bfloat16, True, True, True, False, 0, (60e6, None)],
        [2, 3, 729, 729, 128, 1.0, nl.bfloat16, True, True, True, False, 0, (82e6, None)],
        [2, 3, 729, 729, 128, 1.0, nl.float32, True, True, True, False, 0, (82e6, None)],

        # Liger, with sink, July 15, 2025
        [2, 3, 8192, 8192, 64, 1.0, nl.bfloat16, True, False, False, True, 0, (900e6, None)], # sink, no swa
        [2, 1, 10240, 10240, 64, 1.0, nl.bfloat16, True, False, False, True, 0, (495e6, None)], # sink, no swa, flash
        [2, 3, 8192, 8192, 64, 1.0, nl.bfloat16, False, False, False, True, 0, (145e7, None)],

        # SWA attention (with sink)
        [2, 1, 8192, 8192, 64, 1.0, nl.bfloat16, False, False, False, True, 0, (524e6, None)],
        [2, 1, 8192, 8192, 64, 1.0, nl.bfloat16, True, False, False, True, 0, (354e6, None)],
        [2, 1, 8192, 8192, 64, 1.0, nl.bfloat16, True, False, False, True, 128, (135e6, None)],

        # Liger TP8 perf configuration (with and without SWA)
        [2, 8, 10240, 10240, 64, 1.0, nl.bfloat16, True, False, True, True, 0, (336e7, 263e7)],
        [2, 8, 10240, 10240, 64, 1.0, nl.bfloat16, True, False, True, True, 128, (894e6, 770e6)],
        [2, 8, 10240, 10240, 64, 1.0, nl.bfloat16, True, True, True, True, 0, (338e7, 263e7)],
        [2, 8, 10240, 10240, 64, 1.0, nl.bfloat16, True, True, True, True, 128, (905e6, 770e6)],

        # Liger-like configuration length sweep test for lengths (only use 2 heads to minimize test time)
        [2, 2, 2048, 2048, 64, 1.0, nl.bfloat16, True, False, True, True, 0, (103e6, None)],
        [2, 2, 2048, 2048, 64, 1.0, nl.bfloat16, True, False, True, True, 128, (78e6, None)],
        [2, 2, 4096, 4096, 64, 1.0, nl.bfloat16, True, False, True, True, 0, (211e6, None)],
        [2, 2, 4096, 4096, 64, 1.0, nl.bfloat16, True, False, True, True, 128, (125e6, None)],
        [2, 2, 8192, 8192, 64, 1.0, nl.bfloat16, True, False, True, True, 0, (596e6, None)],
        [2, 2, 8192, 8192, 64, 1.0, nl.bfloat16, True, False, True, True, 128, (211e6, None)],
        [2, 2, 16384, 16384, 64, 1.0, nl.bfloat16, True, False, True, True, 0, (211e7, None)],
        [2, 2, 16384, 16384, 64, 1.0, nl.bfloat16, True, False, True, True, 128, (583e6, None)],
        # [2, 2, 32768, 32768, 64, 1.0, nl.bfloat16, True, False, True, True, 0, (None, None)], # TODO: NKIFE-518
        [2, 2, 32768, 32768, 64, 1.0, nl.bfloat16, True, False, True, True, 128, (156e7, None)],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_no_cp_vnc_param_unit_params, attention_cte_no_cp_vnc_param_unit_perms)
    def test_attention_cte_no_cp_vnc_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        vnc_degree,
        bs,
        seqlen_kv,
        seqlen_q,
        d,
        softmax_scale,
        dtype,
        causal_mask,
        tp_q,
        tp_out,
        sink,
        sliding_window,
        tpbSgCyclesSum,
    ):
        compiler_args = CompilerArgs()
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: causal_mask,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: False,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs,
            cache_softmax=False,
            cp_degree=None,
            d=d,
            dtype=dtype,
            seqlen_kv=seqlen_kv,
            seqlen_kv_prior=None,
            seqlen_q=seqlen_q,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=None,
            use_cp=False,
        )

    # fmt: off
    attention_cte_no_cp_vnc_gqa_unit_params = \
        "vnc_degree, bs, bs_kv, seqlen_kv, seqlen_q, prior_len, prior_used_len, d, softmax_scale, dtype, causal_mask, tp_q, tp_k, tp_out, sink, sliding_window, tpbSgCyclesSum"
    attention_cte_no_cp_vnc_gqa_unit_perms = [
        # test different values of vnc_degree, bs and bs_kv
        [2, 1, 1, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, False, False, False, True, False, 0, (104e6, None)],
        [2, 2, 1, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (125e6, None)],
        [2, 2, 2, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (126e6, None)],
        [2, 9, 3, 2048, 2048, None, None, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (332e6, None)],
        [2, 5, 1, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (259e6, None)],
        [2, 5, 1, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, False, False, False, True, False, 0, (316e6, None)],
        [1, 9, 3, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (777e6, None)],
        [1, 4, 2, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, False, False, False, True, False, 0, (440e6, None)],
        [1, 1, 1, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, False, False, False, True, False, 0, (130e6, None)],

        # play with tp_q/k/out
        [2, 9, 3, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, True, False, False, True, False, 0, (431e6, None)],
        [2, 9, 3, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, True, False, True, True, False, 0, (465e6, None)],
        [2, 9, 3, 2048, 2048, 1024, 512, 128, 1.0, nl.bfloat16, True, True, False, True, False, 0, (448e6, None)],

        # Liger TP8 configuration (with and without SWA)
        [2, 8, 1, 10240, 10240, None, None, 64, 1.0, nl.bfloat16, True, True, True, True, True, 0, (344e7, 282e7)],
        [2, 8, 1, 10240, 10240, None, None, 64, 1.0, nl.bfloat16, True, True, True, True, True, 128, (963e6, 795e6)],
        [2, 8, 1, 10240, 10240, None, None, 64, 1.0, np.float32, True, True, True, True, True, 0, (None, None)],
        [2, 8, 1, 10240, 10240, None, None, 64, 1.0, np.float32, True, True, True, True, True, 128, (None, None)],
        # [2, 8, 1, 16384, 16384, None, None, 64, 1.0, nl.bfloat16, True, True, True, True, True, 0, (None, None)], # TODO: NKIFE-518
        [2, 8, 1, 16384, 16384, None, None, 64, 1.0, nl.bfloat16, True, True, True, True, True, 128, (224e7, None)],
        [2, 8, 1, 4096, 4096, 2048, 2048-45, 64, 1.0, nl.bfloat16, True, False, False, True, True, 0, (118e7, None)],
        [2, 8, 1, 8192, 8192, 4096, 4096-512, 64, 1.0, nl.bfloat16, True, False, True, True, True, 0, (448e7, None)],
        [2, 8, 1, 4096, 4096, 2048, 2048-513, 64, 1.0, nl.bfloat16, True, False, False, True, True, 128, (735e6, None)],
        [2, 8, 1, 6144, 6144, 512, 400, 64, 1.0, nl.bfloat16, True, False, False, True, True, 128, (751e6, None)],
        [2, 8, 1, 8192, 8192, 128, 100, 64, 1.0, nl.bfloat16, True, False, True, True, True, 128, (105e7, None)],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_no_cp_vnc_gqa_unit_params, attention_cte_no_cp_vnc_gqa_unit_perms)
    def test_attention_cte_no_cp_vnc_gqa_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        vnc_degree,
        bs,
        bs_kv,
        seqlen_kv,
        seqlen_q,
        prior_len,
        prior_used_len,
        d,
        softmax_scale,
        dtype,
        causal_mask,
        tp_q,
        tp_k,
        tp_out,
        sink,
        sliding_window,
        tpbSgCyclesSum,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: causal_mask,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: tp_k,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs_kv,
            cache_softmax=False,
            cp_degree=None,
            d=d,
            dtype=dtype,
            seqlen_kv=seqlen_kv,
            seqlen_kv_prior=prior_len,
            seqlen_q=seqlen_q,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=None,
            use_cp=False,
            prior_used_len=prior_used_len,
        )

    # fmt: off
    attention_cte_no_cp_vnc_barameter_params = \
        "vnc_degree, bs, bs_kv, seqlen_kv, seqlen_q, prior_len, prior_used_len, d, softmax_scale, dtype, causal_mask, tp_q, tp_k, tp_out, sink, sliding_window, tpbSgCyclesSum"
    attention_cte_no_cp_vnc_barometer_perms = [
        [1, 3, 1, 12345, 12345, None, None, 127, 1.0, nl.bfloat16, True, True, True, False, True, 128, (140e7, None)],
        [2, 3, 1, 12345, 12345, 1233, 412, 127, 1.0, nl.float32, True, False, True, True, True, 128, (117e7, None)],
        [2, 3, 1, 12345, 12345, 1233, 412, 127, 1.0, nl.bfloat16, True, True, False, True, True, 1024, (133e7, None)],
        [2, 3, 1, 12345, 12345, 1233, 412, 127, 1.0, nl.bfloat16, False, False, False, False, False, 0, (367e7, None)],
        [2, 3, 1, 12345, 12345, 1233, 412, 127, 1.0, nl.bfloat16, True, True, True, True, True, 0, (241e7, None)],
        [2, 8, 4, 16384, 16384, None, None, 64, 0.125, nl.bfloat16, True, True, True, True, True, 0, (856e7, None)],
        [2, 3, 3, 24576, 24576, 123, 0, 128, 1.0, nl.bfloat16, True, True, True, True, False, 23, (221e7, None)],
        [2, 2, 2, 10240, 10240, None, None, 128, 1.0, nl.float32, False, True, True, True, False, 0, None],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_no_cp_vnc_barameter_params, attention_cte_no_cp_vnc_barometer_perms)
    def test_attention_cte_no_cp_vnc_barometer_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        vnc_degree,
        bs,
        bs_kv,
        seqlen_kv,
        seqlen_q,
        prior_len,
        prior_used_len,
        d,
        softmax_scale,
        dtype,
        causal_mask,
        tp_q,
        tp_k,
        tp_out,
        sink,
        sliding_window,
        tpbSgCyclesSum,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: causal_mask,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: tp_k,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs_kv,
            cache_softmax=False,
            cp_degree=None,
            d=d,
            dtype=dtype,
            seqlen_kv=seqlen_kv,
            seqlen_kv_prior=prior_len,
            seqlen_q=seqlen_q,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=None,
            use_cp=False,
            prior_used_len=prior_used_len,
        )

    # fmt: off
    attention_cte_cp_apc_klr_params = \
        "bs, q_seqlen_partial, kv_seqlen, kv_seqlen_prior, kv_prior_used_len, d, softmax_scale, dtype, tp_q, tp_k, tp_out, cp_rank_id, cp_degree, sink, sliding_window, qor"
    attention_cte_cp_apc_klr_perms = [
        # 4-512-1024 (prior 512)
        [4, 512, 1024, 512, 256, 128, 1.0, nl.bfloat16, True, False, False, 0, 2, False, 0, 87e6],
        [4, 512, 1024, 512, 256, 128, 1.0, nl.bfloat16, True, False, False, 1, 2, False, 0, 87e6],
        # 3-512-8192 (prior 2047, different tp_q/tp_k/tp_out settings)
        [3, 512, 8192, 2047, 0, 128, 1.0, nl.bfloat16, True, False, False, 0, 16, False, 0, 278e6],
        [3, 512, 8192, 2047, 1024, 128, 1.0, nl.bfloat16, False, True, False, 6, 16, False, 0, 269e6],
        [3, 512, 8192, 2047, 2046, 128, 1.0, nl.bfloat16, True, False, True, 13, 16, False, 0, 276e6],

        # GPT-OSS, sliding window attention with attention sinks
        [3, 2048, 4096, 2000, 1950, 64, 1.0, nl.bfloat16, True, False, False, 0, 2, True, 0, 357e6],
        [3, 2048, 4096, 4096, 4096, 64, 1.0, nl.bfloat16, True, False, False, 0, 2, True, 256, 404e6],
        [1, 2048, 2048, 1024, 0, 64, 1.0, nl.bfloat16, True, False, False, 0, 1, True, 512, 111e6],
        [3, 2048, 4096, 450, 1, 64, 1.0, nl.bfloat16, True, True, True, 0, 2, True, 256, 278e6],
        [4, 640, 10240, 4096, 3500, 64, 1.0, nl.bfloat16, True, True, False, 13, 16, True, 0, 431e6],
        [4, 640, 10240, 2045, 234, 64, 1.0, nl.bfloat16, True, True, False, 13, 16, True, 128, 151e6],
        [10, 1024, 16384, 4096, 3244, 128, 1.0, nl.bfloat16, True, False, False, 12, 16, False, 0, 198e7],
        [10, 1024, 16384, 4096, 1244, 128, 1.0, nl.bfloat16, True, True, False, 12, 16, False, 128, 706e6],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_cp_apc_klr_params, attention_cte_cp_apc_klr_perms)
    @pytest.mark.parametrize("cp_strided_q_slicing", [False, True])
    def test_attention_cte_cp_apc_klr_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        q_seqlen_partial,
        kv_seqlen,
        kv_seqlen_prior,
        kv_prior_used_len,
        d,
        softmax_scale,
        dtype,
        tp_q,
        tp_k,
        tp_out,
        cp_rank_id,
        cp_degree,
        sink,
        sliding_window,
        qor,
        cp_strided_q_slicing,
    ):
        compiler_args = CompilerArgs()
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: True,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: tp_k,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs,
            cache_softmax=False,
            cp_degree=cp_degree,
            d=d,
            dtype=dtype,
            seqlen_kv=kv_seqlen,
            seqlen_kv_prior=kv_seqlen_prior,
            prior_used_len=kv_prior_used_len,
            seqlen_q=q_seqlen_partial,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=cp_rank_id,
            use_cp=True,
            cp_strided_q_slicing=cp_strided_q_slicing,
        )

    # fmt: off
    attention_cte_cp_bk_klr_params = \
        "bs, q_seqlen_partial, kv_seqlen, d, softmax_scale, dtype, tp_q, tp_k, tp_out, cp_rank_id, cp_degree, sink, sliding_window, qor"
    attention_cte_cp_bk_klr_perms = [
        # 4-512-1024
        [4, 512, 1024, 128, 1.0, nl.bfloat16, True, False, False, 0, 2, False, 0, 77e6],
        [4, 512, 1024, 128, 1.0, nl.bfloat16, True, False, False, 1, 2, False, 0, 77e6],
        # 4-512-4096
        [4, 512, 4096, 128, 1.0, nl.bfloat16, True, False, False, 0, 8, False, 0, 147e6],
        [4, 512, 4096, 128, 1.0, nl.bfloat16, True, False, False, 7, 8, False, 0, 147e6],
        # 4-256-4096
        [8, 256, 4096, 128, 1.0, nl.float32, True, False, True, 3, 16, False, 0, 200e6],
        # 10-512-8192
        [10, 512, 8192, 128, 1.0, nl.bfloat16, True, False, False, 6, 16, False, 0, 530e6],
        [10, 512, 8192, 128, 1.0, nl.bfloat16, True, False, False, 13, 16, False, 0, 530e6],
        # 10-640-10k
        [10, 640, 10240, 128, 1.0, nl.bfloat16, True, False, False, 4, 16, False, 0, 725e6],
        [10, 640, 10240, 128, 1.0, nl.bfloat16, True, False, False, 11, 16, False, 0, 725e6],
        # 10-1024-16k
        [10, 1024, 16384, 128, 1.0, nl.bfloat16, True, False, False, 12, 16, False, 0, 155e7],
        # 10-2048-32k
        [10, 2048, 32768, 128, 1.0, nl.bfloat16, True, False, False, 12, 16, False, 0, 527e7],
        # 10-8192-128k
        # [10, 8192, 131072, 128, 1.0, nl.bfloat16, True, False, False, 12, 16, False, 0, None], #TODO: trace time support
        # Odd batch size
        [3, 512, 8192, 128, 1.0, nl.bfloat16, True, False, False, 13, 16, False, 0, 233e6],

        # Liger, sliding window attention with attention sinks
        [4, 640, 10240, 64, 1.0, nl.bfloat16, True, False, False, 15, 16, True, 0, 306e6], # perf ref, no swa
        [4, 640, 10240, 64, 1.0, nl.bfloat16, True, False, False, 1, 16, True, 128, 92e6],
        [4, 640, 10240, 64, 1.0, nl.bfloat16, True, False, False, 13, 16, True, 128, 92e6],
        # Liger, sliding window attention with attention sinks with K in [bs, kv_seqlen, d] layout (tp_k in kernel)
        [4, 640, 10240, 64, 1.0, nl.bfloat16, True, True, False, 15, 16, True, 0, 298e6], # perf ref, no swa
        [4, 640, 10240, 64, 1.0, nl.bfloat16, True, True, False, 1, 16, True, 128, 105e6],
        [4, 640, 10240, 64, 1.0, nl.bfloat16, True, True, False, 13, 16, True, 128, 105e6],
        # Functional test for swa + sink
        [3, 2048, 4096, 64, 1.0, nl.bfloat16, True, False, False, 0, 2, True, 0, 263e6],
        [3, 2048, 4096, 64, 1.0, nl.bfloat16, True, False, False, 0, 2, True, 256, 212e6],
        [1, 2048, 2048, 64, 1.0, nl.bfloat16, True, False, False, 0, 1, True, 512, 95e6],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_cp_bk_klr_params, attention_cte_cp_bk_klr_perms)
    @pytest.mark.parametrize("cp_strided_q_slicing", [False, True])
    def test_attention_cte_cp_bk_klr_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        q_seqlen_partial,
        kv_seqlen,
        d,
        softmax_scale,
        dtype,
        tp_q,
        tp_k,
        tp_out,
        cp_rank_id,
        cp_degree,
        sink,
        sliding_window,
        qor,
        cp_strided_q_slicing,
    ):
        compiler_args = CompilerArgs()
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: True,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: tp_k,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs,
            cache_softmax=False,
            cp_degree=cp_degree,
            d=d,
            dtype=dtype,
            seqlen_kv=kv_seqlen,
            seqlen_kv_prior=None,
            seqlen_q=q_seqlen_partial,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=cp_rank_id,
            use_cp=True,
            cp_strided_q_slicing=cp_strided_q_slicing,
        )

    # fmt: off
    attention_cte_cp_gqa_klr_params = \
        "bs, bs_kv, q_seqlen_partial, kv_seqlen, kv_seqlen_prior, kv_prior_used_len, d, softmax_scale, dtype, tp_q, tp_k, tp_out, cp_rank_id, cp_degree, sink, sliding_window, qor"
    attention_cte_cp_gqa_klr_perms = [
        # 512-1024 (prior 512)
        [1, 1, 512, 1024, 512, 256, 128, 1.0, nl.bfloat16, True, False, False, 0, 2, False, 0, 60e6],
        [2, 1, 512, 1024, 512, 256, 128, 1.0, nl.bfloat16, True, False, False, 1, 2, False, 0, 62e6],
        [2, 2, 512, 1024, 512, 256, 128, 1.0, nl.bfloat16, True, False, False, 1, 2, False, 0, 62e6],
        [9, 3, 512, 1024, 512, 256, 128, 1.0, nl.bfloat16, True, False, False, 0, 2, False, 0, 176e6],
        [5, 1, 512, 1024, 512, 256, 128, 1.0, nl.bfloat16, True, False, False, 1, 2, False, 0, 117e6],

        # 512-8192 (prior 2047, different tp_q/tp_k/tp_out settings)
        [9, 3, 512, 8192, 2047, 1024, 128, 1.0, nl.bfloat16, True, False, False, 0, 16, False, 0, 660e6],
        [9, 3, 512, 8192, None, None, 128, 1.0, nl.bfloat16, True, False, False, 12, 16, False, 0, 529e6],
        [9, 3, 512, 8192, 2047, 1024, 128, 1.0, nl.bfloat16, True, True, False, 4, 16, False, 0, 615e6],
        [9, 3, 512, 8192, 2047, 1024, 128, 1.0, nl.bfloat16, True, True, True, 6, 16, False, 0, 599e6],
        [9, 3, 512, 8192, 2047, 1024, 128, 1.0, nl.bfloat16, True, True, True, 7, 16, True, 0, None],
        [9, 3, 512, 8192, 2047, 1024, 128, 1.0, nl.bfloat16, True, True, True, 10, 16, True, 128, None],
        [9, 3, 512, 8192, 2047, 1024, 128, 1.0, nl.float32, True, True, True, 7, 16, True, 0, None],
        [9, 3, 512, 8192, 2047, 1024, 128, 1.0, nl.float32, True, True, True, 10, 16, True, 128, None],

        # GPT-OSS, sliding window attention with attention sinks
        [16, 2, 1024, 16384, 4096, 3244, 128, 1.0, nl.bfloat16, True, False, False, 12, 16, False, 0, 315e7],
        [16, 2, 1024, 16384, 4096, 3244, 128, 1.0, nl.bfloat16, True, False, False, 12, 16, False, 128, 942e6],

        # perf reference from non-GQA mode, with and without SWA
        [4, 1, 640, 10240, None, None, 64, 1.0, nl.bfloat16, True, True, False, 15, 16, True, 0, 298e6],
        [4, 1, 640, 10240, None, None, 64, 1.0, nl.bfloat16, True, True, False, 11, 16, True, 128, 106e6],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_cp_gqa_klr_params, attention_cte_cp_gqa_klr_perms)
    @pytest.mark.parametrize("cp_strided_q_slicing", [False, True])
    def test_attention_cte_cp_gqa_klr_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        bs_kv,
        q_seqlen_partial,
        kv_seqlen,
        kv_seqlen_prior,
        kv_prior_used_len,
        d,
        softmax_scale,
        dtype,
        tp_q,
        tp_k,
        tp_out,
        cp_rank_id,
        cp_degree,
        sink,
        sliding_window,
        qor,
        cp_strided_q_slicing,
    ):
        compiler_args = CompilerArgs()
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: True,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: tp_k,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs_kv,
            cache_softmax=False,
            cp_degree=cp_degree,
            d=d,
            dtype=dtype,
            seqlen_kv=kv_seqlen,
            seqlen_kv_prior=kv_seqlen_prior,
            prior_used_len=kv_prior_used_len,
            seqlen_q=q_seqlen_partial,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=cp_rank_id,
            use_cp=True,
            cp_strided_q_slicing=cp_strided_q_slicing,
        )

    # fmt: off
    attention_cte_cp_barometer_klr_params = \
        "bs, bs_kv, q_seqlen_partial, kv_seqlen, kv_seqlen_prior, kv_prior_used_len, d, softmax_scale, dtype, tp_q, tp_k, tp_out, cp_rank_id, cp_degree, sink, sliding_window, qor"
    attention_cte_cp_barometer_klr_perms = [
        [2, 1, 729, 14580, None, None, 64, 1.0, nl.float32, True, True, True, 19, 20, True, 0, 319e6],
        [3, 3, 729, 14580, 123, 1, 127, 1.0, nl.bfloat16, True, False, True, 1, 20, True, 0, 462e6],
        [4, 1, 640, 10240, None, None, 128, 1.0, nl.bfloat16, True, True, True, 7, 16, False, 128, 104e6],
        [4, 1, 640, 10240, 2044, 123, 128, 1.0, nl.bfloat16, True, True, True, 5, 16, True, 121, 159e6],
        [4, 2, 640, 10240, 1245, 142, 64, 1.0, nl.bfloat16, False, False, False, 11, 16, False, 1024, 155e6],
    ]
    # fmt: on

    @pytest.mark.parametrize(attention_cte_cp_barometer_klr_params, attention_cte_cp_barometer_klr_perms)
    @pytest.mark.parametrize("cp_strided_q_slicing", [False, True])
    def test_attention_cte_cp_barometer_klr_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        bs_kv,
        q_seqlen_partial,
        kv_seqlen,
        kv_seqlen_prior,
        kv_prior_used_len,
        d,
        softmax_scale,
        dtype,
        tp_q,
        tp_k,
        tp_out,
        cp_rank_id,
        cp_degree,
        sink,
        sliding_window,
        qor,
        cp_strided_q_slicing,
    ):
        compiler_args = CompilerArgs()
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: True,
                TP_Q_DIM_NAME: tp_q,
                TP_OUT_DIM_NAME: tp_out,
                TP_K_DIM_NAME: tp_k,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs_kv,
            cache_softmax=False,
            cp_degree=cp_degree,
            d=d,
            dtype=dtype,
            seqlen_kv=kv_seqlen,
            seqlen_kv_prior=kv_seqlen_prior,
            prior_used_len=kv_prior_used_len,
            seqlen_q=q_seqlen_partial,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            cp_rank_id=cp_rank_id,
            use_cp=True,
            cp_strided_q_slicing=cp_strided_q_slicing,
        )

    @staticmethod
    def sweep_attention_cte_config(sliding_window=False, prefix_caching=False, context_parallel=False):
        """
        Returns BoundedRange objects for attention_cte parameters.
        Use with @pytest.mark.coverage_parametrize and filter_attention_cte_combinations.
        """
        config = {}

        # Core parameters
        config[BATCH_DIM_NAME] = BoundedRange(
            np.random.choice(range(1, _MAX_BS + 1), size=2, replace=False), boundary_values=[]
        )
        config[GQA_FACTOR_DIM_NAME] = BoundedRange(
            np.random.choice(range(1, _MAX_BS + 1), size=2, replace=False), boundary_values=[]
        )
        config[D_DIM_NAME] = BoundedRange(np.random.choice(range(1, _MAX_HEAD_DIM + 1), size=2), boundary_values=[])
        config[SEQLEN_KV_DIM_NAME] = BoundedRange(
            np.random.choice(range(1, _MAX_SEQLEN + 1), size=4), boundary_values=[]
        )

        # Bool dims
        config[TP_Q_DIM_NAME] = BoundedRange([0, 1], boundary_values=[])
        config[TP_K_DIM_NAME] = BoundedRange([0, 1], boundary_values=[])
        config[TP_OUT_DIM_NAME] = BoundedRange([0, 1], boundary_values=[])
        config[SINK_DIM_NAME] = BoundedRange([0, 1], boundary_values=[])

        # Dtype (0=bfloat16, 1=float16, 2=float32)
        config[DTYPE_DIM_NAME] = BoundedRange([0, 1, 2], boundary_values=[])

        # Causal mask depends on context_parallel/sliding_window
        if not (context_parallel or sliding_window):
            config[CAUSAL_MASK_DIM_NAME] = BoundedRange([0, 1], boundary_values=[])
        else:
            config[CAUSAL_MASK_DIM_NAME] = BoundedRange([1], boundary_values=[])

        # Context parallel or regular seqlen_q
        if context_parallel:
            config[CP_DEGREE_DIM_NAME] = BoundedRange(
                list(
                    map(
                        lambda n: 2**n,
                        filter(
                            lambda n: _MIN_GLOBAL_CP_DEGREE <= 2**n <= _MAX_GLOBAL_CP_DEGREE,
                            range(_MIN_GLOBAL_CP_DEGREE - 1, _MAX_GLOBAL_CP_DEGREE),
                        ),
                    )
                ),
                boundary_values=[],
            )
            config[CP_STRIDED_Q_DIM_NAME] = BoundedRange([0, 1], boundary_values=[])
        else:
            config[SEQLEN_Q_DIM_NAME] = BoundedRange(
                np.random.choice(range(1, _MAX_SEQLEN + 1), size=4, replace=False), boundary_values=[]
            )

        # Optional features
        if sliding_window:
            config[SLIDING_WINDOW_DIM_NAME] = BoundedRange(
                np.random.choice(_MAX_SEQLEN + 1, size=4), boundary_values=[]
            )

        if prefix_caching:
            config[SEQLEN_KV_PRIOR_DIM_NAME] = BoundedRange(
                np.random.choice(range(1, _MAX_SEQLEN + 1), size=4), boundary_values=[]
            )

        return config

    # Global state for reproducible random sampling
    _filter_state = {"max_tests": None, "seed": 42, "sample_rate": 1.0}

    @classmethod
    def set_test_limit(cls, max_tests=None, seed=42):
        """Set maximum number of test combinations to run with reproducible sampling.

        Args:
            max_tests: Maximum number of tests to run (None for unlimited)
            seed: Random seed for reproducible sampling

        Example:
            # Limit to 100 tests with reproducible sampling
            TestRangedAttentionCTEKernels.set_test_limit(max_tests=100, seed=42)
        """
        cls._filter_state["max_tests"] = max_tests
        cls._filter_state["seed"] = seed
        # Estimate sample rate (rough heuristic: ~10k valid combinations with pairs coverage)
        if max_tests:
            cls._filter_state["sample_rate"] = min(1.0, max_tests / 10000.0)

    def filter_attention_cte_combinations(
        bs,
        gqa,
        d,
        s_kv,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        s_q=None,
        cp_deg=None,
        cp_strided=None,
        sw=None,
        s_kv_prior=None,
    ):
        """Filter invalid parameter combinations for attention_cte tests with reproducible random sampling."""
        # GQA factor validation
        if gqa == 0 or gqa > _MAX_BS:
            return FilterResult.REDUNDANT

        # Batch size must be divisible by GQA factor for GQA case
        if gqa > 1 and bs % gqa != 0:
            return FilterResult.REDUNDANT

        # Determine actual seqlen_q
        if cp_deg:
            actual_seqlen_q = s_kv // cp_deg
        elif s_q:
            actual_seqlen_q = s_q
        else:
            actual_seqlen_q = s_kv

        # Sequence length validation
        if actual_seqlen_q > _MAX_SEQLEN or s_kv > _MAX_SEQLEN:
            return FilterResult.INVALID

        # Reproducible random sampling
        sample_rate = TestRangedAttentionCTEKernels._filter_state.get("sample_rate", 1.0)
        if sample_rate < 1.0:
            import hashlib

            # Create deterministic hash from all parameters
            param_str = f"{bs}_{gqa}_{d}_{s_kv}_{tp_q}_{tp_k}_{tp_out}_{sink}_{dtype}_{causal}_{s_q}_{cp_deg}_{cp_strided}_{sw}_{s_kv_prior}"
            seed = TestRangedAttentionCTEKernels._filter_state.get("seed", 42)
            hash_val = int(hashlib.md5(f"{seed}_{param_str}".encode()).hexdigest(), 16)

            # Use hash to decide if we keep this combination
            threshold = int(sample_rate * 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
            if hash_val > threshold:
                return FilterResult.REDUNDANT

        return FilterResult.VALID

    @staticmethod
    def make_limited_filter(max_samples=20, seed=42):
        """Create a filter function that limits to max_samples using reproducible random sampling."""

        def limited_filter(
            bs,
            gqa,
            d,
            s_kv,
            tp_q,
            tp_k,
            tp_out,
            sink,
            dtype,
            causal,
            s_q=None,
            cp_deg=None,
            cp_strided=None,
            sw=None,
            s_kv_prior=None,
        ):
            # First apply standard validation
            result = TestRangedAttentionCTEKernels.filter_attention_cte_combinations(
                bs, gqa, d, s_kv, tp_q, tp_k, tp_out, sink, dtype, causal, s_q, cp_deg, cp_strided, sw, s_kv_prior
            )
            if result != FilterResult.VALID:
                return result

            # Then apply sampling
            import hashlib

            param_str = f"{bs}_{gqa}_{d}_{s_kv}_{tp_q}_{tp_k}_{tp_out}_{sink}_{dtype}_{causal}_{s_q}_{cp_deg}_{cp_strided}_{sw}_{s_kv_prior}"
            hash_val = int(hashlib.md5(f"{seed}_{param_str}".encode()).hexdigest(), 16)

            # Keep approximately max_samples out of every 10000 valid combinations
            sample_rate = max_samples / 10000.0
            threshold = int(sample_rate * 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
            if hash_val > threshold:
                return FilterResult.REDUNDANT

            return FilterResult.VALID

        return limited_filter

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=1)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    @pytest.mark.parametrize("softmax_scale", [0.125, 1.0])
    def test_ranged_attn_cte_sweep_nocp(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        softmax_scale,
        bs,
        gqa,
        d,
        s_kv,
        s_q,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        # Map dtype index to actual dtype
        dtype_map = {0: nl.bfloat16, 1: np.float16, 2: np.float32}
        dtype = dtype_map[dtype]

        # Calculate bs_kv based on GQA factor
        bs_kv = bs // gqa if bs % gqa == 0 else bs

        compiler_args = CompilerArgs()

        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: bool(causal),
                TP_Q_DIM_NAME: bool(tp_q),
                TP_OUT_DIM_NAME: bool(tp_out),
                TP_K_DIM_NAME: bool(tp_k),
                SINK_DIM_NAME: bool(sink),
            },
            bs=bs,
            bs_kv=bs_kv,
            cache_softmax=False,
            cp_degree=None,
            d=d,
            dtype=dtype,
            seqlen_kv=s_kv,
            seqlen_kv_prior=None,
            seqlen_q=s_q,
            sliding_window=0,
            softmax_scale=softmax_scale,
            cp_rank_id=None,
            use_cp=False,
            cp_strided_q_slicing=False,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(sliding_window=True),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=2)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    def test_ranged_attn_cte_sweep_nocp_swa(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa,
        d,
        s_kv,
        s_q,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        sw,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        test_options = {
            BATCH_DIM_NAME: bs,
            GQA_FACTOR_DIM_NAME: gqa,
            D_DIM_NAME: d,
            SEQLEN_KV_DIM_NAME: s_kv,
            SEQLEN_Q_DIM_NAME: s_q,
            TP_Q_DIM_NAME: tp_q,
            TP_K_DIM_NAME: tp_k,
            TP_OUT_DIM_NAME: tp_out,
            SINK_DIM_NAME: sink,
            DTYPE_DIM_NAME: dtype,
            CAUSAL_MASK_DIM_NAME: causal,
            SLIDING_WINDOW_DIM_NAME: sw,
        }

        compiler_args = CompilerArgs()
        self.run_range_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=test_options,
            collector=collector,
            lnc_degree=compiler_args.logical_nc_config,
            softmax_scale=1.0,
            skip_no_causal=True,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(prefix_caching=True),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=3)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    def test_ranged_attn_cte_sweep_nocp_pc(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa,
        d,
        s_kv,
        s_q,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        s_kv_prior,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        test_options = {
            BATCH_DIM_NAME: bs,
            GQA_FACTOR_DIM_NAME: gqa,
            D_DIM_NAME: d,
            SEQLEN_KV_DIM_NAME: s_kv,
            SEQLEN_Q_DIM_NAME: s_q,
            TP_Q_DIM_NAME: tp_q,
            TP_K_DIM_NAME: tp_k,
            TP_OUT_DIM_NAME: tp_out,
            SINK_DIM_NAME: sink,
            DTYPE_DIM_NAME: dtype,
            CAUSAL_MASK_DIM_NAME: causal,
            SEQLEN_KV_PRIOR_DIM_NAME: s_kv_prior,
        }

        compiler_args = CompilerArgs()
        self.run_range_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=test_options,
            collector=collector,
            lnc_degree=compiler_args.logical_nc_config,
            softmax_scale=1.0,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(sliding_window=True, prefix_caching=True),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=4)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    @pytest.mark.parametrize("lnc_degree", [1, 2])
    def test_ranged_attn_cte_sweep_nocp_pc_swa(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        lnc_degree,
        bs,
        gqa,
        d,
        s_kv,
        s_q,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        sw,
        s_kv_prior,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        test_options = {
            BATCH_DIM_NAME: bs,
            GQA_FACTOR_DIM_NAME: gqa,
            D_DIM_NAME: d,
            SEQLEN_KV_DIM_NAME: s_kv,
            SEQLEN_Q_DIM_NAME: s_q,
            TP_Q_DIM_NAME: tp_q,
            TP_K_DIM_NAME: tp_k,
            TP_OUT_DIM_NAME: tp_out,
            SINK_DIM_NAME: sink,
            DTYPE_DIM_NAME: dtype,
            CAUSAL_MASK_DIM_NAME: causal,
            SLIDING_WINDOW_DIM_NAME: sw,
            SEQLEN_KV_PRIOR_DIM_NAME: s_kv_prior,
        }

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_range_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=test_options,
            collector=collector,
            lnc_degree=lnc_degree,
            softmax_scale=1.0,
            skip_no_causal=True,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(context_parallel=True),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=5)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    def test_ranged_attn_cte_sweep_cp(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa,
        d,
        s_kv,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        cp_deg,
        cp_strided,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        test_options = {
            BATCH_DIM_NAME: bs,
            GQA_FACTOR_DIM_NAME: gqa,
            D_DIM_NAME: d,
            SEQLEN_KV_DIM_NAME: s_kv,
            TP_Q_DIM_NAME: tp_q,
            TP_K_DIM_NAME: tp_k,
            TP_OUT_DIM_NAME: tp_out,
            SINK_DIM_NAME: sink,
            DTYPE_DIM_NAME: dtype,
            CAUSAL_MASK_DIM_NAME: causal,
            CP_DEGREE_DIM_NAME: cp_deg,
            CP_STRIDED_Q_DIM_NAME: cp_strided,
        }

        compiler_args = CompilerArgs()
        self.run_range_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=test_options,
            collector=collector,
            lnc_degree=compiler_args.logical_nc_config,
            softmax_scale=1.0,
            skip_no_causal=True,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(context_parallel=True, sliding_window=True),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=6)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    def test_ranged_attn_cte_sweep_cp_swa(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa,
        d,
        s_kv,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        cp_deg,
        cp_strided,
        sw,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        test_options = {
            BATCH_DIM_NAME: bs,
            GQA_FACTOR_DIM_NAME: gqa,
            D_DIM_NAME: d,
            SEQLEN_KV_DIM_NAME: s_kv,
            TP_Q_DIM_NAME: tp_q,
            TP_K_DIM_NAME: tp_k,
            TP_OUT_DIM_NAME: tp_out,
            SINK_DIM_NAME: sink,
            DTYPE_DIM_NAME: dtype,
            CAUSAL_MASK_DIM_NAME: causal,
            CP_DEGREE_DIM_NAME: cp_deg,
            CP_STRIDED_Q_DIM_NAME: cp_strided,
            SLIDING_WINDOW_DIM_NAME: sw,
        }

        compiler_args = CompilerArgs()
        self.run_range_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=test_options,
            collector=collector,
            lnc_degree=compiler_args.logical_nc_config,
            softmax_scale=1.0,
            skip_no_causal=True,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(context_parallel=True, prefix_caching=True),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=7)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    def test_ranged_attn_cte_sweep_cp_pc(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa,
        d,
        s_kv,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        cp_deg,
        cp_strided,
        s_kv_prior,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        test_options = {
            BATCH_DIM_NAME: bs,
            GQA_FACTOR_DIM_NAME: gqa,
            D_DIM_NAME: d,
            SEQLEN_KV_DIM_NAME: s_kv,
            TP_Q_DIM_NAME: tp_q,
            TP_K_DIM_NAME: tp_k,
            TP_OUT_DIM_NAME: tp_out,
            SINK_DIM_NAME: sink,
            DTYPE_DIM_NAME: dtype,
            CAUSAL_MASK_DIM_NAME: causal,
            CP_DEGREE_DIM_NAME: cp_deg,
            CP_STRIDED_Q_DIM_NAME: cp_strided,
            SEQLEN_KV_PRIOR_DIM_NAME: s_kv_prior,
        }

        compiler_args = CompilerArgs()
        self.run_range_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=test_options,
            collector=collector,
            lnc_degree=compiler_args.logical_nc_config,
            softmax_scale=1.0,
            skip_no_causal=True,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_attention_cte_config(context_parallel=True, prefix_caching=True, sliding_window=True),
        filter=lambda *args, **kwargs: TestRangedAttentionCTEKernels.make_limited_filter(max_samples=20, seed=8)(
            *args, **kwargs
        ),
        coverage="pairs",
    )
    @pytest.mark.parametrize("lnc_degree", [1, 2])
    def test_ranged_attn_cte_sweep_cp_pc_swa(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        lnc_degree,
        bs,
        gqa,
        d,
        s_kv,
        tp_q,
        tp_k,
        tp_out,
        sink,
        dtype,
        causal,
        cp_deg,
        cp_strided,
        sw,
        s_kv_prior,
        is_negative_test_case,
    ):
        if is_negative_test_case:
            pytest.skip("Negative test cases not implemented")

        test_options = {
            BATCH_DIM_NAME: bs,
            GQA_FACTOR_DIM_NAME: gqa,
            D_DIM_NAME: d,
            SEQLEN_KV_DIM_NAME: s_kv,
            TP_Q_DIM_NAME: tp_q,
            TP_K_DIM_NAME: tp_k,
            TP_OUT_DIM_NAME: tp_out,
            SINK_DIM_NAME: sink,
            DTYPE_DIM_NAME: dtype,
            CAUSAL_MASK_DIM_NAME: causal,
            CP_DEGREE_DIM_NAME: cp_deg,
            CP_STRIDED_Q_DIM_NAME: cp_strided,
            SLIDING_WINDOW_DIM_NAME: sw,
            SEQLEN_KV_PRIOR_DIM_NAME: s_kv_prior,
        }

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_range_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=test_options,
            collector=collector,
            lnc_degree=lnc_degree,
            softmax_scale=1.0,
            skip_no_causal=True,
        )

    # fmt: off
    attn_cte_sweep_manual_nocp_params = \
        "bs, gqa_factor, seqlen_kv, seqlen_kv_prior, prior_used_len, seqlen_q, d, sliding_window, causal_mask, tp_q, tp_k, tp_out, sink"
    attn_cte_sweep_manual_nocp_perms = [
    # BATCH, GQA_FACTOR, SEQLEN_KV,  SEQLEN_KV_PRIOR,  PRIOR_USED_LEN,  SEQLEN_Q,  D,  SLIDING_WINDOW, CAUSAL_MASK,  TP_Q, TP_K, TP_OUT, SINK
      [8,     2,          16384,      None,             None,           16384,    128,  0,            True,         True, True, True,   True],
      [4,     2,          15000,      1092,             1000,           16384,    128,  0,            True,         True, True, True,   True],
      [2,     2,          32768,      None,             None,           32768,    128,  0,            False,        False,False,False,  False],
      [2,     1,          32768,      None,             None,           32768,    64,   1,            True,         True,False, False,  False],
      [1,     1,          32768,      None,             None,           32768,    128,  2,            True,         False,True, False,  False],
      [2,     1,          32768,      None,             None,           32768,    128,  3,            True,         False,False,True,   False],
      [1,     1,          32768,      None,             None,           32768,    64,   128,          True,         False,False,False,  True],
      [2,     1,          32768,      None,             None,           32768,    64,   2048,         True,         False,False,False,  False],
      [1,     1,          16384,      16384,            0,              16384,    128,  0,            False,        True, False,True,   True],
      [1,     1,          16384,      16384,            15873,          16384,    128,  1,            True,         True, True, False,  True],
      [2,     2,          123,        30909,            30000,          123,      128,  0,            True,         False,True, False,  True],
      [2,     1,          123,        30909,            1,              123,      128,  10000,        True,         True, True, True,   False],
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(attn_cte_sweep_manual_nocp_params, attn_cte_sweep_manual_nocp_perms)
    @pytest.mark.parametrize("lnc_degree", [2])  # keep lnc 2 onlt
    @pytest.mark.parametrize("cache_softmax", [True])  # keep only cache_softmax true case
    def test_ranged_attn_cte_sweep_nocp_manual(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa_factor,
        seqlen_kv,
        seqlen_kv_prior,
        prior_used_len,
        seqlen_q,
        d,
        sliding_window,
        causal_mask,
        tp_q,
        tp_k,
        tp_out,
        sink,
        lnc_degree,
        cache_softmax,
    ):
        is_negative_test_case = False
        if seqlen_q % 128 != 0 and cache_softmax:
            is_negative_test_case = True
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        with assert_negative_test_case(is_negative_test_case):
            self.run_attention_cte_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                bool_dims={
                    CAUSAL_MASK_DIM_NAME: causal_mask,
                    TP_Q_DIM_NAME: tp_q,
                    TP_OUT_DIM_NAME: tp_out,
                    TP_K_DIM_NAME: tp_k,
                    SINK_DIM_NAME: sink,
                },
                bs=bs,
                bs_kv=bs // gqa_factor,
                cache_softmax=cache_softmax,
                cp_degree=None,
                d=d,
                dtype=nl.bfloat16,
                seqlen_kv=seqlen_kv,
                seqlen_kv_prior=seqlen_kv_prior,
                prior_used_len=prior_used_len,
                seqlen_q=seqlen_q,
                sliding_window=sliding_window,
                softmax_scale=1.0,
                cp_rank_id=None,
                use_cp=False,
            )

    # fmt: off
    attn_cte_sweep_manual_cp_params = \
        "bs, gqa_factor, seqlen_kv, seqlen_kv_prior, prior_used_len, cp_degree, cp_rank_id, d, sliding_window, causal_mask, tp_q, tp_k, tp_out, sink"
    attn_cte_sweep_manual_cp_perms = [
    # BATCH, GQA_FACTOR, SEQLEN_KV,  SEQLEN_KV_PRIOR,  PRIOR_USED_LEN,  CP_DEGREE, CP_RANK_ID,  D,  SLIDING_WINDOW, CAUSAL_MASK,  TP_Q, TP_K, TP_OUT, SINK
      [2,     2,          32768,      None,             None,           2,          1,          128,  0,            True,         False,True, True,   True],
      [2,     1,          32768,      None,             None,           8,          0,          128,  1024,         True,         True, False,True,   True],
      [2,     2,          32768,      None,             None,           32,         16,         128,  0,            True,         True, True, False,  True],
      [1,     1,          32768,      None,             None,           32,         10,         128,  0,            True,         False,False,False,  False],
      [1,     1,          32768,      None,             None,           32,         31,         128,  2,            True,         False,False,False,  False],
      [1,     1,          32768,      None,             None,           32,         31,         128,  127,          True,         True, True, True,   False],
      [2,     2,          32768,      None,             None,           32,         13,         128,  30000,        True,         True, True, True,   False],
      [3,     3,          25600,      5000,             4500,           5,          2,          63,   128,          True,         False,False,False,  False],
      [3,     3,          17000,      15000,            500,            17,         16,         127,  0,            True,         True, True, True,   True],
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(attn_cte_sweep_manual_cp_params, attn_cte_sweep_manual_cp_perms)
    @pytest.mark.parametrize("lnc_degree", [1, 2])
    @pytest.mark.parametrize("cp_strided_q_slicing", [False, True])
    @pytest.mark.parametrize("cache_softmax", [False, True])
    def test_ranged_attn_cte_sweep_cp_manual(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa_factor,
        seqlen_kv,
        seqlen_kv_prior,
        prior_used_len,
        cp_degree,
        cp_rank_id,
        d,
        sliding_window,
        causal_mask,
        tp_q,
        tp_k,
        tp_out,
        sink,
        lnc_degree,
        cp_strided_q_slicing,
        cache_softmax,
    ):
        seqlen_q = seqlen_kv // cp_degree
        is_negative_test_case = False
        if seqlen_q % 128 != 0 and cache_softmax:
            is_negative_test_case = True
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        with assert_negative_test_case(is_negative_test_case):
            self.run_attention_cte_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                bool_dims={
                    CAUSAL_MASK_DIM_NAME: causal_mask,
                    TP_Q_DIM_NAME: tp_q,
                    TP_OUT_DIM_NAME: tp_out,
                    TP_K_DIM_NAME: tp_k,
                    SINK_DIM_NAME: sink,
                },
                bs=bs,
                bs_kv=bs // gqa_factor,
                cache_softmax=cache_softmax,
                cp_degree=cp_degree,
                d=d,
                dtype=nl.bfloat16,
                seqlen_kv=seqlen_kv,
                seqlen_kv_prior=seqlen_kv_prior,
                prior_used_len=prior_used_len,
                seqlen_q=seqlen_q,
                sliding_window=sliding_window,
                softmax_scale=1.0,
                cp_rank_id=cp_rank_id,
                use_cp=True,
                cp_strided_q_slicing=cp_strided_q_slicing,
            )

    # fmt: off
    attention_cte_model_config_params = "bs, gqa_factor, seqlen_kv, seqlen_kv_prior, prior_used_len, cp_degree, cp_rank_id, d, sliding_window, causal_mask, tp_q, tp_k, tp_out, sink"
    attention_cte_model_test_ids = [
        f"{MODEL_TEST_TYPE}_" + "-".join(str(p.value) if hasattr(p, "value") else str(p) for p in params)
        for params in attention_cte_model_configs
    ]
    # fmt: on

    @pytest.mark.parametrize(
        attention_cte_model_config_params, attention_cte_model_configs, ids=attention_cte_model_test_ids
    )
    def test_attention_cte_model(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bs,
        gqa_factor,
        seqlen_kv,
        seqlen_kv_prior,
        prior_used_len,
        cp_degree,
        cp_rank_id,
        d,
        sliding_window,
        causal_mask,
        tp_q,
        tp_k,
        tp_out,
        sink,
    ):
        test_metadata_key = {k: v for k, v in locals().items() if k not in _METADATA_EXCLUDE_KEYS}

        seqlen_q = seqlen_kv // cp_degree
        attention_cte_metadata_list = _get_attention_cte_metadata()
        collector.match_and_add_metadata_dimensions(test_metadata_key, attention_cte_metadata_list)

        compiler_args = CompilerArgs(logical_nc_config=1)
        self.run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: causal_mask,
                TP_Q_DIM_NAME: tp_q,
                TP_K_DIM_NAME: tp_k,
                TP_OUT_DIM_NAME: tp_out,
                SINK_DIM_NAME: sink,
            },
            bs=bs,
            bs_kv=bs // gqa_factor,
            cache_softmax=False,
            cp_degree=cp_degree,
            d=d,
            dtype=nl.bfloat16,
            seqlen_kv=seqlen_kv,
            seqlen_kv_prior=seqlen_kv_prior,
            prior_used_len=prior_used_len,
            seqlen_q=seqlen_q,
            sliding_window=sliding_window,
            softmax_scale=1.0,
            cp_rank_id=cp_rank_id,
            use_cp=True,
            cp_strided_q_slicing=False,
        )
