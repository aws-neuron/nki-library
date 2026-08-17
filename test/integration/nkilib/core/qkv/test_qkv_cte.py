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

# Sweep test uses @pytest.mark.coverage_parametrize; unit tests use @pytest_parametrize

"""
Integration tests for QKV CTE kernel.

Tests cover various configurations including fused operations (normalization, residual add, RoPE),
different data types, batch sizes, sequence lengths, and output layouts.
"""

try:
    from test.integration.nkilib.core.qkv.test_qkv_cte_model_config import (
        FUSED_GAMMA_ROPE_MODELS,
        QK_NORM_MODELS,
        STATIC_DEQUANT_MODELS,
        _get_sharded_head_counts,
        qkv_cte_model_configs,
    )
    from test.integration.nkilib.core.qkv.test_qkv_cte_model_config import (
        MODELS as _QKV_MODELS,
    )
except ImportError:
    QK_NORM_MODELS = set()
    STATIC_DEQUANT_MODELS = set()
    FUSED_GAMMA_ROPE_MODELS = set()
    qkv_cte_model_configs = {}
    _QKV_MODELS = {}
    _get_sharded_head_counts = None

import math
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.qkv.qkv import qkv
from nkilib_src.nkilib.core.qkv.qkv_torch import qkv_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    DtypeMode,
    NormType,
    QKNormConfig,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
    StridedInputConfig,
)
from nkilib_src.nkilib.experimental.qkv.qkv_cte_mla import qkv_mla_mx, qkv_mla_mx_deepseek_v4
from nkilib_src.nkilib.experimental.qkv.qkv_cte_mla_torch import qkv_mla_mx_deepseek_v4_torch_ref, qkv_mla_mx_torch_ref

from test.integration.nkilib.core.qkv.test_qkv_common import (
    build_noncontiguous_slot_mapping,
    build_qkv_input,
    build_qkv_mla_input,
    fuse_gamma_into_rope_caches,
    gamma_unfused_ref,
    real_rope_tensor_generator,
    rope_gaussian_tensor_generator,
    row_mx_unpack_ref,
    run_qkv_test,
    strided_gather_ref,
)
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.coverage_parametrized_tests import BoundedRange
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Constructed once at module load and shared across all callers that rely on the default,
# matching the previous behavior where the default argument expression was evaluated once.
_DEFAULT_TENSOR_GEN = gaussian_tensor_generator()


@pytest_test_metadata(name="QKV CTE", tags=["model"])
@pytest_marks(["qkv", "cte", "mx"])
@final
class TestQkvCteKernel:
    def run_qkv_cte_test_utf(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        B: int,
        H: int,
        S: int,
        fused_qkv_dim: int,
        lnc_degree: int,
        dtype,
        eps: float,
        norm_type: NormType,
        use_dma_transpose: bool = True,
        fused_add: bool = True,
        norm_bias: bool = False,
        qkv_bias: bool = False,
        fused_rope: bool = False,
        output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
        hidden_actual: int | None = None,
        n_q_heads: int | None = None,
        n_kv_heads: int | None = None,
        d_head: int | None = None,
        quantization_type: QuantizationType = QuantizationType.NONE,
        tensor_gen=_DEFAULT_TENSOR_GEN,
        fp8_kv_cache: bool = False,
        bf16_kv_cache: bool = False,
        max_seq_len: int | None = None,
        k_scale_val: float | None = None,
        v_scale_val: float | None = None,
        fp8_max: float = 240.0,
        fp8_min: float = -240.0,
        use_block_kv: bool = False,
        fp8_packed: bool = False,
        transpose_k_cache: bool = False,
        num_blocks: int | None = None,
        block_size: int | None = None,
        slot_mapping: np.ndarray | None = None,
        is_h_dim_4h_transposed: bool = False,
        rtol: float = 2e-2,
        atol: float = 1e-5,
        qkv_in_scale_for_mx: np.ndarray | None = None,
        qkv_w_scale_for_mx: np.ndarray | None = None,
        preserve_lower_precision: bool = False,
        is_negative_test: bool = False,
        qk_norm_pre_rope_config: dict | None = None,
        qk_norm_post_rope_config: dict | None = None,
        dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    ):
        """Run a QKV CTE test using the shared run_qkv_test helper."""
        run_qkv_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S=S,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=lnc_degree,
            dtype=dtype,
            eps=eps,
            norm_type=norm_type,
            fused_add=fused_add,
            norm_bias=norm_bias,
            qkv_bias=qkv_bias,
            output_layout=output_layout,
            hidden_actual=hidden_actual,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            quantization_type=quantization_type,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            use_dma_transpose=use_dma_transpose,
            fused_rope=fused_rope,
            tensor_gen=tensor_gen,
            fp8_kv_cache=fp8_kv_cache,
            bf16_kv_cache=bf16_kv_cache,
            transpose_k_cache=transpose_k_cache,
            max_seq_len=max_seq_len,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            use_block_kv=use_block_kv,
            fp8_packed=fp8_packed,
            num_blocks=num_blocks,
            block_size=block_size,
            slot_mapping=slot_mapping,
            qkv_in_scale_for_mx=qkv_in_scale_for_mx,
            qkv_w_scale_for_mx=qkv_w_scale_for_mx,
            preserve_lower_precision=preserve_lower_precision,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            qk_norm_post_rope_config=qk_norm_post_rope_config,
            rtol=rtol,
            atol=atol,
            is_negative_test=is_negative_test,
            # CTE kernel does not return fused_hidden as a separate output
            expect_fused_hidden_output=False,
            dtype_mode=dtype_mode,
        )

    ################################################################################################
    # QKV RoPE FUSION TEST
    ################################################################################################
    # fmt: off
    qkv_cte_kernel_fused_rope_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, output_layout, eps, fused_rope"
    qkv_cte_kernel_fused_rope_test_perms = [
        [2, 1, 128, 8192, 1, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 192, 8192, 8, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 192, 8320, 8, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 1, 1, 128, NormType.RMS_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 2, 1, 128, NormType.RMS_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 8, 1, 128, NormType.RMS_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 1, 1, 128, NormType.NO_NORM, True, False, True, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 2, 1, 128, NormType.NO_NORM, True, False, True, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 8, 1, 64, NormType.NO_NORM, True, False, True, False, QKVOutputLayout.BSD, 1e-6, True],
    ]
    # fmt: on
    @pytest_parametrize(
        qkv_cte_kernel_fused_rope_test_params,
        qkv_cte_kernel_fused_rope_test_perms,
    )
    def test_qkv_cte_fused_rope_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        use_dma_transpose,
        fused_add,
        add_bias,
        norm_bias,
        output_layout,
        eps,
        fused_rope,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_add,
            norm_bias=norm_bias,
            fused_rope=fused_rope,
            output_layout=output_layout,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=rope_gaussian_tensor_generator(),
            rtol=2e-2,
            atol=1e-5,
        )

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_accuracy
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_accuracy_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, dtype, norm_type, fused_add, norm_bias, eps"
    qkv_cte_kernel_accuracy_test_perms = [
        # Data type test
        pytest.param(1, 1, 128, 512, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6, marks=pytest.mark.fast),
        [1, 1, 128, 512, None, 512, np.float16, NormType.RMS_NORM, True, False, 1e-6],
        [1, 1, 128, 512, None, 512, np.float32, NormType.RMS_NORM, True, False, 1e-6],
        # d < 512
        [1, 1, 512, 1024, None, 256, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        [1, 1, 512, 1024, 1024, 256, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        [1, 1, 512, 1024, 768, 256, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        [1, 1, 512, 1024, None, 384, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        # sequence sweep
        [1, 1, 127, 512, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        [1, 1, 128, 512, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        pytest.param(1, 1, 1000, 512, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6, marks=pytest.mark.fast),
        # H sweep
        # pytest.param(1,1,128,283,None,512,nl.bfloat16,NormType.RMS_NORM,True,False, 1e-6),
        # Marked as skip: 'H must be multiple of 128'
        # [1, 1, 128, 283, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        # Combinations of the fused flags
        [1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        pytest.param(1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.NO_NORM, True, False, 1e-6, marks=pytest.mark.fast),
        [1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.RMS_NORM, False, False, 1e-6],
        pytest.param(1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6, marks=pytest.mark.fast),
        # Test large eps to verify eps is working
        [1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 77.0],
        ### Large hidden QKV kernel test parameters
        # Data type test
        [1, 1, 128, 16384, None, 512, np.float16, NormType.NO_NORM, False, False, 1e-6],
        [1, 1, 128, 16384, None, 512, np.float32, NormType.NO_NORM, False, False, 1e-6],
        # d sweep
        [1, 1, 512, 16384, None, 256, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6],
        [1, 1, 512, 16384, None, 384, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6],
        [1, 1, 512, 16384, None, 768, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6],
        # pytest.param(1, 1, 512, 16384, None, 1024, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6, marks=xfail(passes='trn1')),
        # H sweep
        # pytest.param(1, 1, 128, 8193, None, 512, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6, marks=pytest.mark.skip(reason='H must be multiple of 128')),
        # sequence sweep
        # pytest.param(1,1,512,16384,None,512,nl.bfloat16,NormType.RMS_NORM,True,False,1e-6),
        [1, 1, 512, 16384, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        # pytest.param(1,1,128,16384,None,512,nl.bfloat16,NormType.RMS_NORM,False,False,1e-6),
        [1, 1, 128, 16384, None, 512, nl.bfloat16, NormType.RMS_NORM, False, False, 1e-6],
        # pytest.param(1,1,128,16384,16384,512,nl.bfloat16,NormType.RMS_NORM,False,False,1e-6),
        pytest.param(1, 1, 128, 16384, 16384, 512, nl.bfloat16, NormType.RMS_NORM, False, False, 1e-6, marks=pytest.mark.fast),
        # pytest.param(1,1,128,16384,15104,512,nl.bfloat16,NormType.RMS_NORM,False,False,1e-6),
        [1, 1, 128, 16384, 15104, 512, nl.bfloat16, NormType.RMS_NORM, False, False, 1e-6],
        # pytest.param(1,1,8192,16384,None,512,nl.bfloat16,NormType.NO_NORM,False,False,1e-6),
        [1, 1, 8192, 16384, None, 512, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6],
        # Combinations of the fused flags
        # pytest.param(1,1,512,16384,None,512,nl.bfloat16,NormType.RMS_NORM,True,False,1e-6),
        [1, 1, 512, 16384, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        [1, 1, 512, 16384, None, 512, nl.bfloat16, NormType.NO_NORM, True, False, 1e-6],
        # pytest.param(1,1,512,16384,None,512,nl.bfloat16,NormType.RMS_NORM,False,False,1e-6),
        [1, 1, 512, 16384, None, 512, nl.bfloat16, NormType.RMS_NORM, False, False, 1e-6],
        [1, 1, 512, 16384, None, 512, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6],
        # Test large eps to verify eps is working
        # pytest.param(1,1,512,16384,None,512,nl.bfloat16, NormType.RMS_NORM, True, False, 77.0),
        [1, 1, 512, 16384, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 77.0],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_kernel_accuracy_test_params,
        qkv_cte_kernel_accuracy_test_perms,
    )
    def test_qkv_cte_accuracy_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        hidden_actual,
        fused_qkv_dim,
        dtype,
        norm_type,
        fused_add,
        norm_bias,
        eps,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            fused_add=fused_add,
            norm_bias=norm_bias,
            hidden_actual=hidden_actual,
        )

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_performance_bsd
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_performance_bsd_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, fused_qkv_dim, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, eps"
    qkv_cte_kernel_performance_bsd_test_perms = [
        # New model, 2025-Jul
        [1, 1, 128, 3072, 1280, NormType.NO_NORM, True, False, True, False, 1e-6],
        [2, 1, 1024, 3072, 1280, NormType.NO_NORM, True, False, True, False, 1e-6],
        [2, 1, 2048, 3072, 1280, NormType.NO_NORM, True, False, True, False, 1e-6],
        # LLaMA3.1 405B Config SP
        [2, 1, 1024, 16384, 512, NormType.NO_NORM, True, False, False, False, 1e-6],
        [2, 1, 8192, 16384, 512, NormType.NO_NORM, True, False, False, False, 1e-6],
        [1, 1, 1024, 16384, 384, NormType.NO_NORM, True, False, False, False, 1e-6],
        [1, 1, 8192, 16384, 384, NormType.NO_NORM, True, False, False, False, 1e-6],
        # 405B SP Multibatch
        [1, 2, 1024, 16384, 512, NormType.NO_NORM, True, False, False, False, 1e-6],
        [2, 2, 1024, 16384, 512, NormType.NO_NORM, True, False, False, False, 1e-6],
        # LLaMA3 70B SP
        [2, 1, 16384, 8192, 512, NormType.NO_NORM, True, False, False, False, 1e-6],
        # TP
        [2, 1, 1024, 16384, 512, NormType.RMS_NORM, True, True, False, False, 1e-6],
        [2, 1, 8192, 16384, 512, NormType.RMS_NORM, True, True, False, False, 1e-6],
        # pytest.param(1, 1, 1024, 16384, 384, NormType.RMS_NORM, True, True, False, False, 1e-6),
        [1, 1, 1024, 16384, 384, NormType.RMS_NORM, True, True, False, False, 1e-6],
        # pytest.param(1, 1, 8192, 16384, 384, NormType.RMS_NORM, True, True, False, False, 1e-6),
        [1, 1, 8192, 16384, 384, NormType.RMS_NORM, True, True, False, False, 1e-6],
        # Text
        [2, 1, 256, 7168, 384, NormType.RMS_NORM, True, True, False, False, 1e-6],
        [2, 1, 256, 7168, 384, NormType.RMS_NORM_SKIP_GAMMA, True, True, False, False, 1e-6],
        # Small seqlen
        [2, 1, 128, 8192, 384, NormType.RMS_NORM, True, True, False, False, 1e-6],
        # L4 Vision Encoder
        [1, 1, 578, 1408, 264, NormType.RMS_NORM, True, False, True, False, 1e-6],
        [1, 1, 578, 1408, 264, NormType.RMS_NORM, True, True, True, False, 1e-6],
        [1, 1, 578, 1408, 264, NormType.LAYER_NORM, True, True, True, True, 1e-6],
        [1, 12, 578, 1408, 264, NormType.RMS_NORM, True, False, True, False, 1e-6],
        [1, 12, 578, 1408, 264, NormType.RMS_NORM, True, True, True, False, 1e-6],
        [2, 1, 578, 1408, 264, NormType.RMS_NORM, True, False, False, False, 1e-6],
        [2, 1, 578, 1408, 264, NormType.RMS_NORM, True, False, True, False, 1e-6],
        [2, 1, 578, 1408, 264, NormType.RMS_NORM, True, True, True, False, 1e-6],
        [2, 12, 578, 1408, 264, NormType.RMS_NORM, True, False, False, False, 1e-6],
        [2, 12, 578, 1408, 264, NormType.RMS_NORM, True, False, True, False, 1e-6],
        [2, 12, 578, 1408, 264, NormType.RMS_NORM, True, True, True, False, 1e-6],
        [2, 14, 578, 1408, 264, NormType.RMS_NORM, True, False, True, False, 1e-6],
        [2, 14, 578, 1408, 264, NormType.RMS_NORM, True, True, True, False, 1e-6],
        [2, 14, 578, 1408, 264, NormType.LAYER_NORM, True, True, True, True, 1e-6],
        [2, 14, 578, 1408, 264, NormType.LAYER_NORM, True, False, True, True, 1e-6],
        # 123B
        [2, 8, 2048, 12288, 640, NormType.RMS_NORM, True, False, False, False, 1e-6],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_kernel_performance_bsd_test_params,
        qkv_cte_kernel_performance_bsd_test_perms,
    )
    def test_qkv_cte_performance_bsd_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        fused_qkv_dim,
        norm_type,
        use_dma_transpose,
        fused_add,
        add_bias,
        norm_bias,
        eps,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_add,
            qkv_bias=add_bias,
            norm_bias=norm_bias,
            rtol=2e-2,
            atol=1e-5,
        )

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_performance_nbsd
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_performance_nbsd_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, output_layout, eps"
    qkv_cte_kernel_performance_nbsd_test_perms = [
        # 405B
        [2, 1, 1024, 16384, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # 70B
        [2, 1, 1024, 8192, 2, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # PS
        [2, 1, 2048, 8448, 5, 5, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # DIT
        [2, 1, 35520, 4096, 2, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # small unittests
        [2, 1, 512, 256, 2, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # higher batch sizes to validate moving N above B
        [2, 4, 512, 256, 8, 2, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # 470B
        [2, 1, 1024, 20480, 3, 1, 128, NormType.RMS_NORM, True, False, False, False, QKVOutputLayout.NBSd, 1e-6],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_kernel_performance_nbsd_test_params,
        qkv_cte_kernel_performance_nbsd_test_perms,
    )
    def test_qkv_cte_performance_nbsd_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        use_dma_transpose,
        fused_add,
        add_bias,
        norm_bias,
        output_layout,
        eps,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_add,
            norm_bias=norm_bias,
            output_layout=output_layout,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            rtol=2e-2,
            atol=1e-5,
        )

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_no_dma_transpose
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_no_dma_transpose_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, output_layout, eps"
    qkv_cte_kernel_no_dma_transpose_test_perms = [
        # 405B
        [2, 1, 1024, 16384, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        [2, 1, 1024, 16384, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 8192, 16384, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        # 70B
        [2, 1, 1024, 8192, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 16384, 8192, 1, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 8192, 1, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 16384, 8192, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 8192, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        # PS
        [2, 1, 2048, 8448, 5, 5, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # DIT
        [2, 1, 35520, 4096, 2, 2, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # small unittests
        [2, 1, 512, 256, 2, 2, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # higher batch sizes to validate moving N above B
        [2, 4, 512, 256, 8, 2, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # 470B
        [2, 1, 1024, 20480, 3, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.NBSd, 1e-6],
        # L4 Vision
        [2, 1, 578, 1408, 1, 1, 88, NormType.RMS_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 12, 578, 1408, 1, 1, 88, NormType.RMS_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_kernel_no_dma_transpose_test_params,
        qkv_cte_kernel_no_dma_transpose_test_perms,
    )
    def test_qkv_cte_no_dma_transpose_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        use_dma_transpose,
        fused_add,
        add_bias,
        norm_bias,
        output_layout,
        eps,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_add,
            norm_bias=norm_bias,
            output_layout=output_layout,
            hidden_actual=None,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=rope_gaussian_tensor_generator(),
            rtol=2e-2,
            atol=1e-5,
        )

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_static_quantization
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_static_quantization_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, output_layout, eps"
    qkv_cte_kernel_static_quantization_test_perms = [
        # 70B
        [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 1024, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 16384, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 1024, 8192, 8, 1, 128, NormType.NO_NORM, False, False, True, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, False, False, True, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 1024, 8192, 8, 1, 128, NormType.RMS_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 8192, 8, 1, 128, NormType.RMS_NORM, False, True, False, False, QKVOutputLayout.BSD, 1e-6],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_kernel_static_quantization_test_params,
        qkv_cte_kernel_static_quantization_test_perms,
    )
    def test_qkv_cte_static_quantization(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        use_dma_transpose,
        fused_add,
        add_bias,
        norm_bias,
        output_layout,
        eps,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_add,
            qkv_bias=add_bias,
            norm_bias=norm_bias,
            output_layout=output_layout,
            hidden_actual=None,
            d_head=d_head,
            quantization_type=QuantizationType.STATIC,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            rtol=5e-2,
            atol=2e-2,
        )

        # nl.float8_e4m3

    qkv_cte_kernel_static_quantization_fp8_inputs_test_params = (
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, output_layout, eps"
    )
    qkv_cte_kernel_static_quantization_fp8_inputs_test_perms = [
        # 70B
        [2, 1, 128, 8192, 8, 1, 128, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 1024, 8192, 8, 1, 128, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 16384, 8192, 8, 1, 128, QKVOutputLayout.BSD, 1e-6],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_kernel_static_quantization_fp8_inputs_test_params,
        qkv_cte_kernel_static_quantization_fp8_inputs_test_perms,
    )
    def test_qkv_cte_static_quantization_fp8_inputs(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        eps,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.float8_e4m3,
            eps=eps,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            qkv_bias=False,
            norm_bias=False,
            output_layout=output_layout,
            hidden_actual=None,
            d_head=d_head,
            quantization_type=QuantizationType.STATIC,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            rtol=5e-2,
            atol=2e-2,
        )

    ####################################################################################################################
    # FP8 quant mode canary for QKV CTE.
    # Tests NON_OCP, OCP, AUTO across STATIC and ROW.
    ####################################################################################################################
    _QKV_CTE_BY_DTYPE_MODE_CONFIG = {
        "B": 1,
        "H": 8192,
        "S": 128,
        "lnc_degree": 2,
        "dtype": nl.bfloat16,
        "eps": 1e-6,
        "norm_type": NormType.NO_NORM,
        "use_dma_transpose": True,
        "fused_add": False,
        "qkv_bias": False,
        "norm_bias": False,
        "output_layout": QKVOutputLayout.BSD,
        "d_head": 128,
        "n_q_heads": 8,
        "n_kv_heads": 1,
        "rtol": 5e-2,
        "atol": 2e-2,
    }

    @pytest.mark.parametrize("dtype_mode", [DtypeMode.NON_OCP, DtypeMode.OCP, DtypeMode.AUTO])
    @pytest.mark.parametrize(
        "quantization_type",
        [QuantizationType.STATIC, QuantizationType.ROW],
    )
    def test_qkv_cte_by_dtype_mode(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        quantization_type: QuantizationType,
        dtype_mode: DtypeMode,
    ):
        """Smoke-test QKV CTE STATIC/ROW quant with explicit dtype_mode.

        NON_OCP → ``nl.float8_e4m3`` (240), any platform.
        OCP     → ``nl.float8_e4m3fn`` (448), TRN3 only.
        AUTO    → ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3`` elsewhere.
        """
        if dtype_mode == DtypeMode.OCP and not platform_target.is_trn3():
            pytest.skip("OCP dtype_mode requires TRN3")
        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
        cfg = self._QKV_CTE_BY_DTYPE_MODE_CONFIG
        n_q_heads, n_kv_heads, d_head = cfg["n_q_heads"], cfg["n_kv_heads"], cfg["d_head"]
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            fused_qkv_dim=fused_qkv_dim,
            quantization_type=quantization_type,
            dtype_mode=dtype_mode,
            **cfg,
        )

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_row_quantization
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_row_quantization_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, output_layout, eps"
    qkv_cte_kernel_row_quantization_test_perms = [
        # Mirror of STATIC tests with ROW quantization
        [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 1024, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 16384, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 1024, 8192, 8, 1, 128, NormType.NO_NORM, False, False, True, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, False, False, True, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 1024, 8192, 8, 1, 128, NormType.RMS_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 8192, 8, 1, 128, NormType.RMS_NORM, False, True, False, False, QKVOutputLayout.BSD, 1e-6],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_kernel_row_quantization_test_params,
        qkv_cte_kernel_row_quantization_test_perms,
    )
    def test_qkv_cte_row_quantization(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        use_dma_transpose,
        fused_add,
        add_bias,
        norm_bias,
        output_layout,
        eps,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_add,
            qkv_bias=add_bias,
            norm_bias=norm_bias,
            output_layout=output_layout,
            hidden_actual=None,
            d_head=d_head,
            quantization_type=QuantizationType.ROW,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            rtol=5e-2,
            atol=2e-2,
        )

    ####################################################################################################################
    # QKV MX CTE MX Test - test_qkv_cte_mx_kernel_accuracy
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_fused_rope_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, fused_add, add_bias, norm_bias, output_layout, eps, fused_rope, is_h_dim_4h_transposed"
    qkv_cte_kernel_fused_rope_test_perms = [
        [2, 1, 124, 512, 2, 1, 128, NormType.RMS_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 128, 512, 2, 1, 128, NormType.RMS_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 128, 512, 5, 5, 128, NormType.RMS_NORM, True, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 128, 512, 3, 1, 128, NormType.LAYER_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 128, 512, 2, 2, 128, NormType.RMS_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 512, 1024, 2, 2, 128, NormType.RMS_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 512, 1024, 2, 1, 128, NormType.RMS_NORM_SKIP_GAMMA, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 512, 1024, 2, 1, 128, NormType.NO_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 512, 1024, 5, 5, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.NBSd, 1e-6, True, False],
        [2, 1, 512, 1024, 2, 2, 128, NormType.LAYER_NORM, True, True, True, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 1024, 2048, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 1000, 512, 2, 1, 128, NormType.RMS_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 128, 8192, 2, 2, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 128, 16384, 3, 1, 128, NormType.LAYER_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 512, 16384, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 512, 16384, 1, 1, 128, NormType.RMS_NORM_SKIP_GAMMA, False, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 512, 16384, 2, 2, 128, NormType.RMS_NORM, True, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        [2, 1, 2048, 16384, 2, 1, 128, NormType.RMS_NORM, True, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 2048, 16384, 2, 1, 128, NormType.RMS_NORM, True, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 2048, 16384, 2, 1, 128, NormType.LAYER_NORM, True, True, True, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 2048, 16384, 1, 1, 128, NormType.RMS_NORM_SKIP_GAMMA, True, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 8192, 16384, 1, 1, 128, NormType.RMS_NORM, True, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 8192, 16384, 3, 1, 128, NormType.RMS_NORM, True, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 8192, 16384, 2, 1, 128, NormType.LAYER_NORM, True, True, False, QKVOutputLayout.BSD, 1e-6, True, False],
        [2, 1, 512, 2048, 2, 1, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 77.0, False, False],

        # --- Partial H (non-512-aligned) PE transpose path ---
        # H=896 NO_NORM basic
        [2, 1, 128, 896, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # H=896 RMS_NORM (accuracy-sensitive)
        [2, 1, 128, 896, 2, 1, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # H=896 RMS_NORM_SKIP_GAMMA
        [2, 1, 128, 896, 2, 1, 128, NormType.RMS_NORM_SKIP_GAMMA, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # H=896 LAYER_NORM
        [2, 1, 128, 896, 3, 1, 128, NormType.LAYER_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # H=896 LAYER_NORM + bias
        [2, 1, 128, 896, 2, 1, 128, NormType.LAYER_NORM, False, True, True, QKVOutputLayout.BSD, 1e-6, False, False],
        # H=896 RMS_NORM + RoPE
        [2, 1, 128, 896, 2, 1, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, True, False],
        # H=640 (h_pack_last=1)
        [2, 1, 128, 640, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # H=768 (h_pack_last=2) + RMS_NORM
        [2, 1, 128, 768, 2, 2, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # H=1024 regression (no partial tile)
        [2, 1, 128, 1024, 2, 1, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # Gemma 3 H=5376 + RMS_NORM
        [2, 1, 128, 5376, 2, 1, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, False],
        # Gemma 3 H=5376 + RMS_NORM + RoPE
        [2, 1, 128, 5376, 2, 1, 128, NormType.RMS_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, True, False],

        # Swizzled input tests
        [2, 1, 124, 512, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 1024, 512, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 1024, 512, 2, 1, 128, NormType.NO_NORM, False, True, False, QKVOutputLayout.BSD, 1e-6, True, True],
        [2, 1, 128, 512, 5, 5, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 128, 512, 2, 1, 128, NormType.NO_NORM, False, True, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 1000, 512, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, True, True],
        [2, 1, 512, 1024, 2, 1, 128, NormType.NO_NORM, False, True, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 512, 1024, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.NBSd, 1e-6, False, True],
        [2, 1, 1024, 2048, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 128, 16384, 3, 1, 128, NormType.NO_NORM, False, True, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 2048, 16384, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 2048, 16384, 2, 1, 128, NormType.NO_NORM, False, True, False, QKVOutputLayout.BSD, 1e-6, True, True],
        [2, 1, 8192, 16384, 1, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 1e-6, False, True],
        [2, 1, 8192, 16384, 1, 1, 128, NormType.NO_NORM, False, True, False, QKVOutputLayout.BSD, 1e-6, True, True],
        [2, 1, 512, 2048, 2, 1, 128, NormType.NO_NORM, False, False, False, QKVOutputLayout.BSD, 77.0, True, True],
    ]
    # fmt: on
    @pytest_parametrize(
        qkv_cte_kernel_fused_rope_test_params,
        qkv_cte_kernel_fused_rope_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_mxfp8_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        fused_add,
        add_bias,
        norm_bias,
        output_layout,
        eps,
        fused_rope,
        is_h_dim_4h_transposed,
    ):
        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            quantization_type=QuantizationType.MX,
            fused_add=fused_add,
            qkv_bias=add_bias,
            norm_bias=norm_bias,
            fused_rope=fused_rope,
            output_layout=output_layout,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=rope_gaussian_tensor_generator(),
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            rtol=5e-2,
            atol=1e-4,
        )

    # fmt: off
    qkv_cte_mxfp8_neutral_mx_scales_test_params = \
        "batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, output_layout, qkv_bias"
    qkv_cte_mxfp8_neutral_mx_scales_test_perms = [
        # --- Dynamic weight scales (original cases) ---
        # Small H (likely prefetch)
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Large H (likely chunked)
        [1, 128, 2048, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Multiple S tiles
        [1, 512, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Large S forcing multi-S-tile buffering
        [1, 2048, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # GQA config (many Q heads, few KV heads)
        [1, 128, 1024, 5, 1, 128, QKVOutputLayout.BSD, False],
        # Equal Q/KV heads
        [1, 256, 1024, 2, 2, 128, QKVOutputLayout.BSD, False],
        # NBSd output layout
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False],
        # Large H + large S
        [1, 512, 2048, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Non-tile-aligned S
        [1, 124, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Large H chunked, NBSd
        [1, 256, 2048, 2, 2, 128, QKVOutputLayout.NBSd, False],
        # Many heads, large S
        [1, 1024, 1024, 5, 5, 128, QKVOutputLayout.BSD, False],
        # Very large H
        [1, 128, 4096, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Minimal head config (1Q, 1KV)
        [1, 512, 1024, 1, 1, 128, QKVOutputLayout.BSD, False],
        # Non-aligned S=1000
        [1, 1000, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Very large H (8192)
        [1, 128, 8192, 2, 2, 128, QKVOutputLayout.BSD, False],
        # Very large H (16384)
        [1, 128, 16384, 3, 1, 128, QKVOutputLayout.BSD, False],
        # Large H + large S
        [1, 2048, 16384, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Very large S
        [1, 8192, 16384, 1, 1, 128, QKVOutputLayout.BSD, False],
        # Large S, NBSd
        [1, 512, 1024, 5, 5, 128, QKVOutputLayout.NBSd, False],
        # I=1024 (exactly 2 PSUM banks) - 4Q/2KV
        [1, 128, 512, 4, 2, 128, QKVOutputLayout.BSD, False],
        # I=256 (partial single PSUM bank) - 1Q/0.5KV not possible, use 1Q/1KV with partial
        # I=1152 (9 heads, not 512-aligned) - 7Q/1KV
        [1, 128, 1024, 7, 1, 128, QKVOutputLayout.BSD, False],
        # I=2048 (exactly 4 PSUM banks) - 8Q/4KV
        [1, 128, 1024, 8, 4, 128, QKVOutputLayout.BSD, False],
        # --- Both input and weight static scales ---
        # FP8 DMA transpose path with optional static weight scales (qkv_w_scale=None).
        # Small H (likely prefetch)
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Large H (likely chunked)
        [1, 128, 2048, 2, 1, 128, QKVOutputLayout.BSD, False],
        # Large S forcing multi-S-tile buffering
        [1, 2048, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # GQA config (many Q heads, few KV heads)
        [1, 128, 1024, 5, 1, 128, QKVOutputLayout.BSD, False],
        # NBSd output layout
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False],
        # Very large H (16384)
        [1, 128, 16384, 3, 1, 128, QKVOutputLayout.BSD, False],
        # --- Single S tile with static weight scales ---
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        # --- With QKV bias ---
        # Small H with bias
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True],
        # Large H (chunked) with bias
        [1, 128, 2048, 2, 1, 128, QKVOutputLayout.BSD, True],
        # Static weight scales + bias
        [1, 128, 1024, 5, 1, 128, QKVOutputLayout.BSD, True],
        # --- Partial H (non-512-aligned) ---
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False],
        [1, 128, 768, 2, 2, 128, QKVOutputLayout.BSD, False],
        [1, 128, 640, 2, 1, 128, QKVOutputLayout.BSD, False],
    ]
    # fmt: on
    @pytest_parametrize(
        qkv_cte_mxfp8_neutral_mx_scales_test_params,
        qkv_cte_mxfp8_neutral_mx_scales_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_mxfp8_neutral_mx_scales(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        B, S, H = batch, seqlen, hidden_dim
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S=S,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            use_dma_transpose=True,
            fused_add=False,
            qkv_bias=qkv_bias,
            output_layout=output_layout,
            quantization_type=QuantizationType.MX,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            is_h_dim_4h_transposed=False,
            rtol=5e-2,
            atol=1e-4,
        )

    # fmt: off
    qkv_cte_mxfp8_static_dequant_test_params = \
        "batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, output_layout, " \
        "qkv_bias, in_scale_shape, w_scale_shape, fused_rope"
    qkv_cte_mxfp8_static_dequant_test_perms = [
        # --- Broadcast combinations ---
        # Both broadcast
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # in_scale broadcast, w_scale pre-broadcast
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (128, 3), False],
        # in_scale pre-broadcast, w_scale broadcast
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (128, 1), (1, 3), False],
        # Both pre-broadcast
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (128, 1), (128, 3), False],
        # --- With bias ---
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), False],
        [1, 128, 2048, 2, 1, 128, QKVOutputLayout.BSD, True, (128, 1), (128, 3), False],
        # --- Various H/S/head configs ---
        [1, 512, 2048, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 2048, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 128, 1024, 5, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # --- NBSd output layout ---
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, (1, 1), (1, 3), False],
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, True, (128, 1), (128, 3), False],
        # --- Large H / large S (chunked weight loading, multi-S-tile buffering) ---
        [1, 128, 4096, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 128, 8192, 2, 2, 128, QKVOutputLayout.BSD, False, (128, 1), (128, 3), False],
        [1, 2048, 2048, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 128, 16384, 3, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), False],
        [1, 4096, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (128, 1), (128, 3), False],
        # --- Fused RoPE ---
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), True],
        [1, 128, 1024, 5, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), True],
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, (128, 1), (128, 3), True],
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, True, (128, 1), (128, 3), True],
        # --- Non-aligned seqlen (not a multiple of 128, must be multiple of 4 for even S_shard with vnc_degree=2) ---
        [1, 132, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 200, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 1000, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # --- Llama3 70B TP16 S=10240 ---
        [1, 10240, 8192, 4, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # --- Qwen3 32B TP16 S=10240 ---
        [1, 10240, 5120, 4, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # --- Gemma3 27B TP16 S=10240 ---
        [1, 10240, 5632, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # --- Partial H (non-512-aligned) ---
        # H=896: 1 full + 1 partial (h_pack_last=3)
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # H=896 + RoPE
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), True],
        # H=896 + bias
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), False],
        # H=640 (h_pack_last=1)
        [1, 128, 640, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # H=768 (h_pack_last=2)
        [1, 128, 768, 2, 2, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # H=1152: 2 full + 1 partial
        [1, 128, 1152, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
    ]
    # fmt: on
    # NOTE: test_qkv_cte_mxfp8_static_dequant and test_qkv_cte_mx_bf16_static_dequant use
    # preserve_lower_precision=True in torch_ref_wrapper so the torch ref can distinguish
    # FP8 input (arrives as float32) from BF16 input (arrives as torch.bfloat16) to choose
    # the correct static dequant math path.
    @pytest_parametrize(
        qkv_cte_mxfp8_static_dequant_test_params,
        qkv_cte_mxfp8_static_dequant_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_mxfp8_static_dequant(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
        in_scale_shape,
        w_scale_shape,
        fused_rope,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        B, S, H = batch, seqlen, hidden_dim
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        # === Generate FP8 input directly (no quantization error) ===
        np.random.seed(42)
        input_f32 = np.random.randn(B, S, H).astype(np.float32)
        fp8_input = input_f32.astype(nl.float8_e4m3fn)

        # === Generate FP8 weights in [H//4, I, 4] unpacked format ===
        weights_f32 = (np.random.randn(H, fused_qkv_dim) / np.sqrt(H)).astype(np.float32)
        weights_fp8_orig = weights_f32.astype(nl.float8_e4m3fn)

        weight_layout = QKVWeightLayout.MX_INTERLEAVED
        if weight_layout == QKVWeightLayout.MX_INTERLEAVED:
            h_idx = np.empty(H, dtype=np.int64)
            for p in range(H // 4):
                h_idx[4 * p] = 2 * p
                h_idx[4 * p + 1] = 2 * p + 1
                h_idx[4 * p + 2] = H // 2 + 2 * p
                h_idx[4 * p + 3] = H // 2 + 2 * p + 1
            w_reordered = weights_fp8_orig[h_idx, :]
        elif weight_layout == QKVWeightLayout.MX_CONTIGUOUS:
            w_reordered = weights_fp8_orig
        mx_weights_reordered = w_reordered.reshape(H // 4, 4, fused_qkv_dim).transpose(0, 2, 1)

        # === Static dequant scales ===
        in_scale_val = 0.5
        w_scale_val = np.array([0.8, 0.9, 1.2], dtype=np.float32)

        # === Optional bias and RoPE caches ===
        bias = np.random.randn(1, fused_qkv_dim).astype(np.float32) * 0.1 if qkv_bias else None

        cos_cache = None
        sin_cache = None
        if fused_rope:
            gen = rope_gaussian_tensor_generator()
            cos_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
            sin_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")

        def generate_inputs(test_config):
            return {
                "input": fp8_input,
                "fused_qkv_weights": mx_weights_reordered,
                "output_layout": output_layout,
                "bias": bias,
                "quantization_type": QuantizationType.STATIC_MX,
                "qkv_w_scale": np.broadcast_to(w_scale_val.reshape(1, 3), w_scale_shape).astype(np.float32).copy(),
                "qkv_in_scale": np.full(in_scale_shape, in_scale_val, dtype=np.float32),
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": 1e-6,
                "hidden_actual": None,
                "fused_rope": fused_rope,
                "cos_cache": cos_cache,
                "sin_cache": sin_cache,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": weight_layout,
            }

        def output_tensor_descriptor(kernel_input):
            if output_layout == QKVOutputLayout.NBSd:
                num_heads = n_q_heads + 2 * n_kv_heads
                return {"out": np.zeros((num_heads, B, S, d_head), dtype=nl.bfloat16)}
            return {"out": np.zeros((B, S, fused_qkv_dim), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=torch_ref_wrapper(qkv_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
            atol=1e-2,
        )

    # fmt: off
    qkv_cte_mx_bf16_static_dequant_test_params = \
        "batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, output_layout, " \
        "qkv_bias, in_scale_shape, w_scale_shape, fused_rope"
    qkv_cte_mx_bf16_static_dequant_test_perms = [
        # --- Broadcast combinations ---
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (128, 3), False],
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (128, 1), (1, 3), False],
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (128, 1), (128, 3), False],
        # --- With bias ---
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), False],
        [1, 128, 2048, 2, 1, 128, QKVOutputLayout.BSD, True, (128, 1), (128, 3), False],
        # --- Various H/S/head configs ---
        [1, 512, 2048, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 2048, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 128, 1024, 5, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # --- NBSd output layout ---
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, (1, 1), (1, 3), False],
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, True, (128, 1), (128, 3), False],
        # --- Large H / large S ---
        [1, 128, 4096, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 128, 8192, 2, 2, 128, QKVOutputLayout.BSD, False, (128, 1), (128, 3), False],
        [1, 2048, 2048, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 128, 16384, 3, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), False],
        [1, 4096, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (128, 1), (128, 3), False],
        # --- Fused RoPE ---
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), True],
        [1, 128, 1024, 5, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), True],
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, (128, 1), (128, 3), True],
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, True, (128, 1), (128, 3), True],
        # --- Non-aligned seqlen (not a multiple of 128, must be multiple of 4 for even S_shard with vnc_degree=2) ---
        [1, 132, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 200, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        [1, 1000, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # --- Partial H (non-512-aligned) ---
        # H=896: BF16 DMA xpose + quantize_mx
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # H=896 + bias
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, True, (1, 1), (1, 3), False],
        # H=896 + RoPE
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), True],
        # H=640 (h_pack_last=1)
        [1, 128, 640, 2, 1, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
        # H=768 (h_pack_last=2)
        [1, 128, 768, 2, 2, 128, QKVOutputLayout.BSD, False, (1, 1), (1, 3), False],
    ]
    # fmt: on
    @pytest_parametrize(
        qkv_cte_mx_bf16_static_dequant_test_params,
        qkv_cte_mx_bf16_static_dequant_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_mx_bf16_static_dequant(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
        in_scale_shape,
        w_scale_shape,
        fused_rope,
    ):
        """BF16 input with static dequant scales routed through MX engine via BF16→FP32 DMA transpose."""
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        B, S, H = batch, seqlen, hidden_dim
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        np.random.seed(42)
        # BF16 input (the key difference from FP8 test)
        input_bf16 = np.random.randn(B, S, H).astype(np.float32).astype(nl.bfloat16)

        # Generate FP8 weights with DMA transpose reordering (same as FP8 test)
        weights_f32 = (np.random.randn(H, fused_qkv_dim) / np.sqrt(H)).astype(np.float32)
        weights_fp8_orig = weights_f32.astype(nl.float8_e4m3fn)

        h_idx = np.empty(H, dtype=np.int64)
        for p in range(H // 4):
            h_idx[4 * p] = 2 * p
            h_idx[4 * p + 1] = 2 * p + 1
            h_idx[4 * p + 2] = H // 2 + 2 * p
            h_idx[4 * p + 3] = H // 2 + 2 * p + 1
        w_reordered = weights_fp8_orig[h_idx, :]
        mx_weights_reordered = w_reordered.reshape(H // 4, 4, fused_qkv_dim).transpose(0, 2, 1)

        # Static dequant scales
        in_scale_val = 0.5
        w_scale_val = np.array([0.8, 0.9, 1.2], dtype=np.float32)

        bias = np.random.randn(1, fused_qkv_dim).astype(np.float32) * 0.1 if qkv_bias else None

        cos_cache = None
        sin_cache = None
        if fused_rope:
            gen = rope_gaussian_tensor_generator()
            cos_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
            sin_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")

        def generate_inputs(test_config):
            return {
                "input": input_bf16,
                "fused_qkv_weights": mx_weights_reordered,
                "output_layout": output_layout,
                "bias": bias,
                "quantization_type": QuantizationType.STATIC_MX,
                "qkv_w_scale": np.broadcast_to(w_scale_val.reshape(1, 3), w_scale_shape).astype(np.float32).copy(),
                "qkv_in_scale": np.full(in_scale_shape, in_scale_val, dtype=np.float32),
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": 1e-6,
                "hidden_actual": None,
                "fused_rope": fused_rope,
                "cos_cache": cos_cache,
                "sin_cache": sin_cache,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": QKVWeightLayout.MX_INTERLEAVED,
            }

        def output_tensor_descriptor(kernel_input):
            if output_layout == QKVOutputLayout.NBSd:
                num_heads = n_q_heads + 2 * n_kv_heads
                return {"out": np.zeros((num_heads, B, S, d_head), dtype=nl.bfloat16)}
            return {"out": np.zeros((B, S, fused_qkv_dim), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=torch_ref_wrapper(qkv_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
            atol=1e-2,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        B=[1, 2, 3, 4],
        S=BoundedRange(
            [128, 256, 512, 1024, 2048, 4096, 8192],
            boundary_values=[],
        ),
        H=BoundedRange(
            [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            boundary_values=[],
        ),
        n_q_heads=[1, 2, 3],
        n_kv_heads=[1, 2],
        d_head=[128],  # CTE kernel only supports d_head=128
        norm_type=[NormType.NO_NORM, NormType.RMS_NORM, NormType.LAYER_NORM],
        fused_add=[False, True],
        output_layout=[QKVOutputLayout.BSD, QKVOutputLayout.NBSd],
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_qkv_cte_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        B,
        S,
        H,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        fused_add,
        output_layout,
        is_negative_test_case,
    ):
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S=S,
            dtype=nl.bfloat16,
            eps=1e-6,
            fused_add=fused_add,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_type=norm_type,
            output_layout=output_layout,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
            is_negative_test=is_negative_test_case,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        B=[1],
        S=BoundedRange(
            [256, 1024],
            boundary_values=[],
        ),
        H=BoundedRange(
            [1152, 1280],
            boundary_values=[],
        ),
        n_q_heads=[16],
        n_kv_heads=[16],
        d_head=[72, 80],  # ViT head dim
        norm_type=[NormType.NO_NORM, NormType.RMS_NORM, NormType.LAYER_NORM],
        fused_add=[False, True],
        output_layout=[QKVOutputLayout.BSD],
        coverage="full",
        enable_automatic_boundary_tests=False,
    )
    def test_qkv_cte_vit_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        B,
        S,
        H,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        fused_add,
        output_layout,
        is_negative_test_case,
    ):
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S=S,
            dtype=nl.bfloat16,
            eps=1e-6,
            fused_add=fused_add,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_type=norm_type,
            output_layout=output_layout,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
            is_negative_test=is_negative_test_case,
        )

    ####################################################################################################################
    # QKV CTE Test - FP8 KV Cache Quantization Tests
    ####################################################################################################################

    qkv_cte_non_quantized_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head"
    qkv_cte_non_quantized_test_perms = [
        [2, 1, 1024, 2048, 1, 1, 128],
    ]

    @pytest_parametrize(
        qkv_cte_non_quantized_test_params,
        qkv_cte_non_quantized_test_perms,
    )
    def test_qkv_cte_non_quantized(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            use_dma_transpose=True,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            rtol=2e-2,
            atol=1e-5,
        )

    qkv_cte_kv_cache_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, max_seq_len, k_scale_val, v_scale_val, transpose_k_cache"
    qkv_cte_kv_cache_test_perms = [
        # FP8 KV cache (with quantization scales)
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 1.67, 1.67, False],
        [2, 1, 512, 2048, 1, 1, 128, 4096, 1.67, 1.67, False],
        [2, 1, 2048, 2048, 1, 1, 128, 8192, 1.67, 1.67, False],
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 1.0, 1.0, False],
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 2.5, 2.5, False],
        [2, 1, 128, 2048, 1, 1, 128, 8192, 1.67, 1.67, True],
        [2, 1, 128, 2048, 1, 2, 128, 8192, 1.67, 1.67, True],
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 1.67, 1.67, True],
        [2, 1, 512, 2048, 4, 4, 128, 4096, 1.67, 1.67, True],
        [2, 1, 2048, 2048, 2, 1, 128, 8192, 1.67, 1.67, True],
        [2, 1, 1024, 2048, 1, 4, 128, 8192, 1.0, 1.0, True],
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 2.5, 2.5, True],
        # BF16 KV cache (no quantization, scales=None)
        [2, 1, 1024, 2048, 1, 1, 128, 8192, None, None, False],
        [2, 1, 512, 2048, 1, 1, 128, 4096, None, None, False],
        [2, 1, 2048, 2048, 1, 1, 128, 8192, None, None, False],
        [2, 1, 128, 2048, 1, 2, 128, 8192, None, None, False],
        [2, 1, 512, 2048, 4, 4, 128, 4096, None, None, False],
        [2, 1, 2048, 2048, 2, 1, 128, 8192, None, None, False],
        [2, 1, 128, 2048, 1, 1, 128, 8192, None, None, True],
        [2, 1, 128, 2048, 1, 2, 128, 8192, None, None, True],
        [2, 1, 1024, 2048, 1, 1, 128, 8192, None, None, True],
        [2, 1, 512, 2048, 4, 4, 128, 4096, None, None, True],
        [2, 1, 2048, 2048, 2, 1, 128, 8192, None, None, True],
    ]

    @pytest_parametrize(
        qkv_cte_kv_cache_test_params,
        qkv_cte_kv_cache_test_perms,
    )
    def test_qkv_cte_kv_cache(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        max_seq_len,
        k_scale_val,
        v_scale_val,
        transpose_k_cache,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head
        _is_bf16 = k_scale_val is None

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fp8_kv_cache=not _is_bf16,
            bf16_kv_cache=_is_bf16,
            transpose_k_cache=transpose_k_cache,
            max_seq_len=max_seq_len,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            tensor_gen=gaussian_tensor_generator(seed=42),
            rtol=2e-2 if _is_bf16 else 1e-1,
            atol=1e-5,
        )

    qkv_cte_fp8_extreme_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, max_seq_len, k_scale_val, v_scale_val, input_scale, weight_scale, fp8_max, fp8_min"
    qkv_cte_fp8_extreme_test_perms = [
        # Original test case - triggers clamping
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 0.01, 0.01, 5.0, 2.0, 240.0, -240.0],
        # Different sequence length
        [2, 1, 512, 2048, 1, 1, 128, 4096, 0.01, 0.01, 5.0, 2.0, 240.0, -240.0],
        # More extreme scaling
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 0.005, 0.005, 10.0, 3.0, 240.0, -240.0],
    ]

    @pytest_parametrize(
        qkv_cte_fp8_extreme_test_params,
        qkv_cte_fp8_extreme_test_perms,
    )
    def test_qkv_cte_fp8_extreme_values_clamping(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        max_seq_len,
        k_scale_val,
        v_scale_val,
        input_scale,
        weight_scale,
        fp8_max,
        fp8_min,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)

        def scaled_tensor_gen(shape, dtype, name):
            scale = input_scale if name == "input" else weight_scale if name == "fused_qkv_weights" else 1.0
            return (np.random.randn(*shape) * scale).astype(dtype)

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=scaled_tensor_gen,
            fp8_kv_cache=True,
            max_seq_len=max_seq_len,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            rtol=1e-1,
            atol=1e-5,
        )

    qkv_cte_block_kv_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, num_blocks, block_size, k_scale_val, v_scale_val, fp8_max, fp8_min, transpose_k_cache"
    qkv_cte_block_kv_test_perms = [
        # FP8 block KV cache (with quantization scales)
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0, False],
        [2, 1, 1024, 2048, 1, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0, False],
        [2, 1, 512, 2048, 1, 1, 128, 128, 64, 1.67, 1.67, 240.0, -240.0, False],
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 1.0, 1.0, 240.0, -240.0, False],
        [2, 1, 128, 2048, 1, 1, 128, 128, 16, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 128, 2048, 2, 2, 128, 16, 128, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 256, 2048, 1, 1, 128, 32, 128, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 1024, 2048, 1, 1, 128, 32, 64, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 1024, 2048, 4, 4, 128, 16, 128, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 2048, 2048, 1, 2, 128, 16, 128, 1.0, 1.0, 240.0, -240.0, True],
        [2, 1, 2048, 2048, 1, 2, 128, 16, 128, 2.5, 2.5, 240.0, -240.0, True],
        # BF16 block KV cache (no quantization, scales=None)
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, None, None, None, None, False],
        [2, 1, 1024, 2048, 1, 1, 128, 64, 128, None, None, None, None, False],
        [2, 1, 512, 2048, 1, 1, 128, 128, 64, None, None, None, None, False],
        [2, 1, 128, 2048, 1, 1, 128, 128, 16, None, None, None, None, True],
        [2, 1, 128, 2048, 2, 2, 128, 16, 128, None, None, None, None, True],
        [2, 1, 256, 2048, 1, 1, 128, 32, 128, None, None, None, None, True],
        [2, 1, 1024, 2048, 1, 1, 128, 32, 64, None, None, None, None, True],
        [2, 1, 1024, 2048, 4, 4, 128, 16, 128, None, None, None, None, True],
        [2, 1, 2048, 2048, 1, 2, 128, 16, 128, None, None, None, None, True],
    ]

    @pytest_parametrize(
        qkv_cte_block_kv_test_params,
        qkv_cte_block_kv_test_perms,
    )
    def test_qkv_cte_block_kv_cache(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        num_blocks,
        block_size,
        k_scale_val,
        v_scale_val,
        fp8_max,
        fp8_min,
        transpose_k_cache,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head
        _is_bf16 = k_scale_val is None

        slot_mapping = np.arange(0, seqlen, dtype=np.int32).reshape(batch, seqlen)

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fp8_kv_cache=not _is_bf16,
            bf16_kv_cache=_is_bf16,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            use_block_kv=True,
            transpose_k_cache=transpose_k_cache,
            num_blocks=num_blocks,
            block_size=block_size,
            slot_mapping=slot_mapping,
            tensor_gen=gaussian_tensor_generator(seed=42),
            rtol=2e-2 if _is_bf16 else 1e-1,
            atol=1e-5,
        )

    # fmt: off
    qkv_cte_block_kv_noncontig_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, num_blocks, block_size, k_scale_val, v_scale_val, fp8_max, fp8_min, transpose_k_cache"
    qkv_cte_block_kv_noncontig_test_perms = [
        [2, 1, 128, 2048, 1, 1, 128, 32, 64, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 128, 2048, 1, 1, 128, 128, 16, 1.67, 1.67, 240.0, -240.0, True],
        # Multi-head with non-contiguous blocks
        [2, 1, 128, 2048, 2, 2, 128, 32, 64, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 256, 2048, 1, 1, 128, 64, 64, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 128, 2048, 4, 1, 128, 32, 64, 1.67, 1.67, 240.0, -240.0, True],
        [2, 1, 128, 2048, 1, 1, 128, 16, 128, 1.0, 1.0, 240.0, -240.0, True],
        # Tests multi-block with smaller block_size
        [2, 1, 192, 2048, 1, 1, 128, 32, 32, 1.67, 1.67, 240.0, -240.0, True],
        # Tests exact single-block fit per core
        [2, 1, 128, 2048, 1, 1, 128, 32, 64, 2.0, 2.0, 240.0, -240.0, True],
        # Multi-tile + multi-block
        [2, 1, 512, 2048, 1, 1, 128, 64, 64, 1.67, 1.67, 240.0, -240.0, True],
        # Multi-tile + different scale
        [2, 1, 512, 2048, 1, 1, 128, 32, 128, 2.0, 2.0, 240.0, -240.0, True],
        # Non-contiguous + pure remainder
        [2, 1, 256, 2048, 1, 1, 128, 16, 256, 1.67, 1.67, 240.0, -240.0, True],
        # BF16 non-contiguous block KV cache (no quantization, scales=None)
        [2, 1, 128, 2048, 1, 1, 128, 32, 64, None, None, None, None, True],
        [2, 1, 128, 2048, 2, 2, 128, 32, 64, None, None, None, None, True],
        [2, 1, 256, 2048, 1, 1, 128, 64, 64, None, None, None, None, True],
        [2, 1, 128, 2048, 4, 1, 128, 32, 64, None, None, None, None, True],
        [2, 1, 512, 2048, 1, 1, 128, 64, 64, None, None, None, None, True],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_block_kv_noncontig_test_params,
        qkv_cte_block_kv_noncontig_test_perms,
    )
    def test_qkv_cte_block_kv_cache_noncontiguous(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        num_blocks,
        block_size,
        k_scale_val,
        v_scale_val,
        fp8_max,
        fp8_min,
        transpose_k_cache,
    ):
        """Test transposed K cache with non-contiguous block allocation.

        Simulates a fragmented block table where consecutive logical blocks
        map to scattered physical blocks, as happens in vLLM after
        preemption/reallocation or with prefix caching.
        """
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head
        _is_bf16 = k_scale_val is None

        np.random.seed(42)
        slot_mapping = build_noncontiguous_slot_mapping(seqlen, batch, block_size, num_blocks)

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fp8_kv_cache=not _is_bf16,
            bf16_kv_cache=_is_bf16,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            use_block_kv=True,
            transpose_k_cache=transpose_k_cache,
            num_blocks=num_blocks,
            block_size=block_size,
            slot_mapping=slot_mapping,
            rtol=2e-2 if _is_bf16 else 1e-1,
            atol=1e-5,
        )

    ####################################################################################################################
    # QKV CTE FP8 Packed Block KV Test
    ####################################################################################################################

    # fmt: off
    qkv_cte_block_kv_fp8_packed_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, num_blocks, block_size, k_scale_val, v_scale_val, fp8_max, fp8_min, quantization_type"
    qkv_cte_block_kv_fp8_packed_test_perms = [
        # Single KV head, d_head=128, various seqlen and block sizes
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        [2, 1, 1024, 2048, 1, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        [2, 1, 512, 2048, 1, 1, 128, 128, 64, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # Smaller d_head
        [2, 1, 512, 1024, 1, 1, 64, 64, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # Multi KV heads (GQA configurations)
        [2, 1, 512, 2048, 4, 2, 128, 32, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        [2, 1, 512, 2048, 8, 4, 128, 16, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        [2, 1, 256, 2048, 2, 2, 128, 16, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # Multi KV heads with smaller d_head
        [2, 1, 512, 1024, 4, 2, 64, 32, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # Different scale values
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 1.0, 1.0, 240.0, -240.0, QuantizationType.NONE],
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 2.5, 2.5, 240.0, -240.0, QuantizationType.NONE],
        # Larger seqlen (multiple tiles)
        [2, 1, 2048, 2048, 1, 1, 128, 16, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # Smaller block_size
        [2, 1, 512, 2048, 1, 1, 128, 256, 32, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # Seqlen not multiple of 128 (partial last tile)
        [2, 1, 384, 2048, 1, 1, 128, 64, 64, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        [2, 1, 640, 2048, 1, 1, 128, 64, 64, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        [2, 1, 96, 1024, 1, 1, 64, 64, 32, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # Seqlen not multiple of block_size (partial last block)
        [2, 1, 224, 2048, 1, 1, 128, 64, 64, 1.67, 1.67, 240.0, -240.0, QuantizationType.NONE],
        # STATIC FP8 matmul quantization
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.STATIC],
        [2, 1, 1024, 2048, 8, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0, QuantizationType.STATIC],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_block_kv_fp8_packed_test_params,
        qkv_cte_block_kv_fp8_packed_test_perms,
    )
    def test_qkv_cte_block_kv_fp8_packed(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        num_blocks,
        block_size,
        k_scale_val,
        v_scale_val,
        fp8_max,
        fp8_min,
        quantization_type,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        slot_mapping = build_noncontiguous_slot_mapping(seqlen, batch, block_size, num_blocks)

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fp8_kv_cache=True,
            bf16_kv_cache=False,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            use_block_kv=True,
            fp8_packed=True,
            transpose_k_cache=False,
            num_blocks=num_blocks,
            block_size=block_size,
            slot_mapping=slot_mapping,
            tensor_gen=gaussian_tensor_generator(seed=42),
            quantization_type=quantization_type,
            rtol=1e-1,
            atol=1e-5,
        )

    ####################################################################################################################
    # QKV CTE Model Config Tests
    ####################################################################################################################

    ################################################################################################
    # QK-NORM NON-MX DEV TARGET
    ################################################################################################

    # fmt: off
    qkv_cte_qk_norm_non_mx_test_params = (
        "seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,"
        " quantization_type, output_layout, qkv_bias, fused_rope,"
        " use_pre_rope, use_post_rope, use_gamma,"
        " fp8_kv_cache, fused_norm_type, use_dma_transpose"
    )
    qkv_cte_qk_norm_non_mx_test_perms = [
        # --- NONE quant, no RoPE ---
        # 1: Basic pre-RoPE, NONE quant
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, True],
        # 2: Basic post-RoPE, NONE quant
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, False, True, True, False, NormType.NO_NORM, True],
        # 3: Pure RMSNorm (no gamma), NONE quant
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, True, False, False, False, NormType.NO_NORM, True],
        # 4: GQA heads, NONE quant
        [128, 512, 4, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, True],
        # 5: Multi-S-tile (S > 128), NONE quant
        [256, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, True],
        # 6: NBSd output layout, NONE quant
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.NBSd, False, False, True, False, True, False, NormType.NO_NORM, True],
        # --- STATIC quant, no RoPE ---
        # 7: Basic pre-RoPE, STATIC quant
        [128, 512, 2, 1, 128, QuantizationType.STATIC, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, True],
        # 8: With bias, STATIC quant
        [128, 512, 2, 1, 128, QuantizationType.STATIC, QKVOutputLayout.BSD, True, False, True, False, True, False, NormType.NO_NORM, True],
        # 9: GQA heads, STATIC quant
        [128, 512, 4, 1, 128, QuantizationType.STATIC, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, True],
        # --- RoPE, NONE quant ---
        # 10: Pre-RoPE + RoPE, NONE quant
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, True, True, False, True, False, NormType.NO_NORM, True],
        # 11: Post-RoPE + RoPE, NONE quant
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, True, False, True, True, False, NormType.NO_NORM, True],
        # 12: Pre + post RoPE, NONE quant
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, True, True, True, True, False, NormType.NO_NORM, True],
        # 13: GQA + RoPE, NONE quant
        [128, 512, 4, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, True, True, False, True, False, NormType.NO_NORM, True],
        # --- RoPE, STATIC quant ---
        # 14: Pre-RoPE + RoPE, STATIC quant
        [128, 512, 2, 1, 128, QuantizationType.STATIC, QKVOutputLayout.BSD, False, True, True, False, True, False, NormType.NO_NORM, True],
        # 15: With bias + RoPE, STATIC quant
        [128, 512, 2, 1, 128, QuantizationType.STATIC, QKVOutputLayout.BSD, True, True, True, False, True, False, NormType.NO_NORM, True],
        # --- FP8 KV cache smoke tests (NONE quant) ---
        # 16: QK-norm + FP8 KV cache, no RoPE
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, True, False, True, True, NormType.NO_NORM, True],
        # 17: QK-norm + FP8 KV cache + RoPE
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, True, True, False, True, True, NormType.NO_NORM, True],
        # --- Cross-feature interaction tests (NONE quant) ---
        # 18: Input-side RMSNorm + output-side QK-norm
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.RMS_NORM, True],
        # 19: QK-norm with PE-array transpose (no DMA transpose)
        [128, 512, 2, 1, 128, QuantizationType.NONE, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, False],
        # --- ROW quant, no RoPE ---
        # 20: Basic pre-RoPE, ROW quant
        [128, 512, 2, 1, 128, QuantizationType.ROW, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, True],
        # 21: With bias, ROW quant
        [128, 512, 2, 1, 128, QuantizationType.ROW, QKVOutputLayout.BSD, True, False, True, False, True, False, NormType.NO_NORM, True],
        # 22: GQA heads, ROW quant
        [128, 512, 4, 1, 128, QuantizationType.ROW, QKVOutputLayout.BSD, False, False, True, False, True, False, NormType.NO_NORM, True],
        # --- ROW quant, RoPE ---
        # 23: Pre-RoPE + RoPE, ROW quant
        [128, 512, 2, 1, 128, QuantizationType.ROW, QKVOutputLayout.BSD, False, True, True, False, True, False, NormType.NO_NORM, True],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_qk_norm_non_mx_test_params,
        qkv_cte_qk_norm_non_mx_test_perms,
    )
    def test_qkv_cte_qk_norm_non_mx(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        quantization_type,
        output_layout,
        qkv_bias,
        fused_rope,
        use_pre_rope,
        use_post_rope,
        use_gamma,
        fp8_kv_cache,
        fused_norm_type,
        use_dma_transpose,
    ):
        vnc_degree = 2
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = rope_gaussian_tensor_generator() if fused_rope else gaussian_tensor_generator(seed=42)
        q_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="q_gamma_norm_weights") if use_gamma else None
        k_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="k_gamma_norm_weights") if use_gamma else None

        qk_norm_pre_rope_config = {"eps": 1e-6, "q_gamma": q_gamma, "k_gamma": k_gamma} if use_pre_rope else None
        qk_norm_post_rope_config = {"eps": 1e-6, "q_gamma": q_gamma, "k_gamma": k_gamma} if use_post_rope else None

        # Override STATIC quant scales: build_qkv_input uses 1/240 for both w_scale
        # and in_scale (combined dequant 1/57600) with values in range [-1, 1]. This compresses post-dequant values,
        # causing QK-norm to amplify bf16 rounding errors. CTE fuses dequant into eviction so it can handle
        # larger scales. Use larger w_scale and in_scale values to improve BF16 range in this test.
        _is_static = quantization_type == QuantizationType.STATIC
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=1,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=fused_norm_type,
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_norm_type != NormType.NO_NORM,
            qkv_bias=qkv_bias,
            fused_rope=fused_rope,
            output_layout=output_layout,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            quantization_type=quantization_type,
            tensor_gen=tensor_gen,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            qk_norm_post_rope_config=qk_norm_post_rope_config,
            fp8_kv_cache=fp8_kv_cache,
            max_seq_len=8192 if fp8_kv_cache else None,
            k_scale_val=1.67 if fp8_kv_cache else None,
            v_scale_val=1.67 if fp8_kv_cache else None,
            qkv_w_scale_for_mx=np.array([[0.8, 0.7, 0.9]], dtype=np.float32) if _is_static else None,
            qkv_in_scale_for_mx=np.array([[0.1]], dtype=np.float32) if _is_static else None,
            rtol=1e-1 if fp8_kv_cache else 5e-2,
            atol=2e-2,
        )

    ################################################################################################
    # QK-NORM ASYMMETRIC (per-head q_norm / k_norm)
    ################################################################################################

    # fmt: off
    qkv_cte_qk_norm_asymmetric_test_params = (
        "seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,"
        " fused_rope, q_norm, k_norm, use_gamma, use_post_rope"
    )
    qkv_cte_qk_norm_asymmetric_test_perms = [
        # --- Pre-RoPE asymmetric ---
        # 1: Q=RMS_NORM, K=None (Q-only norm)
        [128, 512, 2, 1, 128, False, NormType.RMS_NORM, None, False, False],
        # 2: Q=None, K=RMS_NORM (K-only norm)
        [128, 512, 2, 1, 128, False, None, NormType.RMS_NORM, False, False],
        # 3: Q=None, K=None (both disabled, config present but no norm applied)
        [128, 512, 2, 1, 128, False, None, None, False, False],
        # 4: Q=RMS_NORM, K=RMS_NORM + gamma (symmetric, matches existing behavior)
        [128, 512, 2, 1, 128, False, NormType.RMS_NORM, NormType.RMS_NORM, True, False],
        # 5: Q=RMS_NORM + gamma, K=None (asymmetric with gamma)
        [128, 512, 2, 1, 128, False, NormType.RMS_NORM, None, True, False],
        # --- Post-RoPE asymmetric ---
        # 6: Post-RoPE Q=RMS_NORM, K=None
        [128, 512, 2, 1, 128, True, NormType.RMS_NORM, None, False, True],
        # 7: Post-RoPE Q=None, K=RMS_NORM
        [128, 512, 2, 1, 128, True, None, NormType.RMS_NORM, False, True],
        # 8: Post-RoPE Q=RMS_NORM, K=RMS_NORM + gamma
        [128, 512, 2, 1, 128, True, NormType.RMS_NORM, NormType.RMS_NORM, True, True],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_qk_norm_asymmetric_test_params,
        qkv_cte_qk_norm_asymmetric_test_perms,
    )
    def test_qkv_cte_qk_norm_asymmetric(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        fused_rope,
        q_norm,
        k_norm,
        use_gamma,
        use_post_rope,
    ):
        vnc_degree = 2
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = rope_gaussian_tensor_generator() if fused_rope else gaussian_tensor_generator(seed=42)
        q_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="q_gamma_norm_weights") if use_gamma else None
        k_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="k_gamma_norm_weights") if use_gamma else None

        norm_cfg = {
            "eps": 1e-6,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "q_gamma": q_gamma,
            "k_gamma": k_gamma,
        }

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=1,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            use_dma_transpose=True,
            fused_add=False,
            fused_rope=fused_rope,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=tensor_gen,
            qk_norm_pre_rope_config=None if use_post_rope else norm_cfg,
            qk_norm_post_rope_config=norm_cfg if use_post_rope else None,
            rtol=5e-2,
            atol=2e-2,
        )

    def test_qkv_cte_qk_norm_gamma_fused_requires_both_norms(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """gamma_fused_in_rope_caches=True with q_norm=None should fail validation."""
        vnc_degree = 2
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        d_head = 128
        n_q_heads = 2
        n_kv_heads = 1
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = rope_gaussian_tensor_generator()

        qk_norm_pre_rope_config = {
            "eps": 1e-6,
            "q_norm": None,
            "k_norm": NormType.RMS_NORM,
            "gamma_fused_in_rope_caches": True,
        }

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=1,
            H=512,
            S=128,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            fused_rope=True,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=tensor_gen,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            is_negative_test=True,
        )

    def test_qkv_cte_qk_norm_gamma_fused_requires_both_norms_k_none(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """gamma_fused_in_rope_caches=True with k_norm=None should fail validation."""
        vnc_degree = 2
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        d_head = 128
        n_q_heads = 2
        n_kv_heads = 1
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = rope_gaussian_tensor_generator()

        qk_norm_pre_rope_config = {
            "eps": 1e-6,
            "q_norm": NormType.RMS_NORM,
            "k_norm": None,
            "gamma_fused_in_rope_caches": True,
        }

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=1,
            H=512,
            S=128,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            fused_rope=True,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=tensor_gen,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            is_negative_test=True,
        )

    def test_qkv_cte_qk_norm_unsupported_norm_type(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """q_norm=NormType.LAYER_NORM should fail (not registered in _QK_NORM_REGISTRY)."""
        vnc_degree = 2
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        d_head = 128
        n_q_heads = 2
        n_kv_heads = 1
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = gaussian_tensor_generator(seed=42)

        qk_norm_pre_rope_config = {
            "eps": 1e-6,
            "q_norm": NormType.LAYER_NORM,
            "k_norm": NormType.RMS_NORM,
        }

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=1,
            H=512,
            S=128,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            use_dma_transpose=True,
            fused_add=False,
            fused_rope=False,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=tensor_gen,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            is_negative_test=True,
        )

    # fmt: off
    qkv_cte_mxfp8_static_dequant_qk_norm_test_params = (
        "seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,"
        " output_layout, qkv_bias, fused_rope, use_gamma"
    )
    qkv_cte_mxfp8_static_dequant_qk_norm_test_perms = [
        # Basic: dequant + QK-norm + RoPE
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # No RoPE: dequant + QK-norm only
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False, True],
        # With bias
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, True, True],
        # GQA heads
        [128, 512, 4, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # Multi-S-tile
        [256, 512, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # NBSd output layout
        [128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, True, True],
        # No gamma (pure RMSNorm)
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, True, False],
        # --- Llama3 70B TP16 S=10240 ---
        [10240, 8192, 4, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # --- Qwen3 32B TP16 S=10240 ---
        [10240, 5120, 4, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # --- Gemma3 27B TP16 S=10240 ---
        [10240, 5632, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # --- Partial H (non-512-aligned) ---
        [128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        [128, 5376, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # --- Gemma3 27B TP16 S=10240 (no gamma, H=5632) ---
        [10240, 5632, 2, 1, 128, QKVOutputLayout.BSD, False, True, False],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_mxfp8_static_dequant_qk_norm_test_params,
        qkv_cte_mxfp8_static_dequant_qk_norm_test_perms,
    )
    def test_qkv_cte_mxfp8_static_dequant_qk_norm(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
        fused_rope,
        use_gamma,
    ):
        """FP8 input with static dequant + pre-RoPE QK-norm (FP8 DMA transpose path)."""
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        B, S, H = 1, seqlen, hidden_dim
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2
        eps = 1e-6

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        np.random.seed(42)
        input_f32 = np.random.randn(B, S, H).astype(np.float32)
        fp8_input = input_f32.astype(nl.float8_e4m3fn)

        weights_f32 = (np.random.randn(H, fused_qkv_dim) / np.sqrt(H)).astype(np.float32)
        weights_fp8_orig = weights_f32.astype(nl.float8_e4m3fn)

        h_idx = np.empty(H, dtype=np.int64)
        for p in range(H // 4):
            h_idx[4 * p] = 2 * p
            h_idx[4 * p + 1] = 2 * p + 1
            h_idx[4 * p + 2] = H // 2 + 2 * p
            h_idx[4 * p + 3] = H // 2 + 2 * p + 1
        w_reordered = weights_fp8_orig[h_idx, :]
        mx_weights_reordered = w_reordered.reshape(H // 4, 4, fused_qkv_dim).transpose(0, 2, 1)

        in_scale_val = 0.5
        w_scale_val = np.array([0.8, 0.9, 1.2], dtype=np.float32)

        bias = np.random.randn(1, fused_qkv_dim).astype(np.float32) * 0.1 if qkv_bias else None

        cos_cache = None
        sin_cache = None
        if fused_rope:
            gen = rope_gaussian_tensor_generator()
            cos_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
            sin_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")

        q_gamma = np.random.randn(1, d_head).astype(np.float32) if use_gamma else None
        k_gamma = np.random.randn(1, d_head).astype(np.float32) if use_gamma else None

        def generate_inputs(test_config):
            return {
                "input": fp8_input,
                "fused_qkv_weights": mx_weights_reordered,
                "output_layout": output_layout,
                "bias": bias,
                "quantization_type": QuantizationType.STATIC_MX,
                "qkv_w_scale": w_scale_val.reshape(1, 3).copy(),
                "qkv_in_scale": np.full((1, 1), in_scale_val, dtype=np.float32),
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": eps,
                "hidden_actual": None,
                "fused_rope": fused_rope,
                "cos_cache": cos_cache,
                "sin_cache": sin_cache,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": QKVWeightLayout.MX_INTERLEAVED,
                "qk_norm_pre_rope": QKNormConfig(eps=eps, q_gamma_norm_weights=q_gamma, k_gamma_norm_weights=k_gamma),
            }

        def output_tensor_descriptor(kernel_input):
            if output_layout == QKVOutputLayout.NBSd:
                num_heads = n_q_heads + 2 * n_kv_heads
                return {"out": np.zeros((num_heads, B, S, d_head), dtype=nl.bfloat16)}
            return {"out": np.zeros((B, S, fused_qkv_dim), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=torch_ref_wrapper(qkv_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            # Relaxed from 5e-2: torch ref recomputes MX matmul from unpacked weights,
            # introducing minor float32 ordering differences vs kernel's fused path.
            rtol=6e-2,
            atol=1e-2,
        )

    # fmt: off
    qkv_cte_mx_bf16_static_dequant_qk_norm_test_params = (
        "seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,"
        " output_layout, qkv_bias, fused_rope, use_gamma"
    )
    qkv_cte_mx_bf16_static_dequant_qk_norm_test_perms = [
        # Basic: dequant + QK-norm + RoPE
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # No RoPE: dequant + QK-norm only
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False, True],
        # With bias
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, True, True],
        # GQA heads
        [128, 512, 4, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # Multi-S-tile
        [256, 512, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        # NBSd output layout
        [128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, True, True],
        # No gamma (pure RMSNorm)
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, True, False],
        # --- Partial H (non-512-aligned) ---
        [128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, True, True],
        [128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, True, False],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_mx_bf16_static_dequant_qk_norm_test_params,
        qkv_cte_mx_bf16_static_dequant_qk_norm_test_perms,
    )
    def test_qkv_cte_mx_bf16_static_dequant_qk_norm(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
        fused_rope,
        use_gamma,
    ):
        """BF16 input with static dequant + pre-RoPE QK-norm (BF16 DMA transpose path)."""
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        B, S, H = 1, seqlen, hidden_dim
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2
        eps = 1e-6

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        np.random.seed(42)
        input_bf16 = np.random.randn(B, S, H).astype(np.float32).astype(nl.bfloat16)

        weights_f32 = (np.random.randn(H, fused_qkv_dim) / np.sqrt(H)).astype(np.float32)
        weights_fp8_orig = weights_f32.astype(nl.float8_e4m3fn)

        h_idx = np.empty(H, dtype=np.int64)
        for p in range(H // 4):
            h_idx[4 * p] = 2 * p
            h_idx[4 * p + 1] = 2 * p + 1
            h_idx[4 * p + 2] = H // 2 + 2 * p
            h_idx[4 * p + 3] = H // 2 + 2 * p + 1
        w_reordered = weights_fp8_orig[h_idx, :]
        mx_weights_reordered = w_reordered.reshape(H // 4, 4, fused_qkv_dim).transpose(0, 2, 1)

        in_scale_val = 0.5
        w_scale_val = np.array([0.8, 0.9, 1.2], dtype=np.float32)

        bias = np.random.randn(1, fused_qkv_dim).astype(np.float32) * 0.1 if qkv_bias else None

        cos_cache = None
        sin_cache = None
        if fused_rope:
            gen = rope_gaussian_tensor_generator()
            cos_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
            sin_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")

        q_gamma = np.random.randn(1, d_head).astype(np.float32) if use_gamma else None
        k_gamma = np.random.randn(1, d_head).astype(np.float32) if use_gamma else None

        def generate_inputs(test_config):
            return {
                "input": input_bf16,
                "fused_qkv_weights": mx_weights_reordered,
                "output_layout": output_layout,
                "bias": bias,
                "quantization_type": QuantizationType.STATIC_MX,
                "qkv_w_scale": w_scale_val.reshape(1, 3).copy(),
                "qkv_in_scale": np.full((1, 1), in_scale_val, dtype=np.float32),
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": eps,
                "hidden_actual": None,
                "fused_rope": fused_rope,
                "cos_cache": cos_cache,
                "sin_cache": sin_cache,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": QKVWeightLayout.MX_INTERLEAVED,
                "qk_norm_pre_rope": QKNormConfig(eps=eps, q_gamma_norm_weights=q_gamma, k_gamma_norm_weights=k_gamma),
            }

        def output_tensor_descriptor(kernel_input):
            if output_layout == QKVOutputLayout.NBSd:
                num_heads = n_q_heads + 2 * n_kv_heads
                return {"out": np.zeros((num_heads, B, S, d_head), dtype=nl.bfloat16)}
            return {"out": np.zeros((B, S, fused_qkv_dim), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=torch_ref_wrapper(qkv_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
            atol=1e-2,
        )

    ################################################################################################
    # QK-NORM NO-ROPE TESTS
    ################################################################################################

    # fmt: off
    qkv_cte_qk_norm_no_rope_test_params = (
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,"
        " norm_type, fused_add, add_bias, output_layout, eps,"
        " use_pre_rope, use_post_rope, use_gamma, is_h_dim_4h_transposed"
    )
    qkv_cte_qk_norm_no_rope_test_perms = [
        # Basic pre-RoPE QK-norm with gamma
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # Basic post-RoPE QK-norm with gamma (no RoPE, so equivalent to pre)
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, False, True, True, False],
        # Pure RMSNorm (no gamma)
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, False, False],
        # With bias
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, True, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # GQA heads (num_q_heads != num_kv_heads)
        [2, 1, 128, 512, 4, 1, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # Multi-S-tile (S > 128)
        [2, 1, 256, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # NBSd output layout
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.NBSd, 1e-6, True, False, True, False],
        # Swizzled input
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, True],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_qk_norm_no_rope_test_params,
        qkv_cte_qk_norm_no_rope_test_perms,
    )
    def test_qkv_cte_qk_norm_no_rope(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        fused_add,
        add_bias,
        output_layout,
        eps,
        use_pre_rope,
        use_post_rope,
        use_gamma,
        is_h_dim_4h_transposed,
    ):
        if not platform_target.is_trn3():
            pytest.skip("QK-norm is only supported on TRN3 (MX quantization required).")
        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = gaussian_tensor_generator(seed=42)
        q_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="q_gamma_norm_weights") if use_gamma else None
        k_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="k_gamma_norm_weights") if use_gamma else None

        qk_norm_pre_rope_config = None
        qk_norm_post_rope_config = None
        if use_pre_rope:
            qk_norm_pre_rope_config = {"eps": eps, "q_gamma": q_gamma, "k_gamma": k_gamma}
        if use_post_rope:
            qk_norm_post_rope_config = {"eps": eps, "q_gamma": q_gamma, "k_gamma": k_gamma}

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=True,
            fused_add=fused_add,
            qkv_bias=add_bias,
            output_layout=output_layout,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            quantization_type=QuantizationType.MX,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            qk_norm_post_rope_config=qk_norm_post_rope_config,
        )

    ################################################################################################
    # QK-NORM WITH ROPE TESTS
    ################################################################################################

    # fmt: off
    qkv_cte_qk_norm_rope_test_params = (
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,"
        " norm_type, fused_add, add_bias, output_layout, eps,"
        " use_pre_rope, use_post_rope, use_gamma, use_real_rope_freqs"
    )
    qkv_cte_qk_norm_rope_test_perms = [
        # Pre-RoPE QK-norm
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # Post-RoPE QK-norm
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, False, True, True, False],
        # Pre + post RoPE
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, True, True, False],
        # Pure RMSNorm (no gamma)
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, False, False],
        # With bias
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, True, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # GQA heads
        [2, 1, 128, 512, 4, 1, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # Multi-S-tile (S > 128)
        [2, 1, 256, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
        # Real RoPE frequencies
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, True],
        # Real RoPE frequencies, GQA
        [2, 1, 128, 512, 4, 1, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, True],
        # Large config: stress-test SBUF budget with QK-norm + RoPE double-buffered scratch
        [2, 1, 8192, 16384, 1, 1, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6, True, False, True, False],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_qk_norm_rope_test_params,
        qkv_cte_qk_norm_rope_test_perms,
    )
    def test_qkv_cte_qk_norm_rope(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        fused_add,
        add_bias,
        output_layout,
        eps,
        use_pre_rope,
        use_post_rope,
        use_gamma,
        use_real_rope_freqs,
    ):
        if not platform_target.is_trn3():
            pytest.skip("QK-norm is only supported on TRN3 (MX quantization required).")

        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = real_rope_tensor_generator(seed=42) if use_real_rope_freqs else rope_gaussian_tensor_generator()
        q_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="q_gamma_norm_weights") if use_gamma else None
        k_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="k_gamma_norm_weights") if use_gamma else None

        qk_norm_pre_rope_config = None
        qk_norm_post_rope_config = None
        if use_pre_rope:
            qk_norm_pre_rope_config = {"eps": eps, "q_gamma": q_gamma, "k_gamma": k_gamma}
        if use_post_rope:
            qk_norm_post_rope_config = {"eps": eps, "q_gamma": q_gamma, "k_gamma": k_gamma}

        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=vnc_degree,
            dtype=nl.bfloat16,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=True,
            fused_add=fused_add,
            qkv_bias=add_bias,
            output_layout=output_layout,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fused_rope=True,
            quantization_type=QuantizationType.MX,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            qk_norm_post_rope_config=qk_norm_post_rope_config,
            tensor_gen=tensor_gen,
        )

    # GAMMA FUSION (gamma_fused_in_rope_caches) TESTS
    ################################################################################################

    # fmt: off
    qkv_cte_fused_gamma_rope_test_params = (
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head,"
        " norm_type, fused_add, add_bias, output_layout, eps"
    )
    qkv_cte_fused_gamma_rope_test_perms = [
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 512, 2, 2, 128, NormType.NO_NORM, False, True, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 128, 512, 4, 1, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 256, 512, 2, 2, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6],
        [2, 1, 10240, 5120, 4, 1, 128, NormType.NO_NORM, False, False, QKVOutputLayout.BSD, 1e-6],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_fused_gamma_rope_test_params,
        qkv_cte_fused_gamma_rope_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_fused_gamma_rope(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        fused_add,
        add_bias,
        output_layout,
        eps,
    ):
        if not platform_target.is_trn3():
            pytest.skip("QK-norm is only supported on TRN3 (MX quantization required).")

        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        tensor_gen = gaussian_tensor_generator(seed=42)
        q_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="q_gamma_norm_weights")
        k_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="k_gamma_norm_weights")

        base_input = build_qkv_input(
            batch=batch,
            seqlen=seqlen,
            hidden_dim=hidden_dim,
            fused_qkv_dim=fused_qkv_dim,
            dtype=nl.bfloat16,
            eps=eps,
            d_head=d_head,
            norm_type=norm_type,
            fused_add=fused_add,
            output_layout=output_layout,
            lnc_degree=vnc_degree,
            use_dma_transpose=True,
            qkv_bias=add_bias,
            fused_rope=True,
            num_q_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
            quantization_type=QuantizationType.MX,
            tensor_gen=tensor_gen,
        )
        base_input["is_h_dim_4h_transposed"] = False

        original_cos = base_input["cos_cache"].copy()
        original_sin = base_input["sin_cache"].copy()
        q_cos, q_sin, k_cos, k_sin = fuse_gamma_into_rope_caches(
            original_cos,
            original_sin,
            q_gamma,
            k_gamma,
            d_head,
        )

        def generate_inputs(test_config):
            inputs = dict(base_input)
            inputs["cos_cache"] = q_cos
            inputs["sin_cache"] = q_sin
            inputs["k_cos_cache"] = k_cos
            inputs["k_sin_cache"] = k_sin
            inputs["qk_norm_pre_rope"] = QKNormConfig(eps=eps, gamma_fused_in_rope_caches=True)
            inputs["qk_norm_pre_rope_q_gamma"] = None
            inputs["qk_norm_pre_rope_k_gamma"] = None
            return inputs

        def output_tensor_descriptor(kernel_input):
            return {"out": np.zeros((batch, seqlen, fused_qkv_dim), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=gamma_unfused_ref(
                torch_ref_wrapper(qkv_torch_ref),
                original_cos=original_cos,
                original_sin=original_sin,
                q_gamma=q_gamma,
                k_gamma=k_gamma,
                eps=eps,
            ),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2,
            atol=1e-5,
        )

    # fmt: off
    qkv_cte_fused_gamma_rope_fp8_dequant_test_params = (
        "seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, output_layout, qkv_bias"
    )
    qkv_cte_fused_gamma_rope_fp8_dequant_test_perms = [
        [10240, 5120, 4, 1, 128, QKVOutputLayout.BSD, False],
        [10240, 5632, 2, 1, 128, QKVOutputLayout.BSD, False],
        [10240, 8192, 4, 1, 128, QKVOutputLayout.BSD, False],
        [128, 512, 2, 1, 128, QKVOutputLayout.BSD, False],
        [128, 512, 4, 1, 128, QKVOutputLayout.BSD, False],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_fused_gamma_rope_fp8_dequant_test_params,
        qkv_cte_fused_gamma_rope_fp8_dequant_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_fused_gamma_rope_fp8_dequant(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
    ):
        """FP8 static dequant + fused rsqrt-in-RoPE path (gamma pre-multiplied into cos/sin)."""
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        B, S, H = 1, seqlen, hidden_dim
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2
        eps = 1e-6

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        np.random.seed(42)
        input_f32 = np.random.randn(B, S, H).astype(np.float32)
        fp8_input = input_f32.astype(nl.float8_e4m3fn)

        weights_f32 = (np.random.randn(H, fused_qkv_dim) / np.sqrt(H)).astype(np.float32)
        weights_fp8_orig = weights_f32.astype(nl.float8_e4m3fn)

        h_idx = np.empty(H, dtype=np.int64)
        for p in range(H // 4):
            h_idx[4 * p] = 2 * p
            h_idx[4 * p + 1] = 2 * p + 1
            h_idx[4 * p + 2] = H // 2 + 2 * p
            h_idx[4 * p + 3] = H // 2 + 2 * p + 1
        w_reordered = weights_fp8_orig[h_idx, :]
        mx_weights_reordered = w_reordered.reshape(H // 4, 4, fused_qkv_dim).transpose(0, 2, 1)

        in_scale_val = 0.5
        w_scale_val = np.array([0.8, 0.9, 1.2], dtype=np.float32)
        bias = np.random.randn(1, fused_qkv_dim).astype(np.float32) * 0.1 if qkv_bias else None

        q_gamma = np.random.randn(1, d_head).astype(np.float32)
        k_gamma = np.random.randn(1, d_head).astype(np.float32)

        gen = rope_gaussian_tensor_generator()
        cos_raw = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
        sin_raw = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")
        q_cos, q_sin, k_cos, k_sin = fuse_gamma_into_rope_caches(
            cos_raw,
            sin_raw,
            q_gamma,
            k_gamma,
            d_head,
        )

        def generate_inputs(test_config):
            return {
                "input": fp8_input,
                "fused_qkv_weights": mx_weights_reordered,
                "output_layout": output_layout,
                "bias": bias,
                "quantization_type": QuantizationType.STATIC_MX,
                "qkv_w_scale": w_scale_val.reshape(1, 3).copy(),
                "qkv_in_scale": np.full((1, 1), in_scale_val, dtype=np.float32),
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": eps,
                "hidden_actual": None,
                "fused_rope": True,
                "cos_cache": q_cos,
                "sin_cache": q_sin,
                "k_cos_cache": k_cos,
                "k_sin_cache": k_sin,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": QKVWeightLayout.MX_INTERLEAVED,
                "qk_norm_pre_rope": QKNormConfig(eps=eps, gamma_fused_in_rope_caches=True),
                "qk_norm_pre_rope_q_gamma": None,
                "qk_norm_pre_rope_k_gamma": None,
            }

        def output_tensor_descriptor(kernel_input):
            return {"out": np.zeros((B, S, fused_qkv_dim), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=gamma_unfused_ref(
                torch_ref_wrapper(qkv_torch_ref, preserve_lower_precision=True),
                original_cos=cos_raw,
                original_sin=sin_raw,
                q_gamma=q_gamma,
                k_gamma=k_gamma,
                eps=eps,
            ),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
            atol=1e-2,
        )

    def test_qkv_cte_fused_gamma_rope_rejects_no_rope(self, test_manager: Orchestrator, platform_target: Platforms):
        """gamma_fused_in_rope_caches=True without fused_rope should fail validation."""
        if not platform_target.is_trn3():
            pytest.skip("QK-norm is only supported on TRN3.")
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            B=1,
            H=512,
            S=128,
            fused_qkv_dim=768,
            lnc_degree=2,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_rope=False,
            n_q_heads=2,
            n_kv_heads=2,
            d_head=128,
            quantization_type=QuantizationType.MX,
            qk_norm_pre_rope_config={"eps": 1e-6, "q_gamma": None, "k_gamma": None, "gamma_fused_in_rope_caches": True},
            is_negative_test=True,
        )

    def test_qkv_cte_fused_gamma_rope_rejects_gamma_weights(
        self, test_manager: Orchestrator, platform_target: Platforms
    ):
        """gamma_fused_in_rope_caches=True with gamma weights provided should fail validation."""
        if not platform_target.is_trn3():
            pytest.skip("QK-norm is only supported on TRN3.")
        np.random.seed(42)
        tensor_gen = gaussian_tensor_generator(seed=42)
        gamma = tensor_gen(shape=(1, 128), dtype=nl.float32, name="gamma")
        self.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            B=1,
            H=512,
            S=128,
            fused_qkv_dim=768,
            lnc_degree=2,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_rope=True,
            n_q_heads=2,
            n_kv_heads=2,
            d_head=128,
            quantization_type=QuantizationType.MX,
            qk_norm_pre_rope_config={
                "eps": 1e-6,
                "q_gamma": gamma,
                "k_gamma": gamma,
                "gamma_fused_in_rope_caches": True,
            },
            is_negative_test=True,
        )

    ####################################################################################################################
    # QKV CTE ROW_MX (4-byte DMA transpose, tail-scale packed input)
    ####################################################################################################################
    # fmt: off
    qkv_cte_row_mx_test_params = \
        "batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, output_layout, " \
        "qkv_bias, w_scale_shape, fused_rope, qk_norm, use_gamma"
    qkv_cte_row_mx_test_perms = [
        # --- Basic ROW_MX (no QK-norm) ---
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), False, False, False],
        # + bias
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, (1, None), False, False, False],
        # + RoPE
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, False, False],
        # + bias + RoPE
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, (1, None), True, False, False],
        # + GQA + RoPE (4Q/1KV heads)
        [1, 128, 512, 4, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, False, False],
        # multi-S-tile (S=256)
        [1, 256, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, False, False],
        # + NBSd layout
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, (1, None), True, False, False],
        # w_scale [128, I] (direct load path)
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (128, None), False, False, False],
        # large H
        [1, 128, 2048, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, False, False],
        # partial H (non-512-aligned)
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, False, False],
        # large S
        [1, 2048, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), False, False, False],
        # --- ROW_MX + QK-norm ---
        # QK-norm only (no RoPE)
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), False, True, True],
        # QK-norm + RoPE
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
        # QK-norm + RoPE + bias
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, (1, None), True, True, True],
        # QK-norm without gamma
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, False],
        # QK-norm + GQA
        [1, 128, 512, 4, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
        # QK-norm + NBSd
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, (1, None), True, True, True],
        # QK-norm multi-S-tile
        [1, 256, 512, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
        # --- QK-norm large model configs ---
        [1, 10240, 8192, 4, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
        [1, 10240, 5120, 4, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
        [1, 10240, 5632, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
        # --- QK-norm partial H ---
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
        [1, 128, 5376, 2, 1, 128, QKVOutputLayout.BSD, False, (1, None), True, True, True],
    ]
    # fmt: on
    @pytest_parametrize(
        qkv_cte_row_mx_test_params,
        qkv_cte_row_mx_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_row_mx(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
        w_scale_shape,
        fused_rope,
        qk_norm,
        use_gamma,
    ):
        if not platform_target.is_trn3():
            pytest.skip("ROW_MX is only supported on TRN3.")

        B, S, H = batch, seqlen, hidden_dim
        I = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2
        eps = 1e-6

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        # === Generate FP8 input data and per-row scales ===
        np.random.seed(42)
        input_f32 = np.random.randn(B, S, H).astype(np.float32).clip(-240, 240)
        fp8_data = input_f32.astype(nl.float8_e4m3fn)
        row_scales = np.random.uniform(0.5, 2.0, (B, S, 1)).astype(np.float32)

        # Pack as [B, S, H+4]: [H bytes FP8 data | 4 bytes float32 scale]
        scale_bytes = row_scales.view(np.uint8).reshape(B, S, 4)
        packed_input = np.concatenate(
            [
                fp8_data.view(np.uint8),
                scale_bytes,
            ],
            axis=2,
        ).view(nl.float8_e4m3fn)

        # === Generate FP8 weights in [H//4, I, 4] unpacked format (MX_CONTIGUOUS) ===
        weights_f32 = (np.random.randn(H, I) / np.sqrt(H)).astype(np.float32)
        weights_fp8 = weights_f32.astype(nl.float8_e4m3fn)
        mx_weights = weights_fp8.reshape(H // 4, 4, I).transpose(0, 2, 1)

        # === Per-channel weight scale ===
        w_scale_rows = w_scale_shape[0]  # 1 or 128
        w_scale_1row = np.random.uniform(0.5, 2.0, (1, I)).astype(np.float32)
        w_scale = np.broadcast_to(w_scale_1row, (w_scale_rows, I)).copy()

        # === Optional bias and RoPE caches ===
        bias = np.random.randn(1, I).astype(np.float32) * 0.1 if qkv_bias else None
        cos_cache = None
        sin_cache = None
        if fused_rope:
            gen = rope_gaussian_tensor_generator()
            cos_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
            sin_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")

        # === Optional QK-norm ===
        q_gamma = np.random.randn(1, d_head).astype(np.float32) if use_gamma else None
        k_gamma = np.random.randn(1, d_head).astype(np.float32) if use_gamma else None
        qk_norm_config = (
            QKNormConfig(
                eps=eps,
                q_gamma_norm_weights=q_gamma,
                k_gamma_norm_weights=k_gamma,
            )
            if qk_norm
            else None
        )

        def generate_inputs(test_config):
            inputs = {
                "input": packed_input,
                "fused_qkv_weights": mx_weights,
                "output_layout": output_layout,
                "bias": bias,
                "quantization_type": QuantizationType.ROW_MX,
                "qkv_w_scale": w_scale,
                "qkv_in_scale": None,
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": eps,
                "hidden_actual": None,
                "fused_rope": fused_rope,
                "cos_cache": cos_cache,
                "sin_cache": sin_cache,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": QKVWeightLayout.MX_CONTIGUOUS,
            }
            if qk_norm_config is not None:
                inputs["qk_norm_pre_rope"] = qk_norm_config
            return inputs

        def output_tensor_descriptor(kernel_input):
            if output_layout == QKVOutputLayout.NBSd:
                num_heads = n_q_heads + 2 * n_kv_heads
                return {"out": np.zeros((num_heads, B, S, d_head), dtype=nl.bfloat16)}
            return {"out": np.zeros((B, S, I), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=row_mx_unpack_ref(
                torch_ref_wrapper(qkv_torch_ref, preserve_lower_precision=True),
                unpacked_fp8_data=fp8_data,
                row_scales=row_scales,
            ),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=6e-2,
            atol=1e-2,
        )

    ####################################################################################################################
    # QKV CTE: rmsnorm_quant ROW output → native MX weights (row-quantized FP8 input + per-block MX weight scales)
    ####################################################################################################################
    # fmt: off
    qkv_cte_mx_native_row_input_test_params = \
        "batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, output_layout, qkv_bias, fused_rope"
    qkv_cte_mx_native_row_input_test_perms = [
        # Basic: small H
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # + bias
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, False],
        # + RoPE
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, False, True],
        # + bias + RoPE
        [1, 128, 512, 2, 1, 128, QKVOutputLayout.BSD, True, True],
        # GQA (4Q/1KV)
        [1, 128, 512, 4, 1, 128, QKVOutputLayout.BSD, False, True],
        # multi-S-tile (S=256)
        [1, 256, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # NBSd layout
        [1, 128, 1024, 2, 1, 128, QKVOutputLayout.NBSd, False, True],
        # large H
        [1, 128, 2048, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # partial H (non-512-aligned)
        [1, 128, 896, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # large S
        [1, 2048, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # S=4096
        [1, 4096, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # S=8192
        [1, 8192, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # S=16384
        [1, 16384, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False],
        # S=32768
        [1, 32768, 512, 2, 1, 128, QKVOutputLayout.BSD, False, False],
    ]
    # fmt: on
    @pytest_parametrize(
        qkv_cte_mx_native_row_input_test_params,
        qkv_cte_mx_native_row_input_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_mx_native_row_input(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        output_layout,
        qkv_bias,
        fused_rope,
    ):
        """Test rmsnorm_quant ROW output fed directly into QKV CTE with native MX weights.

        Input is [B, S, H+4] FP8 with per-row float32 dequant scale packed in the tail
        (produced by rmsnorm_quant with QuantizationType.ROW). Weights are native MXFP8
        with per-block scales [H//32, I]. Post-matmul dequant applies only the per-row
        input scale (weight dequant is handled by nc_matmul_mx per-block scales).
        """
        if not platform_target.is_trn3():
            pytest.skip("Native MX with row-quantized FP8 input is only supported on TRN3.")

        B, S, H = batch, seqlen, hidden_dim
        I = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2
        eps = 1e-6

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        # === Generate FP8 input data and per-row scales (simulating rmsnorm_quant ROW output) ===
        np.random.seed(42)
        input_f32 = np.random.randn(B, S, H).astype(np.float32).clip(-240, 240)
        fp8_data = input_f32.astype(nl.float8_e4m3fn)
        row_scales = np.abs(input_f32).max(axis=-1, keepdims=True).astype(np.float32) / 240.0
        row_scales = np.maximum(row_scales, 1e-12)

        # Pack as [B, S, H+4]: [H bytes FP8 data | 4 bytes float32 scale]
        scale_bytes = row_scales.view(np.uint8).reshape(B, S, 4)
        packed_input = np.concatenate(
            [
                fp8_data.view(np.uint8),
                scale_bytes,
            ],
            axis=2,
        ).view(nl.float8_e4m3fn)

        # === Generate FP8 weights in [H//4, I, 4] (MX_CONTIGUOUS layout) with per-block scales ===
        weights_f32 = (np.random.randn(H, I) / np.sqrt(H)).astype(np.float32)
        weights_fp8 = weights_f32.astype(nl.float8_e4m3fn)
        mx_weights = weights_fp8.reshape(H // 4, 4, I).transpose(0, 2, 1)

        # Per-block MX weight scales [H//32, I] as uint8 exponents (2^(val-127))
        # Use real non-neutral scales to validate per-block dequant in the kernel
        w_scale = np.random.randint(120, 135, size=(H // 32, I)).astype(np.uint8)

        # === Optional bias and RoPE caches ===
        bias = np.random.randn(1, I).astype(np.float32) * 0.1 if qkv_bias else None
        cos_cache = None
        sin_cache = None
        if fused_rope:
            gen = rope_gaussian_tensor_generator()
            cos_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
            sin_cache = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")

        def generate_inputs(test_config):
            inputs = {
                "input": packed_input,
                "fused_qkv_weights": mx_weights,
                "output_layout": output_layout,
                "bias": bias,
                "quantization_type": QuantizationType.MX,
                "qkv_w_scale": w_scale,
                "qkv_in_scale": None,
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": eps,
                "hidden_actual": None,
                "fused_rope": fused_rope,
                "cos_cache": cos_cache,
                "sin_cache": sin_cache,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "weight_layout": QKVWeightLayout.MX_CONTIGUOUS,
            }
            return inputs

        def output_tensor_descriptor(kernel_input):
            if output_layout == QKVOutputLayout.NBSd:
                num_heads = n_q_heads + 2 * n_kv_heads
                return {"out": np.zeros((num_heads, B, S, d_head), dtype=nl.bfloat16)}
            return {"out": np.zeros((B, S, I), dtype=nl.bfloat16)}

        def mx_native_row_input_ref(
            input,
            fused_qkv_weights,
            output_layout,
            bias,
            quantization_type,
            qkv_w_scale,
            qkv_in_scale,
            fused_residual_add,
            mlp_prev,
            attention_prev,
            fused_norm_type,
            gamma_norm_weights,
            layer_norm_bias,
            norm_eps,
            hidden_actual,
            fused_rope,
            cos_cache,
            sin_cache,
            d_head,
            num_q_heads,
            num_kv_heads,
            store_output_in_sbuf,
            sbm,
            use_auto_allocation,
            load_input_with_DMA_transpose,
            weight_layout,
            block_size=None,
            dtype_mode=None,
            fp8_max=None,
            fp8_min=None,
            fp8_packed=None,
            is_h_dim_4h_transposed=None,
            k_cache=None,
            k_cos_cache=None,
            k_scale=None,
            k_sin_cache=None,
            kv_dtype=None,
            output_hbm=None,
            qk_norm_post_rope=None,
            qk_norm_post_rope_k_beta=None,
            qk_norm_post_rope_k_gamma=None,
            qk_norm_post_rope_q_beta=None,
            qk_norm_post_rope_q_gamma=None,
            qk_norm_pre_rope=None,
            qk_norm_pre_rope_k_beta=None,
            qk_norm_pre_rope_k_gamma=None,
            qk_norm_pre_rope_q_beta=None,
            qk_norm_pre_rope_q_gamma=None,
            slot_mapping=None,
            strided_input_config=None,
            transpose_k_cache=None,
            transposed_in=None,
            use_block_kv=None,
            v_cache=None,
            v_scale=None,
            k_squared_sum_out=None,
            q_squared_sum_out=None,
            v_squared_sum_out=None,
        ):
            """Reference for native MX weights + row-quantized FP8 input.

            Applies per-block MX weight dequant (2^(scale-127)) then:
            Reference = dequant_input @ dequant_weights [+ bias] [+ RoPE].
            """
            import torch

            # Unpack MX weights [H//4, I, 4] fp8 -> [H, I] float32
            w_np = fused_qkv_weights
            if isinstance(w_np, torch.Tensor):
                w_np = w_np.float().numpy()
            else:
                w_np = w_np.astype(np.float32)
            _, I_dim, _ = w_np.shape
            w_f32 = w_np.transpose(0, 2, 1).reshape(-1, I_dim)

            # Apply per-block MX weight scales: each block of 32 rows shares a scale
            ws_np = qkv_w_scale if isinstance(qkv_w_scale, np.ndarray) else qkv_w_scale.numpy()
            block_scales = 2.0 ** (ws_np.astype(np.float32) - 127.0)  # [H//32, I]
            for blk in range(H // 32):
                w_f32[blk * 32 : (blk + 1) * 32, :] *= block_scales[blk : blk + 1, :]

            # Dequantize input: fp8_data * row_scales
            inp_dequant = fp8_data.astype(np.float32) * row_scales

            # Plain matmul
            qkv_out = (inp_dequant.reshape(B * S, H) @ w_f32).reshape(B, S, I)

            # Apply bias if present
            if bias is not None:
                bias_np = bias.float().numpy() if isinstance(bias, torch.Tensor) else bias.astype(np.float32)
                qkv_out = qkv_out + bias_np

            # Apply RoPE if enabled
            if fused_rope:
                cos_np = (
                    cos_cache.float().numpy() if isinstance(cos_cache, torch.Tensor) else cos_cache.astype(np.float32)
                )
                sin_np = (
                    sin_cache.float().numpy() if isinstance(sin_cache, torch.Tensor) else sin_cache.astype(np.float32)
                )
                d_half = d_head // 2
                for head_idx in range(n_q_heads + n_kv_heads):
                    offset = head_idx * d_head
                    x1 = qkv_out[:, :, offset : offset + d_half].copy()
                    x2 = qkv_out[:, :, offset + d_half : offset + d_head].copy()
                    cos_v = cos_np[:, :, :d_half]
                    sin_v = sin_np[:, :, :d_half]
                    qkv_out[:, :, offset : offset + d_half] = x1 * cos_v - x2 * sin_v
                    qkv_out[:, :, offset + d_half : offset + d_head] = x2 * cos_v + x1 * sin_v

            if output_layout == QKVOutputLayout.NBSd:
                num_heads = n_q_heads + 2 * n_kv_heads
                qkv_out = qkv_out.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
                qkv_out = qkv_out.reshape(num_heads, B, S, d_head)

            return {"out": qkv_out.astype(np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=mx_native_row_input_ref,
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=False,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=6e-2,
            atol=1e-2,
        )

    ####################################################################################################################
    # QKV CTE ROW_MX + fused gamma RoPE
    ####################################################################################################################
    # fmt: off
    qkv_cte_row_mx_fused_gamma_test_params = \
        "seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head"
    qkv_cte_row_mx_fused_gamma_test_perms = [
        [128, 512, 2, 2, 128],
        [128, 512, 4, 1, 128],
        [256, 512, 2, 2, 128],
        # Llama3 70B TP16 S=10240
        [10240, 8192, 4, 1, 128],
        # Qwen3 32B TP16 S=10240
        [10240, 5120, 4, 1, 128],
        # Gemma3 27B TP16 S=10240
        [10240, 5632, 2, 1, 128],
    ]
    # fmt: on
    @pytest_parametrize(
        qkv_cte_row_mx_fused_gamma_test_params,
        qkv_cte_row_mx_fused_gamma_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_cte_row_mx_fused_gamma_rope(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
    ):
        if not platform_target.is_trn3():
            pytest.skip("ROW_MX is only supported on TRN3.")

        B, S, H = 1, seqlen, hidden_dim
        I = (n_q_heads + 2 * n_kv_heads) * d_head
        vnc_degree = 2
        eps = 1e-6

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        np.random.seed(42)
        tensor_gen = gaussian_tensor_generator(seed=42)

        input_f32 = np.random.randn(B, S, H).astype(np.float32).clip(-240, 240)
        fp8_data = input_f32.astype(nl.float8_e4m3fn)
        row_scales = np.random.uniform(0.5, 2.0, (B, S, 1)).astype(np.float32)

        scale_bytes = row_scales.view(np.uint8).reshape(B, S, 4)
        packed_input = np.concatenate(
            [
                fp8_data.view(np.uint8),
                scale_bytes,
            ],
            axis=2,
        ).view(nl.float8_e4m3fn)

        weights_f32 = (np.random.randn(H, I) / np.sqrt(H)).astype(np.float32)
        weights_fp8 = weights_f32.astype(nl.float8_e4m3fn)
        mx_weights = weights_fp8.reshape(H // 4, 4, I).transpose(0, 2, 1)

        w_scale = np.random.uniform(0.5, 2.0, (1, I)).astype(np.float32)

        q_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="q_gamma")
        k_gamma = tensor_gen(shape=(1, d_head), dtype=nl.float32, name="k_gamma")

        gen = rope_gaussian_tensor_generator()
        original_cos = gen(shape=(B, S, d_head), dtype=np.float32, name="cos_cache")
        original_sin = gen(shape=(B, S, d_head), dtype=np.float32, name="sin_cache")

        q_cos, q_sin, k_cos, k_sin = fuse_gamma_into_rope_caches(
            original_cos,
            original_sin,
            q_gamma,
            k_gamma,
            d_head,
        )

        def generate_inputs(test_config):
            return {
                "input": packed_input,
                "fused_qkv_weights": mx_weights,
                "output_layout": QKVOutputLayout.BSD,
                "bias": None,
                "quantization_type": QuantizationType.ROW_MX,
                "qkv_w_scale": w_scale,
                "qkv_in_scale": None,
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": NormType.NO_NORM,
                "gamma_norm_weights": None,
                "layer_norm_bias": None,
                "norm_eps": eps,
                "hidden_actual": None,
                "fused_rope": True,
                "cos_cache": q_cos,
                "sin_cache": q_sin,
                "k_cos_cache": k_cos,
                "k_sin_cache": k_sin,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": QKVWeightLayout.MX_CONTIGUOUS,
                "qk_norm_pre_rope": QKNormConfig(eps=eps, gamma_fused_in_rope_caches=True),
            }

        def output_tensor_descriptor(kernel_input):
            return {"out": np.zeros((B, S, I), dtype=nl.bfloat16)}

        # Torch ref receives unfused caches + explicit gamma (via gamma_unfused_ref),
        # and unpacked ROW_MX input + separate scale (via row_mx_unpack_ref).
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=row_mx_unpack_ref(
                gamma_unfused_ref(
                    torch_ref_wrapper(qkv_torch_ref, preserve_lower_precision=True),
                    original_cos=original_cos,
                    original_sin=original_sin,
                    q_gamma=q_gamma,
                    k_gamma=k_gamma,
                    eps=eps,
                ),
                unpacked_fp8_data=fp8_data,
                row_scales=row_scales,
            ),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=6e-2,
            atol=1e-2,
        )

    ####################################################################################################################
    # QKV CTE Test - strided input (gather uniformly-spaced blocks along S)
    ####################################################################################################################
    # fmt: off
    # Covers: basic accuracy (small S), BSD performance-style (larger S), non-1 batch, plus STATIC quant.
    # Constraints: BSD only, no fused_rope, no fused_residual_add, DMA transpose on, non-MX.
    # All cases use vnc_degree=2 (LNC2), matching the primary deployment target for qkv_cte.
    qkv_cte_strided_input_test_params = \
        "vnc, S_full, H, I, norm, bl, bs, bo, nl_tok, qt, layout, nq, nkv, dh, rope, kvq, qkn"
    qkv_cte_strided_input_test_perms = [
        # Small-S accuracy: pick every other 16-token block
        [2, 512, 512, 512, NormType.NO_NORM,        16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # Non-zero offset, block_len=16
        [2, 512, 1024, 512, NormType.NO_NORM,       16,  32, 16, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # LNC2 with block_len=32
        [2, 512, 8192, 512, NormType.NO_NORM,       32,  64,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # Performance-style: larger S, block_len=16, stride of 4 blocks (context-parallel shard)
        [2, 1024, 8192, 512, NormType.NO_NORM,      16,  64,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # STATIC quantization path, non-zero offset
        [2, 512, 8192, 1280, NormType.NO_NORM,      32,  64, 32, 256, QuantizationType.STATIC, QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # block_len=64 (blocks_per_tile=2): smallest non-trivial block count
        [2, 1024, 2048, 512, NormType.NO_NORM,      64, 128,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # block_len=64 with non-zero offset
        [2, 1024, 2048, 512, NormType.NO_NORM,      64, 128, 64, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # NBSd output layout (d_head=128 required); fused_qkv_dim = (n_q + 2*n_kv)*d_head = 4*128 = 512
        [2, 512, 8192, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.NBSd, 2,    1,    128,  False, False, None],
        # fused_rope + BSD with pre-gathered cos/sin caches (shape matches num_local_tokens)
        [2, 512, 8192, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  2,    1,    128,  True,  False, None],
        # fused_rope + NBSd
        [2, 512, 8192, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.NBSd, 2,    1,    128,  True,  False, None],
        # FP8 KV cache quantization (caller provides k_cache/v_cache sized for num_local_tokens)
        [2, 512, 2048, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  2,    1,    128,  False, "fp8", None],
        # BF16 KV cache (k_cache/v_cache provided, no scales) — exercises use_kv_cache gate in strided validator
        [2, 512, 2048, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  2,    1,    128,  False, "bf16", None],
        # Pre-RoPE QK-norm (applied to Q/K before RoPE rotation) + fused_rope
        [2, 512, 8192, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  2,    1,    128,  True,  False, "pre"],
        # Post-RoPE QK-norm (applied to Q/K after RoPE rotation) + fused_rope
        [2, 512, 8192, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  2,    1,    128,  True,  False, "post"],
        # Pre-RoPE QK-norm + NBSd output
        [2, 512, 8192, 512, NormType.NO_NORM,       16,  32,  0, 256, QuantizationType.NONE,   QKVOutputLayout.NBSd, 2,    1,    128,  True,  False, "pre"],
        # fused_rope + FP8 KV cache (combo stress)
        [2, 512, 2048, 512, NormType.NO_NORM,       32,  64,  0, 256, QuantizationType.NONE,   QKVOutputLayout.BSD,  2,    1,    128,  True,  "fp8", None],
        # Large block_offset (deep into S_full): block_offset + 15*32 + 16 = 2032 <= 2048
        [2, 2048, 4096, 512, NormType.NO_NORM,      16,  32, 1536, 256, QuantizationType.NONE, QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # --- block_len >= pmax (contiguous-tile regime) ---
        # block_len=128 (regime boundary, packed with blocks_per_tile=1)
        [2, 1024, 2048, 512, NormType.NO_NORM,     128, 256,   0, 512, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # block_len=128, non-zero offset
        [2, 1024, 2048, 512, NormType.NO_NORM,     128, 256, 128, 512, QuantizationType.NONE,   QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # block_len=256, 2 tiles/block
        [2, 2048, 2048, 512, NormType.NO_NORM,     256, 512,   0, 1024, QuantizationType.NONE,  QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # block_len=256, non-zero offset (exercises intra-block local_pos_in_block walk)
        [2, 2048, 2048, 512, NormType.NO_NORM,     256, 512, 256, 1024, QuantizationType.NONE,  QKVOutputLayout.BSD,  None, None, None, False, False, None],
        # block_len=256 + STATIC quant
        [2, 2048, 8192, 1280, NormType.NO_NORM,    256, 512,   0, 1024, QuantizationType.STATIC, QKVOutputLayout.BSD, None, None, None, False, False, None],
        # block_len=256 + NBSd + fused_rope
        [2, 2048, 8192, 512, NormType.NO_NORM,     256, 512,   0, 1024, QuantizationType.NONE,  QKVOutputLayout.NBSd, 2,    1,    128,  True,  False, None],
        # block_len=256 + FP8 KV cache
        [2, 2048, 2048, 512, NormType.NO_NORM,     256, 512,   0, 1024, QuantizationType.NONE,  QKVOutputLayout.BSD,  2,    1,    128,  False, "fp8", None],
        # block_len=512, 4 tiles/block (local_pos_in_block in {0,128,256,384})
        [2, 4096, 2048, 512, NormType.NO_NORM,     512, 1024,  0, 2048, QuantizationType.NONE,  QKVOutputLayout.BSD,  None, None, None, False, False, None],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_strided_input_test_params,
        qkv_cte_strided_input_test_perms,
    )
    def test_qkv_cte_strided_input_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc,
        S_full,
        H,
        I,
        norm,
        bl,
        bs,
        bo,
        nl_tok,
        qt,
        layout,
        nq,
        nkv,
        dh,
        rope,
        kvq,
        qkn,
    ):
        """Strided input gather: kernel reads num_local_tokens // block_len blocks along S,
        writes contiguous [B, num_local_tokens, I] output. Torch ref uses pre-gathered input."""
        # Unpack short param names to readable locals used in the test body.
        vnc_degree = vnc
        batch = 1
        seqlen_full = S_full
        hidden_dim = H
        fused_qkv_dim = I
        norm_type = norm
        block_len = bl
        block_stride = bs
        block_offset = bo
        num_local_tokens = nl_tok
        quantization_type = qt
        output_layout = layout
        n_q_heads = nq
        n_kv_heads = nkv
        d_head = dh
        fused_rope = rope
        fp8_kv_cache = kvq == "fp8"
        bf16_kv_cache = kvq == "bf16"
        uses_kv_cache = fp8_kv_cache or bf16_kv_cache
        qk_norm_phase = qkn  # None, "pre", or "post"

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        B = batch
        S_full = seqlen_full
        H = hidden_dim
        I = fused_qkv_dim
        eps = 1e-6
        dtype = nl.bfloat16

        np.random.seed(42)
        gen = gaussian_tensor_generator(seed=42)
        input_full = gen(shape=(B, S_full, H), dtype=dtype, name="input")

        weight_dtype = dtype if quantization_type == QuantizationType.NONE else nl.float8_e4m3
        fused_qkv_weights = gen(shape=(H, I), dtype=weight_dtype, name="fused_qkv_weights")

        qkv_w_scale = None
        qkv_in_scale = None
        if quantization_type == QuantizationType.STATIC:
            input_full = np.clip(input_full, -1.0, 1.0).astype(input_full.dtype)
            fused_qkv_weights = np.clip(fused_qkv_weights, -1.0, 1.0).astype(fused_qkv_weights.dtype)
            base_scale = 1.0 / 240.0
            qkv_w_scale = np.broadcast_to(
                base_scale * np.random.uniform(0.995, 1.005, (1, 3)).astype(np.float32), (128, 3)
            ).copy()
            qkv_in_scale = np.broadcast_to(
                base_scale * np.random.uniform(0.995, 1.005, (1, 1)).astype(np.float32), (128, 1)
            ).copy()

        gamma_norm_weights = (
            gen(shape=(1, H), dtype=dtype, name="gamma_norm_weights")
            if norm_type in (NormType.RMS_NORM, NormType.LAYER_NORM)
            else None
        )

        # Pre-gather input for torch reference: select blocks using same formula as the kernel.
        num_blocks = num_local_tokens // block_len
        block_indices = block_offset + np.arange(num_blocks) * block_stride  # start token of each block
        row_indices = (block_indices[:, None] + np.arange(block_len)[None, :]).reshape(-1)  # flat token indices
        gathered_input = input_full[:, row_indices, :]  # [B, num_local_tokens, H]

        # Pre-gathered cos/sin caches for fused_rope: shape (B, num_local_tokens, d_head).
        cos_cache = None
        sin_cache = None
        if fused_rope:
            rope_gen = rope_gaussian_tensor_generator()
            cos_cache = rope_gen(shape=(B, num_local_tokens, d_head), dtype=dtype, name="cos_cache")
            sin_cache = rope_gen(shape=(B, num_local_tokens, d_head), dtype=dtype, name="sin_cache")

        # KV cache (flat, non-transposed): cache sized for num_local_tokens. FP8 path provides
        # scales; BF16 path omits them (exercises use_kv_cache-without-scales validator gap).
        k_cache = v_cache = k_scale = v_scale = None
        kv_dtype = None
        if uses_kv_cache:
            kv_dim = n_kv_heads * d_head
            kv_dtype = nl.float8_e4m3 if fp8_kv_cache else nl.bfloat16
            k_cache = np.zeros((B, num_local_tokens, kv_dim), dtype=kv_dtype)
            v_cache = np.zeros((B, num_local_tokens, kv_dim), dtype=kv_dtype)
            if fp8_kv_cache:
                k_scale = np.full((128, 1), 1.67, dtype=np.float32)
                v_scale = np.full((128, 1), 1.67, dtype=np.float32)

        # QK-norm: per-head RMSNorm applied pre- or post-RoPE. Gamma weights shape [1, d_head].
        qk_norm_pre = qk_norm_post = None
        qkn_q_gamma_pre = qkn_k_gamma_pre = None
        qkn_q_gamma_post = qkn_k_gamma_post = None
        if qk_norm_phase is not None:
            qkn_q_gamma = gen(shape=(1, d_head), dtype=np.float32, name="qkn_q_gamma")
            qkn_k_gamma = gen(shape=(1, d_head), dtype=np.float32, name="qkn_k_gamma")
            qkn_cfg = QKNormConfig(
                eps=eps,
                q_gamma_norm_weights=qkn_q_gamma,
                k_gamma_norm_weights=qkn_k_gamma,
            )
            if qk_norm_phase == "pre":
                qk_norm_pre = qkn_cfg
                qkn_q_gamma_pre = qkn_q_gamma
                qkn_k_gamma_pre = qkn_k_gamma
            else:  # "post"
                qk_norm_post = qkn_cfg
                qkn_q_gamma_post = qkn_q_gamma
                qkn_k_gamma_post = qkn_k_gamma

        strided_cfg = StridedInputConfig(
            block_len=block_len,
            block_stride=block_stride,
            block_offset=block_offset,
            num_local_tokens=num_local_tokens,
        )

        # Determine kernel output dtype (mirrors qkv_cte output_dtype logic).
        out_dtype = nl.bfloat16 if quantization_type == QuantizationType.STATIC else dtype
        if output_layout == QKVOutputLayout.BSD:
            out_shape = (B, num_local_tokens, I)
        else:  # NBSd
            num_heads = n_q_heads + 2 * n_kv_heads
            out_shape = (num_heads, B, num_local_tokens, d_head)
        output_hbm = np.zeros(out_shape, dtype=out_dtype) if not uses_kv_cache else None

        def generate_inputs(test_config):
            inputs = {
                "input": input_full,
                "fused_qkv_weights": fused_qkv_weights,
                "output_layout": output_layout,
                "bias": None,
                "quantization_type": quantization_type,
                "qkv_w_scale": qkv_w_scale,
                "qkv_in_scale": qkv_in_scale,
                "fused_residual_add": False,
                "mlp_prev": None,
                "attention_prev": None,
                "fused_norm_type": norm_type,
                "gamma_norm_weights": gamma_norm_weights,
                "layer_norm_bias": None,
                "norm_eps": eps,
                "hidden_actual": None,
                "fused_rope": fused_rope,
                "cos_cache": cos_cache,
                "sin_cache": sin_cache,
                "d_head": d_head,
                "num_q_heads": n_q_heads,
                "num_kv_heads": n_kv_heads,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": QKVWeightLayout.CONTIGUOUS,
                "strided_input_config": strided_cfg,
            }
            if uses_kv_cache:
                inputs["k_cache.must_alias_input"] = k_cache
                inputs["v_cache.must_alias_input"] = v_cache
                if fp8_kv_cache:
                    inputs["k_scale"] = k_scale
                    inputs["v_scale"] = v_scale
                    inputs["kv_dtype"] = kv_dtype
                    inputs["fp8_max"] = 240.0
                    inputs["fp8_min"] = -240.0
            else:
                inputs["output_hbm.must_alias_input"] = output_hbm
            if qk_norm_pre is not None:
                inputs["qk_norm_pre_rope"] = qk_norm_pre
                inputs["qk_norm_pre_rope_q_gamma"] = qkn_q_gamma_pre
                inputs["qk_norm_pre_rope_k_gamma"] = qkn_k_gamma_pre
            if qk_norm_post is not None:
                inputs["qk_norm_post_rope"] = qk_norm_post
                inputs["qk_norm_post_rope_q_gamma"] = qkn_q_gamma_post
                inputs["qk_norm_post_rope_k_gamma"] = qkn_k_gamma_post
            if quantization_type == QuantizationType.STATIC and n_q_heads is None:
                # STATIC requires head counts; default to 2/1 for BSD test configs that leave them unset.
                inputs["num_q_heads"] = 2
                inputs["num_kv_heads"] = 1
                inputs["d_head"] = I // (2 + 2 * 1)
            return inputs

        def output_tensor_descriptor(kernel_input):
            if uses_kv_cache:
                q_dim = n_q_heads * d_head
                return {
                    "q_tensor_hbm": np.zeros((B, num_local_tokens, q_dim), dtype=dtype),
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                }
            return {"output_hbm": np.zeros(out_shape, dtype=out_dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=strided_gather_ref(
                torch_ref_wrapper(qkv_torch_ref),
                gathered_input=gathered_input,
            ),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=1e-1 if uses_kv_cache else 2e-2,
            atol=1e-5 if quantization_type == QuantizationType.NONE and not uses_kv_cache else 2e-2,
        )

    # fmt: off
    # Each case violates exactly one validation rule.
    qkv_cte_strided_input_negative_test_params = "case_id, block_len, block_stride, block_offset, num_local_tokens, S_full, fused_residual_add, fused_norm_type, quantization_type, bad_output_shape"
    qkv_cte_strided_input_negative_test_perms = [
        # block_len: not a divisor or multiple of pmax
        ["block_len_96",      96, 192,   0, 192,  512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # block_len: zero / negative
        ["block_len_0",        0,  64,   0, 256,  512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # block_stride < block_len (overlap)
        ["stride_lt_block",   32,  16,   0, 256,  512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # block_stride not a multiple of block_len
        ["stride_not_mul",    32,  48,   0, 256,  512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # num_local_tokens not a multiple of block_len
        ["nlt_not_mul",       32,  64,   0, 248,  512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # num_local_tokens zero
        ["nlt_zero",          32,  64,   0,   0,  512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # Negative block_offset
        ["offset_negative",   32,  64,  -32, 256, 512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # Reads past end of input S
        ["reads_past_end",    32,  64,  512, 256, 512, False, NormType.NO_NORM, QuantizationType.NONE, False],
        # Incompatible: fused_residual_add
        ["fused_residual",    16,  32,   0, 256,  512, True,  NormType.NO_NORM, QuantizationType.NONE, False],
        # Incompatible: non-NO_NORM
        ["rms_norm",          16,  32,   0, 256,  512, False, NormType.RMS_NORM, QuantizationType.NONE, False],
        # Incompatible: MX quantization
        ["mx_quant",          16,  32,   0, 256,  512, False, NormType.NO_NORM, QuantizationType.MX,    False],
        # output_hbm shape mismatch
        ["bad_out_shape",     16,  32,   0, 256,  512, False, NormType.NO_NORM, QuantizationType.NONE, True],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_cte_strided_input_negative_test_params,
        qkv_cte_strided_input_negative_test_perms,
    )
    def test_qkv_cte_strided_input_invalid(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        case_id,
        block_len,
        block_stride,
        block_offset,
        num_local_tokens,
        S_full,
        fused_residual_add,
        fused_norm_type,
        quantization_type,
        bad_output_shape,
    ):
        """Validator must reject each bad strided_input_config or incompatible combination."""
        vnc_degree = 2
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        B, H, I = 1, 512, 512
        dtype = nl.bfloat16

        np.random.seed(42)
        gen = gaussian_tensor_generator(seed=42)
        weight_dtype = dtype if quantization_type != QuantizationType.MX else nl.float8_e4m3
        input_full = gen(shape=(B, S_full, H), dtype=dtype, name="input")
        fused_qkv_weights = gen(
            shape=(H // 4 if quantization_type == QuantizationType.MX else H, I),
            dtype=weight_dtype,
            name="fused_qkv_weights",
        )

        residual = gen(shape=(B, S_full, H), dtype=dtype, name="residual") if fused_residual_add else None
        gamma = gen(shape=(1, H), dtype=dtype, name="gamma") if fused_norm_type != NormType.NO_NORM else None
        strided_cfg = StridedInputConfig(
            block_len=block_len,
            block_stride=block_stride,
            block_offset=block_offset,
            num_local_tokens=num_local_tokens,
        )
        # num_local_tokens may be 0/bad; clamp shape to >=1 to still construct a numpy array.
        out_S = max(num_local_tokens, 1) if not bad_output_shape else 99
        output_hbm = np.zeros((B, out_S, I), dtype=dtype)

        def generate_inputs(_):
            return {
                "input": input_full,
                "fused_qkv_weights": fused_qkv_weights,
                "output_layout": QKVOutputLayout.BSD,
                "bias": None,
                "quantization_type": quantization_type,
                "qkv_w_scale": None,
                "qkv_in_scale": None,
                "fused_residual_add": fused_residual_add,
                "mlp_prev": residual,
                "attention_prev": residual,
                "fused_norm_type": fused_norm_type,
                "gamma_norm_weights": gamma,
                "layer_norm_bias": None,
                "norm_eps": 1e-6,
                "hidden_actual": None,
                "fused_rope": False,
                "cos_cache": None,
                "sin_cache": None,
                "d_head": None,
                "num_q_heads": None,
                "num_kv_heads": None,
                "store_output_in_sbuf": False,
                "sbm": None,
                "use_auto_allocation": False,
                "load_input_with_DMA_transpose": True,
                "is_h_dim_4h_transposed": False,
                "weight_layout": (
                    QKVWeightLayout.MX_CONTIGUOUS
                    if quantization_type == QuantizationType.MX
                    else QKVWeightLayout.CONTIGUOUS
                ),
                "strided_input_config": strided_cfg,
                "output_hbm.must_alias_input": output_hbm,
            }

        def output_tensor_descriptor(_):
            return {"output_hbm": np.zeros((B, out_S, I), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv,
            torch_ref=torch_ref_wrapper(qkv_torch_ref),
            kernel_input_generator=generate_inputs,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, is_negative_test=True)

    ####################################################################################################################
    # QKV MLA MX Test
    ####################################################################################################################
    qkv_mla_mx_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_heads, qk_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, norm_eps"
    qkv_mla_mx_test_perms = [
        [2, 1, 1, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 32, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 128, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 128, 7168, 2, 1536, 512, 64, 32, 64, 1e-6],
        [2, 1, 256, 4096, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 512, 7168, 2, 1536, 512, 64, 64, 64, 1e-6],
        [2, 1, 512, 3072, 2, 1536, 512, 64, 64, 64, 1e-6],
        [2, 1, 512, 3072, 3, 1536, 512, 64, 64, 64, 1e-6],
        [2, 1, 512, 3072, 3, 1536, 512, 64, 32, 32, 1e-6],
        [2, 1, 512, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 1024, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 1024, 8192, 1, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 2048, 7168, 3, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 2048, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 4096, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 8192, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 16384, 7168, 2, 1536, 512, 64, 128, 128, 1e-6],
        # High-H exercises the K-slab dispatch (9216/512=18, 10240/512=20).
        [2, 1, 4096, 9216, 2, 1536, 512, 64, 128, 128, 1e-6],
        [2, 1, 4096, 10240, 2, 1536, 512, 64, 128, 128, 1e-6],
        # Higher head counts. n_heads=8 currently fits but is at the edge.
        [2, 1, 2048, 7168, 8, 1536, 512, 64, 128, 128, 1e-6],
        # Larger qk_lora_rank (next valid 512-multiple after 1536).
        [2, 1, 2048, 7168, 2, 2048, 512, 64, 128, 128, 1e-6],
        # Larger kv_lora_rank.
        [2, 1, 2048, 7168, 2, 1536, 1024, 64, 128, 128, 1e-6],
        # Different qk_rope_head_dim (must be even).
        [2, 1, 2048, 7168, 2, 1536, 512, 32, 128, 128, 1e-6],
        [2, 1, 2048, 7168, 2, 1536, 512, 128, 128, 128, 1e-6],
    ]

    @pytest.mark.parametrize(qkv_mla_mx_test_params, qkv_mla_mx_test_perms)
    def test_qkv_cte_mla_mx_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        norm_eps,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        def input_generator(test_config):
            kernel_input = build_qkv_mla_input(
                batch=batch,
                seqlen=seqlen,
                hidden_dim=hidden_dim,
                qk_lora_rank=qk_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
                n_heads=n_heads,
                variant="v32",
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=v_head_dim,
            )
            return kernel_input

        def output_tensor_descriptor(kernel_input):
            return {
                "q": np.zeros((batch, seqlen, n_heads, qk_head_dim), dtype=nl.bfloat16),
                "k": np.zeros((batch, seqlen, n_heads, qk_head_dim), dtype=nl.bfloat16),
                "v": np.zeros((batch, seqlen, n_heads, v_head_dim), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv_mla_mx,
            torch_ref=torch_ref_wrapper(qkv_mla_mx_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=5e-2, atol=1e-2)

    ####################################################################################################################
    # QKV MLA MX Test (DeepSeek v4 Arch)
    ####################################################################################################################
    qkv_mla_v4_mx_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_heads, qk_lora_rank, kv_lora_rank, qk_rope_head_dim, head_dim, norm_eps"
    qkv_mla_v4_mx_test_perms = [
        # Seq-length sweep at the standard config.
        [2, 1, 128, 7168, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 256, 4096, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 512, 7168, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 1024, 7168, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 2048, 7168, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 4096, 7168, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 8192, 7168, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 16384, 7168, 2, 1536, 512, 64, 512, 1e-6],
        # Larger head counts (N-pressure on wq_b — exercises the head-chunk dispatch).
        [2, 1, 8192, 7168, 4, 1536, 512, 64, 512, 1e-6],
        [2, 1, 8192, 7168, 8, 1536, 512, 64, 512, 1e-6],
        # Smaller seq lengths (sharding edge cases, S<P_MAX).
        [2, 1, 1, 7168, 2, 1536, 512, 64, 512, 1e-6],
        [2, 1, 32, 7168, 2, 1536, 512, 64, 512, 1e-6],
        # n_heads=1 (extreme low-N).
        [2, 1, 8192, 7168, 1, 1536, 512, 64, 512, 1e-6],
        # Smaller head_dim (must satisfy head_dim > qk_rope_head_dim).
        [2, 1, 8192, 7168, 2, 1536, 512, 64, 256, 1e-6],
        # Different qk_rope_head_dim (even).
        [2, 1, 8192, 7168, 2, 1536, 512, 32, 512, 1e-6],
        [2, 1, 8192, 7168, 2, 1536, 512, 128, 512, 1e-6],
    ]

    @pytest.mark.parametrize(qkv_mla_v4_mx_test_params, qkv_mla_v4_mx_test_perms)
    def test_qkv_cte_mla_v4_mx_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        head_dim,
        norm_eps,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MX Quantization is only supported on TRN3.")

        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )
        kv_dim = kv_lora_rank + qk_rope_head_dim

        def input_generator(test_config):
            return build_qkv_mla_input(
                batch=batch,
                seqlen=seqlen,
                hidden_dim=hidden_dim,
                qk_lora_rank=qk_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
                n_heads=n_heads,
                variant="v4",
                head_dim=head_dim,
            )

        def output_tensor_descriptor(kernel_input):
            return {
                "q": np.zeros((batch, seqlen, n_heads, head_dim), dtype=nl.bfloat16),
                "kv": np.zeros((batch, seqlen, kv_dim), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv_mla_mx_deepseek_v4,
            torch_ref=torch_ref_wrapper(qkv_mla_mx_deepseek_v4_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=5e-2, atol=1e-2)


@pytest_marks(["qkv", "cte", "model", "mx"])
@final
class TestQkvCteModel:
    """Model-driven tests for QKV CTE kernel, organized by tier."""

    _QKV_MODEL_PARAMS = "model_name, quant_type, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, qkv_bias, fused_rope, use_gamma, in_scale_shape, w_scale_shape"

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize({ModelTestType.OPTIMAL: qkv_cte_model_configs.get(ModelTestType.OPTIMAL, [])})
        if qkv_cte_model_configs
        else ([], [])
    )

    def _run_model_test(self, **kwargs):
        """Common test logic for model tiers."""
        test_manager = kwargs["test_manager"]
        collector = kwargs["collector"]
        platform_target = kwargs["platform_target"]
        model_name = kwargs["model_name"]
        quant_config = kwargs["quant_type"]
        batch = kwargs["batch"]
        seqlen = kwargs["seqlen"]
        hidden_dim = kwargs["hidden_dim"]
        n_q_heads = kwargs["n_q_heads"]
        n_kv_heads = kwargs["n_kv_heads"]
        d_head = kwargs["d_head"]
        qkv_bias = kwargs["qkv_bias"]
        fused_rope = kwargs["fused_rope"]
        use_gamma = kwargs["use_gamma"]
        in_scale_shape = kwargs["in_scale_shape"]
        w_scale_shape = kwargs["w_scale_shape"]

        kernel = TestQkvCteKernel()

        if quant_config == QuantizationType.NONE:
            compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
            fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head
            kernel.run_qkv_cte_test_utf(
                test_manager=test_manager,
                compiler_args=compiler_args,
                B=batch,
                H=hidden_dim,
                S=seqlen,
                fused_qkv_dim=fused_qkv_dim,
                lnc_degree=2,
                dtype=nl.bfloat16,
                eps=1e-6,
                norm_type=NormType.NO_NORM,
                use_dma_transpose=False,
                fused_add=False,
                qkv_bias=qkv_bias,
                norm_bias=False,
                output_layout=QKVOutputLayout.BSD,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
            )
        elif quant_config == QuantizationType.STATIC:
            if platform_target != Platforms.TRN2:
                pytest.skip("static FP8 only on TRN2")
            kernel.test_qkv_cte_static_quantization(
                test_manager=test_manager,
                platform_target=platform_target,
                vnc_degree=2,
                batch=batch,
                seqlen=seqlen,
                hidden_dim=hidden_dim,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
                norm_type=NormType.NO_NORM,
                use_dma_transpose=False,
                fused_add=False,
                add_bias=qkv_bias,
                norm_bias=False,
                output_layout=QKVOutputLayout.BSD,
                eps=1e-6,
            )
        elif quant_config == QuantizationType.STATIC_MX:
            if not platform_target.is_trn3():
                pytest.skip("MX only on TRN3")
            if model_name in STATIC_DEQUANT_MODELS:
                kernel.test_qkv_cte_mxfp8_static_dequant(
                    test_manager=test_manager,
                    collector=collector,
                    platform_target=platform_target,
                    batch=batch,
                    seqlen=seqlen,
                    hidden_dim=hidden_dim,
                    n_q_heads=n_q_heads,
                    n_kv_heads=n_kv_heads,
                    d_head=d_head,
                    output_layout=QKVOutputLayout.BSD,
                    qkv_bias=qkv_bias,
                    in_scale_shape=tuple(in_scale_shape),
                    w_scale_shape=tuple(w_scale_shape),
                    fused_rope=fused_rope,
                )
            elif model_name in QK_NORM_MODELS or model_name in FUSED_GAMMA_ROPE_MODELS:
                kernel.test_qkv_cte_mxfp8_static_dequant_qk_norm(
                    test_manager=test_manager,
                    collector=collector,
                    platform_target=platform_target,
                    seqlen=seqlen,
                    hidden_dim=hidden_dim,
                    n_q_heads=n_q_heads,
                    n_kv_heads=n_kv_heads,
                    d_head=d_head,
                    output_layout=QKVOutputLayout.BSD,
                    qkv_bias=qkv_bias,
                    fused_rope=fused_rope,
                    use_gamma=use_gamma,
                )
            else:
                pytest.skip(f"No MX variant defined for model {model_name}")

    @pytest.mark.optimal
    @pytest.mark.parametrize(_QKV_MODEL_PARAMS, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        model_name,
        quant_type,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        qkv_bias,
        fused_rope,
        use_gamma,
        in_scale_shape,
        w_scale_shape,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)

    ####################################################################################################################
    # QKV CTE FP8 Packed K-cache model perf sweep (llama3_70b / gptoss_120b)
    #
    # Reproduces the CR-275918024 comparison: BF16 QKV projection + FP8 block KV cache,
    # comparing the packed K/V cache layout against the unpacked baseline across model x
    # seqlen x TP. The 'packed' variant exercises the new head-major layouts
    #   K: [num_blocks, num_kv_heads, block_size//2, d_head, 2]
    #   V: [num_blocks, num_kv_heads, block_size, d_head]
    # The 'unpacked' variant is the fp8 block-KV baseline (no packing).
    # Run on hardware (shared-fleet trn3_a0) and read latency/MFU from the QOR CSV.
    ####################################################################################################################
    # fmt: off
    qkv_cte_fp8_packed_perf_sweep_params = "model_name, seqlen, tp, variant"
    qkv_cte_fp8_packed_perf_sweep_perms = [
        [m, s, tp, v]
        for m in ["llama3_70b", "gptoss_120b"]
        for s in [1024, 8192]
        for tp in [4, 8]
        for v in ["packed", "unpacked"]
    ]
    # fmt: on

    @pytest.mark.optimal
    @pytest_parametrize(
        qkv_cte_fp8_packed_perf_sweep_params,
        qkv_cte_fp8_packed_perf_sweep_perms,
    )
    def test_qkv_cte_fp8_packed_perf_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        model_name,
        seqlen,
        tp,
        variant,
    ):
        """Perf sweep: BF16 projection + FP8 block KV cache, packed vs unpacked layout."""
        m = _QKV_MODELS[model_name]
        n_q_heads, n_kv_heads = _get_sharded_head_counts(tp, m["n_q_heads"], m["n_kv_heads"])
        d_head = m["d_head"]
        hidden_dim = math.ceil(m["hidden"] / 512) * 512
        batch = 1
        block_size = 128
        # Enough physical blocks to hold all sequence positions (round up + headroom).
        num_blocks = math.ceil(seqlen / block_size) + 1

        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        np.random.seed(42)
        slot_mapping = build_noncontiguous_slot_mapping(seqlen, batch, block_size, num_blocks)

        kernel = TestQkvCteKernel()
        kernel.run_qkv_cte_test_utf(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=2,
            dtype=nl.bfloat16,
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            qkv_bias=m["bias"],
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fp8_kv_cache=True,
            bf16_kv_cache=False,
            k_scale_val=1.67,
            v_scale_val=1.67,
            fp8_max=240.0,
            fp8_min=-240.0,
            use_block_kv=True,
            fp8_packed=(variant == "packed"),
            transpose_k_cache=False,
            num_blocks=num_blocks,
            block_size=block_size,
            slot_mapping=slot_mapping,
            tensor_gen=gaussian_tensor_generator(seed=42),
            quantization_type=QuantizationType.NONE,
            rtol=1e-1,
            atol=1e-5,
        )
