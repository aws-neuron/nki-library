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
Integration tests for QKV CTE kernel.

Tests cover various configurations including fused operations (normalization, residual add, RoPE),
different data types, batch sizes, sequence lengths, and output layouts.
"""

from test.utils.model_test_configs import no_model_configs

try:
    from test.integration.nkilib.core.qkv.test_qkv_cte_model_config import (
        qkv_cte_model_configs,
    )
except ImportError:
    qkv_cte_model_configs = no_model_configs()

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import (
    NormType,
    QKVOutputLayout,
    QuantizationType,
)

from test.integration.nkilib.core.qkv.test_qkv_common import (
    rope_gaussian_tensor_generator,
    run_qkv_test,
)
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    Platforms,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework  # noqa: F401

# Constructed once at module load and shared across all callers that rely on the default,
# matching the previous behavior where the default argument expression was evaluated once.
_DEFAULT_TENSOR_GEN = gaussian_tensor_generator()

# Utility maps to serialize type so that generated test name don't contain Python objects
dtype_int_to_type = {0: nl.bfloat16, 1: np.float16, 2: np.float32}
dtype_type_to_int = {nl.bfloat16: 0, np.float16: 1, np.float32: 2}


class TestQkvCteKernel:
    def run_qkv_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        B: int,
        H: int,
        S: int,
        fused_qkv_dim: int,
        lnc_degree: int,
        dtype,
        eps: int | float,
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
        max_seq_len: int | None = None,
        k_scale_val: float | None = None,
        v_scale_val: float | None = None,
        fp8_max: float = 240.0,
        fp8_min: float = -240.0,
        use_block_kv: bool = False,
        num_blocks: int | None = None,
        block_size: int | None = None,
        slot_mapping: np.ndarray | None = None,
        is_h_dim_4h_transposed: bool = False,
        threshold: tuple[float, float] = (2e-2, 1e-5),
        qkv_in_scale_for_mx: np.ndarray | None = None,
        qkv_w_scale_for_mx: np.ndarray | None = None,
    ):
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
            use_dma_transpose=use_dma_transpose,
            fused_add=fused_add,
            norm_bias=norm_bias,
            qkv_bias=qkv_bias,
            fused_rope=fused_rope,
            output_layout=output_layout,
            hidden_actual=hidden_actual,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            quantization_type=quantization_type,
            tensor_gen=tensor_gen,
            fp8_kv_cache=fp8_kv_cache,
            max_seq_len=max_seq_len,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            use_block_kv=use_block_kv,
            num_blocks=num_blocks,
            block_size=block_size,
            slot_mapping=slot_mapping,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            qkv_in_scale_for_mx=qkv_in_scale_for_mx,
            qkv_w_scale_for_mx=qkv_w_scale_for_mx,
            rtol=threshold[0],
            atol=threshold[1],
        )

    ################################################################################################
    # QKV RoPE FUSION TEST
    ################################################################################################
    # fmt: off
    qkv_cte_kernel_fused_rope_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, output_layout, eps, fused_rope"
    qkv_cte_kernel_fused_rope_test_perms = [
        #[2, 1, 128, 8192, 1, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        #[2, 1, 128, 8192, 2, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        #[2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 1, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 2, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 8, 1, 128, NormType.NO_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        #[2, 1, 192, 8192, 8, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        #[2, 1, 192, 8320, 8, 1, 128, NormType.NO_NORM, False, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        #[2, 1, 128, 8192, 1, 1, 128, NormType.RMS_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        #[2, 1, 128, 8192, 2, 1, 128, NormType.RMS_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        #[2, 1, 128, 8192, 8, 1, 128, NormType.RMS_NORM, True, False, False, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 1, 1, 128, NormType.NO_NORM, True, False, True, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 2, 1, 128, NormType.NO_NORM, True, False, True, False, QKVOutputLayout.BSD, 1e-6, True],
        [2, 1, 128, 8192, 8, 1, 64, NormType.NO_NORM, True, False, True, False, QKVOutputLayout.BSD, 1e-6, True],
    ]
    # fmt: on
    @pytest.mark.parametrize(
        qkv_cte_kernel_fused_rope_test_params,
        qkv_cte_kernel_fused_rope_test_perms,
    )
    def test_qkv_cte_fused_rope_unit_primitives(
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
        self.run_qkv_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
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
        )
