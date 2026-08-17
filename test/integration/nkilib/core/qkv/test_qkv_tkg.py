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

import enum
from typing import Optional, TypedDict, final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.qkv.qkv_tkg import qkv_tkg
from nkilib_src.nkilib.core.qkv.qkv_torch import qkv_torch_ref
from nkilib_src.nkilib.core.subkernels.layernorm_tkg import (
    SHARDING_THRESHOLD as layernorm_sharding_threshold,
)
from nkilib_src.nkilib.core.subkernels.rmsnorm_tkg import (
    SHARDING_THRESHOLD as rmsnorm_sharding_threshold,
)
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.common_types import (
    DtypeMode,
    NormType,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
)
from nkilib_src.nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib_src.nkilib.core.utils.logging import Logger
from typing_extensions import override

from test.integration.nkilib.core.qkv.test_qkv_common import build_qkv_input, run_qkv_test
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    Platforms,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


class QkvTkgClassification(enum.Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @staticmethod
    def classify(B: int, S: int, H: int, fused_qkv_dim: int):
        flops_estimate = B * S * H * fused_qkv_dim

        # TODO: proper classification
        if flops_estimate <= 130000000000:
            return QkvTkgClassification.SMALL
        elif flops_estimate <= 626000000000:
            return QkvTkgClassification.MEDIUM
        else:
            return QkvTkgClassification.LARGE

    @override
    def __str__(self):
        return self.name


@nki.jit
def qkv_tkg_sb2sb_wrapper_kernel(
    input: nl.ndarray,
    fused_qkv_weights: nl.ndarray,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[nl.ndarray] = None,
    # -- Quantization
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[nl.ndarray] = None,
    qkv_in_scale: Optional[nl.ndarray] = None,
    # -- Fused Residual Add
    fused_residual_add: bool = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attention_prev: Optional[nl.ndarray] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[nl.ndarray] = None,
    layer_norm_bias: Optional[nl.ndarray] = None,
    norm_eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    # --- Fused RoPE Related
    fused_rope: Optional[bool] = False,
    cos_cache: Optional[nl.ndarray] = None,
    sin_cache: Optional[nl.ndarray] = None,
    k_cos_cache: Optional[nl.ndarray] = None,
    k_sin_cache: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # --- FP8 KV Cache Quantization Related (unused, accepted for signature match with qkv)
    k_cache: Optional[nl.ndarray] = None,
    v_cache: Optional[nl.ndarray] = None,
    k_scale: Optional[nl.ndarray] = None,
    v_scale: Optional[nl.ndarray] = None,
    fp8_max: Optional[float] = None,
    fp8_min: Optional[float] = None,
    kv_dtype: Optional[type] = None,
    transpose_k_cache: bool = False,
    # --- Block KV Cache Related (unused, accepted for signature match with qkv)
    use_block_kv: bool = False,
    fp8_packed: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[nl.ndarray] = None,
    # -----------------------------------------
    store_output_in_sbuf: bool = False,
    # -----------------------------------------
    # User can optionally PASS Sbuf manager
    # -----------------------------------------
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    # ----------------------------------------
    load_input_with_DMA_transpose: bool = True,
    # ----------------------------------------
    is_h_dim_4h_transposed: bool = False,
    # ----------------------------------------
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    # --- QK-Norm Related (unused, accepted for signature match with qkv_torch_ref)
    qk_norm_pre_rope=None,
    qk_norm_post_rope=None,
    qk_norm_pre_rope_q_gamma: Optional[nl.ndarray] = None,
    qk_norm_pre_rope_k_gamma: Optional[nl.ndarray] = None,
    qk_norm_post_rope_q_gamma: Optional[nl.ndarray] = None,
    qk_norm_post_rope_k_gamma: Optional[nl.ndarray] = None,
    qk_norm_pre_rope_q_beta: Optional[nl.ndarray] = None,
    qk_norm_pre_rope_k_beta: Optional[nl.ndarray] = None,
    qk_norm_post_rope_q_beta: Optional[nl.ndarray] = None,
    qk_norm_post_rope_k_beta: Optional[nl.ndarray] = None,
    transposed_in: bool = False,
    # --- Strided Input / Output (unused, accepted for signature match with qkv / qkv_torch_ref)
    strided_input_config=None,
    output_hbm: Optional[nl.ndarray] = None,
    # --- Squared-sum outputs (unused here; accepted for signature match with qkv / qkv_torch_ref)
    q_squared_sum_out: Optional[nl.ndarray] = None,  # noqa: ARG001
    k_squared_sum_out: Optional[nl.ndarray] = None,  # noqa: ARG001
    v_squared_sum_out: Optional[nl.ndarray] = None,  # noqa: ARG001
    # Signature parity with qkv_torch_ref; unused in wrapper (qkv_tkg is dtype-passing).
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,  # noqa: ARG001
) -> nl.ndarray:
    hidden = input
    qkv_w = fused_qkv_weights
    qkv_bias = bias
    fused_add = fused_residual_add
    mlp_prev = mlp_prev
    attn_prev = attention_prev
    norm_type = fused_norm_type
    norm_w = gamma_norm_weights
    norm_bias = layer_norm_bias
    eps = norm_eps
    hidden_actual = hidden_actual
    output_layout = output_layout
    output_in_sbuf = store_output_in_sbuf
    d_head = d_head
    sbm = sbm

    assert output_in_sbuf

    B, S, H = hidden.shape
    BxS = B * S
    H0 = nl.tile_size.pmax
    H1 = H // H0
    _, I = qkv_w.shape

    sbm = SbufManager(
        0,
        # Leave 16KB for compiler dynamic scratch buffer usage
        # Compilation fails for some tests without this extra space
        nl.tile_size.total_available_sbuf_size - 16 * 1024,
        logger=Logger("qkv-tkg-sb2sb-wrapper"),
        use_auto_alloc=False,
    )
    assert hidden.buffer == nl.hbm or hidden.buffer == nl.shared_hbm

    sbm.open_scope()

    # Load input from HBM to SBUF: (B, S, H) -> (H0, BxS, H1)
    # For LNC>1, H must be interleaved as [num_shards, H0, H1_shard] to match physical memory layout
    _, num_shards, _ = get_verified_program_sharding_info()
    H1_shard = H1 // num_shards
    hidden_sb = sbm.alloc_heap(
        shape=(H0, BxS, H1),
        dtype=hidden.dtype,
        buffer=nl.sbuf,
        name="hidden_sb",
    )
    input_view = (
        hidden.flatten_dims(start_dim=0, end_dim=1)  # (BxS, H)
        .reshape_dim(dim=1, shape=[num_shards, H0, H1_shard])  # (BxS, num_shards, H0, H1_shard)
        .permute(dims=[2, 0, 1, 3])  # (H0, BxS, num_shards, H1_shard)
    )
    dst_view = hidden_sb.reshape_dim(dim=2, shape=[num_shards, H1_shard])  # (H0, BxS, num_shards, H1_shard)
    nisa.dma_copy(
        dst=dst_view,
        src=input_view,
    )

    output_sb = qkv_tkg(
        hidden=hidden_sb,
        qkv_w=qkv_w,
        norm_w=norm_w,
        fused_add=fused_add,
        mlp_prev=mlp_prev,
        attn_prev=attn_prev,
        d_head=d_head,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        output_layout=output_layout,
        eps=eps,
        norm_type=norm_type,
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        output_in_sbuf=True,
        qkv_bias=qkv_bias,
        norm_bias=norm_bias,
        hidden_actual=hidden_actual,
        sbm=sbm,
    )

    assert not isinstance(output_sb, tuple), "fused_residual_add is not supported by this wrapper"
    assert output_sb.shape == (BxS, I)

    # Allocate output tensor with layout-specific shape
    if output_layout == QKVOutputLayout.BSD:
        output_hbm = nl.ndarray((BxS, I), dtype=hidden_sb.dtype, buffer=nl.shared_hbm)
        output_pattern = [[I, BxS], [1, I]]
        nisa.dma_copy(dst=output_hbm.ap(pattern=output_pattern, offset=0), src=output_sb)
    elif output_layout == QKVOutputLayout.NBSd:
        assert d_head is not None, "output_layout NBSd requires d_head"
        nh = I // d_head
        output_hbm = nl.ndarray((nh, BxS, d_head), dtype=hidden_sb.dtype, buffer=nl.shared_hbm)
        # output_sb_pattern = [[d_head, nh], [I, BxS], [1, d_head]]
        # nisa.dma_copy(output_hbm[...], output_sb.ap(pattern=output_sb_pattern, offset=0))
        for i_n in range(nh):
            output_pattern = [[d_head, BxS], [1, d_head]]
            output_offset = i_n * BxS * d_head
            output_sb_pattern = [[I, BxS], [1, d_head]]
            output_sb_offset = i_n * d_head
            nisa.dma_copy(
                dst=output_hbm.ap(pattern=output_pattern, offset=output_offset),
                src=output_sb.ap(pattern=output_sb_pattern, offset=output_sb_offset),
            )

    sbm.close_scope()

    return output_hbm


def filter_qkv_tkg_combinations(
    B=None,
    S=None,
    H=None,
    n_q_heads=None,
    n_kv_heads=None,
    d_head=None,
    norm_type=None,
    quantization_type=None,
    fused_add=None,
    output_layout=None,
) -> FilterResult:
    """Filter invalid QKV TKG parameter combinations.

    Constraints:
    - H sharding: H // 128 must be divisible by lnc_degree=2
    - B*S norm sharding: when fused_add=True and norm_type != NO_NORM,
      B*S must be divisible by 2 if B*S > sharding threshold
    """
    # H sharding constraint: H // H0 must be divisible by lnc_degree=2
    if H is not None and (H // 128) % 2 != 0:
        return FilterResult.INVALID

    # B*S norm sharding constraint
    if fused_add is not None and norm_type is not None and B is not None and S is not None:
        if fused_add is True and norm_type is not NormType.NO_NORM:
            BxS = B * S
            if norm_type is NormType.RMS_NORM:
                threshold = rmsnorm_sharding_threshold
            else:
                threshold = layernorm_sharding_threshold
            if BxS > threshold and BxS % 2 != 0:
                return FilterResult.INVALID

    return FilterResult.VALID


class QkvTkgDtypeModeConfig(TypedDict):
    """Shapes and fusion settings shared by every dtype_mode canary case."""

    B: int
    H: int
    S: int
    dtype: str
    eps: float
    fused_add: bool
    lnc_degree: int
    norm_type: NormType
    output_layout: QKVOutputLayout
    n_q_heads: int
    n_kv_heads: int
    d_head: int


@pytest_test_metadata(name="QKV TKG")
@pytest_marks(["qkv", "tkg"])
@final
class TestQkvTkgKernel:
    def run_qkv_tkg_sb2sb_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        B: int,
        H: int,
        S: int,
        dtype,
        eps,
        fused_qkv_dim: int,
        lnc_degree,
        norm_type: NormType,
        norm_bias: bool = False,
        output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
        quantization_type: QuantizationType = QuantizationType.NONE,
        qkv_bias: bool = False,
        hidden_actual: int | None = None,
        n_kv_heads: int | None = None,
        n_q_heads: int | None = None,
        d_head: int | None = None,
    ):
        fused_add = False

        def input_generator(test_config):
            kernel_input = build_qkv_input(
                batch=B,
                seqlen=S,
                hidden_dim=H,
                fused_qkv_dim=fused_qkv_dim,
                dtype=dtype,
                hidden_actual=hidden_actual,
                d_head=d_head,
                eps=eps,
                norm_type=norm_type,
                quantization_type=quantization_type,
                fused_add=fused_add,
                output_layout=output_layout,
                lnc_degree=lnc_degree,
                norm_bias=norm_bias,
                qkv_bias=qkv_bias,
                num_q_heads=n_q_heads,
                num_kv_heads=n_kv_heads,
            )
            kernel_input["store_output_in_sbuf"] = True
            return kernel_input

        def output_tensors(kernel_input):
            return {"out": np.zeros((B, S, fused_qkv_dim), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=qkv_tkg_sb2sb_wrapper_kernel,
            torch_ref=torch_ref_wrapper(qkv_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2 if quantization_type == QuantizationType.NONE else 4e-2,
            atol=1e-5,
            inference_args=TKG_INFERENCE_ARGS,
        )

    # fmt: off
    qkv_tkg_sb2sb_kernel_accuracy_test_params = "lnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, n_q_heads, n_kv_heads, d_head, norm_type, quantization_type, norm_bias, qkv_bias, eps, output_layout"
    qkv_tkg_sb2sb_kernel_accuracy_test_perms = [
        # H support test & I support test
        [1, 1, 4, 16384, None, 384, 1, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 32768, None, 384, 1, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 1, 4, 4096, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 4096, None, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 4096, 4096, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 4096, 3072, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        # EPS test
        [1, 1, 7, 8192, None, 384, 1, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 1e-5, QKVOutputLayout.BSD],
        # LayerNorm bias test
        [1, 1, 5, 8192, None, 384, 1, 1, 128, NormType.LAYER_NORM, QuantizationType.NONE, True, False, 1e-6, QKVOutputLayout.BSD],
        [1, 1, 5, 8192, None, 384, 1, 1, 128, NormType.LAYER_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        # QKV bias test
        [1, 1, 3, 16384, None, 896, 5, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, True, 1e-6, QKVOutputLayout.BSD],
        # NBSD test : 405B
        [2, 1, 1, 16384, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.NBSd],
        # NBSD test : 70B
        [2, 1, 1, 8192, None, 768, 2, 2, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.NBSd],
        # NBSD test : PS
        [2, 1, 1, 8448, None, 1920, 5, 5, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.NBSd],
        # Large eps test
        [2, 1, 1, 8192, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 77.0, QKVOutputLayout.BSD],
        # static quantization
        [2, 1, 5, 8192, None, 1280, 8, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 16384, None, 3200, 15, 5, 128, NormType.NO_NORM, QuantizationType.STATIC, False, True, 1e-6, QKVOutputLayout.NBSd],
        [1, 1, 1, 8192, None, 1408, 7, 2, 128, NormType.RMS_NORM, QuantizationType.STATIC, False, True, 1e-6, QKVOutputLayout.NBSd],
        # Large I (I > 4096) tests — multi I-block sbuf output
        [1, 1, 4, 8192, None, 4608, 34, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 8192, None, 8192, 56, 4, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 1, 1, 16384, None, 7168, 48, 4, 128, NormType.RMS_NORM, QuantizationType.NONE, False, True, 1e-6, QKVOutputLayout.NBSd],
        [2, 1, 1, 16384, None, 4608, 34, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 1, 4, 8192, None, 5120, 32, 4, 128, NormType.RMS_NORM, QuantizationType.STATIC, False, False, 1e-6, QKVOutputLayout.BSD],
    ]
    # ROW quantization tests
    qkv_tkg_sb2sb_kernel_accuracy_test_perms += [
        [2, 1, 5, 8192, None, 1280, 8, 1, 128, NormType.RMS_NORM, QuantizationType.ROW, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 1, 8192, None, 1408, 7, 2, 128, NormType.NO_NORM, QuantizationType.ROW, False, True, 1e-6, QKVOutputLayout.NBSd],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_tkg_sb2sb_kernel_accuracy_test_params,
        qkv_tkg_sb2sb_kernel_accuracy_test_perms,
    )
    def test_qkv_tkg_sb2sb_accuracy_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        hidden_dim,
        hidden_actual,
        fused_qkv_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        quantization_type,
        norm_bias,
        qkv_bias,
        eps,
        output_layout,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)
        self.run_qkv_tkg_sb2sb_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            dtype=nl.float16,
            eps=eps,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_bias=norm_bias,
            norm_type=norm_type,
            quantization_type=quantization_type,
            output_layout=output_layout,
            qkv_bias=qkv_bias,
            hidden_actual=hidden_actual,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
        )

    def run_qkv_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        B: int,
        H: int,
        S: int,
        dtype,
        eps,
        fused_add: bool,
        fused_qkv_dim: int,
        lnc_degree,
        norm_type: NormType,
        norm_bias: bool = False,
        output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
        quantization_type: QuantizationType = QuantizationType.NONE,
        is_h_dim_4h_transposed: bool = False,
        qkv_bias: bool = False,
        hidden_actual: int | None = None,
        n_kv_heads: int | None = None,
        n_q_heads: int | None = None,
        d_head: int | None = None,
        is_negative_test: bool = False,
        transposed_in: bool = False,
        dtype_mode: DtypeMode = DtypeMode.NON_OCP,
        inference_args=TKG_INFERENCE_ARGS,
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
            transposed_in=transposed_in,
            rtol=2e-2 if quantization_type == QuantizationType.NONE else 4e-2,
            atol=1e-5,
            is_negative_test=is_negative_test,
            inference_args=inference_args,
            expect_fused_hidden_output=True,
            dtype_mode=dtype_mode,
        )

    # fmt: off
    qkv_tkg_kernel_accuracy_test_params = \
        "lnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, n_q_heads, n_kv_heads, d_head, norm_type, quantization_type, fused_add, norm_bias, qkv_bias, eps, output_layout"
    qkv_tkg_kernel_accuracy_test_perms = [
        # H support test & I support test
        [1, 1, 4, 16384, None, 384, 1, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 32768, None, 384, 1, 1, 128, NormType.NO_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 1, 4, 4096, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 4096, None, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        pytest.param(1, 4, 1, 4096, 4096, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        [1, 4, 1, 4096, 3072, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        # H remainder tile only test
        [2, 2, 1, 3840, None, 10240, 64, 8, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        pytest.param(1, 1, 1, 128, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        # EPS test
        [1, 1, 7, 8192, None, 384, 1, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-5, QKVOutputLayout.BSD],
        # LayerNorm bias test
        [1, 1, 5, 8192, None, 384, 1, 1, 128, NormType.LAYER_NORM, QuantizationType.NONE, False, True, False, 1e-6, QKVOutputLayout.BSD],
        pytest.param(1, 1, 5, 8192, None, 384, 1, 1, 128, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        # QKV bias test
        [1, 1, 3, 16384, None, 896, 5, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, True, 1e-6, QKVOutputLayout.BSD],
        pytest.param(2, 8, 5, 3072, None, 640, 8, 1, 64, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        # NBSD test: 405B
        pytest.param(2, 1, 1, 16384, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.NBSd, marks=pytest.mark.fast),
        # NBSD test: 70B
        [2, 1, 1, 8192, None, 768, 2, 2, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.NBSd],
        [2, 1, 1, 8448, None, 4352, 24, 5, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.NBSd],
        [1, 1, 1, 8192, None, 5120, 32, 4, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.NBSd],
        [2, 1, 1, 8192, None, 10240, 64, 8, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # NBSD test: PS
        [2, 1, 1, 8448, None, 1920, 5, 5, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.NBSd],
        # Large eps test
        [2, 1, 1, 8192, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 77.0, QKVOutputLayout.BSD],
        # static quantization
        [2, 1, 5, 8192, None, 1280, 8, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC, False, False, False, 1e-6, QKVOutputLayout.BSD],
        pytest.param(2, 4, 1, 16384, None, 4352, 24, 5, 128, NormType.NO_NORM, QuantizationType.STATIC, False, False, True, 1e-6, QKVOutputLayout.NBSd, marks=pytest.mark.fast),
        pytest.param(1, 1, 1, 8192, None, 5120, 32, 4, 128, NormType.RMS_NORM, QuantizationType.STATIC, False, False, False, 1e-6, QKVOutputLayout.NBSd, marks=pytest.mark.fast),
    ]
    # ROW quantization tests
    qkv_tkg_kernel_accuracy_test_perms += [
        [2, 1, 5, 8192, None, 1280, 8, 1, 128, NormType.RMS_NORM, QuantizationType.ROW, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 16384, None, 4352, 24, 5, 128, NormType.NO_NORM, QuantizationType.ROW, False, False, True, 1e-6, QKVOutputLayout.NBSd],
        [1, 1, 1, 8192, None, 5120, 32, 4, 128, NormType.RMS_NORM, QuantizationType.ROW, False, False, False, 1e-6, QKVOutputLayout.NBSd],
        pytest.param(2, 1, 3, 8192, None, 1280, 8, 1, 128, NormType.RMS_NORM, QuantizationType.ROW, True, False, False, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        pytest.param(2, 2, 1, 16384, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.ROW, False, False, False, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        [2, 1, 1, 8192, None, 10240, 64, 8, 128, NormType.RMS_NORM, QuantizationType.ROW, False, False, False, 1e-6, QKVOutputLayout.BSD],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_tkg_kernel_accuracy_test_params,
        qkv_tkg_kernel_accuracy_test_perms,
    )
    def test_qkv_tkg_accuracy_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        hidden_dim,
        hidden_actual,
        fused_qkv_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        quantization_type,
        fused_add,
        norm_bias,
        qkv_bias,
        eps,
        output_layout,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            dtype=nl.bfloat16,
            eps=eps,
            fused_add=fused_add,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_bias=norm_bias,
            norm_type=norm_type,
            quantization_type=quantization_type,
            output_layout=output_layout,
            qkv_bias=qkv_bias,
            hidden_actual=hidden_actual,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
        )

    # fmt: off
    # transposed_in tests: input in [H0, n_prgs, H1_shard, BxS] layout
    qkv_tkg_transposed_in_test_params = \
        "lnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, n_q_heads, n_kv_heads, d_head, norm_type, quantization_type"
    qkv_tkg_transposed_in_test_perms = [
        # llama3_70b: RMSNorm enabled
        pytest.param(2, 1, 1, 8192, None, 768, 4, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, marks=pytest.mark.fast),
        # llama3_70b: RMSNorm enabled, larger batch
        [2, 16, 1, 8192, None, 768, 4, 1, 128, NormType.RMS_NORM, QuantizationType.NONE],
        # qwen3_32b: NO_NORM (rmsnorm_X=False in model)
        pytest.param(2, 1, 1, 5120, None, 384, 1, 1, 128, NormType.NO_NORM, QuantizationType.NONE, marks=pytest.mark.fast),
        # STATIC quant with RMSNorm
        pytest.param(2, 8, 1, 8192, None, 768, 4, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC, marks=pytest.mark.fast),
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_tkg_transposed_in_test_params,
        qkv_tkg_transposed_in_test_perms,
    )
    def test_qkv_tkg_transposed_in_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        hidden_dim,
        hidden_actual,
        fused_qkv_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        quantization_type,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            dtype=nl.bfloat16,
            eps=1e-6,
            fused_add=False,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_type=norm_type,
            quantization_type=quantization_type,
            output_layout=QKVOutputLayout.BSD,
            hidden_actual=hidden_actual,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
            transposed_in=True,
        )

    # fmt: off
    qkv_tkg_kernel_performance_bsd_test_params = \
        "lnc_degree, batch, seqlen, hidden_dim, fused_qkv_dim, d_head, cyclesQoR, norm_type, quantization_type, fused_add, norm_bias, qkv_bias, eps, output_layout"
    qkv_tkg_kernel_performance_bsd_test_perms = [
        # LayerNorm with batch larger than 1
        [2, 2, 4, 8448, 1408, 128, (None, 88_383_528), NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 2, 4096, 1024, 128, (None, 65_302_814), NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # LNC2 BxS > 64
        [2, 16, 5, 8192, 104, 26, (None, 75_977_297), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 14, 5, 8192, 104, 26, (None, 72_189_720), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 14, 5, 8192, 1280, 128, (None, 72_189_720), NormType.RMS_NORM, QuantizationType.STATIC, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # LNC2 with I not a multiple of 128
        [2, 3, 1, 2048, 104, 26, (None, None), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 1, 2048, 104, 26, (None, None), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 5, 1, 4096, 512, 128, (None, None), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 5, 4096, 512, 128, (None, None), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # hidden=16384 (LLaMA 405B cases)
        [2, 3, 1, 16384, 384, 128, (None, 74_415_967), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 5, 1, 16384, 384, 128, (None, 76_411_380), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 6, 16384, 896, 128, (None, 105_265_002), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 6, 16384, 384, 128, (None, 77_188_129), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 3, 16384, 384, 128, (None, 74_358_717), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 5, 16384, 384, 128, (None, 76_409_880), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 16384, 512, 128, (None, 74_809_216), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 7, 16384, 512, 128, (None, 123_832_889), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 7, 16384, 512, 128, (None, 80_773_207), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # hidden=8192 (LLaMA 70B cases)
        [2, 3, 1, 8192, 512, 128, (None, 62_466_402), NormType.NO_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 1, 8192, 896, 128, (None, 70_471_973), NormType.NO_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 5, 8192, 384, 128, (None, 70_544_972), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 5, 8192, 384, 128, (None, 61_577_737), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 3, 8192, 512, 128, (None, 62_620_152), NormType.NO_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 5, 8192, 5120, 1024, (None, 265_156_919), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 5, 8192, 10240, 1024, (None, 447_899_383), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # Llama3 470B and 2T
        [2, 1, 1, 32768, 896, 128, (None, 137_652_784), NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 1, 20480, 896, 128, (None, 104_202_753), NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 3, 32768, 896, 128, (None, 141_587_695), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 3, 20480, 896, 128, (None, 107_279_332), NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 5, 32768, 896, 128, (None, 151_882_512), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 5, 20480, 896, 128, (None, 117_113_150), NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        # GPT-OSS
        [2, 4, 4, 3072, 640, 64, (None, 59_411_907), NormType.RMS_NORM, QuantizationType.NONE, False, False, True, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 3072, 640, 64, (None, 55_897_912), NormType.RMS_NORM, QuantizationType.NONE, False, False, True, 1e-6, QKVOutputLayout.BSD],
        # PS
        [2, 1, 1, 8448, 1920, 128, (None, 105_167_419), NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 1, 8448, 1920, 128, (None, 104_287_503), NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_tkg_kernel_performance_bsd_test_params,
        qkv_tkg_kernel_performance_bsd_test_perms,
    )
    def test_qkv_tkg_performance_bsd_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        hidden_dim,
        fused_qkv_dim,
        d_head,
        cyclesQoR,
        norm_type,
        quantization_type,
        fused_add,
        norm_bias,
        qkv_bias,
        eps,
        output_layout,
    ):
        # Quantized QKV kernel need to specify number of heads
        n_kv_heads = 1
        n_q_heads = (fused_qkv_dim // d_head) - 2

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            dtype=nl.bfloat16,
            eps=eps,
            fused_add=fused_add,
            fused_qkv_dim=fused_qkv_dim,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
            lnc_degree=compiler_args.logical_nc_config,
            norm_bias=norm_bias,
            norm_type=norm_type,
            quantization_type=quantization_type,
            output_layout=output_layout,
            qkv_bias=qkv_bias,
        )

    # fmt: off
    qkv_tkg_kernel_mxfp_test_params = \
        "lnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, n_q_heads, n_kv_heads, d_head, norm_type, quantization_type, is_h_dim_4h_transposed, fused_add, norm_bias, qkv_bias, eps, output_layout"
    qkv_tkg_kernel_mxfp_test_perms = [
        # is_h_dim_4h_transposed = True required for now.
        # NO_NORM,
        pytest.param(2, 12, 1, 512, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        [2, 12, 1, 1024, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 12, 1, 2048, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048*2, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048*8, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        pytest.param(2, 4, 1, 1024, None, (2 +2*5)*128, 2, 5, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        [2, 4, 1, 1024, None, (6 +2*3)*128, 6, 3, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 1024, None, (2 +2*8)*128, 2, 8, 128, NormType.NO_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        # RMS_NORM
        [2, 12, 1, 512, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 12, 1, 1024, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 12, 1, 2048, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048*2, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048*8, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 1024, None, (2 +2*5)*128, 2, 5, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 1024, None, (6 +2*3)*128, 6, 3, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 1024, None, (2 +2*8)*128, 2, 8, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        [2, 128, 1, 6144, None, 2176, 32, 1, 64, NormType.RMS_NORM, QuantizationType.MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        # BxS not divisible by 4
        [2, 1, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 2, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 3, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 5, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 3, 6, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
    ]
    # fmt: on

    # fmt: off
    qkv_tkg_kernel_static_mxfp_test_params = \
        "lnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, n_q_heads, n_kv_heads, d_head, norm_type, quantization_type, is_h_dim_4h_transposed, fused_add, norm_bias, qkv_bias, eps, output_layout"
    qkv_tkg_kernel_static_mxfp_test_perms = [
        # NO_NORM
        [2, 12, 1, 512, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 12, 1, 4096, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        pytest.param(2, 4, 1, 4096, None, (2 +2*5)*128, 2, 5, 128, NormType.NO_NORM, QuantizationType.STATIC_MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        # RMS_NORM
        [2, 12, 1, 512, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 12, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 4096, None, (2 +2*5)*128, 2, 5, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        # BxS not divisible by 4
        [2, 1, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 2, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 3, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 5, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 3, 6, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.STATIC_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
    ]
    # fmt: on

    @pytest_parametrize(
        qkv_tkg_kernel_mxfp_test_params,
        qkv_tkg_kernel_mxfp_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_tkg_mxfp_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        hidden_dim,
        hidden_actual,
        fused_qkv_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        quantization_type,
        is_h_dim_4h_transposed,
        fused_add,
        norm_bias,
        qkv_bias,
        eps,
        output_layout,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            dtype=nl.bfloat16,
            eps=eps,
            fused_add=fused_add,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_bias=norm_bias,
            norm_type=norm_type,
            quantization_type=quantization_type,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            output_layout=output_layout,
            qkv_bias=qkv_bias,
            hidden_actual=hidden_actual,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
        )

    @pytest_parametrize(
        qkv_tkg_kernel_static_mxfp_test_params,
        qkv_tkg_kernel_static_mxfp_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_tkg_static_mxfp_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        hidden_dim,
        hidden_actual,
        fused_qkv_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        quantization_type,
        is_h_dim_4h_transposed,
        fused_add,
        norm_bias,
        qkv_bias,
        eps,
        output_layout,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            dtype=nl.bfloat16,
            eps=eps,
            fused_add=fused_add,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_bias=norm_bias,
            norm_type=norm_type,
            quantization_type=quantization_type,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            output_layout=output_layout,
            qkv_bias=qkv_bias,
            hidden_actual=hidden_actual,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
        )

    # fmt: off
    qkv_tkg_kernel_row_mxfp_test_params = \
        "lnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, n_q_heads, n_kv_heads, d_head, norm_type, quantization_type, is_h_dim_4h_transposed, fused_add, norm_bias, qkv_bias, eps, output_layout"
    qkv_tkg_kernel_row_mxfp_test_perms = [
        # NO_NORM
        [2, 12, 1, 512, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 12, 1, 4096, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 4096, None, (2 +2*5)*128, 2, 5, 128, NormType.NO_NORM, QuantizationType.ROW_MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        # RMS_NORM
        [2, 12, 1, 512, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 12, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 1, 2048, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 4, 1, 4096, None, (2 +2*5)*128, 2, 5, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, True, 1e-6, QKVOutputLayout.BSD],
        # BxS not divisible by 4
        [2, 1, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 2, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 3, 1, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 1, 5, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [2, 3, 6, 4096, None, 512, 2, 1, 128, NormType.RMS_NORM, QuantizationType.ROW_MX, True, False, False, False, 1e-6, QKVOutputLayout.BSD],
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest_parametrize(
        qkv_tkg_kernel_row_mxfp_test_params,
        qkv_tkg_kernel_row_mxfp_test_perms,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_qkv_tkg_row_mxfp_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        hidden_dim,
        hidden_actual,
        fused_qkv_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        quantization_type,
        is_h_dim_4h_transposed,
        fused_add,
        norm_bias,
        qkv_bias,
        eps,
        output_layout,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=batch,
            H=hidden_dim,
            S=seqlen,
            dtype=nl.bfloat16,
            eps=eps,
            fused_add=fused_add,
            fused_qkv_dim=fused_qkv_dim,
            lnc_degree=compiler_args.logical_nc_config,
            norm_bias=norm_bias,
            norm_type=norm_type,
            quantization_type=quantization_type,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            output_layout=output_layout,
            qkv_bias=qkv_bias,
            hidden_actual=hidden_actual,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
        )

    @pytest.mark.coverage_parametrize(
        B=[1, 4, 8, 12],
        S=[1, 3, 5, 8],
        H=BoundedRange(
            [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            boundary_values=[129, 255, 384],
        ),
        n_q_heads=[1, 4, 8, 16, 32, 64],
        n_kv_heads=[1, 2, 4, 8],
        d_head=[1, 2, 4, 8, 16, 32, 64, 128],
        norm_type=[NormType.NO_NORM, NormType.RMS_NORM, NormType.LAYER_NORM],
        quantization_type=[QuantizationType.NONE, QuantizationType.STATIC],
        fused_add=[False, True],
        output_layout=[QKVOutputLayout.BSD, QKVOutputLayout.NBSd],
        filter=filter_qkv_tkg_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=True,
    )
    def test_qkv_tkg_sweep(
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
        quantization_type,
        fused_add,
        output_layout,
        is_negative_test_case,
    ):
        # Pre-existing kernel bug: fused_hidden output is incorrect when
        # fused_add=True + NO_NORM + STATIC quantization.
        # Tracked in NKILIB-561
        if fused_add and norm_type is NormType.NO_NORM and quantization_type is QuantizationType.STATIC:
            pytest.xfail("Known kernel bug: fused_hidden accuracy with fused_add+NO_NORM+STATIC quantization")

        # Pre-existing hardware non-determinism on large hidden_dim=32768 with LAYER_NORM + STATIC quantization.
        # Tracked in NKILIB-849
        if H == 32768 and norm_type is NormType.LAYER_NORM and quantization_type is QuantizationType.STATIC:
            pytest.xfail("Hardware non-determinism on large hidden_dim=32768 with LAYER_NORM+STATIC (NKILIB-849)")

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_qkv_tkg_test(
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
            quantization_type=quantization_type,
            output_layout=output_layout,
            n_kv_heads=n_kv_heads,
            n_q_heads=n_q_heads,
            d_head=d_head,
            is_negative_test=is_negative_test_case,
        )

    ####################################################################################################################
    # FP8 quant mode canary for QKV TKG.
    # Tests NON_OCP, OCP, AUTO across STATIC and ROW.
    ####################################################################################################################
    _DTYPE_MODE_D_HEAD = 128
    _DTYPE_MODE_N_Q_HEADS = 8
    _DTYPE_MODE_N_KV_HEADS = 1
    _QKV_TKG_BY_DTYPE_MODE_CONFIG: QkvTkgDtypeModeConfig = {
        "B": 1,
        "H": 8192,
        "S": 1,
        "dtype": nl.bfloat16,
        "eps": 1e-6,
        "fused_add": False,
        "lnc_degree": 2,
        "norm_type": NormType.RMS_NORM,
        "output_layout": QKVOutputLayout.BSD,
        "n_q_heads": _DTYPE_MODE_N_Q_HEADS,
        "n_kv_heads": _DTYPE_MODE_N_KV_HEADS,
        "d_head": _DTYPE_MODE_D_HEAD,
    }

    @pytest.mark.parametrize("dtype_mode", [DtypeMode.NON_OCP, DtypeMode.OCP, DtypeMode.AUTO])
    @pytest.mark.parametrize(
        "quantization_type",
        [QuantizationType.STATIC, QuantizationType.ROW],
    )
    def test_qkv_tkg_by_dtype_mode(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        quantization_type: QuantizationType,
        dtype_mode: DtypeMode,
    ):
        """Smoke-test QKV TKG STATIC/ROW quant with explicit dtype_mode.

        NON_OCP → ``nl.float8_e4m3`` (240), any platform.
        OCP     → ``nl.float8_e4m3fn`` (448), TRN3 only.
        AUTO    → ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3`` elsewhere.
        """
        if dtype_mode == DtypeMode.OCP and not platform_target.is_trn3():
            pytest.skip("OCP dtype_mode requires TRN3")
        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
        cfg = self._QKV_TKG_BY_DTYPE_MODE_CONFIG
        fused_qkv_dim = (self._DTYPE_MODE_N_Q_HEADS + 2 * self._DTYPE_MODE_N_KV_HEADS) * self._DTYPE_MODE_D_HEAD
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            fused_qkv_dim=fused_qkv_dim,
            quantization_type=quantization_type,
            dtype_mode=dtype_mode,
            inference_args=None,
            **cfg,
        )
