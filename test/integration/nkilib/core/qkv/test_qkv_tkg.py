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
from test.integration.nkilib.core.qkv.test_qkv_common import (
    BATCH_DIM_NAME,
    D_HEAD_DIM_NAME,
    DUMMY_TENSOR_NAME,
    FUSED_ADD_DIM_NAME,
    HIDDEN_DIM_NAME,
    N_KV_HEADS_DIM_NAME,
    N_Q_HEADS_DIM_NAME,
    NORM_TYPE_DIM_NAME,
    OUTPUT_LAYOUT_DIM_NAME,
    QUANTIZATION_TYPE_DIM_NAME,
    SEQUENCE_LEN_DIM_NAME,
    build_qkv_input,
    norm_qkv_ref,
)
from test.integration.nkilib.utils.tensor_generators import guess_tensor_dtype
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeMonotonicGeneratorStrategy,
    RangeRandomGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import Optional, final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.qkv.qkv import qkv
from nkilib_src.nkilib.core.qkv.qkv_tkg import qkv_tkg
from nkilib_src.nkilib.core.subkernels.layernorm_tkg import (
    SHARDING_THRESHOLD as layernorm_sharding_threshold,
)
from nkilib_src.nkilib.core.subkernels.rmsnorm_tkg import (
    SHARDING_THRESHOLD as rmsnorm_sharding_threshold,
)
from nkilib_src.nkilib.core.utils.allocator import SbufManager, create_auto_alloc_manager
from nkilib_src.nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from nkilib_src.nkilib.core.utils.logging import Logger
from typing_extensions import override


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


def golden_func_fused_add_qkv(
    inp_np,
    dtype,
    fused_add,
    norm_type,
    eps,
    head_dim=None,
    n_kv_heads=None,
    n_q_heads=None,
    quantization_type=QuantizationType.NONE,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    hidden_actual=None,
):
    guessed_type = guess_tensor_dtype(dtype)

    hidden = inp_np["input"].astype(guessed_type)
    added = hidden
    if fused_add:
        prev = inp_np["mlp_prev"].astype(guessed_type)
        attn = inp_np["attention_prev"].astype(guessed_type)
        added = hidden + prev + attn
    w = inp_np["fused_qkv_weights"].astype(guessed_type)
    ln_w = inp_np["gamma_norm_weights"].astype(guessed_type) if inp_np["gamma_norm_weights"] is not None else None
    bias_t = inp_np["bias"].astype(guessed_type) if inp_np["bias"] is not None else None
    norm_b = inp_np["layer_norm_bias"].astype(guessed_type) if inp_np["layer_norm_bias"] is not None else None
    qkv_w_scale = inp_np["qkv_w_scale"][0, :].astype(np.float32) if inp_np["qkv_w_scale"] is not None else None
    qkv_in_scale = inp_np["qkv_in_scale"][0, 0].astype(np.float32) if inp_np["qkv_in_scale"] is not None else None
    qkv_out = norm_qkv_ref(
        hidden=added,
        gamma=ln_w,
        qkv_weights=w,
        dtype=dtype,
        norm_type=norm_type,
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        eps=eps,
        bias=bias_t,
        norm_b=norm_b,
        hidden_actual=hidden_actual,
        head_dim=head_dim,
        n_kv_heads=n_kv_heads,
        n_q_heads=n_q_heads,
    )

    if output_layout == QKVOutputLayout.NBSd:
        B, S, I = qkv_out.shape
        qkv_out = qkv_out.reshape((B, S, I // head_dim, head_dim))
        qkv_out = qkv_out.transpose((2, 0, 1, 3))
    elif output_layout == QKVOutputLayout.NBdS:
        B, S, I = qkv_out.shape
        qkv_out = qkv_out.reshape((B, S, I // head_dim, head_dim))
        qkv_out = qkv_out.transpose((2, 0, 3, 1))

    if fused_add:
        return {"out": qkv_out.astype(dtype), "fused_hidden": added.astype(dtype)}
    else:
        return {"out": qkv_out.astype(dtype)}


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
    fused_residual_add: Optional[bool] = False,
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
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # -----------------------------------------
    store_output_in_sbuf: bool = False,
    # -----------------------------------------
    # User can optionally PASS Sbuf manager
    # -----------------------------------------
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    # ----------------------------------------
    load_input_with_DMA_transpose: bool = True,
) -> nl.ndarray:
    # construct qkv_tkg inputs
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

    sbm = create_auto_alloc_manager(logger=Logger("qkv-tkg-sb2sb-wrapper"))
    assert hidden.buffer == nl.hbm

    sbm.open_scope()
    hidden_hbm = hidden.reshape((BxS, H0, H1))
    hidden_sb = sbm.alloc_stack(
        shape=(H0, BxS, H1),
        dtype=hidden.dtype,
        buffer=nl.sbuf,
        name="hidden_sb",
    )
    hidden_hbm = hidden_hbm.reshape((BxS, H0, H1))
    hidden_hbm_pattern = [[H1, H0], [H, BxS], [1, H1]]
    hidden_hbm_offset = 0
    nisa.dma_copy(
        hidden_sb,
        hidden_hbm.ap(pattern=hidden_hbm_pattern, offset=hidden_hbm_offset),
    )

    hidden_sb = hidden

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

    assert output_sb.shape == (BxS, I)

    # Allocate output tensor with layout-specific shape
    if output_layout == QKVOutputLayout.BSD:
        output_hbm = nl.ndarray((BxS, I), dtype=hidden_sb.dtype, buffer=nl.shared_hbm)
        output_pattern = [[I, BxS], [1, I]]
        nisa.dma_copy(dst=output_hbm.ap(pattern=output_pattern, offset=0), src=output_sb)
    elif output_layout == QKVOutputLayout.NBSd:
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


@pytest_test_metadata(
    name="QKV TKG",
    pytest_marks=["qkv", "tkg"],
)
@final
class TestQkvTkgKernel:
    def run_range_qkv_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree,
        dtype,
        collector: IMetricsCollector,
    ):
        dummy_tensor = test_options.tensors[DUMMY_TENSOR_NAME]

        B = dummy_tensor[BATCH_DIM_NAME]
        S = dummy_tensor[SEQUENCE_LEN_DIM_NAME]
        H = dummy_tensor[HIDDEN_DIM_NAME]
        n_q_heads = dummy_tensor[N_Q_HEADS_DIM_NAME]
        n_kv_heads = dummy_tensor[N_KV_HEADS_DIM_NAME]
        d_head = dummy_tensor[D_HEAD_DIM_NAME]
        norm_type = NormType(dummy_tensor[NORM_TYPE_DIM_NAME])
        quantization_type = QuantizationType(value=dummy_tensor[QUANTIZATION_TYPE_DIM_NAME])
        fused_add = bool(dummy_tensor[FUSED_ADD_DIM_NAME])
        output_layout = QKVOutputLayout(dummy_tensor[OUTPUT_LAYOUT_DIM_NAME])

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head

        is_negative_test_case = test_options.is_negative_test_case
        H0 = 128
        # Tkg kernel requires (H // H0) to be divisible by num_shards.
        if (H // H0) % lnc_degree != 0:
            is_negative_test_case = True

        BxS = B * S
        sharding_threshold = (
            rmsnorm_sharding_threshold if norm_type == NormType.RMS_NORM else layernorm_sharding_threshold
        )
        # Norm kernels expect B*S to be shardable when B*S is over the sharding threshold
        if fused_add and norm_type != NormType.NO_NORM and BxS > sharding_threshold and BxS % 2 != 0:
            is_negative_test_case = True

        test_size_classification = QkvTkgClassification.classify(B=B, S=S, H=H, fused_qkv_dim=fused_qkv_dim)

        with assert_negative_test_case(is_negative_test_case):
            self.run_qkv_tkg_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                B=B,
                H=H,
                S=S,
                dtype=dtype,
                eps=1e-6,
                fused_add=fused_add,
                fused_qkv_dim=fused_qkv_dim,
                lnc_degree=lnc_degree,
                norm_type=norm_type,
                quantization_type=quantization_type,
                output_layout=output_layout,
                n_kv_heads=n_kv_heads,
                n_q_heads=n_q_heads,
                d_head=d_head,
            )

    def run_qkv_tkg_sb2sb_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
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

        # Create lazy golden generator with closure to capture all parameters
        def create_lazy_golden():
            return golden_func_fused_add_qkv(
                inp_np=kernel_input,
                dtype=dtype,
                fused_add=fused_add,
                norm_type=norm_type,
                eps=eps,
                head_dim=d_head,
                output_layout=output_layout,
                hidden_actual=hidden_actual,
                n_kv_heads=n_kv_heads,
                n_q_heads=n_q_heads,
                quantization_type=quantization_type,
            )

        # Create output tensor placeholders for shape/dtype
        # Shape: (B, S, fused_qkv_dim) based on build_qkv_input
        output_placeholder = {"out": np.zeros((B, S, fused_qkv_dim), dtype=dtype)}
        if fused_add:
            # fused_hidden has same shape as input: (B, S, H)
            output_placeholder["fused_hidden"] = np.zeros((B, S, H), dtype=dtype)

        test_manager.execute(
            KernelArgs(
                kernel_func=qkv_tkg_sb2sb_wrapper_kernel,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_placeholder,
                    ),
                    relative_accuracy=2e-2 if quantization_type == QuantizationType.NONE else 4e-2,
                    absolute_accuracy=1e-5,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
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
    ]
    # fmt: on

    @pytest.mark.parametrize(
        qkv_tkg_sb2sb_kernel_accuracy_test_params,
        qkv_tkg_sb2sb_kernel_accuracy_test_perms,
    )
    def test_qkv_tkg_sb2sb_accuracy_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_qkv_tkg_sb2sb_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
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
        collector: IMetricsCollector,
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
        qkv_bias: bool = False,
        hidden_actual: int | None = None,
        n_kv_heads: int | None = None,
        n_q_heads: int | None = None,
        d_head: int | None = None,
    ):
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

        # Create lazy golden generator with closure to capture all parameters
        def create_lazy_golden():
            return golden_func_fused_add_qkv(
                inp_np=kernel_input,
                dtype=dtype,
                fused_add=fused_add,
                norm_type=norm_type,
                eps=eps,
                head_dim=d_head,
                output_layout=output_layout,
                hidden_actual=hidden_actual,
                n_kv_heads=n_kv_heads,
                n_q_heads=n_q_heads,
                quantization_type=quantization_type,
            )

        # Create output tensor placeholders for shape/dtype
        # Shape: (B, S, fused_qkv_dim) based on build_qkv_input
        output_placeholder = {"out": np.zeros((B, S, fused_qkv_dim), dtype=dtype)}
        if fused_add:
            # fused_hidden has same shape as input: (B, S, H)
            output_placeholder["fused_hidden"] = np.zeros((B, S, H), dtype=dtype)

        test_manager.execute(
            KernelArgs(
                kernel_func=qkv,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_placeholder,
                    ),
                    relative_accuracy=2e-2 if quantization_type == QuantizationType.NONE else 4e-2,
                    absolute_accuracy=1e-5,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
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
        [1, 4, 1, 4096, 4096, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 4, 1, 4096, 3072, 896, 5, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, True, False, False, 1e-6, QKVOutputLayout.BSD],
        # H remainder tile only test
        [2, 2, 1, 3840, None, 10240, 64, 8, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        [1, 1, 1, 128, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # EPS test
        [1, 1, 7, 8192, None, 384, 1, 1, 128, NormType.RMS_NORM, QuantizationType.NONE, False, False, False, 1e-5, QKVOutputLayout.BSD],
        # LayerNorm bias test
        [1, 1, 5, 8192, None, 384, 1, 1, 128, NormType.LAYER_NORM, QuantizationType.NONE, False, True, False, 1e-6, QKVOutputLayout.BSD],
        [1, 1, 5, 8192, None, 384, 1, 1, 128, NormType.LAYER_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.BSD],
        # QKV bias test
        [1, 1, 3, 16384, None, 896, 5, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, True, 1e-6, QKVOutputLayout.BSD],
        [2, 8, 5, 3072, None, 640, 8, 1, 64, NormType.RMS_NORM, QuantizationType.NONE, False, False, True, 1e-6, QKVOutputLayout.BSD],
        # NBSD test: 405B
        [2, 1, 1, 16384, None, 512, 2, 1, 128, NormType.NO_NORM, QuantizationType.NONE, False, False, False, 1e-6, QKVOutputLayout.NBSd],
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
        [2, 4, 1, 16384, None, 4352, 24, 5, 128, NormType.NO_NORM, QuantizationType.STATIC, False, False, True, 1e-6, QKVOutputLayout.NBSd],
        [1, 1, 1, 8192, None, 5120, 32, 4, 128, NormType.RMS_NORM, QuantizationType.STATIC, False, False, False, 1e-6, QKVOutputLayout.NBSd],      
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(
        qkv_tkg_kernel_accuracy_test_params,
        qkv_tkg_kernel_accuracy_test_perms,
    )
    def test_qkv_tkg_accuracy_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
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

    @pytest.mark.parametrize(
        qkv_tkg_kernel_performance_bsd_test_params,
        qkv_tkg_kernel_performance_bsd_test_perms,
    )
    def test_qkv_tkg_performance_bsd_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_qkv_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
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

    @staticmethod
    def qkv_tkg_sweep_config() -> RangeTestConfig:
        B = 12
        S = 8
        H = 32768
        n_q_heads = 64
        n_kv_heads = 8
        d_head = 128

        tc = TensorRangeConfig(
            tensor_configs={
                DUMMY_TENSOR_NAME: TensorConfig(
                    [
                        DimensionRangeConfig(max=B, name=BATCH_DIM_NAME),
                        DimensionRangeConfig(max=S, name=SEQUENCE_LEN_DIM_NAME),
                        DimensionRangeConfig(min=128, max=H, multiple_of=128, name=HIDDEN_DIM_NAME),
                        DimensionRangeConfig(min=1, max=n_q_heads, name=N_Q_HEADS_DIM_NAME),
                        DimensionRangeConfig(min=1, max=n_kv_heads, name=N_KV_HEADS_DIM_NAME),
                        DimensionRangeConfig(min=1, max=d_head, power_of=2, name=D_HEAD_DIM_NAME),
                        DimensionRangeConfig(
                            min=0, max=2, name=NORM_TYPE_DIM_NAME
                        ),  # NO_NORM=0, RMS_NORM=1, LAYER_NORM=2
                        DimensionRangeConfig(min=0, max=1, name=QUANTIZATION_TYPE_DIM_NAME),  # NO_NORM=0, STATIC=1
                        DimensionRangeConfig(min=0, max=1, name=FUSED_ADD_DIM_NAME),
                        DimensionRangeConfig(min=0, max=1, name=OUTPUT_LAYOUT_DIM_NAME),  # BSD=0, NBSd=1
                    ],
                ),
            },
            monotonic_step_percent=10,
        )

        tc.custom_generators = [
            RangeRandomGeneratorStrategy(
                tc.random_sample_size,
            ),
            RangeMonotonicGeneratorStrategy(
                tc.monotonic_step_size,
                tc.monotonic_step_percent,
            ),
        ]

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=tc,
        )

    @range_test_config(qkv_tkg_sweep_config())
    def test_qkv_tkg_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
    ):
        compiler_args = CompilerArgs()
        self.run_range_qkv_tkg_test(
            dtype=nl.bfloat16,
            test_manager=test_manager,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            collector=collector,
        )
