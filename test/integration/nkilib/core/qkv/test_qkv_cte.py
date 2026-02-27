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
    SEQUENCE_LEN_DIM_NAME,
    build_qkv_input,
    norm_qkv_ref,
    norm_qkv_ref_mx,
    rope_gaussian_tensor_generator,
)
from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
    guess_tensor_dtype,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metrics_collector import IMetricsCollector, MetricName, MetricsCollector
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
from typing import Any, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.qkv.qkv import qkv
from nkilib_src.nkilib.core.utils.common_types import (
    NormType,
    QKVOutputLayout,
    QuantizationType,
)
from nkilib_src.nkilib.core.utils.kernel_helpers import get_max_positive_value_for_dtype
from typing_extensions import override

# Utility maps to serialize type so that generated test name don't contain Python objects
dtype_int_to_type = {0: nl.bfloat16, 1: np.float16, 2: np.float32}
dtype_type_to_int = {nl.bfloat16: 0, np.float16: 1, np.float32: 2}


class QkvCteClassification(enum.Enum):
    """
    Classification of QKV CTE test cases by computational complexity.

    Used to categorize tests based on estimated FLOPs for performance tracking
    and test organization.

    Attributes:
        SMALL: Low computational complexity (FLOPs <= 130B)
        MEDIUM: Medium computational complexity (130B < FLOPs <= 626B)
        LARGE: High computational complexity (FLOPs > 626B)
    """

    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @staticmethod
    def classify(B: int, S: int, H: int, fused_qkv_dim: int):
        """
        Classify test case based on estimated FLOPs.

        Args:
            B (int): Batch size
            S (int): Sequence length
            H (int): Hidden dimension
            fused_qkv_dim (int): Fused QKV dimension

        Returns:
            QkvCteClassification: Classification category
        """
        flops_estimate = B * S * H * fused_qkv_dim

        # TODO: proper classification
        if flops_estimate <= 130000000000:
            return QkvCteClassification.SMALL
        elif flops_estimate <= 626000000000:
            return QkvCteClassification.MEDIUM
        else:
            return QkvCteClassification.LARGE

    @override
    def __str__(self):
        return self.name


def numpy_rope_golden(x_in, cos, sin, first_second_half_impl):
    """
    NumPy reference implementation of Rotary Position Embedding (RoPE).

    Applies RoPE transformation to input embeddings using precomputed cosine and sine values.
    Supports two layout modes: interleaved (first/second half) or contiguous.

    Args:
        x_in (np.ndarray): [d_head, S], Input embeddings
        cos (np.ndarray): [d_head//2, S], Cosine frequencies
        sin (np.ndarray): [d_head//2, S], Sine frequencies
        first_second_half_impl (bool): If True, interleave first/second halves into even/odd positions

    Returns:
        np.ndarray: [d_head, S], Rotated embeddings with RoPE applied
    """
    d_head, S = x_in.shape

    x = x_in.transpose(1, 0)  # [d_head, S] -> [S, d_head]

    if first_second_half_impl:  # Interleave first/second halves into even/odd positions.
        new_x = np.empty_like(x)
        new_x[:, ::2] = x[:, : d_head // 2]  # Fill even positions w/ first half.
        new_x[:, 1::2] = x[:, d_head // 2 :]  # Fill odd positions w/ second half.
        x = new_x

    freqs_cos = cos.transpose(1, 0)
    freqs_sin = sin.transpose(1, 0)

    xri = x.reshape(x.shape[:-1] + (-1, 2))  # -> [S, d_head//2, 2]
    # Split into two [S, d_head//2]
    x_r, x_i = np.split(xri, 2, axis=-1)
    x_r = x_r.squeeze(-1)
    x_i = x_i.squeeze(-1)

    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Stack and flatten last two dimensions.
    x_out = np.stack([x_out_r, x_out_i], axis=-1)  # [S, d_head//2, 2]
    x_out = x_out.reshape(x_out.shape[:-2] + (-1,))  # [S, d_head]

    if first_second_half_impl:  # Put even/odd indices into first/second halves.
        x_out = np.concatenate((x_out[:, 0::2], x_out[:, 1::2]), axis=1)

    return x_out.transpose(1, 0)


def golden_func_fused_add_qkv(
    inp_np: dict[str, Any],
    dtype,
    fused_add,
    norm_type,
    eps,
    head_dim=None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    hidden_actual=None,
    fused_rope=False,
    num_q_heads=None,
    num_kv_heads=None,
    fp8_kv_cache=False,
    max_seq_len=None,
    k_scale_val=None,
    v_scale_val=None,
    fp8_max=None,
    fp8_min=None,
    use_block_kv=False,
    num_blocks=None,
    block_size=None,
    slot_mapping=None,
    quantization_type: QuantizationType = QuantizationType.NONE,
):
    """
    Golden reference function for QKV projection with optional fused operations.

    Computes expected output for QKV kernel including optional residual addition,
    normalization, bias, and RoPE transformations.

    Args:
        inp_np (dict[str, Any]): Dictionary of input tensors
        dtype: Data type for computation
        fused_add (bool): Whether to perform residual addition
        norm_type (NormType): Type of normalization to apply
        eps (float): Epsilon for normalization stability
        head_dim (Optional[int]): Dimension per attention head
        output_layout (QKVOutputLayout): Output tensor layout
        hidden_actual (Optional[int]): Actual hidden dimension if padded
        fused_rope (bool): Whether to apply RoPE
        num_q_heads (Optional[int]): Number of query heads
        num_kv_heads (Optional[int]): Number of key/value heads

    Returns:
        dict: Dictionary with 'out' key containing QKV output tensor
    """
    guessed_type = guess_tensor_dtype(dtype)

    hidden = inp_np["input"].astype(guessed_type)
    added = hidden
    if fused_add:
        prev = inp_np["mlp_prev"].astype(guessed_type)
        attn = inp_np["attention_prev"].astype(guessed_type)
        added = hidden + prev + attn

    qkv_w_scale = None
    qkv_in_scale = None
    if quantization_type == QuantizationType.STATIC:
        w = inp_np["fused_qkv_weights"]
        qkv_w_scale = inp_np.get("qkv_w_scale")
        qkv_in_scale = inp_np.get("qkv_in_scale")
        if qkv_w_scale is not None:
            qkv_w_scale = qkv_w_scale[0, :].astype(np.float32)
        if qkv_in_scale is not None:
            qkv_in_scale = qkv_in_scale[0, 0].astype(np.float32)

    elif quantization_type == QuantizationType.MX:
        w = inp_np["fused_qkv_weights"]
        qkv_w_scale = inp_np["qkv_w_scale"]
    else:
        w = inp_np["fused_qkv_weights"].astype(guessed_type)
    ln_w = inp_np["gamma_norm_weights"].astype(guessed_type) if inp_np["gamma_norm_weights"] is not None else None
    bias_t = inp_np["bias"].astype(guessed_type) if inp_np["bias"] is not None else None
    norm_b = inp_np["layer_norm_bias"].astype(guessed_type) if inp_np["layer_norm_bias"] is not None else None

    cos_cache = inp_np["cos_cache"].astype(guessed_type) if inp_np["cos_cache"] is not None else None
    sin_cache = inp_np["sin_cache"].astype(guessed_type) if inp_np["sin_cache"] is not None else None
    is_input_swizzled = inp_np["is_input_swizzled"]

    if quantization_type == QuantizationType.MX:
        qkv_out = norm_qkv_ref_mx(
            hidden=added,
            gamma=ln_w,
            qkv_weights=w,
            qkv_w_scale=qkv_w_scale,
            norm_type=norm_type,
            eps=eps,
            bias=bias_t,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            is_input_swizzled=is_input_swizzled,
        )
    else:
        qkv_out = norm_qkv_ref(
            hidden=added,
            gamma=ln_w,
            qkv_weights=w,
            dtype=dtype,
            norm_type=norm_type,
            eps=eps,
            bias=bias_t,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            head_dim=head_dim,
            n_kv_heads=num_kv_heads,
            n_q_heads=num_q_heads,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            quantization_type=quantization_type,
        )

    if fused_rope:
        B, S, I = qkv_out.shape
        qkv_rotated = []
        for batch_idx in range(B):
            cos_cache_golden_input = cos_cache[batch_idx, :, : head_dim // 2].transpose(1, 0)
            sin_cache_golden_input = sin_cache[batch_idx, :, : head_dim // 2].transpose(1, 0)
            rotated_heads = []

            # rotate qk
            for i_head in range(num_q_heads + num_kv_heads):
                head = qkv_out[batch_idx, :, head_dim * i_head : head_dim * (i_head + 1)].transpose(1, 0)
                head_rotated = numpy_rope_golden(head, cos_cache_golden_input, sin_cache_golden_input, True).transpose(
                    1, 0
                )
                rotated_heads.append(head_rotated)

            # copy v
            rotated_heads.append(qkv_out[batch_idx, :, -head_dim * num_kv_heads :])
            rotated_heads = np.concatenate(rotated_heads, axis=-1)

            qkv_rotated.append(rotated_heads)
        qkv_out = np.stack(qkv_rotated, axis=0)

    if fp8_kv_cache:
        B, S, _ = qkv_out.shape
        q_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        cpu_q = qkv_out[:, :, :q_dim]
        k_scale_typed = np.array(k_scale_val, dtype=qkv_out.dtype)
        v_scale_typed = np.array(v_scale_val, dtype=qkv_out.dtype)
        max_val = get_max_positive_value_for_dtype(nl.float8_e4m3)
        clip_max = fp8_max if fp8_max is not None else max_val
        clip_min = fp8_min if fp8_min is not None else -max_val
        cpu_k = np.clip(qkv_out[:, :, q_dim : q_dim + kv_dim] / k_scale_typed, clip_min, clip_max)
        cpu_v = np.clip(qkv_out[:, :, q_dim + kv_dim :] / v_scale_typed, clip_min, clip_max)

        if use_block_kv:
            # Block KV layout: [num_blocks, block_size, kv_dim]
            k_golden = np.zeros((num_blocks, block_size, kv_dim), dtype=nl.float8_e4m3)
            v_golden = np.zeros((num_blocks, block_size, kv_dim), dtype=nl.float8_e4m3)
            for b in range(B):
                for s in range(S):
                    slot = slot_mapping[b, s]
                    block_idx = slot // block_size
                    pos_in_block = slot % block_size
                    k_golden[block_idx, pos_in_block, :] = cpu_k[b, s, :]
                    v_golden[block_idx, pos_in_block, :] = cpu_v[b, s, :]
        else:
            # Non-block KV layout: [B, max_seq_len, kv_dim]
            k_golden = np.zeros((B, max_seq_len, kv_dim), dtype=nl.float8_e4m3)
            v_golden = np.zeros((B, max_seq_len, kv_dim), dtype=nl.float8_e4m3)
            k_golden[:, :S, :] = cpu_k.astype(nl.float8_e4m3)
            v_golden[:, :S, :] = cpu_v.astype(nl.float8_e4m3)

        return {
            "q_tensor_hbm": cpu_q.astype(dtype),
            "k_cache": k_golden,
            "v_cache": v_golden,
        }

    if output_layout == QKVOutputLayout.NBSd:
        B, S, I = qkv_out.shape
        qkv_out = qkv_out.reshape((B, S, I // head_dim, head_dim))
        qkv_out = qkv_out.transpose((2, 0, 1, 3))
    elif output_layout == QKVOutputLayout.NBdS:
        B, S, I = qkv_out.shape
        qkv_out = qkv_out.reshape((B, S, I // head_dim, head_dim))
        qkv_out = qkv_out.transpose((2, 0, 3, 1))

    return {"out": qkv_out.astype(dtype)}


@pytest_test_metadata(
    name="QKV CTE",
    pytest_marks=["qkv", "cte"],
)
@final
class TestQkvCteKernel:
    def run_range_qkv_cte_test(
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
        fused_add = bool(dummy_tensor[FUSED_ADD_DIM_NAME])
        output_layout = QKVOutputLayout(dummy_tensor[OUTPUT_LAYOUT_DIM_NAME])

        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head

        test_size_classification = QkvCteClassification.classify(B=B, S=S, H=H, fused_qkv_dim=fused_qkv_dim)

        is_negative_test_case = test_options.is_negative_test_case
        with assert_negative_test_case(is_negative_test_case):
            self.run_qkv_cte_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                B=B,
                H=H,
                S=S,
                fused_qkv_dim=fused_qkv_dim,
                lnc_degree=lnc_degree,
                dtype=dtype,
                eps=1e-6,
                norm_type=norm_type,
                fused_add=fused_add,
                output_layout=output_layout,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
            )

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
        tensor_gen=gaussian_tensor_generator(),
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
        is_input_swizzled: bool = False,
        threshold: tuple[float, float] = (2e-2, 1e-5),
    ):
        kernel_input = build_qkv_input(
            batch=B,
            seqlen=S,
            hidden_dim=H,
            fused_qkv_dim=fused_qkv_dim,
            dtype=dtype,
            eps=eps,
            d_head=d_head,
            norm_type=norm_type,
            fused_add=fused_add,
            output_layout=output_layout,
            lnc_degree=lnc_degree,
            use_dma_transpose=use_dma_transpose,
            qkv_bias=qkv_bias,
            norm_bias=norm_bias,
            hidden_actual=hidden_actual,
            fused_rope=fused_rope,
            num_q_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
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
            is_input_swizzled=is_input_swizzled,
        )
        kernel_input["is_input_swizzled"] = is_input_swizzled

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
                fused_rope=fused_rope,
                num_q_heads=n_q_heads,
                num_kv_heads=n_kv_heads,
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
                quantization_type=quantization_type,
            )

        if fp8_kv_cache:
            q_dim = n_q_heads * d_head
            kv_dim = n_kv_heads * d_head
            if use_block_kv:
                output_placeholder = {
                    "q_tensor_hbm": np.zeros((B, S, q_dim), dtype=dtype),
                    "k_cache": np.zeros((num_blocks, block_size, kv_dim), dtype=nl.float8_e4m3),
                    "v_cache": np.zeros((num_blocks, block_size, kv_dim), dtype=nl.float8_e4m3),
                }
            else:
                output_placeholder = {
                    "q_tensor_hbm": np.zeros((B, S, q_dim), dtype=dtype),
                    "k_cache": np.zeros((B, max_seq_len, kv_dim), dtype=nl.float8_e4m3),
                    "v_cache": np.zeros((B, max_seq_len, kv_dim), dtype=nl.float8_e4m3),
                }
            kernel_func = qkv
            for key in ["quantization_type", "qkv_w_scale", "qkv_in_scale"]:
                kernel_input.pop(key, None)
        else:
            output_placeholder = {"out": np.zeros((B, S, fused_qkv_dim), dtype=dtype)}
            kernel_func = qkv

        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_func,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_placeholder,
                    ),
                    relative_accuracy=threshold[0],
                    absolute_accuracy=threshold[1],
                ),
            )
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
    @pytest.mark.parametrize(
        qkv_cte_kernel_fused_rope_test_params,
        qkv_cte_kernel_fused_rope_test_perms,
    )
    def test_qkv_cte_fused_rope_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)

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

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_accuracy
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_accuracy_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, hidden_actual, fused_qkv_dim, dtype, norm_type, fused_add, norm_bias, eps"
    qkv_cte_kernel_accuracy_test_perms = [
        # Data type test
        [1, 1, 128, 512, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
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
        [1, 1, 1000, 512, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        # H sweep
        # pytest.param(1,1,128,283,None,512,nl.bfloat16,NormType.RMS_NORM,True,False, 1e-6),
        # Marked as skip: 'H must be multiple of 128'
        # [1, 1, 128, 283, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        # Combinations of the fused flags
        [1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.RMS_NORM, True, False, 1e-6],
        [1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.NO_NORM, True, False, 1e-6],
        [1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.RMS_NORM, False, False, 1e-6],
        [1, 1, 512, 1024, None, 512, nl.bfloat16, NormType.NO_NORM, False, False, 1e-6],
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
        [1, 1, 128, 16384, 16384, 512, nl.bfloat16, NormType.RMS_NORM, False, False, 1e-6],
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

    @pytest.mark.fast
    @pytest.mark.parametrize(
        qkv_cte_kernel_accuracy_test_params,
        qkv_cte_kernel_accuracy_test_perms,
    )
    def test_qkv_cte_accuracy_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)

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
            fused_add=fused_add,
            norm_bias=norm_bias,
        )

    ####################################################################################################################
    # QKV CTE Test - test_qkv_cte_kernel_performance_bsd
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_performance_bsd_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, fused_qkv_dim, norm_type, use_dma_transpose, fused_add, add_bias, norm_bias, eps"
    qkv_cte_kernel_performance_bsd_test_perms = [
        # New model, 2025-Jul
        [2, 1, 128, 3072, 1280, NormType.NO_NORM, True, False, True, False, 1e-6],
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

    @pytest.mark.parametrize(
        qkv_cte_kernel_performance_bsd_test_params,
        qkv_cte_kernel_performance_bsd_test_perms,
    )
    def test_qkv_cte_performance_bsd_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)

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

    @pytest.mark.parametrize(
        qkv_cte_kernel_performance_nbsd_test_params,
        qkv_cte_kernel_performance_nbsd_test_perms,
    )
    def test_qkv_cte_performance_nbsd_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)

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
            output_layout=output_layout,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
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

    @pytest.mark.parametrize(
        qkv_cte_kernel_no_dma_transpose_test_params,
        qkv_cte_kernel_no_dma_transpose_test_perms,
    )
    def test_qkv_cte_no_dma_transpose_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)

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
            output_layout=output_layout,
            hidden_actual=None,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            tensor_gen=rope_gaussian_tensor_generator(),
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

    @pytest.mark.parametrize(
        qkv_cte_kernel_static_quantization_test_params,
        qkv_cte_kernel_static_quantization_test_perms,
    )
    def test_qkv_cte_static_quantization(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)

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
            qkv_bias=add_bias,
            norm_bias=norm_bias,
            output_layout=output_layout,
            hidden_actual=None,
            d_head=d_head,
            quantization_type=QuantizationType.STATIC,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            threshold=(5e-2, 2e-2),
        )

    ####################################################################################################################
    # QKV MX CTE MX Test - test_qkv_cte_mx_kernel_accuracy
    ####################################################################################################################
    # fmt: off
    qkv_cte_kernel_fused_rope_test_params = \
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, norm_type, fused_add, add_bias, norm_bias, output_layout, eps, fused_rope, is_input_swizzled"
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
    @pytest.mark.parametrize(
        qkv_cte_kernel_fused_rope_test_params,
        qkv_cte_kernel_fused_rope_test_perms,
    )
    def test_qkv_cte_mxfp8_unit(
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
        norm_bias,
        output_layout,
        eps,
        fused_rope,
        is_input_swizzled,
    ):
        if platform_target is not Platforms.TRN3:
            pytest.skip("MX Quantization is only supported on TRN3.")

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
            is_input_swizzled=is_input_swizzled,
            threshold=(5e-2, 1e-2),
        )

    @staticmethod
    def qkv_cte_sweep_config() -> RangeTestConfig:
        MIN_B = 1
        MAX_B = 4
        # ---------
        MIN_S = 128
        MAX_S = 8192
        # ---------
        MIN_H = 128
        MAX_H = 16384

        tc = TensorRangeConfig(
            tensor_configs={
                DUMMY_TENSOR_NAME: TensorConfig(
                    [
                        DimensionRangeConfig(min=MIN_B, max=MAX_B, name=BATCH_DIM_NAME),
                        DimensionRangeConfig(min=MIN_S, max=MAX_S, name=SEQUENCE_LEN_DIM_NAME),
                        DimensionRangeConfig(
                            min=MIN_H, max=MAX_H, multiple_of=128, name=HIDDEN_DIM_NAME
                        ),  # kernel requires H % 128 == 0
                        DimensionRangeConfig(min=1, max=3, name=N_Q_HEADS_DIM_NAME),
                        DimensionRangeConfig(min=1, max=2, name=N_KV_HEADS_DIM_NAME),
                        DimensionRangeConfig(min=128, max=128, name=D_HEAD_DIM_NAME),
                        # cte kernel only supports d_head = 128
                        DimensionRangeConfig(min=0, max=2, name=NORM_TYPE_DIM_NAME),
                        # NO_NORM=0, RMS_NORM=1, LAYER_NORM=2
                        DimensionRangeConfig(min=0, max=1, name=FUSED_ADD_DIM_NAME),
                        DimensionRangeConfig(min=0, max=1, name=OUTPUT_LAYOUT_DIM_NAME),
                        # BSD=0, NBSd=1
                    ]
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

    @range_test_config(qkv_cte_sweep_config())
    def test_qkv_cte_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
    ):
        compiler_args = CompilerArgs()
        self.run_range_qkv_cte_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            collector=collector,
        )

    ####################################################################################################################
    # QKV CTE Test - FP8 KV Cache Quantization Tests
    ####################################################################################################################

    qkv_cte_non_quantized_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head"
    qkv_cte_non_quantized_test_perms = [
        [2, 1, 1024, 2048, 1, 1, 128],
    ]

    @pytest.mark.parametrize(
        qkv_cte_non_quantized_test_params,
        qkv_cte_non_quantized_test_perms,
    )
    def test_qkv_cte_non_quantized(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        vnc_degree,
        batch,
        seqlen,
        hidden_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

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
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            use_dma_transpose=True,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
        )

    qkv_cte_fp8_kv_cache_test_params = (
        "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, max_seq_len, k_scale_val, v_scale_val"
    )
    qkv_cte_fp8_kv_cache_test_perms = [
        # Original test case
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 1.67, 1.67],
        # Different sequence lengths
        [2, 1, 512, 2048, 1, 1, 128, 4096, 1.67, 1.67],
        [2, 1, 2048, 2048, 1, 1, 128, 8192, 1.67, 1.67],
        # Different scales
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 1.0, 1.0],
        [2, 1, 1024, 2048, 1, 1, 128, 8192, 2.5, 2.5],
    ]

    @pytest.mark.parametrize(
        qkv_cte_fp8_kv_cache_test_params,
        qkv_cte_fp8_kv_cache_test_perms,
    )
    def test_qkv_cte_fp8_kv_cache_quantization(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

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
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fp8_kv_cache=True,
            max_seq_len=max_seq_len,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            threshold=(1e-1, 1e-5),
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

    @pytest.mark.parametrize(
        qkv_cte_fp8_extreme_test_params,
        qkv_cte_fp8_extreme_test_perms,
    )
    def test_qkv_cte_fp8_extreme_values_clamping(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        def scaled_tensor_gen(shape, dtype, name):
            scale = input_scale if name == "input" else weight_scale if name == "fused_qkv_weights" else 1.0
            return (np.random.randn(*shape) * scale).astype(dtype)

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
            threshold=(1e-1, 1e-5),
        )

    qkv_cte_block_kv_test_params = "vnc_degree, batch, seqlen, hidden_dim, n_q_heads, n_kv_heads, d_head, num_blocks, block_size, k_scale_val, v_scale_val, fp8_max, fp8_min"
    qkv_cte_block_kv_test_perms = [
        # Original test case
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0],
        # Different sequence length
        [2, 1, 1024, 2048, 1, 1, 128, 64, 128, 1.67, 1.67, 240.0, -240.0],
        # Different block size
        [2, 1, 512, 2048, 1, 1, 128, 128, 64, 1.67, 1.67, 240.0, -240.0],
        # Different scales
        [2, 1, 512, 2048, 1, 1, 128, 64, 128, 1.0, 1.0, 240.0, -240.0],
    ]

    @pytest.mark.parametrize(
        qkv_cte_block_kv_test_params,
        qkv_cte_block_kv_test_perms,
    )
    def test_qkv_cte_block_kv_cache(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
    ):
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree)
        fused_qkv_dim = (n_q_heads + n_kv_heads * 2) * d_head

        slot_mapping = np.arange(seqlen - 1, -1, -1, dtype=np.int32).reshape(batch, seqlen)

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
            eps=1e-6,
            norm_type=NormType.NO_NORM,
            fused_add=False,
            output_layout=QKVOutputLayout.BSD,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            fp8_kv_cache=True,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            use_block_kv=True,
            num_blocks=num_blocks,
            block_size=block_size,
            slot_mapping=slot_mapping,
            threshold=(1e-1, 1e-5),
        )
