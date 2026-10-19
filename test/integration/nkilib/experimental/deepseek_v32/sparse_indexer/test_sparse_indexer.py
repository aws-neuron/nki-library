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

from typing import Any

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.experimental.deepseek_v32.sparse_indexer import (
    deepseek_v32_sparse_indexer,
    deepseek_v32_sparse_indexer_torch_ref,
    get_deepseek_v32_sparse_indexer_shape,
)
from typing_extensions import override

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, CustomValidator, CustomValidatorWithOutputTensorData, Platforms
from test.utils.comparators import maxAllClose
from test.utils.mx_utils import dequantize_mx_golden, quantize_mx_golden
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# DeepSeek-V3.2 DSA indexer constants (config.index_*).
HEAD_DIM = 128
ROPE_DIM = 64

PROD_NUM_TOKENS = 16
PROD_HIDDEN_DIM = 7168
PROD_Q_LORA = 1536
PROD_NUM_HEADS = 64
PROD_INDEX_TOPK = 2048
PROD_MAX_SEQ_LEN = 2560

_SCORE_ATOL = 1e-5
"""
Absolute tolerance for the per-slot scores. Tight, so that the comparison is governed by the
relative term rather than an absolute band wide enough to reorder slots near the top-k cut line.
"""

_SCORE_RTOL = 5e-3
"""
Relative tolerance for the per-slot scores. The golden models the same MX quantization of both
query-projection operands that the kernel performs, which leaves a measured 0.16% relative
difference from the bf16 key cache and bf16 score matmul. This allows roughly 3x that.
"""

# fmt: off
PARAM_NAMES = "n_tokens, max_seq_len, hidden, q_lora_rank, n_heads, index_topk, dtype"
TEST_PARAMS = [
    pytest.param(1, 256,  256,  512,  4,  64,   np.float32,         id="ds32_sparse_indexer_t1_s256_lora512_h4_f32",  marks=pytest.mark.fast),
    pytest.param(2, 256,  256,  512,  4,  64,   ml_dtypes.bfloat16, id="ds32_sparse_indexer_t2_s256_lora512_h4_bf16", marks=pytest.mark.fast),
    pytest.param(1, 512,  1024, 512,  8,  128,  ml_dtypes.bfloat16, id="ds32_sparse_indexer_t1_s512_lora512_h8_bf16", marks=pytest.mark.fast),
    pytest.param(4, 384,  512,  512,  16, 128,  np.float32,         id="ds32_sparse_indexer_t4_s384_lora512_h16_f32"),
    pytest.param(
        PROD_NUM_TOKENS, PROD_MAX_SEQ_LEN, PROD_HIDDEN_DIM, PROD_Q_LORA, PROD_NUM_HEADS, PROD_INDEX_TOPK, ml_dtypes.bfloat16,
        id="DS32_SPARSE_INDEXER_PROD_CONFIG_B16",
    ),
]
# fmt: on


class TestSparseIndexer:
    """Test executor for DeepseekV32 Sparse Indexer tests."""

    @pytest.mark.parametrize("lnc", [1])
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_sparse_indexer(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_tokens: int,
        max_seq_len: int,
        hidden: int,
        q_lora_rank: int,
        n_heads: int,
        index_topk: int,
        dtype: type,
        lnc: int,
    ):
        # The query projection is always MX, and nisa.quantize_mx is unavailable before trn3.
        if platform_target in (Platforms.TRN1, Platforms.TRN2):
            pytest.skip(f"test_sparse_indexer: quantize-mx requires trn3 or later, got {platform_target}")

        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return self._generate_inputs(n_tokens, max_seq_len, hidden, q_lora_rank, n_heads, index_topk, dtype)

        def topk_comparator(golden_dict, output_tensors):
            golden_score = np.asarray(golden_dict["output_0"], dtype=np.float32).reshape(n_tokens, max_seq_len)
            k = min(index_topk, max_seq_len)

            class TopkValidator(CustomValidator):
                @override
                def validate(self, inference_output):
                    hw = np.frombuffer(inference_output, dtype=np.int32).reshape(n_tokens, k)
                    sb = golden_score.astype(ml_dtypes.bfloat16).astype(np.float32)
                    hw_vals = np.full((n_tokens, k), -np.inf, dtype=np.float32)
                    ref_vals = np.full((n_tokens, k), -np.inf, dtype=np.float32)
                    for token in range(n_tokens):
                        for j in range(k):
                            pos = int(hw[token, j])
                            if 0 <= pos < max_seq_len:
                                hw_vals[token, j] = sb[token, pos]
                        ref_vals[token] = np.sort(sb[token])[::-1][:k]
                    self._print_with_log("DeepSeek Indexer TopK (gathered values, sorted):")
                    return maxAllClose(
                        np.sort(hw_vals, axis=-1),
                        np.sort(ref_vals, axis=-1),
                        rtol=5e-2,
                        atol=1e-5,
                        verbose=1,
                        logfile=self.logfile,
                    )

            class ScoreValidator(CustomValidator):
                @override
                def validate(self, inference_output):
                    actual = np.frombuffer(inference_output, dtype=np.float32).reshape(n_tokens, max_seq_len)
                    self._print_with_log("DeepSeek Indexer scores:")
                    return maxAllClose(
                        actual,
                        golden_score,
                        rtol=_SCORE_RTOL,
                        atol=_SCORE_ATOL,
                        equal_nan_inf=True,
                        verbose=1,
                        logfile=self.logfile,
                    )

            return {
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=ScoreValidator,
                    output_ndarray=output_tensors["output_0"],
                ),
                "output_1": CustomValidatorWithOutputTensorData(
                    validator=TopkValidator,
                    output_ndarray=output_tensors["output_1"],
                ),
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            shapes = get_deepseek_v32_sparse_indexer_shape(
                kernel_input["hidden_states"],
                kernel_input["qr"],
                kernel_input["wk"],
                kernel_input["weights_proj"],
                kernel_input["cos"],
                kernel_input["key_cache"],
                kernel_input["index_topk"],
                kernel_input["valid_len"],
            )
            n_tokens, max_seq_len, index_topk = shapes.n_tokens, shapes.max_seq_len, shapes.index_topk
            return {
                "output_0": np.zeros((n_tokens, max_seq_len), dtype=np.float32),
                "output_1": np.zeros((n_tokens, index_topk), dtype=np.int32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            torch_ref=torch_ref_wrapper(_deepseek_v32_sparse_indexer_torch_ref_mx),
            kernel_entry=deepseek_v32_sparse_indexer,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            atol=1e-2,
            rtol=3e-2 if is_bf16 else 1e-3,
            custom_comparator=topk_comparator,
        )

    @staticmethod
    def _generate_inputs(
        n_tokens: int,
        max_seq_len: int,
        hidden: int,
        q_lora_rank: int,
        n_heads: int,
        index_topk: int,
        dtype: type,
    ) -> dict[str, Any]:
        """Wrapper around input generator for TestSparseIndexer."""

        def random_tensor(
            name: str, *shape, dtype: type = dtype, mean: float = 0.0, std: float = 0.2
        ) -> npt.NDArray[Any]:
            """Generates a random tensor for the provided name, shape, dtype and optionally mean and std."""
            generator = gaussian_tensor_generator(mean, std)
            return generator(shape, dtype, name)

        def scaled_hadamard_matrix(size: int, dtype=np.float32) -> npt.NDArray[Any]:
            """Builds a Sylvester Walsh-Hadamard matrix of shape ``(size, size)``, scaled by ``1 / sqrt(size)``."""
            assert size > 0 and (size & (size - 1)) == 0, f"Hadamard size must be a power of two, got {size}"
            hadamard = np.ones((1, 1), dtype=np.float32)

            while hadamard.shape[0] < size:
                hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])

            return (hadamard / np.sqrt(size)).astype(dtype)

        rope_dim_half = ROPE_DIM // 2
        q_out_dim = n_heads * HEAD_DIM

        positions = np.full((n_tokens,), max_seq_len - 1, dtype=np.int32)

        # The kernel always takes MX-packed weights, so q_lora_rank must tile evenly into MX K-tiles.
        wq_b_f32 = random_tensor("wq_b", q_lora_rank, q_out_dim, dtype=np.float32, std=0.02)
        wq_b, wq_b_scale, _ = _quantize_weight_for_mx(wq_b_f32)

        return {
            "hidden_states": random_tensor("hidden_states", n_tokens, hidden, std=1.0),
            "qr": random_tensor("qr", n_tokens, q_lora_rank, std=1.0),
            "wq_b": wq_b,
            "wq_b_scale": wq_b_scale,
            "wk": random_tensor("wk", hidden, HEAD_DIM),
            "k_norm_weight": random_tensor("k_norm_weight", 1, HEAD_DIM, dtype=np.float32, mean=1.0, std=0.1),
            "k_norm_bias": random_tensor("k_norm_bias", 1, HEAD_DIM, dtype=np.float32, std=0.05),
            "weights_proj": random_tensor("weights_proj", hidden, n_heads),
            "hadamard": scaled_hadamard_matrix(HEAD_DIM, dtype),
            "cos": random_tensor("cos", n_tokens, rope_dim_half, std=1.0),
            "sin": random_tensor("sin", n_tokens, rope_dim_half, std=1.0),
            "key_cache": random_tensor("key_cache", n_tokens, max_seq_len, HEAD_DIM, std=1.0),
            "positions": positions,
            "index_topk": index_topk,
            # the batch's max position + 1; slots beyond it are masked, so they are not scored
            "valid_len": int(positions.max()) + 1,
        }


def _deepseek_v32_sparse_indexer_torch_ref_mx(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    wq_b: npt.NDArray[Any],
    wq_b_scale: torch.Tensor,
    wk: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    weights_proj: torch.Tensor,
    hadamard: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    positions: torch.Tensor,
    index_topk: int,
    valid_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Quantize both query-projection operands the way the kernel does, then delegate to the torch
    reference.

    ``torch_ref_wrapper`` converts every numpy input to a torch tensor except MX-packed ``x4``
    dtypes, which torch cannot represent. ``wq_b`` therefore arrives as numpy while its scale
    arrives as a tensor.
    """

    q_lora_rank = qr.shape[1]
    q_out_dim = weights_proj.shape[1] * wk.shape[1]
    dequantized_expanded = dequantize_mx_golden(np.asarray(wq_b), wq_b_scale.numpy())

    # dequantize_mx_golden returns float64 despite its docstring: its scale factors come from
    # ``2.0 ** exponent`` with a Python float base, which numpy widens. Cast back so the matmul
    # against the float32 query and Hadamard operands does not hit a dtype mismatch.
    dequantized = (
        dequantized_expanded.reshape(q_lora_rank // 4, q_out_dim, 4)
        .transpose(0, 2, 1)
        .reshape(q_lora_rank, q_out_dim)
        .astype(np.float32)
    )

    # The kernel loads qr into a bf16 buffer, so the DMA rounds f32 inputs on the wire, then
    # MX-quantizes along the contraction axis. _quantize_weight_for_mx expects the contraction
    # dimension on axis 0, so transpose in and back out.
    qr_bf16 = qr.numpy().astype(ml_dtypes.bfloat16).astype(np.float32)
    _, _, qr_dequantized = _quantize_weight_for_mx(np.ascontiguousarray(qr_bf16.T))
    qr = torch.from_numpy(np.ascontiguousarray(qr_dequantized.T))

    return deepseek_v32_sparse_indexer_torch_ref(
        hidden_states,
        qr,
        torch.from_numpy(dequantized),
        None,  # the scale is consumed by the dequantization above
        wk,
        k_norm_weight,
        k_norm_bias,
        weights_proj,
        hadamard,
        cos,
        sin,
        key_cache,
        positions,
        index_topk,
        valid_len,
    )


def _quantize_weight_for_mx(
    weight_f32: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    K, F = weight_f32.shape
    assert K % 32 == 0, f"K must be a multiple of 32 for MX, got {K}"
    expanded = weight_f32.reshape(K // 4, 4, F).transpose(0, 2, 1).reshape(K // 4, F * 4)
    packed_x4, scale = quantize_mx_golden(expanded, nl.float8_e4m3fn_x4)
    dequantized_expanded = dequantize_mx_golden(packed_x4, scale)  # [K // 4, F * 4], float64
    dequantized_f32 = dequantized_expanded.reshape(K // 4, F, 4).transpose(0, 2, 1).reshape(K, F).astype(np.float32)
    return packed_x4, scale, dequantized_f32
