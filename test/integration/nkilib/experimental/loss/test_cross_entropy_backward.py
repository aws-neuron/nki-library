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

# LEGACY SWEEP TEST FRAMEWORK - Uses @range_test_config / RangeTestHarness
# New tests should use @pytest.mark.coverage_parametrize instead

"""
Comprehensive tests for Cross Entropy backward pass kernel.
"""

from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import final

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.loss import cross_entropy_backward
from nkilib_src.nkilib.experimental.loss.cross_entropy_torch import (
    cross_entropy_backward_torch_ref,
    cross_entropy_forward_torch_ref,
)

# Dimension names
CE_CONFIG = "ce_config"
B_DIM = "B"
T_DIM = "T"
V_DIM = "V"
DTYPE_DIM = "dtype"
PB_DIM = "PB"
CS_DIM = "CS"
REDUCTION_DIM = "reduction"


def _make_config(test_grid) -> RangeTestConfig:
    """Helper to generate test configuration from test grid (minimizes duplication)."""
    test_cases = []
    for tc in test_grid:
        test_cases.append(
            {
                CE_CONFIG: {
                    B_DIM: tc[0],
                    T_DIM: tc[1],
                    V_DIM: tc[2],
                    DTYPE_DIM: tc[3],
                    PB_DIM: tc[4],
                    CS_DIM: tc[5],
                    REDUCTION_DIM: tc[6] if len(tc) > 6 else "mean",
                }
            }
        )

    return RangeTestConfig(
        additional_params={},
        global_tensor_configs=TensorRangeConfig(
            tensor_configs={
                CE_CONFIG: TensorConfig(
                    [
                        DimensionRangeConfig(name=B_DIM),
                        DimensionRangeConfig(name=T_DIM),
                        DimensionRangeConfig(name=V_DIM),
                        DimensionRangeConfig(name=DTYPE_DIM),
                        DimensionRangeConfig(name=PB_DIM),
                        DimensionRangeConfig(name=CS_DIM),
                        DimensionRangeConfig(name=REDUCTION_DIM),
                    ]
                ),
            },
            monotonic_step_size=1,
            custom_generators=[RangeManualGeneratorStrategy(test_cases=test_cases)],
        ),
    )


def generate_kernel_inputs(test_case: RangeTestCase):
    """Generate cross entropy backward kernel inputs from test configuration."""
    test_cfg = test_case.tensors[CE_CONFIG]
    B = test_cfg[B_DIM]
    T = test_cfg[T_DIM]
    V = test_cfg[V_DIM]
    dtype_str = test_cfg.get(DTYPE_DIM, "bfloat16")

    num_positions = B * T

    numpy_dtype_map = {
        "bfloat16": ml_dtypes.bfloat16,
        "float32": np.float32,
    }
    numpy_dtype = numpy_dtype_map.get(dtype_str, np.float32)

    np.random.seed(42)
    logits = np.random.randn(num_positions, V).astype(numpy_dtype)
    targets = np.random.randint(0, V, size=(num_positions,), dtype=np.int32)

    # Compute LSE using forward pass (needed for backward)
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype_str, torch.bfloat16)

    if logits.dtype == ml_dtypes.bfloat16:
        logits_torch = torch.from_numpy(logits.astype(np.float32)).to(torch_dtype)
    else:
        logits_torch = torch.from_numpy(logits).to(torch_dtype)

    targets_torch = torch.from_numpy(targets).long()
    _, lse_torch = cross_entropy_forward_torch_ref(logits_torch, targets_torch)

    if dtype_str == "float32":
        lse_state = lse_torch.float().numpy().astype(np.float32)
    else:
        lse_state = lse_torch.float().numpy().astype(ml_dtypes.bfloat16)

    return {
        "logits_hbm": logits,
        "targets_hbm": targets,
        "lse_state_hbm": lse_state,
    }


def golden_cross_entropy_backward_ref(inp_np, dtype_str="bfloat16", reduction="mean"):
    """PyTorch reference implementation of cross entropy backward using torch reference."""
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype_str, torch.bfloat16)

    # Convert inputs to torch
    logits_np = inp_np['logits_hbm']

    if logits_np.dtype == ml_dtypes.bfloat16:
        logits = torch.from_numpy(logits_np.astype(np.float32)).to(torch_dtype)
    else:
        logits = torch.from_numpy(logits_np).to(torch_dtype)

    targets = torch.from_numpy(inp_np['targets_hbm']).long()

    # Use torch reference with reduction (mean or sum)
    grad_logits = cross_entropy_backward_torch_ref(logits, targets, reduction)

    if dtype_str == "float32":
        return {
            'grad_logits_hbm': grad_logits.float().numpy().astype(np.float32),
        }
    else:
        return {
            'grad_logits_hbm': grad_logits.float().numpy().astype(ml_dtypes.bfloat16),
        }


@final
class TestCrossEntropyBackwardSweep:
    """
    Comprehensive tests for Cross Entropy backward kernel - 99 total tests (11 marked as fast).

    Tests the cross_entropy_backward kernel with hardcoded optimal PB (positions_per_batch)
    and CS (chunk_size) parameters determined from SBUF memory constraint analysis.

    Kernel sweep tests (100 tests total, 11 marked with pytest.mark.fast for pre-commit):
    - Switch Transformers: V=32,128 (27 tests, 3 fast)
    - DeepSeek V3: V=129,280 (21 tests, 2 fast)
    - Qwen3-8B: V=151,936 (19 tests, 3 fast)
    - GPT-OSS 20B: V=201,088 (16 tests, 2 fast)
    - Llama 4 Maverick: V=202,408 (17 tests, 2 fast)

    Fast tests are marked with pytest.param(..., marks=pytest.mark.fast) and selected to provide:
    - Coverage of all vocabulary sizes
    - Representative mid-size configurations
    - Corner cases (minimal sizes, odd dimensions)
    - Both dtypes (bfloat16 and float32)
    - Both reduction modes (mean and sum)
    - Different chunk configurations (single-chunk vs multi-chunk)

    Each test configuration includes:
    - B (batch size), T (sequence length), V (vocabulary size)
    - dtype (bfloat16 or float32)
    - PB (positions_per_batch): up to 128 (P_MAX), optimized for hardware throughput
    - CS (chunk_size): dtype-dependent due to tensor_tensor per-partition constraint
    - reduction (mean or sum)

    SBUF Memory Constraint:
    - SBUF has 128 partitions, each with 229,376 bytes (224 KiB)
    - Per-partition constraint for tensor_tensor op: 2 buffers × CS × dtype_bytes ≤ 229,376
    - bf16: CS ≤ 57,344 (use 32768)
    - fp32: CS ≤ 28,672 (use 16384)
    - PB is independent of this constraint (only affects total SBUF, not per-partition)

    Chunks needed per vocab size:
    - V=32,128 with CS=32128: 1 chunk (bf16), 2 chunks (fp32 with CS=16384)
    - V=129,280 with CS=32768: 4 chunks (bf16), 8 chunks (fp32 with CS=16384)
    - V=151,936 with CS=32768: 5 chunks (bf16), 10 chunks (fp32 with CS=16384)
    - V=201,088 with CS=32768: 7 chunks (bf16), 13 chunks (fp32 with CS=16384)
    - V=202,408 with CS=32768: 7 chunks (bf16), 13 chunks (fp32 with CS=16384)
    """

    def prepare_test_parametrized(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
    ):
        """Prepare and execute a single sweep test case."""
        test_cfg = test_options.tensors[CE_CONFIG]
        B = test_cfg[B_DIM]
        T = test_cfg[T_DIM]
        V = test_cfg[V_DIM]
        dtype_str = test_cfg.get(DTYPE_DIM, "bfloat16")
        PB = test_cfg[PB_DIM]
        CS = test_cfg[CS_DIM]
        reduction = test_cfg.get(REDUCTION_DIM, "mean")

        # Map dtype string to NKI dtype
        nki_dtype_map = {
            "bfloat16": nl.bfloat16,
            "float32": nl.float32,
        }
        nki_dtype = nki_dtype_map.get(dtype_str, nl.bfloat16)

        # Map dtype string to numpy dtype
        numpy_dtype_map = {
            "bfloat16": ml_dtypes.bfloat16,
            "float32": np.float32,
        }
        numpy_dtype = numpy_dtype_map.get(dtype_str, ml_dtypes.bfloat16)

        # Generate inputs
        kernel_input = generate_kernel_inputs(test_options)

        # Create output placeholder for LazyGoldenGenerator
        num_positions = B * T
        output_placeholder = {
            'grad_logits_hbm': np.zeros((num_positions, V), dtype=numpy_dtype),
        }

        # Create lazy golden generator
        def create_lazy_golden():
            return golden_cross_entropy_backward_ref(kernel_input, dtype_str, reduction)

        # Build kernel input dict
        kernel_input_dict = {
            "logits_hbm": kernel_input["logits_hbm"],
            "targets_hbm": kernel_input["targets_hbm"],
            "lse_state_hbm": kernel_input["lse_state_hbm"],
            "reduction": reduction,
            "positions_per_batch": PB,
            "chunk_size": CS,
            "dtype": nki_dtype,
        }

        # Execute test
        test_manager.execute(
            KernelArgs(
                kernel_func=cross_entropy_backward,
                compiler_input=compiler_args,
                kernel_input=kernel_input_dict,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_placeholder,
                    ),
                    absolute_accuracy=1e-2 if dtype_str == "bfloat16" else 1e-5,
                    relative_accuracy=1e-2 if dtype_str == "bfloat16" else 1e-5,
                ),
            )
        )

    @staticmethod
    def switch_transformers_fast_config():
        """Switch Transformers (V=32,128) - 3 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            [4,  4096,  32128,  "bfloat16", 64, 16384, "mean"],  # FAST: Representative (1 chunk)
            [1,  1,     32128,  "bfloat16",   1, 16384, "mean"],  # FAST: Minimal size
            [4,  4096,  32128,  "float32",  64, 8192, "sum"],   # FAST: FP32 sum reduction (1 chunk)
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def switch_transformers_full_config():
        """Switch Transformers (V=32,128) - 24 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            # Batch sweep T=8192 (mean reduction)
            [1,  8192,  32128,  "bfloat16", 128, 16384, "mean"],
            [2,  8192,  32128,  "bfloat16", 128, 16384, "mean"],
            [4,  8192,  32128,  "bfloat16", 128, 16384, "mean"],
            [6,  8192,  32128,  "bfloat16", 128, 16384, "mean"],
            [8,  8192,  32128,  "bfloat16", 128, 16384, "mean"],
            # Batch sweep T=4096 (mean reduction)
            [16, 4096,  32128,  "bfloat16", 128, 16384, "mean"],
            [2,  4096,  32128,  "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=8 (mean reduction)
            [8,  512,   32128,  "bfloat16", 128, 16384, "mean"],
            [8,  1024,  32128,  "bfloat16", 128, 16384, "mean"],
            [8,  2048,  32128,  "bfloat16", 128, 16384, "mean"],
            [8,  4096,  32128,  "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=4 (mean reduction)
            [4,  1024,  32128,  "bfloat16", 128, 16384, "mean"],
            [4,  2048,  32128,  "bfloat16", 128, 16384, "mean"],
            [4,  16384, 32128,  "bfloat16", 128, 16384, "mean"],
            # Mixed patterns (mean reduction)
            [32, 1024,  32128,  "bfloat16", 128, 16384, "mean"],
            [16, 2048,  32128,  "bfloat16", 128, 16384, "mean"],
            [12, 4096,  32128,  "bfloat16", 128, 16384, "mean"],
            [2,  16384, 32128,  "bfloat16", 128, 16384, "mean"],
            # FP32 (mean reduction)
            [4,  4096,  32128,  "float32",  128, 8192, "mean"],
            [8,  4096,  32128,  "float32",  128, 8192, "mean"],
            [16, 2048,  32128,  "float32",  128, 8192, "mean"],
            # Corner cases (mean reduction)
            [7,  4096,  32128,  "bfloat16", 128, 16384, "mean"],
            # Sum reduction tests
            [4,  4096,  32128,  "bfloat16", 128, 16384, "sum"],
            [8,  2048,  32128,  "bfloat16", 128, 16384, "sum"],
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def deepseek_v3_fast_config():
        """DeepSeek V3 (V=129,280) - 2 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            [2,  4096,  129280, "bfloat16", 64, 16384, "mean"],  # FAST: Multi-chunk bf16 (4 chunks)
            [1,  4096,  129280, "float32",  64, 8192, "mean"],  # FAST: FP32 (4 chunks)
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def deepseek_v3_full_config():
        """DeepSeek V3 (V=129,280) - 19 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            # Batch sweep T=4096 (mean reduction)
            [1,  4096,  129280, "bfloat16", 128, 16384, "mean"],
            [4,  4096,  129280, "bfloat16", 128, 16384, "mean"],
            # Batch sweep T=2048 (mean reduction)
            [2,  2048,  129280, "bfloat16", 128, 16384, "mean"],
            [4,  2048,  129280, "bfloat16", 128, 16384, "mean"],
            [8,  2048,  129280, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=2 (mean reduction)
            [2,  512,   129280, "bfloat16", 128, 16384, "mean"],
            [2,  1024,  129280, "bfloat16", 128, 16384, "mean"],
            [2,  8192,  129280, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=1 (mean reduction)
            [1,  2048,  129280, "bfloat16", 128, 16384, "mean"],
            [1,  8192,  129280, "bfloat16", 128, 16384, "mean"],
            [1,  16384, 129280, "bfloat16", 128, 16384, "mean"],
            # Mixed patterns (mean reduction)
            [8,  1024,  129280, "bfloat16", 128, 16384, "mean"],
            [6,  2048,  129280, "bfloat16", 128, 16384, "mean"],
            [3,  4096,  129280, "bfloat16", 128, 16384, "mean"],
            # FP32 (mean reduction)
            [2,  4096,  129280, "float32",  128, 8192, "mean"],
            [1,  8192,  129280, "float32",  128, 8192, "mean"],
            # Corner cases (mean reduction)
            [1,  128,   129280, "bfloat16", 128, 16384, "mean"],
            # Sum reduction tests
            [2,  4096,  129280, "bfloat16", 128, 16384, "sum"],
            [1,  4096,  129280, "float32",  128, 8192, "sum"],
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def qwen3_8b_fast_config():
        """Qwen3-8B (V=151,936) - 3 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            [2,  4096,  151936, "bfloat16", 128, 16384, "mean"],  # FAST: Large vocab (5 chunks)
            [1,  4096,  151936, "float32", 128, 8192, "mean"],   # Torch titan example
            [1,  13000, 151936, "bfloat16", 128, 16384, "mean"],  # FAST: Max positions stress test
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def qwen3_8b_full_config():
        """Qwen3-8B (V=151,936) - 16 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            # Batch sweep T=4096 (mean reduction)
            [1,  4096,  151936, "bfloat16", 128, 16384, "mean"],
            [3,  4096,  151936, "bfloat16", 128, 16384, "mean"],
            # Batch sweep T=2048 (mean reduction)
            [2,  2048,  151936, "bfloat16", 128, 16384, "mean"],
            [4,  2048,  151936, "bfloat16", 128, 16384, "mean"],
            [6,  2048,  151936, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=2 (mean reduction)
            [2,  512,   151936, "bfloat16", 128, 16384, "mean"],
            [2,  1024,  151936, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=1 (mean reduction)
            [1,  2048,  151936, "bfloat16", 128, 16384, "mean"],
            [1,  8192,  151936, "bfloat16", 128, 16384, "mean"],
            [1,  12288, 151936, "bfloat16", 128, 16384, "mean"],
            # Mixed patterns (mean reduction)
            [4,  3072,  151936, "bfloat16", 128, 16384, "mean"],
            # FP32 (mean reduction)
            [2,  2048,  151936, "float32",  128, 8192, "mean"],
            # Max position tests (mean reduction)
            [1,  14000, 151936, "bfloat16", 128, 16384, "mean"],
            # Corner cases (mean reduction)
            [1,  256,   151936, "bfloat16", 128, 16384, "mean"],
            # Sum reduction tests
            [2,  4096,  151936, "bfloat16", 128, 16384, "sum"],
            [2,  2048,  151936, "float32",  128, 8192, "sum"],
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def gpt_oss_20b_fast_config():
        """GPT-OSS 20B (V=201,088) - 2 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            [1,  4096,  201088, "bfloat16", 64, 16384, "mean"],  # FAST: Very large vocab representative (7 chunks)
            [3,  3413,  201088, "bfloat16", 64, 16384, "mean"],  # FAST: Odd dimensions (7 chunks)
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def gpt_oss_20b_full_config():
        """GPT-OSS 20B (V=201,088) - 14 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            # Batch sweep T=4096 (mean reduction)
            [2,  4096,  201088, "bfloat16", 128, 16384, "mean"],
            # Batch sweep T=2048 (mean reduction)
            [2,  2048,  201088, "bfloat16", 128, 16384, "mean"],
            [4,  2048,  201088, "bfloat16", 128, 16384, "mean"],
            [5,  2048,  201088, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=2 (mean reduction)
            [2,  512,   201088, "bfloat16", 128, 16384, "mean"],
            [2,  1024,  201088, "bfloat16", 128, 16384, "mean"],
            [2,  5120,  201088, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=1 (mean reduction)
            [1,  2048,  201088, "bfloat16", 128, 16384, "mean"],
            [1,  8192,  201088, "bfloat16", 128, 16384, "mean"],
            [1,  10240, 201088, "bfloat16", 128, 16384, "mean"],
            # Mixed patterns (mean reduction)
            [4,  2560,  201088, "bfloat16", 128, 16384, "mean"],
            # FP32 (mean reduction)
            [1,  5120,  201088, "float32",  128, 8192, "mean"],
            # Sum reduction tests
            [2,  4096,  201088, "bfloat16", 128, 16384, "sum"],
            [1,  5120,  201088, "float32",  128, 8192, "sum"],
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def llama4_maverick_fast_config():
        """Llama 4 Maverick (V=202,408) - 2 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            [1,  4096,  202408, "bfloat16", 64, 16384, "mean"],  # FAST: Very large vocab (7 chunks)
            [5,  2047,  202408, "bfloat16", 64, 16384, "mean"],  # FAST: Odd dimensions (7 chunks)
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def llama4_maverick_full_config():
        """Llama 4 Maverick (V=202,408) - 15 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS, reduction]
        # SBUF per-partition constraint for tensor_tensor: 2 × CS × dtype_bytes ≤ 229,376
        test_grid = [
            # Batch sweep T=4096 (mean reduction)
            [2,  4096,  202408, "bfloat16", 128, 16384, "mean"],
            # Batch sweep T=2048 (mean reduction)
            [2,  2048,  202408, "bfloat16", 128, 16384, "mean"],
            [4,  2048,  202408, "bfloat16", 128, 16384, "mean"],
            [5,  2048,  202408, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=2 (mean reduction)
            [2,  512,   202408, "bfloat16", 128, 16384, "mean"],
            [2,  1024,  202408, "bfloat16", 128, 16384, "mean"],
            [2,  5120,  202408, "bfloat16", 128, 16384, "mean"],
            # Seq sweep B=1 (mean reduction)
            [1,  2048,  202408, "bfloat16", 128, 16384, "mean"],
            [1,  8192,  202408, "bfloat16", 128, 16384, "mean"],
            [1,  10240, 202408, "bfloat16", 128, 16384, "mean"],
            # Mixed patterns (mean reduction)
            [4,  2560,  202408, "bfloat16", 128, 16384, "mean"],
            # FP32 (mean reduction)
            [1,  4096,  202408, "float32",  128, 8192, "mean"],
            [1,  5120,  202408, "float32",  128, 8192, "mean"],
            # Sum reduction tests
            [2,  4096,  202408, "bfloat16", 128, 16384, "sum"],
            [1,  4096,  202408, "float32",  128, 8192, "sum"],
        ]
        # fmt: on
        return _make_config(test_grid)

    @pytest.mark.fast
    @range_test_config(switch_transformers_fast_config())
    def test_switch_transformers_fast(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """Switch Transformers (V=32,128) - 3 fast tests for pre-commit."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @range_test_config(switch_transformers_full_config())
    def test_switch_transformers_full(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """Switch Transformers (V=32,128) - 24 full sweep tests."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @pytest.mark.fast
    @range_test_config(deepseek_v3_fast_config())
    def test_deepseek_v3_fast(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """DeepSeek V3 (V=129,280) - 2 fast tests for pre-commit."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @range_test_config(deepseek_v3_full_config())
    def test_deepseek_v3_full(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """DeepSeek V3 (V=129,280) - 19 full sweep tests."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @pytest.mark.fast
    @range_test_config(qwen3_8b_fast_config())
    def test_qwen3_8b_fast(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """Qwen3-8B (V=151,936) - 2 fast tests for pre-commit."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @range_test_config(qwen3_8b_full_config())
    def test_qwen3_8b_full(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """Qwen3-8B (V=151,936) - 16 full sweep tests."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @pytest.mark.fast
    @range_test_config(gpt_oss_20b_fast_config())
    def test_gpt_oss_20b_fast(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """GPT-OSS 20B (V=201,088) - 2 fast tests for pre-commit."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @range_test_config(gpt_oss_20b_full_config())
    def test_gpt_oss_20b_full(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """GPT-OSS 20B (V=201,088) - 14 full sweep tests."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @pytest.mark.fast
    @range_test_config(llama4_maverick_fast_config())
    def test_llama4_maverick_fast(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """Llama 4 Maverick (V=202,408) - 2 fast tests for pre-commit."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )

    @range_test_config(llama4_maverick_full_config())
    def test_llama4_maverick_full(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
    ):
        """Llama 4 Maverick (V=202,408) - 15 full sweep tests."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )
