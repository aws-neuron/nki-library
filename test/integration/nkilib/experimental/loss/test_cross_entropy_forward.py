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
Comprehensive tests for Cross Entropy loss kernel.

Based on the test grid from: https://quip-amazon.com/AnOFAD11m20n/Cross-Entropy-Test-Grid

Tests the cross_entropy_forward_kernel with hardcoded optimal PB (positions_per_batch)
and CS (chunk_size) parameters. The PB/CS values were determined through optimization
analysis to maximize performance while respecting hardware constraints.

Includes:
- Kernel sweep tests: 88 configurations across 5 model vocabularies
- Each test specifies: [B, T, V, dtype, PB, CS]
- Covers batch/sequence sweeps, FP32 precision, and corner cases
"""

from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.pytest_test_metadata import pytest_test_metadata
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
from nkilib_src.nkilib.experimental.loss import cross_entropy_forward
from nkilib_src.nkilib.experimental.loss.cross_entropy_torch import cross_entropy_forward_torch_ref

# Dimension names
CE_CONFIG = "ce_config"
B_DIM = "B"
T_DIM = "T"
V_DIM = "V"
DTYPE_DIM = "dtype"
PB_DIM = "PB"
CS_DIM = "CS"


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
                    ]
                ),
            },
            monotonic_step_size=1,
            custom_generators=[RangeManualGeneratorStrategy(test_cases=test_cases)],
        ),
    )


def generate_kernel_inputs(test_case: RangeTestCase):
    """Generate cross entropy kernel inputs from test configuration."""
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

    return {
        "logits_hbm": logits,
        "targets_hbm": targets,
    }


def golden_cross_entropy_ref(inp_np, dtype_str="bfloat16"):
    """PyTorch reference implementation of cross entropy using torch reference."""
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype_str, torch.bfloat16)

    logits_np = inp_np['logits_hbm']
    if logits_np.dtype == ml_dtypes.bfloat16:
        logits = torch.from_numpy(logits_np.astype(np.float32)).to(torch_dtype)
    else:
        logits = torch.from_numpy(logits_np).to(torch_dtype)

    targets = torch.from_numpy(inp_np['targets_hbm']).long()

    # Use the torch reference implementation
    loss, lse = cross_entropy_forward_torch_ref(logits, targets)

    if dtype_str == "float32":
        return {
            'loss_hbm': loss.float().numpy().astype(np.float32),
            'lse_state_hbm': lse.float().numpy().astype(np.float32),
        }
    else:
        return {
            'loss_hbm': loss.float().numpy().astype(ml_dtypes.bfloat16),
            'lse_state_hbm': lse.float().numpy().astype(ml_dtypes.bfloat16),
        }


@pytest_test_metadata(
    name="CrossEntropySweep",
    pytest_marks=["loss", "cross_entropy", "sweep"],
)
@final
class TestCrossEntropyForwardSweep:
    """
    Comprehensive tests for Cross Entropy forward kernel - 88 total tests (9 marked as fast).

    Tests the cross_entropy_forward kernel with hardcoded optimal PB (positions_per_batch)
    and CS (chunk_size) parameters determined from optimization analysis.

    Test Structure:
    - Each model has two test methods: test_<model>_fast and test_<model>_full
    - Fast tests are marked with @pytest.mark.fast for quick pre-commit validation
    - Full tests contain comprehensive sweep coverage

    Kernel sweep tests (88 tests total, 9 fast tests for pre-commit):
    - Switch Transformers: V=32,128 (2 fast + 22 full = 24 tests)
    - DeepSeek V3: V=129,280 (2 fast + 17 full = 19 tests)
    - Qwen3-8B: V=151,936 (2 fast + 14 full = 16 tests)
    - GPT-OSS 20B: V=201,088 (1 fast + 13 full = 14 tests)
    - Llama 4 Maverick: V=202,408 (2 fast + 13 full = 15 tests)

    Fast tests provide:
    - Coverage of all vocabulary sizes
    - Representative mid-size configurations
    - Corner cases (minimal sizes, odd dimensions)
    - Both dtypes (bfloat16 and float32)
    - Different chunk configurations (single-chunk vs multi-chunk)

    Each test configuration includes:
    - B (batch size), T (sequence length), V (vocabulary size)
    - dtype (bfloat16 or float32)
    - PB (positions_per_batch): typically 64, optimized for hardware
    - CS (chunk_size): varies by vocabulary size (32128-65535 for bf16, 32768 for fp32)

    All configurations stay within hardware constraints:
    - Partition dimension (PB) ≤ 128
    - Free dimension (CS) ≤ 65,535
    - Memory per allocation ≤ 24 MiB
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
            'loss_hbm': np.zeros((num_positions,), dtype=numpy_dtype),
            'lse_state_hbm': np.zeros((num_positions,), dtype=numpy_dtype),
        }

        # Create lazy golden generator
        def create_lazy_golden():
            return golden_cross_entropy_ref(kernel_input, dtype_str)

        # Execute test
        test_manager.execute(
            KernelArgs(
                kernel_func=cross_entropy_forward,
                compiler_input=compiler_args,
                kernel_input={
                    "logits_hbm": kernel_input["logits_hbm"],
                    "targets_hbm": kernel_input["targets_hbm"],
                    "positions_per_batch": PB,
                    "chunk_size": CS,
                    "dtype": nki_dtype,
                },
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
        """Switch Transformers (V=32,128) - 2 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            [4,  4096,  32128,  "bfloat16", 64, 32128],  # FAST: Representative
            [1,  1,     32128,  "bfloat16",  1, 32128],  # FAST: Minimal size
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def switch_transformers_full_config():
        """Switch Transformers (V=32,128) - 22 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            # Batch sweep T=8192
            [1,  8192,  32128,  "bfloat16", 64, 32128],
            [2,  8192,  32128,  "bfloat16", 64, 32128],
            [4,  8192,  32128,  "bfloat16", 64, 32128],
            [6,  8192,  32128,  "bfloat16", 64, 32128],
            [8,  8192,  32128,  "bfloat16", 64, 32128],
            # Batch sweep T=4096
            [16, 4096,  32128,  "bfloat16", 64, 32128],
            [2,  4096,  32128,  "bfloat16", 64, 32128],
            # Seq sweep B=8
            [8,  512,   32128,  "bfloat16", 64, 32128],
            [8,  1024,  32128,  "bfloat16", 64, 32128],
            [8,  2048,  32128,  "bfloat16", 64, 32128],
            [8,  4096,  32128,  "bfloat16", 64, 32128],
            # Seq sweep B=4
            [4,  1024,  32128,  "bfloat16", 64, 32128],
            [4,  2048,  32128,  "bfloat16", 64, 32128],
            [4,  16384, 32128,  "bfloat16", 64, 32128],
            # Mixed patterns
            [32, 1024,  32128,  "bfloat16", 64, 32128],
            [16, 2048,  32128,  "bfloat16", 64, 32128],
            [12, 4096,  32128,  "bfloat16", 64, 32128],
            [2,  16384, 32128,  "bfloat16", 64, 32128],
            # FP32
            [4,  4096,  32128,  "float32",   64, 32128],
            [8,  4096,  32128,  "float32",   64, 32128],
            [16, 2048,  32128,  "float32",   64, 32128],  # FP32 boundary test
            # Corner cases
            [7,  4096,  32128,  "bfloat16", 64, 32128],  # Uneven split: prime batch size
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def deepseek_v3_fast_config():
        """DeepSeek V3 (V=129,280) - 2 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            [2,  4096,  129280, "bfloat16", 64, 65535],  # FAST: Representative, multi-chunk
            [1,  4096,  129280, "float32",   64, 32768],  # FAST: FP32, different chunk size
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def deepseek_v3_full_config():
        """DeepSeek V3 (V=129,280) - 17 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            # Batch sweep T=4096
            [1,  4096,  129280, "bfloat16", 64, 65535],
            [4,  4096,  129280, "bfloat16", 64, 65535],
            # Batch sweep T=2048
            [2,  2048,  129280, "bfloat16", 64, 65535],
            [4,  2048,  129280, "bfloat16", 64, 65535],
            [8,  2048,  129280, "bfloat16", 64, 65535],
            # Seq sweep B=2
            [2,  512,   129280, "bfloat16", 64, 65535],
            [2,  1024,  129280, "bfloat16", 64, 65535],
            [2,  8192,  129280, "bfloat16", 64, 65535],
            # Seq sweep B=1
            [1,  2048,  129280, "bfloat16", 64, 65535],
            [1,  8192,  129280, "bfloat16", 64, 65535],
            [1,  16384, 129280, "bfloat16", 64, 65535],
            # Mixed patterns
            [8,  1024,  129280, "bfloat16", 64, 65535],
            [6,  2048,  129280, "bfloat16", 64, 65535],
            [3,  4096,  129280, "bfloat16", 64, 65535],
            # FP32
            [2,  4096,  129280, "float32",   64, 32768],
            [1,  8192,  129280, "float32",   64, 32768], # FP32 boundary test
            # Corner cases
            [1,  128,   129280, "bfloat16", 64, 65535], # Tiny: very small sequence
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def qwen3_8b_fast_config():
        """Qwen3-8B (V=151,936) - 2 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            [2,  4096,  151936, "bfloat16", 64, 65535],  # FAST: Representative, large vocab
            [1,  14000, 151936, "bfloat16", 64, 65535],  # FAST: Max positions stress test
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def qwen3_8b_full_config():
        """Qwen3-8B (V=151,936) - 14 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            # Batch sweep T=4096
            [1,  4096,  151936, "bfloat16", 64, 65535],
            [3,  4096,  151936, "bfloat16", 64, 65535],
            # Batch sweep T=2048
            [2,  2048,  151936, "bfloat16", 64, 65535],
            [4,  2048,  151936, "bfloat16", 64, 65535],
            [6,  2048,  151936, "bfloat16", 64, 65535],
            # Seq sweep B=2
            [2,  512,   151936, "bfloat16", 64, 65535],
            [2,  1024,  151936, "bfloat16", 64, 65535],
            # Seq sweep B=1
            [1,  2048,  151936, "bfloat16", 64, 65535],
            [1,  8192,  151936, "bfloat16", 64, 65535],
            [1,  12288, 151936, "bfloat16", 64, 65535],
            # Mixed patterns
            [4,  3072,  151936, "bfloat16", 64, 65535],
            # FP32
            [2,  2048,  151936, "float32",   64, 32768],
            # Max position tests
            [1,  13000, 151936, "bfloat16", 64, 65535],
            # Corner cases
            [1,  256,   151936, "bfloat16", 64, 65535],  # Small: minimal viable sequence
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def gpt_oss_20b_fast_config():
        """GPT-OSS 20B (V=201,088) - 1 fast test for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            [2,  4096,  201088, "bfloat16", 64, 65535],  # FAST: Representative, very large vocab
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def gpt_oss_20b_full_config():
        """GPT-OSS 20B (V=201,088) - 13 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            # Batch sweep T=4096
            [1,  4096,  201088, "bfloat16", 64, 65535],
            # Batch sweep T=2048
            [2,  2048,  201088, "bfloat16", 64, 65535],
            [4,  2048,  201088, "bfloat16", 64, 65535],
            [5,  2048,  201088, "bfloat16", 64, 65535],
            # Seq sweep B=2
            [2,  512,   201088, "bfloat16", 64, 65535],
            [2,  1024,  201088, "bfloat16", 64, 65535],
            [2,  5120,  201088, "bfloat16", 64, 65535],
            # Seq sweep B=1
            [1,  2048,  201088, "bfloat16", 64, 65535],
            [1,  8192,  201088, "bfloat16", 64, 65535],
            [1,  10240, 201088, "bfloat16", 64, 65535],
            # Mixed patterns
            [4,  2560,  201088, "bfloat16", 64, 65535],
            # FP32
            [1,  5120,  201088, "float32",  64, 32768],
            # Corner cases
            [3,  3413,  201088, "bfloat16", 64, 32768],  # Prime B×T: tests non-power-of-2 dimensions],
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def llama4_maverick_fast_config():
        """Llama 4 Maverick (V=202,408) - 2 fast tests for pre-commit."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            [2,  4096,  202408, "bfloat16", 64, 65535],  # FAST: Representative, very large vocab
            [5,  2047,  202408, "bfloat16", 64, 65535],  # FAST: Odd dimensions
        ]
        # fmt: on
        return _make_config(test_grid)

    @staticmethod
    def llama4_maverick_full_config():
        """Llama 4 Maverick (V=202,408) - 13 full sweep tests."""
        # fmt: off
        # Format: [B, T, V, dtype, PB, CS]
        test_grid = [
            # Batch sweep T=4096
            [1,  4096,  202408, "bfloat16", 64, 65535],
            # Batch sweep T=2048
            [2,  2048,  202408, "bfloat16", 64, 65535],
            [4,  2048,  202408, "bfloat16", 64, 65535],
            [5,  2048,  202408, "bfloat16", 64, 65535],
            # Seq sweep B=2
            [2,  512,   202408, "bfloat16", 64, 65535],
            [2,  1024,  202408, "bfloat16", 64, 65535],
            [2,  5120,  202408, "bfloat16", 64, 65535],
            # Seq sweep B=1
            [1,  2048,  202408, "bfloat16", 64, 65535],
            [1,  8192,  202408, "bfloat16", 64, 65535],
            [1,  10240, 202408, "bfloat16", 64, 65535],
            # Mixed patterns
            [4,  2560,  202408, "bfloat16", 64, 65535],
            # FP32
            [1,  4096,  202408, "float32",  64, 32768],
            [1,  5120,  202408, "float32",  64, 32768],
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
        """Switch Transformers (V=32,128) - 2 fast tests for pre-commit."""
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
        """Switch Transformers (V=32,128) - 22 full sweep tests."""
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
        """DeepSeek V3 (V=129,280) - 17 full sweep tests."""
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
        """Qwen3-8B (V=151,936) - 14 full sweep tests."""
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
        """GPT-OSS 20B (V=201,088) - 1 fast test for pre-commit."""
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
        """GPT-OSS 20B (V=201,088) - 13 full sweep tests."""
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
        """Llama 4 Maverick (V=202,408) - 13 full sweep tests."""
        compiler_args = CompilerArgs()
        self.prepare_test_parametrized(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
        )
