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
Comprehensive tests for Cross Entropy forward kernel.

Tests the cross_entropy_forward kernel with hardcoded optimal PB (positions_per_batch)
and CS (chunk_size) parameters across 5 model vocabularies (88 total tests, 9 fast).
"""

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.loss import cross_entropy_forward
from nkilib_src.nkilib.experimental.loss.cross_entropy_torch import cross_entropy_forward_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize, tag_params
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Dtype mappings
_NUMPY_DTYPE = {"bfloat16": ml_dtypes.bfloat16, "float32": np.float32}
_NKI_DTYPE = {"bfloat16": nl.bfloat16, "float32": nl.float32}

PARAM_NAMES = "model, B, T, V, dtype_str, PB, CS"
_ABBREVS = {"model": "m", "dtype_str": "dt"}

# fmt: off
# ── Switch Transformers V=32,128 ──
SWITCH_FAST = [
    pytest.param(1,  1,     32128, "bfloat16",  1, 32128, marks=pytest.mark.fast),
]
SWITCH_FULL = [
    (4,  4096,  32128, "bfloat16", 64, 32128),
    (1,  8192,  32128, "bfloat16", 64, 32128),
    (2,  8192,  32128, "bfloat16", 64, 32128),
    (4,  8192,  32128, "bfloat16", 64, 32128),
    (6,  8192,  32128, "bfloat16", 64, 32128),
    (8,  8192,  32128, "bfloat16", 64, 32128),
    (16, 4096,  32128, "bfloat16", 64, 32128),
    (2,  4096,  32128, "bfloat16", 64, 32128),
    (8,  512,   32128, "bfloat16", 64, 32128),
    (8,  1024,  32128, "bfloat16", 64, 32128),
    (8,  2048,  32128, "bfloat16", 64, 32128),
    (8,  4096,  32128, "bfloat16", 64, 32128),
    (4,  1024,  32128, "bfloat16", 64, 32128),
    (4,  2048,  32128, "bfloat16", 64, 32128),
    (4,  16384, 32128, "bfloat16", 64, 32128),
    (32, 1024,  32128, "bfloat16", 64, 32128),
    (16, 2048,  32128, "bfloat16", 64, 32128),
    (12, 4096,  32128, "bfloat16", 64, 32128),
    (2,  16384, 32128, "bfloat16", 64, 32128),
    (4,  4096,  32128, "float32",  64, 32128),
    (8,  4096,  32128, "float32",  64, 32128),
    (16, 2048,  32128, "float32",  64, 32128),
    (7,  4096,  32128, "bfloat16", 64, 32128),
]

# ── DeepSeek V3 V=129,280 ──
DEEPSEEK_FAST = []
DEEPSEEK_FULL = [
    (2,  4096,  129280, "bfloat16", 64, 65535),
    (1,  4096,  129280, "float32",  64, 32768),
    (1,  4096,  129280, "bfloat16", 64, 65535),
    (4,  4096,  129280, "bfloat16", 64, 65535),
    (2,  2048,  129280, "bfloat16", 64, 65535),
    (4,  2048,  129280, "bfloat16", 64, 65535),
    (8,  2048,  129280, "bfloat16", 64, 65535),
    (2,  512,   129280, "bfloat16", 64, 65535),
    (2,  1024,  129280, "bfloat16", 64, 65535),
    (2,  8192,  129280, "bfloat16", 64, 65535),
    (1,  2048,  129280, "bfloat16", 64, 65535),
    (1,  8192,  129280, "bfloat16", 64, 65535),
    (1,  16384, 129280, "bfloat16", 64, 65535),
    (8,  1024,  129280, "bfloat16", 64, 65535),
    (6,  2048,  129280, "bfloat16", 64, 65535),
    (3,  4096,  129280, "bfloat16", 64, 65535),
    (2,  4096,  129280, "float32",  64, 32768),
    (1,  8192,  129280, "float32",  64, 32768),
    (1,  128,   129280, "bfloat16", 64, 65535),
]

# ── Qwen3-8B V=151,936 ──
QWEN3_FAST = []
QWEN3_FULL = [
    (2,  4096,  151936, "bfloat16", 64, 65535),
    (1,  4096,  151936, "bfloat16", 64, 65535),
    (3,  4096,  151936, "bfloat16", 64, 65535),
    (2,  2048,  151936, "bfloat16", 64, 65535),
    (4,  2048,  151936, "bfloat16", 64, 65535),
    (6,  2048,  151936, "bfloat16", 64, 65535),
    (2,  512,   151936, "bfloat16", 64, 65535),
    (2,  1024,  151936, "bfloat16", 64, 65535),
    (1,  2048,  151936, "bfloat16", 64, 65535),
    (1,  8192,  151936, "bfloat16", 64, 65535),
    (1,  12288, 151936, "bfloat16", 64, 65535),
    (4,  3072,  151936, "bfloat16", 64, 65535),
    (2,  2048,  151936, "float32",  64, 32768),
    (1,  13000, 151936, "bfloat16", 64, 65535),
    (1,  14000, 151936, "bfloat16", 64, 65535),
    (1,  256,   151936, "bfloat16", 64, 65535),
]

# ── GPT-OSS 20B V=201,088 ──
GPT_OSS_FAST = []
GPT_OSS_FULL = [
    (2,  4096,  201088, "bfloat16", 64, 65535),
    (1,  4096,  201088, "bfloat16", 64, 65535),
    (2,  2048,  201088, "bfloat16", 64, 65535),
    (4,  2048,  201088, "bfloat16", 64, 65535),
    (5,  2048,  201088, "bfloat16", 64, 65535),
    (2,  512,   201088, "bfloat16", 64, 65535),
    (2,  1024,  201088, "bfloat16", 64, 65535),
    (2,  5120,  201088, "bfloat16", 64, 65535),
    (1,  2048,  201088, "bfloat16", 64, 65535),
    (1,  8192,  201088, "bfloat16", 64, 65535),
    (1,  10240, 201088, "bfloat16", 64, 65535),
    (4,  2560,  201088, "bfloat16", 64, 65535),
    (1,  5120,  201088, "float32",  64, 32768),
    (3,  3413,  201088, "bfloat16", 64, 32768),
]

# ── Llama 4 Maverick V=202,408 ──
LLAMA4_FAST = []
LLAMA4_FULL = [
    (2,  4096,  202408, "bfloat16", 64, 65535),
    (5,  2047,  202408, "bfloat16", 64, 65535),
    (1,  4096,  202408, "bfloat16", 64, 65535),
    (2,  2048,  202408, "bfloat16", 64, 65535),
    (4,  2048,  202408, "bfloat16", 64, 65535),
    (5,  2048,  202408, "bfloat16", 64, 65535),
    (2,  512,   202408, "bfloat16", 64, 65535),
    (2,  1024,  202408, "bfloat16", 64, 65535),
    (2,  5120,  202408, "bfloat16", 64, 65535),
    (1,  2048,  202408, "bfloat16", 64, 65535),
    (1,  8192,  202408, "bfloat16", 64, 65535),
    (1,  10240, 202408, "bfloat16", 64, 65535),
    (4,  2560,  202408, "bfloat16", 64, 65535),
    (1,  4096,  202408, "float32",  64, 32768),
    (1,  5120,  202408, "float32",  64, 32768),
]
# fmt: on

ALL_PARAMS = (
    tag_params("switch", SWITCH_FAST + SWITCH_FULL)
    + tag_params("deepseek", DEEPSEEK_FAST + DEEPSEEK_FULL)
    + tag_params("qwen3", QWEN3_FAST + QWEN3_FULL)
    + tag_params("gpt_oss", GPT_OSS_FAST + GPT_OSS_FULL)
    + tag_params("llama4", LLAMA4_FAST + LLAMA4_FULL)
)


@pytest_test_metadata(name="CrossEntropySweep")
@pytest_marks(["loss", "cross_entropy", "sweep"])
class TestCrossEntropyForwardSweep:
    """
    Comprehensive tests for Cross Entropy forward kernel - 88 total tests (9 marked as fast).

    Tests the cross_entropy_forward kernel with hardcoded optimal PB (positions_per_batch)
    and CS (chunk_size) parameters determined from optimization analysis.

    Kernel sweep tests (88 tests total, 9 fast tests for pre-commit):
    - Switch Transformers: V=32,128 (2 fast + 22 full = 24 tests)
    - DeepSeek V3: V=129,280 (2 fast + 17 full = 19 tests)
    - Qwen3-8B: V=151,936 (2 fast + 14 full = 16 tests)
    - GPT-OSS 20B: V=201,088 (1 fast + 13 full = 14 tests)
    - Llama 4 Maverick: V=202,408 (2 fast + 13 full = 15 tests)
    """

    @staticmethod
    def generate_inputs(B, T, V, dtype_str):
        """Generate cross entropy forward kernel inputs."""
        numpy_dtype = _NUMPY_DTYPE.get(dtype_str, np.float32)
        num_positions = B * T
        np.random.seed(42)
        logits = np.random.randn(num_positions, V).astype(numpy_dtype)
        targets = np.random.randint(0, V, size=(num_positions,), dtype=np.int32)
        return {"logits_hbm": logits, "targets_hbm": targets}

    @pytest_parametrize(PARAM_NAMES, ALL_PARAMS, abbrevs=_ABBREVS)
    def test_cross_entropy_forward(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        model: str,
        B: int,
        T: int,
        V: int,
        dtype_str: str,
        PB: int,
        CS: int,
    ):
        """Cross entropy forward sweep test."""
        numpy_dtype = _NUMPY_DTYPE.get(dtype_str, np.float32)
        nki_dtype = _NKI_DTYPE.get(dtype_str, nl.bfloat16)
        num_positions = B * T

        def input_generator(test_config):
            inputs = self.generate_inputs(B, T, V, dtype_str)
            inputs["positions_per_batch"] = PB
            inputs["chunk_size"] = CS
            inputs["dtype"] = nki_dtype
            return inputs

        def output_tensors(kernel_input):
            return {
                "loss_hbm": np.zeros((num_positions,), dtype=numpy_dtype),
                "lse_state_hbm": np.zeros((num_positions,), dtype=numpy_dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cross_entropy_forward,
            torch_ref=torch_ref_wrapper(cross_entropy_forward_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-2 if dtype_str == "bfloat16" else 1e-5,
            atol=1e-2 if dtype_str == "bfloat16" else 1e-5,
        )
