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

"""Integration tests for indexed_flatten kernel."""

import numpy as np
import pytest
from nkilib_src.nkilib.core.subkernels.indexed_flatten import indexed_flatten
from nkilib_src.nkilib.core.subkernels.indexed_flatten_torch import indexed_flatten_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_input_tensor(T: int, E: int) -> np.ndarray:
    """Generate deterministic input tensor with sequential int32 values.

    Args:
        T (int): Number of elements per row
        E (int): Number of rows

    Returns:
        np.ndarray: [E, T] input tensor with sequential int32 values
    """
    return np.arange(E * T, dtype=np.int32).reshape(E, T)


def _output_tensor_descriptor(kernel_input):
    """Generate output tensor descriptor from kernel inputs."""
    return {"flattened_array": np.zeros(kernel_input["output_len"], dtype=np.int32)}


# fmt: off
FAST_PARAM_NAMES = \
    "T, E, f_len"
FAST_TEST_PARAMS = [
    pytest.param(4096,  1,   128, marks=pytest.mark.fast),
    (4096,  2,   128),
    (4096,  2,   256),
    (4096,  3,   128),
    (10240, 16,  128),
    (10240, 17,  128),
    (65536, 4,   128),
    (65536, 4,   256),
    (4096,  64,  128),
    (4096,  128, 128),
    (4096,  256, 128),
]

ROW_OFFSET_PARAM_NAMES = \
    "T, E, f_len, N, row_offsets_start"
ROW_OFFSET_TEST_PARAMS = [
    (4096,  2, 128, 4,  2),
    (4096,  3, 128, 5,  1),
    (10240, 8, 128, 16, 8),
]

SPARSE_ROUTING_PARAM_NAMES = \
    "T, E, f_len, row_offsets_start, tokens_per_expert"
SPARSE_ROUTING_TEST_PARAMS = [
    # Empty expert: cross-NC race (expert 1 on NC0, expert 2 on NC1 share offset).
    pytest.param(256,  4, 16, 0,  [16, 0, 16, 16], marks=pytest.mark.fast),
    # Empty expert: with row_offsets_start.
    (256,  4, 16, 8,  [16, 0, 16, 16]),
    # Multiple empty experts: cross-NC overlap.
    (256,  4, 16, 0,  [0, 0, 16, 16]),
    # Padding spill: T/f_len=32 but only 2-8 blocks allocated per expert.
    (4096, 4, 128, 0,  [470, 252, 359, 775]),
    # Padding spill: with row_offsets_start.
    (4096, 4, 128, 8,  [470, 252, 359, 775]),
    # Padding spill: minimal tokens — maximum spill.
    (4096, 4, 128, 0,  [1, 1, 1, 1]),
    # Mixed: empty expert + padding spill.
    (4096, 4, 128, 0,  [128, 0, 200, 500]),
]

BOUNDARY_STRADDLE_PARAM_NAMES = \
    "T, E, f_len, tokens_per_expert"
BOUNDARY_STRADDLE_TEST_PARAMS = [
    # TIGHT output_len (no padding past the last owned block) so the last populated
    # expert's fused partitions_per_row-block write straddles the buffer end. Pre-fix the
    # LNC2 oob_mode.skip dropped the ENTIRE straddling tile, losing that expert's in-bounds
    # blocks; the kernel now over-allocates internally so the in-bounds blocks survive.
    # partitions_per_row = T//f_len = 16 here; earlier experts fit, only the last straddles.
    (2048, 3, 128, [2048, 2048, 200]),
    (2048, 4, 128, [2048, 2048, 2048, 128]),
    (4096, 2, 128, [4096, 256]),
]
# fmt: on


@pytest_test_metadata(name="IndexedFlatten")
@pytest_marks(["indexed_flatten"])
class TestIndexedFlattenKernel:
    """Test class for indexed_flatten kernel."""

    def _run_test(self, test_manager: Orchestrator, platform_target: Platforms, input_generator):
        """Run indexed_flatten test with the given input generator."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=indexed_flatten,
            torch_ref=torch_ref_wrapper(indexed_flatten_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            atol=0,
            rtol=0,
        )

    @pytest_parametrize(FAST_PARAM_NAMES, FAST_TEST_PARAMS)
    def test_indexed_flatten_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        T: int,
        E: int,
        f_len: int,
    ):
        """Fast compile-only tests with minimal coverage."""

        def input_generator(test_config):
            partitions_per_row = T // f_len
            row_offsets_list = [partitions_per_row * idx for idx in range(E)]
            output_len = E * T + 1024
            output_len = ((output_len + 127) // 128) * 128
            return {
                "input_tensor": _generate_input_tensor(T, E),
                "f_len": f_len,
                "output_len": output_len,
                "row_offsets": np.array(row_offsets_list, dtype=np.int32),
                "row_offsets_start": None,
                "padding_val": -1,
            }

        self._run_test(test_manager, platform_target, input_generator)

    @pytest_parametrize(ROW_OFFSET_PARAM_NAMES, ROW_OFFSET_TEST_PARAMS)
    def test_indexed_flatten_row_offsets_start(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        T: int,
        E: int,
        f_len: int,
        N: int,
        row_offsets_start: int,
    ):
        """Test row_offsets_start parameter for tensor-parallel MoE use cases."""

        def input_generator(test_config):
            partitions_per_row = T // f_len
            row_offsets_list = [partitions_per_row * idx for idx in range(N)]
            output_len = N * T + 1024
            output_len = ((output_len + 127) // 128) * 128
            return {
                "input_tensor": _generate_input_tensor(T, E),
                "f_len": f_len,
                "output_len": output_len,
                "row_offsets": np.array(row_offsets_list, dtype=np.int32),
                "row_offsets_start": np.array([row_offsets_start], dtype=np.int32),
                "padding_val": -1,
            }

        self._run_test(test_manager, platform_target, input_generator)

    @pytest_parametrize(SPARSE_ROUTING_PARAM_NAMES, SPARSE_ROUTING_TEST_PARAMS)
    def test_indexed_flatten_sparse_token_routing(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        T: int,
        E: int,
        f_len: int,
        row_offsets_start: int,
        tokens_per_expert: list,
    ):
        """Test cross-NC correctness with empty experts and padding spill.

        Covers two scenarios that cause cross-NC write conflicts with shared HBM:
        1. Empty experts: zero-token experts produce duplicate row_offsets (prefix
           sum doesn't advance), so two experts target the same output region.
        2. Padding spill: each expert writes T/f_len blocks but only owns
           ceil(tokens/f_len) blocks. Padding tail spills into neighboring
           experts' regions on the other NC.
        Both rely on max(-1, real_value) = real_value for correctness.
        """
        np.random.seed(42)

        def input_generator(test_config):
            partitions_per_row = T // f_len
            N = row_offsets_start + E

            # Build row_offsets as prefix sum of ceil(tokens/f_len) per expert
            blocks_per_expert = [(t + f_len - 1) // f_len for t in tokens_per_expert]
            row_offsets_list = []
            block_pos = 0
            for i in range(N):
                row_offsets_list.append(block_pos)
                local_idx = i - row_offsets_start
                if 0 <= local_idx < E:
                    block_pos += blocks_per_expert[local_idx]
                else:
                    block_pos += partitions_per_row

            # Build input: real token IDs packed at front, padding at tail
            input_tensor = np.full((E, T), -1, dtype=np.int32)
            for e in range(E):
                n_tokens = tokens_per_expert[e]
                if n_tokens > 0:
                    input_tensor[e, :n_tokens] = np.random.permutation(T)[:n_tokens]

            output_len = (block_pos + partitions_per_row) * f_len
            output_len = ((output_len + 127) // 128) * 128
            return {
                "input_tensor": input_tensor,
                "f_len": f_len,
                "output_len": output_len,
                "row_offsets": np.array(row_offsets_list, dtype=np.int32),
                "row_offsets_start": np.array([row_offsets_start], dtype=np.int32),
                "padding_val": -1,
            }

        self._run_test(test_manager, platform_target, input_generator)

    @pytest.mark.fast
    @pytest_parametrize(BOUNDARY_STRADDLE_PARAM_NAMES, BOUNDARY_STRADDLE_TEST_PARAMS)
    def test_indexed_flatten_boundary_straddle(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        T: int,
        E: int,
        f_len: int,
        tokens_per_expert: list,
    ):
        """Last populated expert's fused write straddles a TIGHT output_len boundary.

        Each expert writes partitions_per_row = T//f_len blocks at its row_offset, but the
        last expert only owns ceil(tokens/f_len) < partitions_per_row blocks, so its fused
        write spans past num_output_blocks. With output_len set tight (no trailing padding),
        the LNC2 oob_mode.skip previously dropped the ENTIRE straddling tile, losing that
        expert's in-bounds blocks (left as padding_val) and mismatching the torch reference
        (which writes each in-bounds block independently). The kernel now over-allocates its
        private buffer by partitions_per_row blocks so the in-bounds blocks survive. The
        atol=rtol=0 exact comparison catches any dropped block.
        """
        np.random.seed(42)

        def input_generator(test_config):
            blocks_per_expert = [(t + f_len - 1) // f_len for t in tokens_per_expert]
            row_offsets_list = []
            block_pos = 0
            for i in range(E):
                row_offsets_list.append(block_pos)
                block_pos += blocks_per_expert[i]

            # Real token IDs packed at the front of each row, padding (-1) at the tail.
            input_tensor = np.full((E, T), -1, dtype=np.int32)
            for e in range(E):
                n_tokens = tokens_per_expert[e]
                if n_tokens > 0:
                    input_tensor[e, :n_tokens] = np.random.permutation(T)[:n_tokens]

            # TIGHT output_len: no trailing partitions_per_row padding, so the last expert's
            # fused write straddles the buffer end (this is what triggers the boundary bug).
            output_len = block_pos * f_len
            output_len = ((output_len + 127) // 128) * 128
            return {
                "input_tensor": input_tensor,
                "f_len": f_len,
                "output_len": output_len,
                "row_offsets": np.array(row_offsets_list, dtype=np.int32),
                "row_offsets_start": None,
                "padding_val": -1,
            }

        self._run_test(test_manager, platform_target, input_generator)
