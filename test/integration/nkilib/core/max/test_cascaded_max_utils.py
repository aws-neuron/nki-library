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
Test suite for predicated_folded_load and unfolded_store utilities using UnitTestFramework.
"""

import math
from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.max.cascaded_max_utils import predicated_folded_load, unfolded_store
from nkilib_src.nkilib.core.utils.kernel_helpers import get_program_sharding_info

FILL_VALUE = -9948.0


# ===================== Kernels =====================


@nki.jit
def folded_load_store_kernel(input_tensor, fold_factor: int, batch_start: int, batch_end: int):
    """Load input[batch_start:batch_end], store to output[0:b_range)."""
    _, n_prgs, prg_id = get_program_sharding_info()
    b, n = input_tensor.shape
    b_range = batch_end - batch_start
    output_tensor = nl.ndarray((b_range, n), dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    data_sb = predicated_folded_load(
        input_tensor,
        fold_factor=fold_factor,
        program_id=prg_id,
        n_programs=n_prgs,
        fill_value=FILL_VALUE,
        batch_start=batch_start,
        batch_end=batch_end,
    )
    unfolded_store(
        data_sb,
        output_tensor,
        fold_factor=fold_factor,
        program_id=prg_id,
        n_programs=n_prgs,
    )
    return output_tensor


@nki.jit
def folded_load_store_dst_kernel(
    input_tensor,
    fold_factor: int,
    src_batch_start: int,
    src_batch_end: int,
    dst_batch_start: int,
    dst_batch_end: int,
):
    """Load input[src_start:src_end], store to output via dst_batch_start/dst_batch_end."""
    _, n_prgs, prg_id = get_program_sharding_info()
    b, n = input_tensor.shape
    output_tensor = nl.ndarray((dst_batch_end, n), dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    data_sb = predicated_folded_load(
        input_tensor,
        fold_factor=fold_factor,
        program_id=prg_id,
        n_programs=n_prgs,
        fill_value=FILL_VALUE,
        batch_start=src_batch_start,
        batch_end=src_batch_end,
    )
    unfolded_store(
        data_sb,
        output_tensor,
        fold_factor=fold_factor,
        program_id=prg_id,
        n_programs=n_prgs,
        batch_start=src_batch_start,
        batch_end=src_batch_end,
        dst_batch_start=dst_batch_start,
        dst_batch_end=dst_batch_end,
    )
    return output_tensor


@nki.jit
def folded_load_oversized_sb_kernel(
    input_tensor,
    fold_factor: int,
    batch_start: int,
    batch_end: int,
    sb_extra_cols: int,
):
    """Load into oversized SBUF, read back full SBUF to verify extra cols stay FILL_VALUE."""
    _, n_prgs, prg_id = get_program_sharding_info()
    b, n = input_tensor.shape
    n_folded = math.ceil(n / fold_factor)
    b_range = batch_end - batch_start
    batch_size_sharded = (b_range + n_prgs - 1) // n_prgs
    sb_rows = batch_size_sharded * fold_factor
    sb_cols = n_folded + sb_extra_cols

    data_sb = nl.ndarray((sb_rows, sb_cols), dtype=input_tensor.dtype, buffer=nl.sbuf)
    predicated_folded_load(
        input_tensor,
        fold_factor=fold_factor,
        program_id=prg_id,
        n_programs=n_prgs,
        fill_value=FILL_VALUE,
        data_sb=data_sb,
        batch_start=batch_start,
        batch_end=batch_end,
    )

    output_tensor = nl.ndarray((sb_rows * n_prgs, sb_cols), dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    rows_bound = min(b_range - prg_id * batch_size_sharded, batch_size_sharded) * fold_factor
    ix_dst = nl.ds(prg_id * sb_rows, rows_bound)
    ix_src = nl.ds(0, rows_bound)
    nisa.dma_copy(dst=output_tensor[ix_dst, :], src=data_sb[ix_src, :])
    return output_tensor


# ===================== Torch References =====================


def folded_load_store_torch_ref(input_tensor: torch.Tensor, fold_factor: int, batch_start: int, batch_end: int) -> dict:
    """Torch reference for folded_load_store_kernel."""
    b_range = batch_end - batch_start
    return {"output_tensor": input_tensor[batch_start:batch_end].clone()}


def folded_load_store_dst_torch_ref(
    input_tensor: torch.Tensor,
    fold_factor: int,
    src_batch_start: int,
    src_batch_end: int,
    dst_batch_start: int,
    dst_batch_end: int,
) -> dict:
    """Torch reference for folded_load_store_dst_kernel."""
    n = input_tensor.shape[1]
    output = torch.zeros((dst_batch_end, n), dtype=input_tensor.dtype)
    output[dst_batch_start:dst_batch_end] = input_tensor[src_batch_start:src_batch_end]
    return {"output_tensor": output}


def folded_load_oversized_sb_torch_ref(
    input_tensor: torch.Tensor,
    fold_factor: int,
    batch_start: int,
    batch_end: int,
    sb_extra_cols: int,
) -> dict:
    """Torch reference for folded_load_oversized_sb_kernel."""
    b, n = input_tensor.shape
    b_range = batch_end - batch_start
    n_folded = math.ceil(n / fold_factor)
    # Simulate single program for reference
    n_programs = 1
    batch_size_sharded = b_range
    sb_rows = batch_size_sharded * fold_factor
    sb_cols = n_folded + sb_extra_cols

    output = torch.full((sb_rows * n_programs, sb_cols), FILL_VALUE, dtype=input_tensor.dtype)
    for i in range(batch_size_sharded):
        if batch_start + i < b:
            row = input_tensor[batch_start + i]
            for f in range(fold_factor):
                cs = f * n_folded
                ce = min(cs + n_folded, n)
                if cs < n:
                    output[i * fold_factor + f, : ce - cs] = row[cs:ce]
    return {"output_tensor": output}


@pytest_test_metadata(name="Cascaded Max Utils", pytest_marks=["max", "utils"])
class TestCascadedMaxUtils:
    # fmt: off
    full_batch_params = "lnc_degree, b, n, fold_factor"
    full_batch_perms = [
        [1, 8, 256, 1], [2, 8, 256, 2], [1, 4, 1024, 4], [2, 16, 512, 8],
        [1, 1, 3168, 4], [2, 8, 4058, 8], [1, 4, 3999, 16], [1, 1, 25600, 32],
        [1, 1, 128, 1], [2, 2, 256, 128],
        [2, 3, 256, 2], [2, 5, 512, 4], [2, 7, 3168, 4], [2, 3, 4058, 8],
    ]
    # fmt: on

    @staticmethod
    def generate_full_batch_inputs(b, n, fold_factor, batch_start=0, batch_end=None):
        np.random.seed(42)
        if batch_end is None:
            batch_end = b
        return {
            "input_tensor": np.random.randn(b, n).astype(np.float32),
            "fold_factor": fold_factor,
            "batch_start": batch_start,
            "batch_end": batch_end,
        }

    @staticmethod
    def output_full_batch(kernel_input):
        b_range = kernel_input["batch_end"] - kernel_input["batch_start"]
        n = kernel_input["input_tensor"].shape[1]
        return {"output_tensor": np.zeros((b_range, n), dtype=np.float32)}

    @pytest.mark.fast
    @pytest.mark.parametrize(full_batch_params, full_batch_perms)
    def test_folded_load_store_full_batch(
        self, test_manager: Orchestrator, lnc_degree: int, b: int, n: int, fold_factor: int
    ):
        def input_generator(test_config, input_tensor_def=None):
            return self.generate_full_batch_inputs(b, n, fold_factor, 0, b)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=folded_load_store_kernel,
            torch_ref=torch_ref_wrapper(folded_load_store_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_full_batch,
        )
        framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=lnc_degree))

    # fmt: off
    partial_batch_params = "lnc_degree, b, n, fold_factor, batch_start, batch_end"
    partial_batch_perms = [
        [1, 16, 256, 2, 0, 8], [1, 16, 256, 2, 4, 12], [1, 16, 256, 2, 8, 16],
        [2, 32, 512, 4, 0, 16], [2, 32, 512, 4, 16, 32],
        [1, 16, 3168, 4, 0, 8], [1, 16, 3168, 4, 4, 12],
        [2, 32, 4058, 8, 0, 16], [2, 32, 4058, 8, 8, 24],
        [1, 8, 256, 2, 3, 4], [1, 8, 3168, 4, 5, 6],
        [2, 16, 256, 2, 0, 7], [2, 32, 3168, 4, 4, 15],
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(partial_batch_params, partial_batch_perms)
    def test_folded_load_store_partial_batch(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        b: int,
        n: int,
        fold_factor: int,
        batch_start: int,
        batch_end: int,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return self.generate_full_batch_inputs(b, n, fold_factor, batch_start, batch_end)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=folded_load_store_kernel,
            torch_ref=torch_ref_wrapper(folded_load_store_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_full_batch,
        )
        framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=lnc_degree))

    # fmt: off
    dst_batch_params = "lnc_degree, b, n, fold_factor, src_start, src_end, dst_start, dst_end"
    dst_batch_perms = [
        [1, 16, 256, 2, 0, 8, 0, 8], [1, 16, 256, 2, 0, 8, 8, 16], [2, 16, 512, 4, 0, 8, 8, 16],
        [1, 16, 3168, 4, 0, 8, 8, 16], [2, 32, 4058, 8, 0, 8, 16, 24],
        [1, 16, 256, 2, 4, 8, 0, 4], [1, 16, 3168, 4, 8, 12, 0, 4],
    ]
    # fmt: on

    @staticmethod
    def generate_dst_batch_inputs(b, n, fold_factor, src_start, src_end, dst_start, dst_end):
        np.random.seed(42)
        return {
            "input_tensor": np.random.randn(b, n).astype(np.float32),
            "fold_factor": fold_factor,
            "src_batch_start": src_start,
            "src_batch_end": src_end,
            "dst_batch_start": dst_start,
            "dst_batch_end": dst_end,
        }

    @staticmethod
    def output_dst_batch(kernel_input):
        dst_end = kernel_input["dst_batch_end"]
        n = kernel_input["input_tensor"].shape[1]
        return {"output_tensor": np.zeros((dst_end, n), dtype=np.float32)}

    @pytest.mark.fast
    @pytest.mark.parametrize(dst_batch_params, dst_batch_perms)
    def test_folded_load_store_dst_batch(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        b: int,
        n: int,
        fold_factor: int,
        src_start: int,
        src_end: int,
        dst_start: int,
        dst_end: int,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return self.generate_dst_batch_inputs(b, n, fold_factor, src_start, src_end, dst_start, dst_end)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=folded_load_store_dst_kernel,
            torch_ref=torch_ref_wrapper(folded_load_store_dst_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_dst_batch,
        )
        framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=lnc_degree))

    # fmt: off
    oversized_sb_params = "lnc_degree, b, n, fold_factor, batch_start, batch_end, sb_extra_cols"
    oversized_sb_perms = [
        [1, 8, 256, 2, 0, 8, 16], [2, 8, 512, 4, 0, 8, 32],
        [1, 4, 3168, 4, 0, 4, 8], [2, 8, 4058, 8, 0, 8, 16],
        [1, 16, 256, 2, 4, 12, 10], [2, 32, 3168, 4, 8, 24, 20],
        [2, 5, 512, 4, 0, 5, 12],
    ]
    # fmt: on

    @staticmethod
    def generate_oversized_inputs(b, n, fold_factor, batch_start, batch_end, sb_extra_cols):
        np.random.seed(42)
        return {
            "input_tensor": np.random.randn(b, n).astype(np.float32),
            "fold_factor": fold_factor,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "sb_extra_cols": sb_extra_cols,
        }

    @staticmethod
    def output_oversized(kernel_input):
        b_range = kernel_input["batch_end"] - kernel_input["batch_start"]
        n = kernel_input["input_tensor"].shape[1]
        fold_factor = kernel_input["fold_factor"]
        sb_extra_cols = kernel_input["sb_extra_cols"]
        n_folded = math.ceil(n / fold_factor)
        # Assume single program for output shape
        batch_size_sharded = b_range
        sb_rows = batch_size_sharded * fold_factor
        sb_cols = n_folded + sb_extra_cols
        return {"output_tensor": np.zeros((sb_rows, sb_cols), dtype=np.float32)}

    @pytest.mark.fast
    @pytest.mark.parametrize(oversized_sb_params, oversized_sb_perms)
    def test_folded_load_oversized_sb(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        b: int,
        n: int,
        fold_factor: int,
        batch_start: int,
        batch_end: int,
        sb_extra_cols: int,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return self.generate_oversized_inputs(b, n, fold_factor, batch_start, batch_end, sb_extra_cols)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=folded_load_oversized_sb_kernel,
            torch_ref=torch_ref_wrapper(folded_load_oversized_sb_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_oversized,
        )
        framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=lnc_degree))
