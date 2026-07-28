# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.

"""Regression tests for scatter_add with duplicate indices within a tile.

The base scatter_add test (test_scatter_add.py) constructs indices as per-tile
permutations, so every 128-row tile has unique destination rows -- it never
exercises duplicates. Real LM embedding-gradient batches repeat rows constantly
(common tokens, BOS/EOS), which the tile-wide gather/scatter drops silently.
These tests pass unique_indices=False and feed sampled-with-replacement indices.
"""

import nki.language as nl
import numpy as np
import pytest

from nkilib_src.nkilib.experimental.misc.scatter_add import scatter_add
from nkilib_src.nkilib.experimental.misc.scatter_add_torch import scatter_add_torch_ref
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_dup_inputs(bs_slen, dim_size, src_rows, dtype):
    """Indices sampled WITH replacement -> duplicates within and across tiles."""
    rng = np.random.RandomState(42)
    return {
        "input.must_alias_input": rng.randn(bs_slen, dim_size).astype(dtype),
        "dim": 0,
        "index": rng.randint(0, bs_slen, size=src_rows).astype(np.int32),
        "src": rng.randn(src_rows, dim_size).astype(dtype),
        "unique_indices": False,
    }


def _output_tensors(kernel_input):
    return {"output": kernel_input["input.must_alias_input"]}


PARAM_NAMES = "bs_slen, dim_size, src_rows, dtype"
TEST_PARAMS = [
    (16, 512, 64, nl.float32),      # all dups inside one <128-row tile
    (16, 512, 64, nl.bfloat16),
    (10, 256, 200, nl.float32),     # dups crossing the 128-row tile boundary
    (64, 1024, 512, nl.float32),
]


@pytest_test_metadata(name="ScatterAddDupIndices")
@pytest_marks(["scatter_add"])
class TestScatterAddDupIndices:
    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_scatter_add_dup_indices(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs_slen,
        dim_size,
        src_rows,
        dtype,
    ):
        is_bf16 = dtype == nl.bfloat16
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=scatter_add,
            torch_ref=torch_ref_wrapper(scatter_add_torch_ref),
            kernel_input_generator=lambda _: _generate_dup_inputs(bs_slen, dim_size, src_rows, dtype),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                dump_after_lowering=False,
                logical_nc_config=2,
            ),
            atol=1e-1 if is_bf16 else 1e-4,
            rtol=1e-2 if is_bf16 else 1e-5,
        )
