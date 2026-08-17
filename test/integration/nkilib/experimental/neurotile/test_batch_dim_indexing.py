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

"""Regression tests for batch dim indexing in _index_multi (kernel-dispatch tests).

CPU-only assertions live in test/unit/.../test_batch_dim_indexing.py.
This file holds the on-device kernel tests: 3D expert select with scalar
indirect, and KV cache load with scalar indirect on sequence dim.
"""

import ml_dtypes
import nki
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental import neurotile as nt

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@nki.jit
def _kernel_expert_select_3key(weights, expert_ids):
    """3D expert select: w_iter[eid, 0, f] with 3 keys."""
    _, P, F = weights.shape[0], weights.shape[1], weights.shape[2]
    T_F = 128
    N = expert_ids.shape[0]
    out = nl.ndarray((N, P, F), dtype=weights.dtype, buffer=nl.shared_hbm)

    w_iter = nt.tiles(weights, tile_size=(P, T_F))
    eid_iter = nt.tiles(expert_ids, tile_size=(1, 1))
    out_iter = nt.tiles(out, tile_size=(P, T_F))

    n_f = F // T_F
    for t in range(N):
        eid_tile = eid_iter[t, 0].load()
        for f in range(n_f):
            data = w_iter[eid_tile, 0, f].load()
            out_iter[t, 0, f].store(data.data)
    return out


@nki.jit
def _kernel_kv_cache_load(kv_cache, seq_offsets):
    """KV cache load with scalar indirect on dim 1."""
    B = kv_cache.shape[0]
    D = kv_cache.shape[2]
    out = nl.ndarray((B, 1, D), dtype=kv_cache.dtype, buffer=nl.shared_hbm)

    kv_iter = nt.tiles(kv_cache, tile_size=(1, D))
    seq_iter = nt.tiles(seq_offsets, tile_size=(1, 1))
    out_iter = nt.tiles(out, tile_size=(1, D))

    for b in range(B):
        seq_tile = seq_iter[b, 0].load()
        kv_data = kv_iter[b, seq_tile, 0].load()
        out_iter[b, 0, 0].store(kv_data.data)
    return out


# Constants matching the original test
_E, _P, _F, _N = 8, 128, 256, 4
_EID_LIST = [5, 2, 7, 0]
_B, _S, _D = 2, 8, 64
_POSITIONS = [3, 5]


def _expert_select_inputs(_):
    np.random.seed(42)
    weights = np.random.randn(_E, _P, _F).astype(ml_dtypes.bfloat16)
    expert_ids = np.array(_EID_LIST, dtype=np.int32).reshape(_N, 1)
    return {"weights": weights, "expert_ids": expert_ids}


def _expert_select_output(kernel_input):
    return {"out": np.zeros((_N, _P, _F), dtype=kernel_input["weights"].dtype)}


def _expert_select_ref(weights: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Out[t] = weights[expert_ids[t, 0]]."""
    out = torch.zeros((_N, _P, _F), dtype=weights.dtype)
    for t in range(_N):
        eid = int(expert_ids[t, 0].item())
        out[t] = weights[eid]
    return out


def _kv_cache_inputs(_):
    np.random.seed(43)
    cache = np.random.randn(_B, _S, _D).astype(ml_dtypes.bfloat16)
    positions = np.array([[p] for p in _POSITIONS], dtype=np.int32)
    return {"kv_cache": cache, "seq_offsets": positions}


def _kv_cache_output(kernel_input):
    return {"out": np.zeros((_B, 1, _D), dtype=kernel_input["kv_cache"].dtype)}


def _kv_cache_ref(kv_cache: torch.Tensor, seq_offsets: torch.Tensor) -> torch.Tensor:
    """Out[b, 0, :] = kv_cache[b, positions[b, 0], :]."""
    out = torch.zeros((_B, 1, _D), dtype=kv_cache.dtype)
    for b in range(_B):
        pos = int(seq_offsets[b, 0].item())
        out[b, 0] = kv_cache[b, pos]
    return out


@pytest_marks(["neurotile"])
class TestExpertSelectOnDevice:
    """3D expert weight select with scalar indirect."""

    @pytest.mark.fast
    def test_expert_select_correctness(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_expert_select_3key,
            torch_ref=torch_ref_wrapper(_expert_select_ref),
            kernel_input_generator=_expert_select_inputs,
            output_tensor_descriptor=_expert_select_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest_marks(["neurotile"])
class TestKVCacheOnDevice:
    """KV cache load with scalar indirect on sequence dim."""

    @pytest.mark.fast
    def test_kv_cache_load(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_kv_cache_load,
            torch_ref=torch_ref_wrapper(_kv_cache_ref),
            kernel_input_generator=_kv_cache_inputs,
            output_tensor_descriptor=_kv_cache_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
