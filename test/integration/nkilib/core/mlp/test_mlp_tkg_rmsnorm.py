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

"""Direct branch-coverage tests for the ``rmsnorm_tkg`` helper in mlp_tkg_rmsnorm.

``rmsnorm_tkg`` is a plain helper (not ``@nki.jit``) that writes its result into a
caller-provided SBUF output tensor and dispatches between a ``_th`` layout
([H0, H1, T], used when T >= 128) and a ``_ht`` layout ([H0, T, H1], used when the
input is already in SBUF or when T < 128). Production callers only exercise it at
hidden=8192/LNC2, which is skipped under CPU simulation, so its small-shape /
LNC1 / layout-permutation branches are otherwise uncovered.

This file drives the helper directly through a thin ``@nki.jit`` wrapper that
loads/allocates the buffers, calls ``rmsnorm_tkg`` (letting it build its own
``SbufManager`` so the ``sbm is None`` branch is covered), and DMAs the per-core
SBUF output slice back to HBM for numerical validation against a torch RMSNorm
reference.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.mlp.mlp_tkg.mlp_tkg_rmsnorm import rmsnorm_tkg

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_EPS = 1e-6
_H0 = 128


# ===================== Kernel wrapper =====================


@nki.jit
def rmsnorm_tkg_wrapper(input, gamma, hidden, out_ndim, in_sbuf, hidden_dim_tp, shard_output):
    """Thin wrapper that drives ``rmsnorm_tkg`` directly.

    Args:
        input: [B, S, H] or [T, H] hidden states in HBM.
        gamma: [1, H] RMSNorm weight in HBM.
        hidden: H (compile-time int).
        out_ndim: 2 or 3 - shape rank of the SBUF output tensor handed to the helper.
        in_sbuf: if True, pre-load the input into SBUF ([H0, T, H1]) so the helper
            takes its SBUF-input (always ``_ht``) path.
        hidden_dim_tp: forwarded to the helper (TP-sharded hidden-dim layout).
        shard_output: if True, the output covers only this core's H-shard
            (shard_H1 = H1 / n_prgs), forcing the helper's shard-on-h path.
    """
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    H1 = hidden // _H0

    # T = product of all leading (non-hidden) dims.
    T = 1
    for d in input.shape[:-1]:
        T = T * d

    shard_H1 = H1 // n_prgs if shard_output else H1

    # Helper dispatch mirror: SBUF input -> _ht; else T >= 128 -> _th, else _ht.
    is_th = (not in_sbuf) and (T >= 128)

    # Allocate the caller-owned SBUF output tensor in the shape/rank the helper expects.
    if is_th:
        if out_ndim == 2:
            output = nl.ndarray((_H0, shard_H1 * T), dtype=input.dtype, buffer=nl.sbuf)
        else:
            output = nl.ndarray((_H0, shard_H1, T), dtype=input.dtype, buffer=nl.sbuf)
    else:
        if out_ndim == 2:
            output = nl.ndarray((_H0, T * shard_H1), dtype=input.dtype, buffer=nl.sbuf)
        else:
            output = nl.ndarray((_H0, T, shard_H1), dtype=input.dtype, buffer=nl.sbuf)

    if in_sbuf:
        # Pre-load full-H input into [H0, T, H1] SBUF (non-TP layout: h = h0 * H1 + h1).
        input_flat = input.flatten_dims(start_dim=0, end_dim=1) if len(input.shape) == 3 else input
        input_sb = nl.ndarray((_H0, T, H1), dtype=input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=input_sb, src=input_flat.reshape_dim(dim=1, shape=[_H0, H1]).permute(dims=[1, 0, 2]))
        helper_input = input_sb
    else:
        helper_input = input

    # Call helper without an sbm so the ``sbm is None`` branch builds one internally.
    out_sb, _out_layout = rmsnorm_tkg(
        input=helper_input,
        gamma=gamma,
        output=output,
        hidden_scale=1.0 / hidden,
        eps=_EPS,
        hidden_dim_tp=hidden_dim_tp,
    )

    # Assemble the (possibly per-core) result into a full-H HBM tensor for validation.
    if is_th:
        out = nl.ndarray((_H0, H1, T), dtype=input.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out[:, prg_id * shard_H1 : (prg_id + 1) * shard_H1, :], src=out_sb)
    else:
        out = nl.ndarray((_H0, T, H1), dtype=input.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out[:, :, prg_id * shard_H1 : (prg_id + 1) * shard_H1], src=out_sb)
    return out


# ===================== Torch reference =====================


def rmsnorm_tkg_ref(input, gamma, hidden, out_ndim, in_sbuf, hidden_dim_tp, shard_output):
    """Torch RMSNorm reference producing the assembled full-H output layout.

    ``input``/``gamma`` arrive as float32 torch tensors (converted by
    ``torch_ref_wrapper``); the remaining args are plain scalars.
    """
    x = input.reshape(-1, hidden).to(torch.float32)
    g = gamma.reshape(1, hidden).to(torch.float32)
    T = x.shape[0]
    H1 = hidden // _H0

    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + _EPS)
    normed = x * inv_rms * g  # [T, H]

    is_th = (not in_sbuf) and (T >= 128)
    if is_th:
        if hidden_dim_tp:
            # h = h1 * H0 + h0 -> [H0, H1, T]
            out = normed.reshape(T, H1, _H0).permute(2, 1, 0)
        else:
            # h = h0 * H1 + h1 -> [H0, H1, T]
            out = normed.reshape(T, _H0, H1).permute(1, 2, 0)
    else:
        if hidden_dim_tp:
            # h = h1 * H0 + h0 -> [H0, T, H1]
            out = normed.reshape(T, H1, _H0).permute(2, 0, 1)
        else:
            # h = h0 * H1 + h1 -> [H0, T, H1]
            out = normed.reshape(T, _H0, H1).permute(1, 0, 2)

    return {"out": out.contiguous()}


# ===================== Tests =====================


def _gen_inputs(batch, seqlen, hidden, input_3d, dtype):
    np.random.seed(42)
    if input_3d:
        inp = np.random.randn(batch, seqlen, hidden).astype(dtype)
    else:
        inp = np.random.randn(batch * seqlen, hidden).astype(dtype)
    return {
        "input": inp,
        "gamma": np.random.randn(1, hidden).astype(dtype),
    }


@pytest_test_metadata(name="RMSNorm TKG mlp_tkg direct")
@pytest_marks(["rmsnorm", "tkg", "mlp"])
class TestMlpTkgRmsnorm:
    # fmt: off
    _PARAMS = "lnc_degree, batch, seqlen, hidden, out_ndim, in_sbuf, hidden_dim_tp, shard_output, input_3d"
    _ABBREVS = {
        "lnc_degree": "lnc", "hidden": "h", "out_ndim": "ond", "in_sbuf": "insb",
        "hidden_dim_tp": "hdtp", "shard_output": "sho", "input_3d": "i3d",
    }
    _CASES = [
        # HBM _th (T>=128), LNC1
        pytest.param(1, 1, 128, 1024, 3, False, False, False, True, marks=pytest.mark.fast),   # th_lnc1_3d
        pytest.param(1, 1, 128, 1024, 2, False, False, False, False, marks=pytest.mark.fast),  # th_lnc1_2d
        pytest.param(1, 1, 128, 1024, 3, False, True, False, True, marks=pytest.mark.fast),    # th_lnc1_tp
        # HBM _ht (T<128), LNC1
        pytest.param(1, 1, 32, 1024, 3, False, False, False, True, marks=pytest.mark.fast),    # ht_lnc1_3d
        pytest.param(1, 1, 32, 1024, 2, False, False, False, False, marks=pytest.mark.fast),   # ht_lnc1_2d
        # SBUF input (always _ht), LNC1
        pytest.param(1, 1, 32, 1024, 3, True, False, False, True, marks=pytest.mark.fast),     # sbuf_lnc1_3d
        pytest.param(1, 1, 32, 1024, 2, True, False, False, True, marks=pytest.mark.fast),     # sbuf_lnc1_2d
        # HBM _ht LNC2, hidden_dim_tp + shard-on-h
        pytest.param(2, 1, 32, 1024, 3, False, True, True, False, marks=pytest.mark.fast),     # ht_lnc2_tp_shard
    ]
    # fmt: on

    @pytest_parametrize(_PARAMS, _CASES, abbrevs=_ABBREVS)
    def test_rmsnorm_tkg(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        hidden: int,
        out_ndim: int,
        in_sbuf: bool,
        hidden_dim_tp: bool,
        shard_output: bool,
        input_3d: bool,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config, input_tensor_def=None):
            base = _gen_inputs(batch, seqlen, hidden, input_3d, dtype)
            base.update(
                {
                    "hidden": hidden,
                    "out_ndim": out_ndim,
                    "in_sbuf": in_sbuf,
                    "hidden_dim_tp": hidden_dim_tp,
                    "shard_output": shard_output,
                }
            )
            return base

        def output_tensors(kernel_input):
            T = batch * seqlen
            H1 = hidden // _H0
            is_th = (not in_sbuf) and (T >= 128)
            shape = (_H0, H1, T) if is_th else (_H0, T, H1)
            return {"out": np.zeros(shape, dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_tkg_wrapper,
            torch_ref=torch_ref_wrapper(rmsnorm_tkg_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target),
            rtol=3e-2,
            atol=2e-2,
        )
