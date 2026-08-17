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
import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.gdn.gdn_tkg import gdn_tkg

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

NUM_V_HEADS = 12


def _generate_inputs(bh, k_dim, v_dim, dtype):
    torch.manual_seed(42)
    # q,k already L2-normed + GQA-expanded (kernel does NOT normalize in folds 1-2).
    q = torch.nn.functional.normalize(torch.randn(bh, k_dim), p=2, dim=-1)
    k = torch.nn.functional.normalize(torch.randn(bh, k_dim), p=2, dim=-1)
    v = torch.randn(bh, v_dim)
    a = torch.randn(bh) * 0.1  # RAW pre-gate a (kernel: g=-exp(A_log)*softplus(a+dt_bias))
    b = torch.randn(bh)  # RAW pre-sigmoid beta (kernel applies sigmoid)
    A_log = torch.randn(NUM_V_HEADS) * 0.5
    dt_bias = torch.randn(NUM_V_HEADS) * 0.1
    z = torch.randn(bh, v_dim)
    norm_weight = torch.randn(v_dim) * 0.2 + 1.0
    state = torch.randn(bh, k_dim, v_dim) * 0.1
    return {
        "q": dt.static_cast(q.numpy(), dtype),
        "k": dt.static_cast(k.numpy(), dtype),
        "v": dt.static_cast(v.numpy(), dtype),
        "b": b.numpy().astype(np.float32),
        "a": a.numpy().astype(np.float32),
        "A_log": A_log.numpy().astype(np.float32),
        "dt_bias": dt_bias.numpy().astype(np.float32),
        "z": dt.static_cast(z.numpy(), dtype),
        "norm_weight": norm_weight.numpy().astype(np.float32),
        "state_in": state.numpy().astype(np.float32),
    }


@torch_ref_wrapper
def _torch_ref(q, k, v, b, a, A_log, dt_bias, z, norm_weight, state_in):
    bh, k_dim = q.shape
    scale = 1.0 / (k_dim**0.5)
    # Fold 3: l2norm(q), l2norm(k) over K (eps=1e-6).
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q_s = q * scale
    beta = torch.sigmoid(b)  # Fold 1: beta = sigmoid(b)
    # Fold 2: g = -exp(A_log) * softplus(a + dt_bias); v-head index cycles fastest.
    B = bh // NUM_V_HEADS
    A_log_bh = A_log.repeat(B)
    dt_bh = dt_bias.repeat(B)
    g = -A_log_bh.exp() * torch.nn.functional.softplus(a + dt_bh)
    exp_g = g.exp().unsqueeze(-1).unsqueeze(-1)
    S = state_in * exp_g
    kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (S * q_s.unsqueeze(-1)).sum(dim=-2)
    # Fold 4: RMSNormGated (eps=1e-6).
    rms = torch.rsqrt((out * out).mean(-1, keepdim=True) + 1e-6)
    silu_z = z * torch.sigmoid(z)
    out = out * rms * norm_weight.unsqueeze(0) * silu_z
    return {"out": out, "state_out": S}


@pytest_test_metadata(name="GDN TKG v2 fused decode")
@pytest_marks(["attention", "gdn", "tkg"])
class TestGdnTkgV2:
    params = "bh, k_dim, v_dim, dtype"
    _ABBREVS = {"bh": "bh", "k_dim": "k", "v_dim": "v", "dtype": "dt"}
    test_cases_basic = [(24, 128, 128, nl.bfloat16)]

    def _run(self, test_manager, platform_target, bh, k_dim, v_dim, dtype, atol=2e-2, rtol=2e-2):
        def inputs(test_config, input_tensor_def=None):
            return _generate_inputs(bh, k_dim, v_dim, dtype)

        def outputs(kin):
            return {
                "out": np.zeros((bh, v_dim), dtype=np.dtype("bfloat16")),
                "state_out": np.zeros((bh, k_dim, v_dim), dtype=np.float32),
            }

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gdn_tkg,
            torch_ref=_torch_ref,
            kernel_input_generator=inputs,
            output_tensor_descriptor=outputs,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.fast
    @pytest_parametrize(params, test_cases_basic, abbrevs=_ABBREVS)
    def test_basic(self, test_manager, platform_target, bh, k_dim, v_dim, dtype):
        self._run(test_manager, platform_target, bh, k_dim, v_dim, dtype)
