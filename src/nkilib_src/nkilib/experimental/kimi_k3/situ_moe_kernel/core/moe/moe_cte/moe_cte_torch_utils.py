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

"""Shared PyTorch utility functions for MoE CTE torch reference implementations."""

from typing import Optional

import torch
import torch.nn.functional as F

from ...utils.common_types import ActFnType


def torch_act_fn(x: torch.Tensor, act_fn: ActFnType) -> torch.Tensor:
    """Apply activation function to tensor.

    Args:
        x (torch.Tensor): Input tensor
        act_fn (ActFnType): Activation function type

    Returns:
        torch.Tensor: Activated tensor
    """
    if act_fn == ActFnType.SiTU:
        raise ValueError(
            "SiTU is two-sided -- it transforms the `up` half as well as the gate -- so "
            "it cannot be expressed as a gate-only activation. Use torch_situ_gate_up()."
        )
    if act_fn == ActFnType.SiLU:
        return F.silu(x)
    elif act_fn == ActFnType.GELU:
        return F.gelu(x)
    elif act_fn == ActFnType.GELU_Tanh_Approx:
        return F.gelu(x, approximate="tanh")
    elif act_fn == ActFnType.Swish:
        return x * torch.sigmoid(1.702 * x)
    elif act_fn == ActFnType.SquaredReLU:
        return torch.pow(F.relu(x), 2)
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def torch_situ_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """``situ(gate) * up'`` --- Kimi-K3's two-sided gated nonlinearity, in torch.

    ::

        situ(gate) = beta * tanh(gate / beta) * sigmoid(gate)
        up'        = linear_beta * tanh(up / linear_beta)     (identity if None)

    Transcribed from the upstream Kimi-K3 ``SituAndMul``, including the fp32 upcast, so
    this is the oracle the kernel is verified against. Computed in fp32 and cast back to
    ``gate``'s dtype -- the kernel accumulates in bf16, so comparisons against it should
    expect ~1 bf16 ulp rather than exactness.

    Note the factored form used by the kernel (constants pulled out to the end, because
    ``nisa.activation`` scales its input and has no output multiplier) is written out
    here in the natural form instead; the two are algebraically identical and this one is
    the readable statement of intent.

    Args:
        gate: gate-projection output.
        up: up-projection output, same shape as ``gate``.
        beta: gate-branch soft-clip scale. Must be > 0.
        linear_beta: up-branch soft-clip scale, or None to leave ``up`` linear.

    Returns:
        ``situ(gate) * up'``, in ``gate``'s dtype.
    """
    g = gate.float()
    u = up.float()
    activated = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (activated * u).to(gate.dtype)
