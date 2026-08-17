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

"""Interleaved PSUM→SBUF copy that alternates between Scalar and Vector engines.

The goal of this helper is to drain PSUM into SBUF (optionally fused with a
scale and/or bias) while spreading the work across the Scalar (activation) and
Vector (DVE) engines so the two can execute in parallel.

Callers pass a monotonically increasing ``index`` per copy. Even indices pick
the Scalar engine, odd indices pick the Vector engine. Within each engine the
helper emits the cheapest legal instruction for the operand shapes:

- ``scale`` / ``bias`` shape ``(P,)`` or ``(P, 1)`` → vector/per-partition
  broadcast: Scalar emits ``nisa.activation``; Vector emits ``nisa.tensor_scalar``
  (which natively accepts a ``(P, 1)`` operand, so no materialized broadcast).
- ``scale`` / ``bias`` shape matches ``dst`` (full ``(P, F)`` or higher-rank) →
  Vector engine falls back to ``nisa.tensor_tensor``. Scalar is not usable for
  this shape, so on even ``index`` we still emit on the Vector engine.
"""

import nki.isa as nisa
import nki.language as nl

from .kernel_assert import kernel_assert


def _is_per_partition_vector(tensor) -> bool:
    """True if tensor is shaped ``(P,)`` or ``(P, 1)`` — a per-partition scalar
    that ``nisa.activation`` / ``nisa.tensor_scalar`` can broadcast internally.
    """
    if tensor is None:
        return False
    shape = tensor.shape
    return len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)


def _validate_partition_dim(name, tensor, expected_p):
    if tensor is None:
        return
    kernel_assert(
        tensor.shape[0] == expected_p,
        f"Partition dim of {name} ({tensor.shape[0]}) must equal dst partition dim ({expected_p}).",
    )


def _emit_scalar_engine(dst, src, scale, bias):
    """One-instruction copy on Scalar (activation) engine.

    Requires any non-None ``scale`` / ``bias`` to be per-partition shaped.
    """
    if scale is not None and bias is not None:
        nisa.activation(dst=dst, data=src, scale=scale, bias=bias, op=nl.copy)
    elif scale is not None:
        nisa.activation(dst=dst, data=src, scale=scale, op=nl.copy)
    elif bias is not None:
        nisa.activation(dst=dst, data=src, bias=bias, op=nl.copy)
    else:
        nisa.activation(dst=dst, data=src, op=nl.copy)


def _emit_vector_engine(dst, src, scale, bias):
    """Copy (with optional scale/bias) on Vector engine.

    Picks the cheapest legal instruction given the operand shapes:

    =============  =============  =================================================
    scale shape    bias shape     Emitted instruction(s)
    =============  =============  =================================================
    None           None           ``tensor_copy``                               (1)
    (P,) / (P,1)   None           ``tensor_scalar(multiply)``                   (1)
    (P, F)         None           ``tensor_tensor(multiply)``                   (1)
    None           (P,) / (P,1)   ``tensor_scalar(add)``                        (1)
    None           (P, F)         ``tensor_tensor(add)``                        (1)
    (P,) / (P,1)   (P,) / (P,1)   ``tensor_scalar(multiply, add)``  — fused    (1)
    (P,) / (P,1)   (P, F)         ``scalar_tensor_tensor(multiply, add)`` — fused (1)
    (P, F)         (P,) / (P,1)   ``tensor_tensor(mul)`` + ``tensor_scalar(add)`` (2)
    (P, F)         (P, F)         ``tensor_tensor(mul)`` + ``tensor_tensor(add)`` (2)
    =============  =============  =================================================

    Note: ``scalar_tensor_tensor`` only applies when the *scale* is per-partition
    and the *bias* is full-shape (STT fuses them at the cost of a single
    ``tensor_tensor``). The reverse (scale full-shape, bias per-partition) would
    require reordering the ops, which would change semantics, so we keep the
    two-instruction fallback there.
    """
    scale_pp = _is_per_partition_vector(scale)
    bias_pp = _is_per_partition_vector(bias)

    # Both present, both per-partition → single fused tensor_scalar.
    if scale is not None and bias is not None and scale_pp and bias_pp:
        nisa.tensor_scalar(
            dst=dst,
            data=src,
            op0=nl.multiply,
            operand0=scale,
            op1=nl.add,
            operand1=bias,
            engine=nisa.vector_engine,
        )
        return

    # Per-partition scale combined with full-shape bias → single scalar_tensor_tensor.
    if scale is not None and bias is not None and scale_pp and not bias_pp:
        nisa.scalar_tensor_tensor(
            dst=dst,
            data=src,
            op0=nl.multiply,
            operand0=scale,
            op1=nl.add,
            operand1=bias,
        )
        return

    # First op: apply scale (multiply) into dst. If no scale, fall through to bias/copy.
    if scale is not None:
        if scale_pp:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=scale, engine=nisa.vector_engine)
        else:
            nisa.tensor_tensor(dst=dst, data1=src, data2=scale, op=nl.multiply)
        # dst now holds src * scale. If bias present, accumulate below.
        if bias is not None:
            if bias_pp:
                nisa.tensor_scalar(dst=dst, data=dst, op0=nl.add, operand0=bias, engine=nisa.vector_engine)
            else:
                nisa.tensor_tensor(dst=dst, data1=dst, data2=bias, op=nl.add)
        return

    # Bias only
    if bias is not None:
        if bias_pp:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.add, operand0=bias, engine=nisa.vector_engine)
        else:
            nisa.tensor_tensor(dst=dst, data1=src, data2=bias, op=nl.add)
        return

    # Pure copy
    nisa.tensor_copy(dst=dst, src=src, engine=nisa.vector_engine)


def interleave_copy(
    dst: nl.NkiTensor,
    src: nl.NkiTensor,
    scale: nl.NkiTensor = None,
    bias: nl.NkiTensor = None,
    index: int = 0,
):
    """Copy ``src`` into ``dst`` with optional fused scale/bias, alternating engines.

    Engine selection:
    - Even ``index`` → Scalar (activation) engine.
    - Odd ``index``  → Vector (DVE) engine.

    The Scalar engine only supports per-partition ``(P,)`` / ``(P, 1)`` operands
    for scale/bias; if a caller passes a full-shape ``scale`` or ``bias`` on an
    even index, the copy is emitted on the Vector engine instead (only legal
    option) to preserve correctness.

    Args:
        dst: Destination tile.
        src: Source tile.
        scale: Optional multiplicative factor. Per-partition ``(P,)`` / ``(P, 1)``
            or same shape as ``dst``.
        bias: Optional additive term. Same shape constraints as ``scale``.
        index: Engine-selection counter. Callers should pass monotonically
            increasing values so consecutive copies alternate engines.
    """

    expected_p = dst.shape[0]
    _validate_partition_dim("src", src, expected_p)
    _validate_partition_dim("scale", scale, expected_p)
    _validate_partition_dim("bias", bias, expected_p)

    # Scalar engine requires per-partition shapes for any scale/bias it fuses in.
    scalar_engine_legal = (scale is None or _is_per_partition_vector(scale)) and (
        bias is None or _is_per_partition_vector(bias)
    )

    if index % 2 == 0 and scalar_engine_legal:
        _emit_scalar_engine(dst, src, scale, bias)
    else:
        _emit_vector_engine(dst, src, scale, bias)
