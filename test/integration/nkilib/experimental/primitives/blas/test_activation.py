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

"""Integration tests for the activation BLAS primitive.

Applies an element-wise activation: dst = op(src * scale + bias). scale and bias are
handled symmetrically -- each can be None, a scalar float, or a (P, 1) per-partition
vector -- so the tests sweep the full 3 x 3 scale-kind x bias-kind grid (including mixed
cases like a scalar scale with a vector bias).
"""

import ml_dtypes
import nki
import nki.language as nl
import numpy as np
import pytest
import torch
from nki.language import tile_size

from nkilib_src.nkilib.experimental.primitives import blas, dma, tile_stream
from nkilib_src.nkilib.experimental.primitives.iter_order import RowMajor
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Baked scalar scale/bias values, shared by kernels and torch references.
SCALE = 2.0
BIAS = 5.0

# scale/bias "kind" axes. Each of scale and bias can independently be:
#   "none"   -> omitted (scale defaults to 1.0, bias to 0)
#   "scalar" -> a constant float (SCALE / BIAS)
#   "vector" -> a (P, 1) per-partition vector, broadcast (P, 1) -> (P, F)
KINDS = ["none", "scalar", "vector"]

# Activation ops, covering the set named in the Activation docstring
# ("copy (cast), silu, exp, sin, rsqrt, etc.") plus common extras.
#   needs_positive: op requires a positive domain (e.g. rsqrt = 1/sqrt), so inputs
#   (and any vector scale/bias) are made strictly positive to keep op(src*scale+bias) valid.
OPS = {
    "copy": dict(nl_op=nl.copy, torch_fn=lambda t: t, needs_positive=False),
    "exp": dict(nl_op=nl.exp, torch_fn=torch.exp, needs_positive=False),
    "relu": dict(nl_op=nl.relu, torch_fn=torch.relu, needs_positive=False),
    "sigmoid": dict(nl_op=nl.sigmoid, torch_fn=torch.sigmoid, needs_positive=False),
    "tanh": dict(nl_op=nl.tanh, torch_fn=torch.tanh, needs_positive=False),
    "silu": dict(nl_op=nl.silu, torch_fn=torch.nn.functional.silu, needs_positive=False),
    "gelu": dict(nl_op=nl.gelu, torch_fn=torch.nn.functional.gelu, needs_positive=False),
    "sin": dict(nl_op=nl.sin, torch_fn=torch.sin, needs_positive=False),
    "rsqrt": dict(nl_op=nl.rsqrt, torch_fn=torch.rsqrt, needs_positive=True),
}


# =============================================================================
# Test Kernels (thin @nki.jit wrappers around the primitive under test)
# =============================================================================


def _make_activation_kernel(nl_op, scale_kind, bias_kind, inplace=False):
    """Build a single-tile (P <= pmax) compact-blas.activation() wrapper computing
    op(src * scale + bias) for one (scale_kind, bias_kind, inplace) point in the grid.
    """

    def _body(x, s, b):
        p, f = x.shape
        y = nl.ndarray((p, f), dtype=x.dtype, buffer=nl.shared_hbm)
        src_sb = tile_stream.alloc_logical((p, f), p, x.dtype, "src")
        dma.load(src_sb, x)

        if scale_kind == "vector":
            scale_sb = tile_stream.alloc_logical((p, 1), p, nl.float32, "scale")  # nisa: fp32 scale
            dma.load(scale_sb, s)
            scale_arg = scale_sb
        else:
            scale_arg = SCALE if scale_kind == "scalar" else None

        if bias_kind == "vector":
            bias_sb = tile_stream.alloc_logical((p, 1), p, x.dtype, "bias")
            dma.load(bias_sb, b)
            bias_arg = bias_sb
        else:
            bias_arg = BIAS if bias_kind == "scalar" else None

        if inplace:
            blas.activation(src_sb, op=nl_op, scale=scale_arg, bias=bias_arg)  # src=None -> in place
            dst_sb = src_sb
        else:
            dst_sb = tile_stream.alloc_logical((p, f), p, x.dtype, "dst")
            blas.activation(dst_sb, src_sb, op=nl_op, scale=scale_arg, bias=bias_arg)

        dma.store(y, dst_sb)
        return y

    return _emit_kernel(_body, scale_kind, bias_kind)


def _make_activation_class_kernel(nl_op, scale_kind, bias_kind):
    """Build a class-API (blas.Activation) wrapper for the multi-tile path: P > pmax is
    tiled into (P_MAX, F) chunks so Activation iterates its tile loop (n_p_tiles > 1). A
    (P, 1) vector scale/bias is tiled along the same partition loop (logical_p=p sizes the
    last tile to the true remainder), advancing in lockstep with the data tiles."""

    def _body(x, s, b):
        p, f = x.shape
        pt = tile_size.pmax
        y = nl.ndarray((p, f), dtype=x.dtype, buffer=nl.shared_hbm)
        src_sb = tile_stream.alloc_logical((p, f), pt, x.dtype, "src")
        dst_sb = tile_stream.alloc_logical((p, f), pt, x.dtype, "dst")
        dma.Load(
            tile_stream.tile(src_sb, (pt, f), iter_order=RowMajor(), logical_p=p),
            tile_stream.tile_hbm(x, (pt, f), iter_order=RowMajor()),
        ).execute()

        if scale_kind == "vector":
            scale_sb = tile_stream.alloc_logical((p, 1), pt, nl.float32, "scale")
            dma.Load(
                tile_stream.tile(scale_sb, (pt, 1), iter_order=RowMajor(), logical_p=p),
                tile_stream.tile_hbm(s, (pt, 1), iter_order=RowMajor()),
            ).execute()
            scale_arg = tile_stream.tile(scale_sb, (pt, 1), iter_order=RowMajor(), logical_p=p)
        else:
            scale_arg = SCALE if scale_kind == "scalar" else None

        if bias_kind == "vector":
            bias_sb = tile_stream.alloc_logical((p, 1), pt, x.dtype, "bias")
            dma.Load(
                tile_stream.tile(bias_sb, (pt, 1), iter_order=RowMajor(), logical_p=p),
                tile_stream.tile_hbm(b, (pt, 1), iter_order=RowMajor()),
            ).execute()
            bias_arg = tile_stream.tile(bias_sb, (pt, 1), iter_order=RowMajor(), logical_p=p)
        else:
            bias_arg = BIAS if bias_kind == "scalar" else None

        blas.Activation(
            dst=tile_stream.tile(dst_sb, (pt, f), iter_order=RowMajor(), logical_p=p),
            src=tile_stream.tile(src_sb, (pt, f), iter_order=RowMajor(), logical_p=p),
            op=nl_op,
            scale=scale_arg,
            bias=bias_arg,
        ).execute()
        dma.Store(
            tile_stream.tile_hbm(y, (pt, f), iter_order=RowMajor()),
            tile_stream.tile(dst_sb, (pt, f), iter_order=RowMajor(), logical_p=p),
        ).execute()
        return y

    return _emit_kernel(_body, scale_kind, bias_kind)


def _emit_kernel(body, scale_kind, bias_kind):
    """Wrap `body(x, s, b)` in an @nki.jit kernel whose signature carries only the vector
    terms: (x) / (x, s) / (x, b) / (x, s, b) depending on which of scale/bias are vectors."""
    has_s = scale_kind == "vector"
    has_b = bias_kind == "vector"

    if has_s and has_b:

        @nki.jit
        def kernel_activation(x: nl.ndarray, s: nl.ndarray, b: nl.ndarray) -> nl.ndarray:
            return body(x, s, b)
    elif has_s:

        @nki.jit
        def kernel_activation(x: nl.ndarray, s: nl.ndarray) -> nl.ndarray:
            return body(x, s, None)
    elif has_b:

        @nki.jit
        def kernel_activation(x: nl.ndarray, b: nl.ndarray) -> nl.ndarray:
            return body(x, None, b)
    else:

        @nki.jit
        def kernel_activation(x: nl.ndarray) -> nl.ndarray:
            return body(x, None, None)

    return kernel_activation


def _make_bad_scale_kernel(scale_p, scale_f):
    """Class-API wrapper feeding a deliberately mis-shaped (scale_p, scale_f) scale to
    blas.Activation, to exercise the (P, 1)-vector shape guard. src is (128, F)."""

    @nki.jit
    def kernel_bad_scale(x: nl.ndarray, s: nl.ndarray) -> nl.ndarray:
        p, f = x.shape
        y = nl.ndarray((p, f), dtype=x.dtype, buffer=nl.shared_hbm)
        x_sb = tile_stream.alloc_logical((p, f), p, x.dtype, "x")
        s_sb = tile_stream.alloc_logical((scale_p, scale_f), scale_p, nl.float32, "s")
        dst_sb = tile_stream.alloc_logical((p, f), p, x.dtype, "dst")
        dma.load(x_sb, x)
        dma.load(s_sb, s)
        blas.Activation(
            dst=tile_stream.tile(dst_sb, (p, f), iter_order=RowMajor()),
            src=tile_stream.tile(x_sb, (p, f), iter_order=RowMajor()),
            op=nl.copy,
            scale=tile_stream.tile(s_sb, (scale_p, scale_f), iter_order=RowMajor()),
        ).execute()
        dma.store(y, dst_sb)
        return y

    return kernel_bad_scale


# =============================================================================
# Torch Reference
# =============================================================================


def _make_torch_ref(torch_fn, scale_kind, bias_kind):
    """Reference mirroring the kernel: out = op(x * scale + bias), where scale/bias come
    from the vector inputs (s / b), the scalar constants (SCALE / BIAS), or the identity
    (1.0 / 0.0). Signature carries only the vector terms, matching the kernel."""

    def compute(x, s, b):
        scale = s if scale_kind == "vector" else (SCALE if scale_kind == "scalar" else 1.0)
        bias = b if bias_kind == "vector" else (BIAS if bias_kind == "scalar" else 0.0)
        return torch_fn(x * scale + bias)

    has_s = scale_kind == "vector"
    has_b = bias_kind == "vector"
    if has_s and has_b:
        return lambda x, s, b: compute(x, s, b)
    if has_s:
        return lambda x, s: compute(x, s, None)
    if has_b:
        return lambda x, b: compute(x, None, b)
    return lambda x: compute(x, None, None)


# =============================================================================
# Inputs
# =============================================================================


def _generate_inputs(P, F, dtype, needs_positive=False, scale_kind="none", bias_kind="none"):
    """Generate inputs matching the kernel signature: {x} plus a (P, 1) vector s and/or b
    when scale/bias is "vector". For needs_positive ops (rsqrt) x and any vector scale/bias
    are made strictly positive so op(x * scale + bias) stays in a valid domain."""
    np.random.seed(42)
    x = np.random.randn(P, F).astype(dtype)
    if needs_positive:
        x = np.abs(x) + 0.5
    inputs = {"x": x.astype(dtype)}
    if scale_kind == "vector":
        s = np.random.randn(P, 1).astype(np.float32)
        inputs["s"] = np.abs(s) + 0.5 if needs_positive else s
    if bias_kind == "vector":
        b = np.random.randn(P, 1).astype(dtype)
        inputs["b"] = (np.abs(b) + 0.5).astype(dtype) if needs_positive else b
    return inputs


def _output_tensors(kernel_input):
    return {"out": np.zeros_like(kernel_input["x"])}


# fmt: off
# Fast smoke tier: a few hand-picked cases, each touching a distinct path (the sweep below
# does the thorough pairwise coverage). Columns: P, F, dtype, op, scale_kind, bias_kind, inplace.
_FAST_CONFIG_NAMES = "P, F, dtype, op_name, scale_kind, bias_kind, inplace"
FAST_CONFIGS = [
    pytest.param(128, 512, np.float32,         "copy",  "none",   "none",   False, id="plain"),
    pytest.param(64,  256, ml_dtypes.bfloat16, "silu",  "scalar", "scalar", False, id="scalar_scale_bias_bf16"),
    pytest.param(128, 512, np.float32,         "gelu",  "vector", "vector", False, id="vector_scale_bias"),
    pytest.param(64,  256, np.float32,         "exp",   "scalar", "vector", False, id="mixed_scalar_scale_vector_bias"),
    pytest.param(1,   128, np.float32,         "rsqrt", "vector", "none",   False, id="vector_scale_rsqrt_P1"),
    pytest.param(128, 512, np.float32,         "copy",  "scalar", "vector", True,  id="inplace"),
]

# Multi-tile (class-API) smoke configs: P > pmax forces n_p_tiles > 1. Each entry pairs a
# distinct shape -- incl. partial last tiles (200 = 128+72, 300 = 128+128+44) -- with a
# scale/bias kind, so the per-partition (P, 1) vector scale/bias is tiled along the loop.
# Columns: P, F, dtype, op, scale_kind, bias_kind.
_MULTITILE_CONFIG_NAMES = "P, F, dtype, op_name, scale_kind, bias_kind"
MULTITILE_CONFIGS = [
    pytest.param(256, 64,  np.float32,         "copy", "scalar", "none",   id="scalar_scale_2tiles"),
    pytest.param(384, 128, ml_dtypes.bfloat16, "exp",  "vector", "none",   id="vector_scale_3tiles"),
    pytest.param(200, 64,  np.float32,         "copy", "none",   "vector", id="vector_bias_partial"),
    pytest.param(300, 128, ml_dtypes.bfloat16, "copy", "vector", "vector", id="vector_scale_bias_partial"),
]
# fmt: on


# =============================================================================
# Tests
# =============================================================================


@pytest_marks(["activation"])
class TestActivationPrimitive:
    """Tests class for the activation BLAS primitive kernel."""

    def _run(self, test_manager, platform_target, kernel_entry, torch_ref, input_gen, dtype):
        is_bf16 = dtype == ml_dtypes.bfloat16
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_entry,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=lambda test_config: input_gen(),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            atol=1e-2 if is_bf16 else 1e-4,
            rtol=1e-2 if is_bf16 else 1e-4,
        )

    def _run_compact(self, test_manager, platform_target, op_name, scale_kind, bias_kind, inplace, P, F, dtype):
        spec = OPS[op_name]
        self._run(
            test_manager,
            platform_target,
            _make_activation_kernel(spec["nl_op"], scale_kind, bias_kind, inplace),
            _make_torch_ref(spec["torch_fn"], scale_kind, bias_kind),
            lambda: _generate_inputs(P, F, dtype, spec["needs_positive"], scale_kind, bias_kind),
            dtype,
        )

    def _run_class(self, test_manager, platform_target, op_name, scale_kind, bias_kind, P, F, dtype):
        spec = OPS[op_name]
        self._run(
            test_manager,
            platform_target,
            _make_activation_class_kernel(spec["nl_op"], scale_kind, bias_kind),
            _make_torch_ref(spec["torch_fn"], scale_kind, bias_kind),
            lambda: _generate_inputs(P, F, dtype, spec["needs_positive"], scale_kind, bias_kind),
            dtype,
        )

    # -- Compact API: full fast + full pairwise sweep -------------------------

    @pytest.mark.fast
    @pytest.mark.parametrize(_FAST_CONFIG_NAMES, FAST_CONFIGS)
    def test_activation_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, P, F, dtype, op_name, scale_kind, bias_kind, inplace
    ):
        """Single-tile compact activation smoke tier: a few hand-picked cases touching plain,
        scalar/vector/mixed scale+bias, a needs-positive op (rsqrt), and in-place. The sweep
        does the thorough pairwise coverage."""
        self._run_compact(test_manager, platform_target, op_name, scale_kind, bias_kind, inplace, P, F, dtype)

    @pytest.mark.coverage_parametrize(
        op_name=list(OPS.keys()),
        scale_kind=KINDS,
        bias_kind=KINDS,
        inplace=[False, True],
        P=[1, 32, 64, 128],
        F=[1, 128, 512, 2048],
        dtype=[np.float32, ml_dtypes.bfloat16],
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_activation_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        op_name,
        scale_kind,
        bias_kind,
        inplace,
        P,
        F,
        dtype,
        is_negative_test_case,
    ):
        """Single-tile compact sweep, pairwise over op x scale-kind x bias-kind x inplace x
        P x F x dtype (P <= pmax). Covers every scale/bias type pairing, including mixes."""
        self._run_compact(test_manager, platform_target, op_name, scale_kind, bias_kind, inplace, P, F, dtype)

    # -- Class API: multi-tile behaviour + shape guard ------------------------

    @pytest.mark.fast
    @pytest.mark.parametrize(_MULTITILE_CONFIG_NAMES, MULTITILE_CONFIGS)
    def test_activation_multitile_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, P, F, dtype, op_name, scale_kind, bias_kind
    ):
        """Multi-tile (P > pmax) class-API activation: the tile loop, with per-partition
        (P, 1) vector scale and/or bias tiled along the partition loop in lockstep."""
        self._run_class(test_manager, platform_target, op_name, scale_kind, bias_kind, P, F, dtype)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "scale_p, scale_f",
        [
            pytest.param(128, 2, id="wrong_free_dim"),  # (P, 2): free dim must be 1
            pytest.param(64, 1, id="wrong_partition"),  # (64, 1): partition != src P (128)
        ],
    )
    def test_activation_bad_scale_shape_rejected(
        self, test_manager: Orchestrator, platform_target: Platforms, scale_p, scale_f
    ):
        """A scale that is not a (P, 1) vector matching the data partition dim is rejected at
        the primitive boundary (clear message), not deep in nisa.activation."""
        P, F = 128, 8

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_make_bad_scale_kernel(scale_p, scale_f),
            torch_ref=torch_ref_wrapper(_make_torch_ref(OPS["copy"]["torch_fn"], "vector", "none")),
            kernel_input_generator=lambda test_config: {
                "x": np.random.randn(P, F).astype(np.float32),
                "s": np.random.randn(scale_p, scale_f).astype(np.float32),
            },
            output_tensor_descriptor=_output_tensors,
        )
        with pytest.raises(Exception, match="must be a .P, 1. vector"):
            framework.run_test(
                test_config=None,
                compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                atol=1e-4,
                rtol=1e-4,
            )
