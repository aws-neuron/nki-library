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

"""Auto-tuning sweep for MXFP8 matmul kernel config optimization.

Generates candidate configs for user-specified shapes and runs each on
hardware to collect performance data for offline cache updates.

"""

import os
from typing import Any, NotRequired, TypedDict

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_kernel
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
    generate_autotune_candidates,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref

from test.integration.nkilib.experimental.matmul_mxfp8 import config_helper, constants
from test.integration.nkilib.experimental.matmul_mxfp8.test_matmul_mxfp8_generic_kernel import (
    _mxfp8_comparator,
    build_matmul_inputs,
    get_output_dtype,
)
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

# ── Shape parsing ─────────────────────────────────────────────────────


def _parse_shapes():
    """Parse shapes from env vars.

    Supports two modes:
      1. AUTOTUNE_SHAPES="MxKxN,MxKxN,..." — explicit shapes
      2. AUTOTUNE_MODEL_CONFIG="/path/to/model.json" — generate shapes from model config
         Optional: AUTOTUNE_TP (default 1), AUTOTUNE_CP (default 1)

    Both can be combined.
    """
    shapes = []

    # Mode 1: explicit shapes
    raw = os.environ.get("AUTOTUNE_SHAPES", "")
    if raw.strip():
        for token in raw.split(","):
            token = token.strip()
            parts = token.split("x")
            if len(parts) != 3:
                raise ValueError(f"Invalid shape '{token}', expected MxKxN")
            M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
            shapes.append((f"{M}x{K}x{N}", M, K, N))

    # Mode 2: model config
    model_config = os.environ.get("AUTOTUNE_MODEL_CONFIG", "")
    if model_config.strip():
        from test.integration.nkilib.experimental.matmul_mxfp8.model_config_reader import (
            TorchTitanModelConfig,
            generate_attention_shapes,
            generate_dense_mlp_shapes,
            generate_moe_expert_mlp_shapes,
        )

        tp = int(os.environ.get("AUTOTUNE_TP", "1"))
        cp = int(os.environ.get("AUTOTUNE_CP", "1"))
        cfg = TorchTitanModelConfig.from_json(model_config)
        is_moe = (
            cfg.get("moe_enabled", "False").lower() == "true"
            if isinstance(cfg.get("moe_enabled"), str)
            else bool(cfg.get("moe_enabled", False))
        )

        if is_moe:
            model_shapes = generate_attention_shapes(model_config, TP=tp, CP=cp) + generate_moe_expert_mlp_shapes(
                model_config, TP=tp
            )
        else:
            model_shapes = generate_attention_shapes(model_config, TP=tp, CP=cp) + generate_dense_mlp_shapes(
                model_config, TP=tp, CP=cp
            )

        seen = set()
        for _name, M, K, N in model_shapes:
            M, K, N = int(M), int(K), int(N)
            key = f"{M}x{K}x{N}"
            if key not in seen:
                seen.add(key)
                shapes.append((key, M, K, N))

    return shapes


def _is_prequant_only():
    return os.environ.get("AUTOTUNE_PREQUANT_ONLY", "0") not in ("0", "", "false")


def _selected_methods():
    """Which input-preparation methods to sweep.

    Controlled by AUTOTUNE_METHODS (comma-separated). Defaults to the swizzled
    pair (prequant, preswizzled) to preserve prior behavior. Set to "all" to
    sweep every method, or list a subset e.g. "prequant,dgt,pe_1x32".

    AUTOTUNE_PREQUANT_ONLY=1 still forces prequant-only (back-compat).
    """
    if _is_prequant_only():
        return ["prequant"]
    raw = os.environ.get("AUTOTUNE_METHODS", "").strip().lower()
    if not raw:
        return ["prequant", "preswizzled"]
    if raw == "all":
        return [m for m in _METHOD_SPECS if m not in _EXCLUDED_FROM_ALL]
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


# ── Candidate generation ─────────────────────────────────────────────


class _InputDistribution(TypedDict):
    """Per-operand random input distribution names and their parameters."""

    dists: list[str]
    params: list[dict[str, Any]]


_UNIFORM: _InputDistribution = {
    "dists": ["uniform", "uniform"],
    "params": [{"a": -1.0, "b": 1.0}, {"a": -1.0, "b": 1.0}],
}
_BF16 = constants.MatrixPrecision.BFLOAT16
_MXFP8 = constants.MatrixPrecision.MXFP8
_MXFP8_X4 = constants.MatrixPrecision.MXFP8_X4


# Per-method operand dtypes + layout. Tiling comes from the swept candidate (kc);
# the cache key is derived by the updater from these emitted fields, so the method
# name here is only a human-readable label in the test id.
class _MethodSpec(TypedDict):
    """Operand dtypes and layout flags describing one input-preparation method.

    Keys marked as not required are only supplied by the methods that exercise
    the corresponding feature; the test config default applies otherwise.
    """

    lhs_dtype: str
    rhs_dtype: str
    lhs_is_swizzled: bool
    rhs_is_swizzled: bool
    load_with_PE_swizzle: NotRequired[bool]
    quant_scheme: NotRequired[str]
    enable_scale_packing: NotRequired[bool]


_METHOD_SPECS: dict[str, _MethodSpec] = {
    "prequant": {"lhs_dtype": _MXFP8_X4, "rhs_dtype": _MXFP8_X4, "lhs_is_swizzled": True, "rhs_is_swizzled": True},
    "preswizzled": {"lhs_dtype": _BF16, "rhs_dtype": _BF16, "lhs_is_swizzled": True, "rhs_is_swizzled": True},
    "dgt": {"lhs_dtype": _BF16, "rhs_dtype": _BF16, "lhs_is_swizzled": False, "rhs_is_swizzled": False},
    "pe_1x32": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "load_with_PE_swizzle": True,
        "quant_scheme": "1x32",
    },
    "lhs_pe_1x32_rhs_prequant": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _MXFP8,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": True,
        "load_with_PE_swizzle": True,
        "quant_scheme": "1x32",
        "enable_scale_packing": True,
    },
}

# Methods with a pre-quantized operand: MXFP8 is quantized offline, never spilled.
_PREQUANT_METHODS = ("prequant", "lhs_pe_1x32_rhs_prequant")
# Methods that load an unswizzled operand on-chip; require K divisible by 128.
_UNSWIZZLED_METHODS = ("dgt", "pe_1x32", "lhs_pe_1x32_rhs_prequant")
# Excluded from AUTOTUNE_METHODS="all" (still selectable explicitly by name):
# the pre-quantized RHS is wrapX-layout while the LHS loads via the 1x32 path, so
# the K-contraction scale groupings don't match and the product is wrong (see the
# known limitation in test_matmul_mxfp8_generic_kernel.py). Re-add to "all" once an
# offline 1x32 pre-quantizer exists.
_EXCLUDED_FROM_ALL = ("lhs_pe_1x32_rhs_prequant",)


def _make_test_config(kc, method, idx, shape_label):
    """Wrap a MatmulMxfp8KernelConfig into a TestConfig for the given method."""
    return config_helper.TestConfig(
        M=kc.M,
        K=kc.K,
        N=kc.N,
        tile_m=kc.tile_m,
        tile_k=kc.tile_k,
        tile_n=kc.tile_n,
        TILES_IN_BLOCK_M=kc.TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=kc.TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=kc.TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=kc.TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=kc.TILES_IN_LOAD_N,
        run_with_lnc2=kc.run_with_lnc2,
        lnc_2_shard_rhs=kc.lnc_2_shard_rhs,
        description=f"{method}_{shape_label}_c{idx}",
        seed=42,
        output_dtype=_BF16,
        spill_reload=kc.spill_reload,
        **_METHOD_SPECS[method],
        **_UNIFORM,
    )


def _generate_all_configs():
    """Generate TestConfig list from AUTOTUNE_SHAPES env var across selected methods."""
    shapes = _parse_shapes()
    methods = _selected_methods()
    configs = []
    for label, M, K, N in shapes:
        base = MatmulMxfp8KernelConfig(M=M, K=K, N=N)
        candidates = generate_autotune_candidates(base)
        for method in methods:
            for idx, kc in enumerate(candidates):
                # Pre-quantized MXFP8 never uses spill_reload; skip those candidates.
                if method in _PREQUANT_METHODS and kc.spill_reload:
                    continue
                # Unswizzled on-chip loads require K divisible by 128.
                if method in _UNSWIZZLED_METHODS and kc.K % 128 != 0:
                    continue
                configs.append(_make_test_config(kc, method, idx, label))
    return configs


AUTOTUNE_CONFIGS = _generate_all_configs()


# ── Test class ────────────────────────────────────────────────────────


@pytest_test_metadata(name="Matmul MXFP8 Autotune Sweep")
@pytest_marks(["matmul_mxfp8", "mx", "mxfp8", "autotune"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestMatmulMxfp8AutotuneSweep:
    @pytest.mark.parametrize(
        "conf",
        AUTOTUNE_CONFIGS,
        ids=[c.description for c in AUTOTUNE_CONFIGS],
    )
    def test_autotune_sweep(self, test_manager, conf, platform_target):
        """Sweep autotune candidates for user-specified shapes."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        test_manager.collector.set_kernel_params(conf.to_metrics_dict())
        output_dtype = get_output_dtype(conf)

        def input_generator(test_config):
            return build_matmul_inputs(conf)

        def output_tensors(kernel_input):
            return {"out": np.zeros((conf.M, conf.N), dtype=output_dtype)}

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=2 if conf.run_with_lnc2 else 1,
            platform_target=platform_target,
        )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=matmul_mxfp8_generic_kernel.matmul_mxfp8,
            torch_ref=matmul_mxfp8_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_mxfp8_comparator(conf, output_dtype, gpu_golden_enabled=False),
        )
