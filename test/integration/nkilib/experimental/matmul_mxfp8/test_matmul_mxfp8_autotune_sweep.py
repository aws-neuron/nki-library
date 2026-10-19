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
import warnings
from typing import Any, NotRequired, TypedDict

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_kernel
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
    _generate_autotune_candidates,
)
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation

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
         Optional: AUTOTUNE_TP (default 1), AUTOTUNE_CP (default 1), AUTOTUNE_UBS
         (micro-batch size, default 1), AUTOTUNE_FUSED_GATE_UP=1 (one fused dense
         gate/up matmul instead of separate w1/w3)

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
        if "AUTOTUNE_UBS" not in os.environ:
            warnings.warn("AUTOTUNE_UBS is not set; model-config shapes assume micro-batch size 1 (uBS1)", stacklevel=2)
        ubs = int(os.environ.get("AUTOTUNE_UBS", "1"))
        fuse_gate_up = os.environ.get("AUTOTUNE_FUSED_GATE_UP", "0") not in ("0", "", "false")
        cfg = TorchTitanModelConfig.from_json(model_config)
        is_moe = (
            cfg.get("moe_enabled", "False").lower() == "true"
            if isinstance(cfg.get("moe_enabled"), str)
            else bool(cfg.get("moe_enabled", False))
        )

        # The per-head attention matmuls (QK^T, AV and their grads) do not run through matmul_mxfp8.
        attention_shapes = [
            shape
            for shape in generate_attention_shapes(model_config, TP=tp, CP=cp, UBS=ubs)
            if "-Attn-" not in shape[0]
        ]
        if is_moe:
            model_shapes = attention_shapes + generate_moe_expert_mlp_shapes(model_config, TP=tp)
        else:
            model_shapes = attention_shapes + generate_dense_mlp_shapes(
                model_config, TP=tp, CP=cp, UBS=ubs, fuse_gate_up=fuse_gate_up
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


def _selected_ratios(env_name):
    """Feature-ratio values to sweep, from a comma-separated env var.

    Used for AUTOTUNE_PSUM_COPY_IN_RATIOS and AUTOTUNE_STRIDED_COPY_1X32_RATIOS, e.g.
    "none,0.5,1". "none" passes no ratio (the kernel default). Unset sweeps only
    the default, preserving prior behavior.
    """
    raw = os.environ.get(env_name, "").strip().lower()
    if not raw:
        return [None]
    ratios = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        ratio = None if tok == "none" else float(tok)
        if ratio is not None and not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{env_name}: '{tok}' must be 'none' or a value in [0, 1]")
        ratios.append(ratio)
    return list(dict.fromkeys(ratios))


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
    fast_dma_transpose: NotRequired[bool]
    quant_scheme: NotRequired[str]
    enable_scale_packing: NotRequired[bool]
    lhs_orientation: NotRequired[Any]
    rhs_orientation: NotRequired[Any]
    disable_dma_transpose: NotRequired[bool]


_METHOD_SPECS: dict[str, _MethodSpec] = {
    "prequant": {"lhs_dtype": _MXFP8_X4, "rhs_dtype": _MXFP8_X4, "lhs_is_swizzled": True, "rhs_is_swizzled": True},
    "preswizzled": {"lhs_dtype": _BF16, "rhs_dtype": _BF16, "lhs_is_swizzled": True, "rhs_is_swizzled": True},
    "dgt": {"lhs_dtype": _BF16, "rhs_dtype": _BF16, "lhs_is_swizzled": False, "rhs_is_swizzled": False},
    "dgt_fast": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "fast_dma_transpose": True,
    },
    "pe_1x32": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "load_with_PE_swizzle": True,
        "quant_scheme": "1x32",
    },
    # Same loader as pe_1x32 but the K-by-F transpose goes through dma_copy +
    # nc_transpose (PE) instead of a HW-DGE DMA transpose, which the kernel keys
    # separately ("_nodma"). This is what the e2e wrapper runs under
    # NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=1, so those keys need their own entries.
    "pe_1x32_no_dma": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "load_with_PE_swizzle": True,
        "quant_scheme": "1x32",
        "disable_dma_transpose": True,
    },
    "pe_wrapx": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "load_with_PE_swizzle": True,
        "quant_scheme": "wrapX",
    },
    # Backward dW config: lhs (dY) is K-by-F (the sole transpose), rhs (X) is F-by-K.
    # dw_pe_1x32 = DMA route (dY via dma_transpose); dw_pe_1x32_no_dma = no-DMA (dma_copy +
    # nc_transpose). dX is both-F-by-K -> use pe_1x32 (DMA == no-DMA there, no K-by-F operand).
    "dw_pe_1x32": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "load_with_PE_swizzle": True,
        "quant_scheme": "1x32",
        "lhs_orientation": TensorOrientation.K_BY_F,
    },
    "dw_pe_1x32_no_dma": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "load_with_PE_swizzle": True,
        "quant_scheme": "1x32",
        "lhs_orientation": TensorOrientation.K_BY_F,
        "disable_dma_transpose": True,
    },
    # matmul_mxfp8_backward phases (natural-layout operands, orientation handled on-chip):
    # dX = dY[M,N] @ W[N,K] -> lhs F-by-K, rhs K-by-F; dW = dY^T @ X -> both K-by-F.
    # wrapX passes fast_dma_transpose (affects only the F-by-K dY), K-by-F -> PE swizzle.
    "bwd_dx_wrapx": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "fast_dma_transpose": True,
        "quant_scheme": "wrapX",
        "rhs_orientation": TensorOrientation.K_BY_F,
    },
    "bwd_dw_wrapx": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "fast_dma_transpose": True,
        "quant_scheme": "wrapX",
        "lhs_orientation": TensorOrientation.K_BY_F,
        "rhs_orientation": TensorOrientation.K_BY_F,
    },
    "bwd_dx_1x32": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "quant_scheme": "1x32",
        "rhs_orientation": TensorOrientation.K_BY_F,
    },
    "bwd_dw_1x32": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _BF16,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": False,
        "quant_scheme": "1x32",
        "lhs_orientation": TensorOrientation.K_BY_F,
        "rhs_orientation": TensorOrientation.K_BY_F,
    },
    "dgt_fast_rhs_prequant": {
        "lhs_dtype": _BF16,
        "rhs_dtype": _MXFP8,
        "lhs_is_swizzled": False,
        "rhs_is_swizzled": True,
        "fast_dma_transpose": True,
        "quant_scheme": "wrapX",
        "enable_scale_packing": True,
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

# Methods where BOTH operands are pre-quantized MXFP8: nothing is loaded on-chip as
# bf16, so there is nothing to swizzle+spill -> spill_reload is a no-op and is skipped.
# Mixed methods (lhs_pe_1x32_rhs_prequant, dgt_fast_rhs_prequant) still load a bf16
# operand on-chip, which spill_reload can amortize, so they DO sweep spill_reload.
_BOTH_PREQUANT_METHODS = ("prequant",)
# Methods that load an unswizzled operand on-chip; require K divisible by 128.
_UNSWIZZLED_METHODS = (
    "dgt",
    "dgt_fast",
    "pe_1x32",
    "pe_1x32_no_dma",
    "pe_wrapx",
    "dw_pe_1x32",
    "dw_pe_1x32_no_dma",
    "bwd_dx_wrapx",
    "bwd_dw_wrapx",
    "bwd_dx_1x32",
    "bwd_dw_1x32",
    "lhs_pe_1x32_rhs_prequant",
    "dgt_fast_rhs_prequant",
)
# Excluded from AUTOTUNE_METHODS="all" (still selectable explicitly by name):
# the pre-quantized RHS is wrapX-layout while the LHS loads via the 1x32 path, so
# the K-contraction scale groupings don't match and the product is wrong (see the
# known limitation in test_matmul_mxfp8_generic_kernel.py). Re-add to "all" once an
# offline 1x32 pre-quantizer exists.
_EXCLUDED_FROM_ALL = ("lhs_pe_1x32_rhs_prequant",)


def _make_test_config(kc, method, idx, shape_label, psum_copy_in_ratio=None, strided_copy_swizzle_1x32_ratio=None):
    """Wrap a MatmulMxfp8KernelConfig into a TestConfig for the given method."""
    # Non-default ratios extend the test id; default ids stay as before.
    suffix = "" if psum_copy_in_ratio is None else f"_psum{psum_copy_in_ratio:g}"
    if strided_copy_swizzle_1x32_ratio is not None:
        suffix += f"_strided{strided_copy_swizzle_1x32_ratio:g}"
    conf = config_helper.TestConfig(
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
        description=f"{method}_{shape_label}_c{idx}{suffix}",
        seed=42,
        output_dtype=_BF16,
        spill_reload=kc.spill_reload,
        **_METHOD_SPECS[method],
        **_UNIFORM,
    )
    conf.psum_copy_in_ratio = psum_copy_in_ratio
    conf.strided_copy_swizzle_1x32_ratio = strided_copy_swizzle_1x32_ratio
    return conf


def _uses_strided_copy_1x32(spec):
    """Whether strided_copy_swizzle_1x32_ratio can affect this method.

    The ratio only selects the loader for unswizzled BF16 K-by-F operands under the
    1x32 scheme (see load_block); every other method ignores it.
    """
    return spec.get("quant_scheme") == "1x32" and any(
        spec[f"{side}_dtype"] == _BF16
        and not spec[f"{side}_is_swizzled"]
        and spec.get(f"{side}_orientation") == TensorOrientation.K_BY_F
        for side in ("lhs", "rhs")
    )


def _uses_psum_copy_in(kc):
    """Whether psum_copy_in_ratio can affect this tiling.

    Copy-in needs an FP32 PSUM accumulating across K blocks. With the sweep's BF16
    output, TILES_IN_BLOCK_K == 1 keeps PSUM in BF16 (see generic_matmul_mxfp8_api).
    """
    if kc.TILES_IN_BLOCK_K is None or kc.tile_k is None:
        return True
    return kc.TILES_IN_BLOCK_K > 1 and kc.K > kc.tile_k * kc.TILES_IN_BLOCK_K


_TILING_FIELDS = (
    "tile_m",
    "tile_k",
    "tile_n",
    "TILES_IN_BLOCK_M",
    "TILES_IN_BLOCK_N",
    "TILES_IN_BLOCK_K",
    "TILES_IN_LOAD_M",
    "TILES_IN_LOAD_N",
    "spill_reload",
)


def _seed_candidates(shape, method, candidates, eligible):
    """Extra candidates from AUTOTUNE_SEEDS_FILE (JSON {"MxKxN": {method: [tiling, ...]}}).

    Lets a capped sweep always include known-good tilings (e.g. the entry tuned for the
    same shape on another load path) on top of the even subsample. Seeds reuse the
    candidates' resolved LNC2 sharding and are skipped when already in the sample.
    Indices start at 10000 so test ids stay unique.
    """
    path = os.environ.get("AUTOTUNE_SEEDS_FILE", "")
    if not path or not candidates:
        return []
    import json

    with open(path) as f:
        seeds = json.load(f).get(shape, {}).get(method, [])
    have = {tuple(getattr(kc, f) for f in _TILING_FIELDS) for _, kc in eligible}
    out = []
    for i, tiling in enumerate(seeds):
        t = tuple(tiling.get(f, False if f == "spill_reload" else None) for f in _TILING_FIELDS)
        if t in have:
            continue
        have.add(t)
        out.append(
            (
                10000 + i,
                MatmulMxfp8KernelConfig(
                    M=candidates[0].M,
                    K=candidates[0].K,
                    N=candidates[0].N,
                    lnc_sharding=candidates[0].lnc_sharding,
                    **dict(zip(_TILING_FIELDS, t, strict=True)),
                ),
            )
        )
    return out


def _generate_all_configs():
    """Generate TestConfig list from AUTOTUNE_SHAPES env var across selected methods.

    AUTOTUNE_MAX_CANDIDATES (optional): cap the number of candidate tilings swept
    per (shape, method). Candidates are evenly subsampled across the full list
    rather than truncated, so the cap keeps a spread of tile sizes instead of one
    family. Default: unlimited.

    AUTOTUNE_PSUM_COPY_IN_RATIOS / AUTOTUNE_STRIDED_COPY_1X32_RATIOS (optional): ratio
    values crossed with every tiling candidate, applied only where the knob can take
    effect. The candidate cap applies before this cross product.
    """
    shapes = _parse_shapes()
    methods = _selected_methods()
    max_cand = int(os.environ.get("AUTOTUNE_MAX_CANDIDATES", "0"))
    psum_ratios = _selected_ratios("AUTOTUNE_PSUM_COPY_IN_RATIOS")
    strided_ratios = _selected_ratios("AUTOTUNE_STRIDED_COPY_1X32_RATIOS")
    configs = []
    for label, M, K, N in shapes:
        base = MatmulMxfp8KernelConfig(M=M, K=K, N=N)
        for method in methods:
            spec = _METHOD_SPECS[method]
            # Prequant-ness of each operand drives the auto LNC2 axis choice (shard the
            # non-pre-quantized operand), which sets the per-core shape the candidates
            # must be tiled for -- so generate per method, not once per shape.
            lhs_is_prequant = spec["lhs_dtype"] in (_MXFP8, _MXFP8_X4)
            rhs_is_prequant = spec["rhs_dtype"] in (_MXFP8, _MXFP8_X4)
            candidates = _generate_autotune_candidates(
                base, lhs_is_prequant=lhs_is_prequant, rhs_is_prequant=rhs_is_prequant
            )
            eligible = [
                (idx, kc)
                for idx, kc in enumerate(candidates)
                # Both-operands-prequant MXFP8 has nothing to spill; skip those candidates.
                if not (method in _BOTH_PREQUANT_METHODS and kc.spill_reload)
                # Unswizzled on-chip loads require K divisible by 128.
                and not (method in _UNSWIZZLED_METHODS and kc.K % 128 != 0)
            ]
            if max_cand > 0 and len(eligible) > max_cand:
                step = len(eligible) / max_cand
                eligible = [eligible[int(i * step)] for i in range(max_cand)]
            eligible += _seed_candidates(f"{M}x{K}x{N}", method, candidates, eligible)
            method_strided_ratios = strided_ratios if _uses_strided_copy_1x32(spec) else [None]
            for idx, kc in eligible:
                for strided in method_strided_ratios:
                    # PSUM copy-in is swept only with the strided-copy loader off.
                    for psum in psum_ratios if _uses_psum_copy_in(kc) and not strided else [None]:
                        configs.append(_make_test_config(kc, method, idx, label, psum, strided))
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


@pytest.mark.fast
@pytest.mark.parametrize("ubs", [None, 2], ids=["default_ubs1", "ubs2"])
def test_model_config_shapes(monkeypatch, ubs):
    """Model-config mode emits exactly the Qwen3-32B TP4 matmul_mxfp8 shapes, warning when uBS defaults to 1."""
    monkeypatch.delenv("AUTOTUNE_SHAPES", raising=False)
    monkeypatch.delenv("AUTOTUNE_FUSED_GATE_UP", raising=False)
    monkeypatch.setenv("AUTOTUNE_MODEL_CONFIG", os.path.join(os.path.dirname(__file__), "qwen3_32B.json"))
    monkeypatch.setenv("AUTOTUNE_TP", "4")
    if ubs is None:
        monkeypatch.delenv("AUTOTUNE_UBS", raising=False)
        with pytest.warns(UserWarning, match="AUTOTUNE_UBS"):
            shapes = _parse_shapes()
    else:
        monkeypatch.setenv("AUTOTUNE_UBS", str(ubs))
        shapes = _parse_shapes()
    tokens = 4096 * (ubs or 1)
    expected = [
        (tokens, 5120, 2560),  # fused QKV fwd
        (tokens, 2560, 5120),  # QKV dX
        (2560, tokens, 5120),  # QKV dW
        (tokens, 2048, 5120),  # o_proj fwd
        (tokens, 5120, 2048),  # o_proj dX
        (5120, tokens, 2048),  # o_proj dW
        (tokens, 5120, 6400),  # w1/w3 fwd, w2 dX
        (tokens, 6400, 5120),  # w1/w3 dX, w2 fwd
        (6400, tokens, 5120),  # w1/w3 dW
        (5120, tokens, 6400),  # w2 dW
    ]
    assert [(M, K, N) for _, M, K, N in shapes] == expected


@pytest.mark.fast
def test_psum_ratio_crossed_only_without_strided_copy(monkeypatch):
    """PSUM copy-in variants are generated only with the strided-copy loader off."""
    monkeypatch.delenv("AUTOTUNE_MODEL_CONFIG", raising=False)
    monkeypatch.delenv("AUTOTUNE_SEEDS_FILE", raising=False)
    monkeypatch.setenv("AUTOTUNE_SHAPES", "2560x8192x5120")
    monkeypatch.setenv("AUTOTUNE_METHODS", "bwd_dw_1x32")
    monkeypatch.setenv("AUTOTUNE_MAX_CANDIDATES", "8")
    monkeypatch.setenv("AUTOTUNE_PSUM_COPY_IN_RATIOS", "none,1")
    monkeypatch.setenv("AUTOTUNE_STRIDED_COPY_1X32_RATIOS", "none,0.5,1")
    ids = [conf.description for conf in _generate_all_configs()]
    assert any("_psum1" in i for i in ids) and any("_strided1" in i for i in ids)
    assert not any("_psum" in i and "_strided" in i for i in ids)
