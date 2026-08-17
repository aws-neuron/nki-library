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

"""Integration tests for the fused rmsnorm_router_topk_tkg kernel."""

import functools
from typing import Any, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.core.utils.common_types import QuantizationType, RouterActFnType
from nkilib_src.nkilib.experimental.moe_block.rmsnorm_router_topk_tkg import rmsnorm_router_topk_tkg
from nkilib_src.nkilib.experimental.moe_block.rmsnorm_router_topk_tkg_torch import rmsnorm_router_topk_tkg_torch_ref

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.comparators import maxAllClose
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Map test input dtypes (numpy/nl) to torch dtypes for re-narrowing rmsnorm input
# inside the wrapped torch ref. The torch_ref_wrapper auto-promotes bf16/fp16 numpy
# inputs to torch.float32, so without this re-narrowing the torch ref's RMSNorm
# operates at higher precision than the kernel.
_INPUT_DTYPE_TO_TORCH = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    nl.bfloat16: torch.bfloat16,
    nl.float16: torch.float16,
    nl.float32: torch.float32,
}


def _make_cosine_validator(
    golden: npt.NDArray, kernel_dtype, name: str, rtol: float, atol: float, min_cos: float, min_pass_rate: float
):
    """Cosine-similarity + allclose-with-pass-rate validator. Used for low-precision
    expert_affinities where ULP-level matmul differences flip top-K positions but
    the overall affinity distribution remains directionally aligned."""
    _g, _shape, _kdtype = golden, golden.shape, kernel_dtype

    class _CosValidator(CustomValidator):
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            # Framework provides raw uint8 bytes; reinterpret as the kernel output dtype.
            raw = np.asarray(inference_output, dtype=np.uint8).reshape(-1)
            actual = raw.view(_kdtype).reshape(_shape).astype(np.float32)
            expected = _g.astype(np.float32)
            a, b = actual.flatten(), expected.flatten()
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            allclose_ok = maxAllClose(
                actual, expected, rtol=rtol, atol=atol, verbose=1, logfile=self.logfile, min_pass_rate=min_pass_rate
            )
            self._print_with_log(f"{name}: cos={cos:.6f} (min={min_cos}), allclose@{min_pass_rate}={allclose_ok}")
            return cos >= min_cos and allclose_ok

    return _CosValidator


def _make_packed_scale_validator(golden_packed: npt.NDArray, H: int, rtol: float = 1e-2, atol: float = 1e-3):
    """Validate the scale portion (last H/4 columns) of MX-packed [T, H + H/4] FP8 output.

    The quant portion is skipped because tiny RMSNorm precision differences get
    amplified across FP8 representable boundaries; correctness is established
    indirectly via the scale portion + the (separately validated) router top-K outputs.
    """
    _golden_scale = golden_packed[:, H:].copy()
    _shape = golden_packed.shape

    class _PackedScaleValidator(CustomValidator):
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            raw = np.asarray(inference_output, dtype=np.uint8).reshape(-1)
            packed = raw.reshape(_shape[0], -1)
            actual_scale = packed[:, H:].view(np.uint8).astype(np.float32)
            golden = _golden_scale.copy().view(np.uint8).astype(np.float32)
            matches = np.isclose(actual_scale, golden, rtol=rtol, atol=atol)
            passed = bool(np.all(matches))
            pct = float(np.mean(matches) * 100)
            self._print_with_log(f"norm_output packed scale: {pct:.2f}% match ({int(np.sum(~matches))} mismatches)")
            return passed

    return _PackedScaleValidator


def _make_topk_set_validator(golden_idx: npt.NDArray, name: str, min_overlap: float):
    """Validate that the *set* of selected experts overlaps with the golden set
    above ``min_overlap`` per token. Order doesn't matter; near-tied flips are tolerated."""
    _g = golden_idx.astype(np.int64)
    _shape = _g.shape

    class _TopKSetValidator(CustomValidator):
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            # Framework provides raw uint8 bytes; reinterpret as int32 (kernel dtype).
            raw = np.asarray(inference_output, dtype=np.uint8).reshape(-1)
            actual = raw.view(np.int32).reshape(_shape).astype(np.int64)
            T, K = _shape
            overlap_per_token = np.array([len(set(actual[t].tolist()) & set(_g[t].tolist())) / K for t in range(T)])
            mean_overlap = float(overlap_per_token.mean())
            self._print_with_log(f"{name}: mean top-K set overlap = {mean_overlap:.4f} (min={min_overlap})")
            return mean_overlap >= min_overlap

    return _TopKSetValidator


def generate_inputs(
    batch,
    seqlen,
    hidden,
    num_experts,
    top_k,
    has_bias,
    input_dtype,
    router_mm_dtype,
    quantization_type,
    router_act_fn,
    hidden_actual=None,
    store_eager_affi_only=False,
):
    """Build inputs for rmsnorm_router_topk_tkg. Reuses small-range uniforms used by
    the rmsnorm and router_topk tests to avoid sigmoid saturation and bf16 top-K ties."""
    rng = np.random.default_rng(42)
    inputs = {
        "hidden_states": dt.static_cast(rng.uniform(-0.1, 0.1, (batch, seqlen, hidden)), input_dtype),
        "gamma": dt.static_cast(rng.uniform(-0.1, 0.1, (1, hidden)), input_dtype),
        "router_weights": dt.static_cast(rng.uniform(-0.1, 0.1, (hidden, num_experts)), router_mm_dtype),
        "top_k": top_k,
        "quantization_type": quantization_type,
        "router_mm_dtype": router_mm_dtype,
        "router_act_fn": router_act_fn,
    }
    if store_eager_affi_only:
        inputs["store_eager_affi_only"] = True
    if has_bias:
        # Tiebreak offset reduces near-ties that flip top-K under bf16 matmul.
        tiebreak = np.linspace(0, 1.0, num_experts).reshape(1, num_experts)
        inputs["router_bias"] = dt.static_cast(rng.uniform(-0.1, 0.1, (1, num_experts)) + tiebreak, router_mm_dtype)
    if hidden_actual is not None:
        inputs["hidden_actual"] = hidden_actual
    return inputs


def output_tensor_descriptor(kernel_input):
    B, S, H = kernel_input["hidden_states"].shape
    T = B * S
    _, E = kernel_input["router_weights"].shape
    top_k = kernel_input["top_k"]
    qtype = kernel_input["quantization_type"]
    router_mm_dtype = kernel_input["router_mm_dtype"]
    store_eager_affi_only = kernel_input.get("store_eager_affi_only", False)

    if qtype == QuantizationType.MX:
        norm_shape = (T, H + H // 4)
        norm_dtype = nl.float8_e4m3fn
    else:
        norm_shape = (T, H)
        norm_dtype = router_mm_dtype

    # Eager: expert_affinities is the dense [T, K] bf16 tensor (co-indexed with expert_index),
    # not the sparse [T, E] tensor.
    affinities_shape = (T, top_k) if store_eager_affi_only else (T, E)

    return {
        "norm_output": np.zeros(norm_shape, dtype=norm_dtype),
        "expert_index": np.zeros((T, top_k), dtype=np.int32),
        "expert_affinities": np.zeros(affinities_shape, dtype=nl.bfloat16),
    }


# fmt: off
PARAMS = "batch, seqlen, hidden, num_experts, top_k, has_bias, input_dtype, router_mm_dtype, quant_type, router_act_fn, hidden_actual"
_ABBREVS = {
    "batch": "b", "seqlen": "s", "hidden": "h", "num_experts": "e", "top_k": "k",
    "has_bias": "bias", "input_dtype": "idt", "router_mm_dtype": "rdt",
    "quant_type": "qt", "router_act_fn": "rf", "hidden_actual": "ha",
}
_PARAM_NAMES = [n.strip() for n in PARAMS.split(",")]


def _format_value(val):
    """Format param values for test IDs. Use enum names instead of integer values
    so 'qt-MX' / 'rf-SIGMOID' appear instead of 'qt-3' / 'rf-1'."""
    if hasattr(val, "name") and hasattr(val, "value"):  # Enum
        return val.name
    if isinstance(val, bool):
        return str(int(val))
    if isinstance(val, type):
        return val.__name__
    return str(val).replace(" ", "")


def _make_test_id(params):
    values = params.values if hasattr(params, "values") else params
    return "_".join(f"{_ABBREVS[name]}-{_format_value(val)}" for name, val in zip(_PARAM_NAMES, values, strict=True))

# NONE: T must be a multiple of 256 (DLoC tiling); H must be divisible by 128.
# MX:   T can be small (>=1); H must be divisible by 512 (MX block size).
TEST_CASES = [
    # fp32 (high-precision baseline)
    pytest.param(256, 1, 3072, 128, 4, True,  np.float32,  nl.float32,  QuantizationType.NONE, RouterActFnType.SIGMOID, None),
    pytest.param(256, 1, 3072, 128, 1, False, np.float32,  nl.float32,  QuantizationType.NONE, RouterActFnType.SOFTMAX, None),
    pytest.param(512, 1, 4096, 128, 8, True,  np.float32,  nl.float32,  QuantizationType.NONE, RouterActFnType.SOFTMAX, None),
    pytest.param(256, 1, 5120, 128, 4, True,  np.float32,  nl.float32,  QuantizationType.NONE, RouterActFnType.SIGMOID, None),
    # bf16 / fp16 — torch ref narrows to router_mm_dtype to mimic kernel precision.
    pytest.param(256, 1, 3072, 128, 4, True,  nl.bfloat16, nl.bfloat16, QuantizationType.NONE, RouterActFnType.SIGMOID, None),
    pytest.param(256, 1, 3072, 128, 1, False, nl.bfloat16, nl.bfloat16, QuantizationType.NONE, RouterActFnType.SOFTMAX, None),
    pytest.param(256, 1, 3072, 128, 4, True,  nl.bfloat16, nl.bfloat16, QuantizationType.NONE, RouterActFnType.SIGMOID, 2880),
    pytest.param(256, 1, 4096, 128, 4, True,  np.float16,  nl.bfloat16, QuantizationType.NONE, RouterActFnType.SIGMOID, None),
    pytest.param(512, 1, 4096, 128, 8, True,  np.float16,  nl.bfloat16, QuantizationType.NONE, RouterActFnType.SOFTMAX, None),
    pytest.param(256, 1, 5120, 128, 4, True,  np.float16,  nl.bfloat16, QuantizationType.NONE, RouterActFnType.SIGMOID, None),
    # MX (TRN3 only) — sweep small T (1..8, 16) and H (3K, 4K, 5K).
    pytest.param(1,  1, 3072, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(2,  1, 3072, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(3,  1, 3072, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(4,  1, 3072, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(5,  1, 4096, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(6,  1, 4096, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(7,  1, 5120, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(8,  1, 5120, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(16, 1, 3072, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, None),
    pytest.param(16, 1, 3072, 128, 1, False, np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SOFTMAX, None),
    pytest.param(16, 1, 3072, 128, 4, True,  np.float16, nl.bfloat16, QuantizationType.MX, RouterActFnType.SIGMOID, 2880),
]
# fmt: on


@pytest_test_metadata(name="RMSNorm Router TopK TKG")
@pytest_marks(["rmsnorm", "router_topk", "moe", "tkg"])
@final
class TestRmsNormRouterTopkTkg:
    @pytest.mark.fast
    @pytest.mark.parametrize("lnc", [1, 2], ids=["lnc-1", "lnc-2"])
    @pytest.mark.parametrize(PARAMS, TEST_CASES, ids=[_make_test_id(p) for p in TEST_CASES])
    def test_rmsnorm_router_topk_tkg(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden,
        num_experts,
        top_k,
        has_bias,
        input_dtype,
        router_mm_dtype,
        quant_type,
        router_act_fn,
        hidden_actual,
        lnc,
    ):
        # LNC=1 support was added for the NONE (DLoC RMSNorm) path; the MX
        # small-T path is validated at LNC=2 only (out of scope for this change).
        if quant_type == QuantizationType.MX and lnc == 1:
            pytest.skip("MX router path is validated at LNC=2 only.")
        if quant_type == QuantizationType.MX and not platform_target.is_trn3():
            pytest.skip("MX quantization only supported on TRN3.")

        def input_generator(test_config):
            return generate_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                num_experts=num_experts,
                top_k=top_k,
                has_bias=has_bias,
                input_dtype=input_dtype,
                router_mm_dtype=router_mm_dtype,
                quantization_type=quant_type,
                router_act_fn=router_act_fn,
                hidden_actual=hidden_actual,
            )

        # Re-narrow rmsnorm operands to the test's input dtype to mimic the kernel's
        # precision flow (the wrapper auto-promotes bf16/fp16 to fp32, losing this).
        narrow_dtype = _INPUT_DTYPE_TO_TORCH[input_dtype]

        @functools.wraps(rmsnorm_router_topk_tkg_torch_ref)
        def narrowed_torch_ref(**kwargs):
            kwargs["hidden_states"] = kwargs["hidden_states"].to(narrow_dtype).float()
            kwargs["gamma"] = kwargs["gamma"].to(narrow_dtype).float()
            return rmsnorm_router_topk_tkg_torch_ref(**kwargs)

        is_low_precision = input_dtype != np.float32
        is_mx = quant_type == QuantizationType.MX

        # Low-precision (bf16/fp16) router matmul flips top-K at near-tied logits.
        # Validate via cosine similarity + min_pass_rate (affinities) and set-overlap
        # (indices), following test_attention_block_tkg's approach for fp8 KV cache.
        # For MX, the packed FP8 norm_output is validated via its scale portion only
        # (matches test_rmsnorm_mx_quantize_tkg's _PackedScaleValidator pattern).
        def _custom_comparator(golden_dict, output_tensors):
            if is_mx:
                norm_validator = CustomValidatorWithOutputTensorData(
                    validator=_make_packed_scale_validator(golden_dict["norm_output"], H=hidden),
                    output_ndarray=output_tensors["norm_output"],
                )
            else:
                norm_validator = golden_dict["norm_output"]
            return {
                "norm_output": norm_validator,
                "expert_index": CustomValidatorWithOutputTensorData(
                    validator=_make_topk_set_validator(golden_dict["expert_index"], "expert_index", min_overlap=0.9),
                    output_ndarray=output_tensors["expert_index"],
                ),
                "expert_affinities": CustomValidatorWithOutputTensorData(
                    validator=_make_cosine_validator(
                        golden_dict["expert_affinities"],
                        output_tensors["expert_affinities"].dtype,
                        "expert_affinities",
                        rtol=5e-2,
                        atol=1e-3,
                        # MX has tighter precision budget than bf16/fp16 for the affinity values.
                        min_cos=0.98 if is_mx else 0.99,
                        min_pass_rate=0.95,
                    ),
                    output_ndarray=output_tensors["expert_affinities"],
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_router_topk_tkg,
            torch_ref=torch_ref_wrapper(narrowed_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        use_custom = is_low_precision or is_mx
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=5e-2 if is_low_precision else 2e-2,
            atol=1e-3,
            custom_comparator=_custom_comparator if use_custom else None,
        )

    # MX-only: matches the gpt-oss-120b decode config (H=3072, E=128, K=4, SOFTMAX).
    @pytest.mark.trn3
    @pytest.mark.parametrize(
        "batch, seqlen, hidden, num_experts, top_k, router_act_fn",
        [
            (1, 16, 3072, 128, 4, RouterActFnType.SOFTMAX),
            (1, 64, 3072, 128, 4, RouterActFnType.SOFTMAX),
            (1, 128, 3072, 128, 4, RouterActFnType.SOFTMAX),
        ],
        ids=lambda v: _format_value(v),
    )
    def test_rmsnorm_router_topk_tkg_eager(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden,
        num_experts,
        top_k,
        router_act_fn,
    ):
        """Eager MX path: store_eager_affi_only=True returns dense [T, K] bf16 affinities
        (co-indexed with expert_index) instead of the sparse [T, E] tensor; matches torch ref."""
        if not platform_target.is_trn3():
            pytest.skip("MX quantization only supported on TRN3.")

        input_dtype = np.float16
        router_mm_dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                num_experts=num_experts,
                top_k=top_k,
                has_bias=True,
                input_dtype=input_dtype,
                router_mm_dtype=router_mm_dtype,
                quantization_type=QuantizationType.MX,
                router_act_fn=router_act_fn,
                store_eager_affi_only=True,
            )

        # Re-narrow rmsnorm operands to the test's input dtype to mimic the kernel's
        # precision flow (the wrapper auto-promotes bf16/fp16 to fp32, losing this).
        narrow_dtype = _INPUT_DTYPE_TO_TORCH[input_dtype]

        @functools.wraps(rmsnorm_router_topk_tkg_torch_ref)
        def narrowed_torch_ref(**kwargs):
            kwargs["hidden_states"] = kwargs["hidden_states"].to(narrow_dtype).float()
            kwargs["gamma"] = kwargs["gamma"].to(narrow_dtype).float()
            kwargs["store_eager_affi_only"] = True
            return rmsnorm_router_topk_tkg_torch_ref(**kwargs)

        def _custom_comparator(golden_dict, output_tensors):
            return {
                "norm_output": CustomValidatorWithOutputTensorData(
                    validator=_make_packed_scale_validator(golden_dict["norm_output"], H=hidden),
                    output_ndarray=output_tensors["norm_output"],
                ),
                "expert_index": CustomValidatorWithOutputTensorData(
                    validator=_make_topk_set_validator(golden_dict["expert_index"], "expert_index", min_overlap=0.9),
                    output_ndarray=output_tensors["expert_index"],
                ),
                "expert_affinities": CustomValidatorWithOutputTensorData(
                    validator=_make_cosine_validator(
                        golden_dict["expert_affinities"],
                        output_tensors["expert_affinities"].dtype,
                        "expert_affinities_eager",
                        rtol=5e-2,
                        atol=1e-3,
                        min_cos=0.98,
                        min_pass_rate=0.95,
                    ),
                    output_ndarray=output_tensors["expert_affinities"],
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_router_topk_tkg,
            torch_ref=torch_ref_wrapper(narrowed_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            rtol=5e-2,
            atol=1e-3,
            custom_comparator=_custom_comparator,
        )
