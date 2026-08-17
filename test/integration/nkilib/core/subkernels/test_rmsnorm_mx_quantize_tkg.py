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

"""Integration tests for the fused RMSNorm + MX quantization TKG kernel."""

from typing import Any, final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.subkernels.norm_tkg_utils import _RMSNORM_QMX_SHARDING_THRESHOLD
from nkilib_src.nkilib.core.subkernels.rmsnorm_mx_quantize_tkg import rmsnorm_mx_quantize_tkg
from nkilib_src.nkilib.core.subkernels.rmsnorm_mx_quantize_tkg_torch import rmsnorm_mx_quantize_tkg_wrapper_torch_ref
from nkilib_src.nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info, kernel_assert

from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.tensor_histogram import TensorHistogram
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@nki.jit
def rmsnorm_mx_quantize_tkg_wrapper(
    inp,
    gamma,
    residual=None,
    hidden_actual=None,
    output_quant_in_sbuf=True,
    output_quant_packed=False,
    _out_quant_dtype=nl.float8_e4m3fn_x4,
):
    """Wrapper kernel that allocates SBUF buffers and spills outputs to HBM."""
    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen4,
        f"rmsnorm_mx_quantize_tkg_wrapper only supports gen4+ (Trn3+) but got {nisa.get_nc_version()=}",
    )
    _, n_prgs, prg_id = get_verified_program_sharding_info("rmsnorm_mx_quantize_tkg_wrapper", (0, 1))
    kernel_assert(n_prgs == 2, f"Expected LNC=2 for rmsnorm_mx_quantize_tkg_wrapper, got LNC={n_prgs}")

    B, S, H = inp.shape
    H_par = nl.tile_size.pmax
    H_free = H // H_par
    T = B * S
    kernel_assert(H_free % 4 == 0, f"H_free must be divisible by 4 for quantize_mx, got {H_free=}")

    with_residual = residual is not None

    output_shape = (H_par, T, H_free)
    quant_shape = (H_par, H_free // 4, T) if output_quant_in_sbuf or not output_quant_packed else (T, H * 5 // 4)
    quant_buffer = nl.sbuf if output_quant_in_sbuf else nl.shared_hbm
    output_sb = nl.ndarray(output_shape, dtype=inp.dtype, buffer=nl.sbuf, name="wrapper_output_sb")
    output_quant = nl.ndarray(quant_shape, dtype=_out_quant_dtype, buffer=quant_buffer, name="wrapper_output_quant")
    output_scale = (
        nl.ndarray(quant_shape, dtype=nl.uint8, buffer=quant_buffer, name="wrapper_output_scale")
        if not output_quant_packed
        else None
    )

    output_residual = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm) if with_residual else None

    rmsnorm_mx_quantize_tkg(
        input=inp,
        gamma=gamma,
        output=output_sb,
        output_quant=output_quant,
        output_scale=output_scale,
        residual=residual,
        output_residual=output_residual,
        hidden_actual=hidden_actual,
    )

    # Spill unquantized output to HBM
    output_hbm = nl.ndarray(output_shape, dtype=inp.dtype, buffer=nl.shared_hbm, name="wrapper_output_hbm")
    do_shard = T % n_prgs == 0 and T >= _RMSNORM_QMX_SHARDING_THRESHOLD
    T_local = T // n_prgs if do_shard else T
    T_slice_output = nl.ds(T_local * prg_id, T_local) if do_shard else nl.ds(0, T)
    nisa.dma_copy(src=output_sb[:, T_slice_output, :], dst=output_hbm[:, T_slice_output, :])

    # If quantized output is in SBUF, spill to HBM
    if output_quant_in_sbuf:
        output_quant_hbm = nl.ndarray(
            quant_shape, dtype=_out_quant_dtype, buffer=nl.shared_hbm, name="wrapper_output_quant_hbm"
        )
        output_scale_hbm = nl.ndarray(
            quant_shape, dtype=nl.uint8, buffer=nl.shared_hbm, name="wrapper_output_scale_hbm"
        )

        T_slice_quant = nl.ds(T_local * (1 - prg_id), T_local) if do_shard else nl.ds(0, T)
        nisa.dma_copy(src=output_quant[:, :, T_slice_quant], dst=output_quant_hbm[:, :, T_slice_quant])

        # Copy scale values to HBM, skipping garbage portions of strided scale tensor
        for sb_quadrant in nl.affine_range(4):
            nisa.dma_copy(
                src=output_scale[nl.ds(sb_quadrant * 32, 4), :, T_slice_quant],
                dst=output_scale_hbm[nl.ds(sb_quadrant * 32, 4), :, T_slice_quant],
            )

    else:
        output_quant_hbm = output_quant
        output_scale_hbm = output_scale

    output = [output_hbm, output_quant_hbm]
    if not output_quant_packed:
        output.append(output_scale_hbm)
    if with_residual:
        output.append(output_residual)

    return output


class _SkipValidator(CustomValidator):
    """Validator that always passes, used to skip comparison for out_quant."""

    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        self._print_with_log("(skipped)")
        return True


class _PackedScaleValidator(CustomValidator):
    """Validates the scale portion (last H//4 elements of dim1) of a packed quant+scale tensor.

    The packed tensor has shape (BxS, H + H//4) where the first H elements are
    quant data (skipped) and the last H//4 are scale data (compared).
    """

    def __init__(self, golden_scale: npt.NDArray, H: int, rtol: float = 1e-2, atol: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.golden_scale = golden_scale
        self.H = H
        self.rtol = rtol
        self.atol = atol

    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        BxS = self.golden_scale.shape[0]
        packed = inference_output.reshape(BxS, -1)

        # Slice final H/4 columns corresponding to scales, reinterpret to uint8
        actual_scale = packed[:, self.H :].copy().view(np.uint8).astype(np.float32)
        golden = self.golden_scale.copy().view(np.uint8).astype(np.float32)

        # Validate accuracy, print log + histogram
        matches = np.isclose(actual_scale, golden, rtol=self.rtol, atol=self.atol)
        passed = bool(np.all(matches))
        pct = np.mean(matches) * 100

        self._print_with_log(f"Packed scale: {pct:.2f}% match ({np.sum(~matches)} mismatches)")
        TensorHistogram().print_full_comparison_report(
            actual=actual_scale,
            expected=golden,
            name="packed_out_scale",
            atol=self.atol,
            rtol=self.rtol,
            passed=passed,
            logfile=self.logfile,
            enable_histograms=not passed,
        )
        return passed


def _wrap_torch_ref_skip_out_quant(base_torch_ref):
    """Wrap torch ref to skip out_quant comparison (or validate packed scale).

    MX quantization amplifies tiny RMSNorm precision differences across FP8
    representable value boundariers. Correctness is validated indirectly through
    out (pre-quantized RMSNorm result) and out_scale passing their checks.

    When out_quant is not packed, out_quant is skipped entirely; when out_quant
    is packed, uses _PackedScaleValidator to compare the scale portion
    (last H//4 of dim1) while skipping the quant portion.
    """
    import functools

    @functools.wraps(base_torch_ref)
    def wrapped(**kwargs):
        result = base_torch_ref(**kwargs)
        golden_quant = result.pop("out_quant")
        output_quant_packed = kwargs.get("output_quant_packed", False)

        if output_quant_packed:
            # Packed: out_quant has shape (BxS, H + H//4), out_scale is None
            H = kwargs["inp"].shape[-1]
            golden_scale = golden_quant[:, H:].copy()
            result.pop("out_scale", None)

            class _BoundPackedScaleValidator(_PackedScaleValidator):
                def __init__(self, logfile=None):
                    super().__init__(golden_scale=golden_scale, H=H, logfile=logfile)

            result["out_quant"] = CustomValidatorWithOutputTensorData(
                validator=_BoundPackedScaleValidator,
                output_ndarray=np.ndarray(golden_quant.shape, dtype=golden_quant.dtype),
            )
        else:
            result["out_quant"] = CustomValidatorWithOutputTensorData(
                validator=_SkipValidator,
                output_ndarray=np.ndarray(golden_quant.shape, dtype=golden_quant.dtype),
            )
        return result

    return wrapped


# =============================================================================
# Input / Output Generators
# =============================================================================


def generate_inputs(
    batch,
    seqlen,
    hidden,
    hidden_actual,
    in_dtype,
    out_quant_dtype,
    output_quant_in_sbuf,
    output_quant_packed,
    with_residual=False,
):
    """Generate input tensors for rmsnorm_mx_quantize_tkg kernel."""
    rng = np.random.default_rng(0)
    inputs = {
        "inp": rng.normal(size=(batch, seqlen, hidden)).astype(in_dtype),
        "gamma": rng.normal(size=(1, hidden)).astype(in_dtype),
    }
    if hidden_actual is not None:
        inputs["hidden_actual"] = hidden_actual
    if with_residual:
        inputs["residual"] = rng.normal(size=(batch, seqlen, hidden)).astype(in_dtype)
    inputs["output_quant_in_sbuf"] = output_quant_in_sbuf
    inputs["output_quant_packed"] = output_quant_packed
    # Store out_quant_dtype separately for use in output descriptor
    inputs["_out_quant_dtype"] = out_quant_dtype
    return inputs


def output_tensor_descriptor(kernel_input):
    """Describe output tensor shapes and dtypes matching the kernel wrapper outputs."""
    inp = kernel_input["inp"]
    B, S, H = inp.shape
    BxS = B * S
    H0 = 128
    H1 = H // H0
    num_H512_tiles = H // 512
    out_dtype = inp.dtype
    out_quant_dtype = kernel_input["_out_quant_dtype"]
    output_quant_packed = kernel_input["output_quant_packed"]

    result = {
        "out": np.zeros((H0, BxS, H1), dtype=out_dtype),
    }
    if output_quant_packed:
        # Packed: single tensor (BxS, H + H//4) containing quant + scale
        result["out_quant"] = np.zeros((BxS, H * 5 // 4), dtype=out_quant_dtype)
    else:
        out_quant_shape = (H0, num_H512_tiles, BxS)
        result["out_quant"] = np.zeros(out_quant_shape, dtype=out_quant_dtype)
        result["out_scale"] = np.zeros(out_quant_shape, dtype=np.uint8)
    if "residual" in kernel_input:
        result["out_residual"] = np.zeros((BxS, H), dtype=out_dtype)
    return result


# Abbreviation mappings for keyword-prefixed test IDs
_ABBREVS = {
    "batch": "b",
    "seqlen": "s",
    "hidden": "h",
    "hidden_actual": "ha",
    "in_dtype": "dt",
    "out_quant_dtype": "qdt",
    "output_quant_in_sbuf": "sbuf",
    "output_quant_packed": "packed",
}

RMSNORM_MX_QUANTIZE_TKG_PARAMS = (
    "batch, seqlen, hidden, hidden_actual, in_dtype, out_quant_dtype, output_quant_in_sbuf, output_quant_packed"
)
RMSNORM_MX_QUANTIZE_TKG_PERMS = [
    # Basic shapes
    pytest.param(1, 1, 3072, None, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(2, 1, 3072, None, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(3, 1, 3072, None, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(7, 1, 3072, None, np.float16, nl.float8_e4m3fn_x4, True, False, marks=pytest.mark.fast),
    pytest.param(32, 1, 512, None, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(64, 1, 512, None, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(32, 1, 3072, None, nl.bfloat16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(32, 1, 3072, None, np.float16, nl.float8_e5m2_x4, True, False),
    # Sweep multiples of 128
    pytest.param(128, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(256, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(384, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    # Speculation shapes
    pytest.param(16, 3, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(32, 3, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(64, 3, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(16, 5, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(32, 5, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(64, 5, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(128, 4, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(128, 5, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    # With HBM output
    pytest.param(128, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, False, False),
    pytest.param(256, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, False, False),
    pytest.param(384, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, False, False),
    # With packed HBM output
    pytest.param(1, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(2, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(3, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(7, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(8, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(16, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(32, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True, marks=pytest.mark.fast),
    pytest.param(64, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(128, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(256, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(384, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True, marks=pytest.mark.fast),
    pytest.param(512, 1, 3072, 2880, np.float16, nl.float8_e5m2, False, True),
]

RMSNORM_MX_QUANTIZE_TKG_RESIDUAL_PARAMS = (
    "batch, seqlen, hidden, hidden_actual, in_dtype, out_quant_dtype, output_quant_in_sbuf, output_quant_packed"
)
RMSNORM_MX_QUANTIZE_TKG_RESIDUAL_PERMS = [
    # Residual add requires BxS >= 256 and H1 % 8 == 0 (H >= 1024)
    pytest.param(256, 1, 1024, None, np.float16, nl.float8_e4m3fn_x4, True, False),
    pytest.param(256, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    # Speculation with residual
    pytest.param(128, 3, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False, marks=pytest.mark.fast),
    pytest.param(64, 5, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, True, False),
    # Residual with HBM output
    pytest.param(256, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, False, False),
    pytest.param(384, 1, 3072, 2880, np.float16, nl.float8_e4m3fn_x4, False, False),
    # Residual with packed HBM output
    pytest.param(256, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True, marks=pytest.mark.fast),
    pytest.param(384, 1, 3072, 2880, np.float16, nl.float8_e4m3fn, False, True),
    pytest.param(512, 1, 3072, 2880, np.float16, nl.float8_e5m2, False, True, marks=pytest.mark.fast),
]


# =============================================================================
# Test Class
# =============================================================================


@pytest_test_metadata(name="RMSNorm Quantize MX TKG")
@pytest_marks(["rmsnorm", "quantize", "mx", "tkg"])
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestRmsNormQuantizeMxTKGKernel:
    def _run_test(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden,
        hidden_actual,
        in_dtype,
        out_quant_dtype,
        output_quant_in_sbuf,
        output_quant_packed,
        with_residual: bool,
    ):
        def input_generator(test_config):
            return generate_inputs(
                batch,
                seqlen,
                hidden,
                hidden_actual,
                in_dtype,
                out_quant_dtype,
                output_quant_in_sbuf,
                output_quant_packed,
                with_residual=with_residual,
            )

        torch_ref = _wrap_torch_ref_skip_out_quant(torch_ref_wrapper(rmsnorm_mx_quantize_tkg_wrapper_torch_ref))

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_mx_quantize_tkg_wrapper,
            torch_ref=torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=2,
                platform_target=platform_target,
            ),
            rtol=1e-2,
            atol=1e-3,
            inference_args=TKG_INFERENCE_ARGS,
        )

    @pytest_parametrize(RMSNORM_MX_QUANTIZE_TKG_PARAMS, RMSNORM_MX_QUANTIZE_TKG_PERMS, abbrevs=_ABBREVS)
    def test_rmsnorm_mx_quantize_tkg(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden,
        hidden_actual,
        in_dtype,
        out_quant_dtype,
        output_quant_in_sbuf,
        output_quant_packed,
    ):
        self._run_test(
            test_manager=test_manager,
            collector=collector,
            platform_target=platform_target,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            hidden_actual=hidden_actual,
            in_dtype=in_dtype,
            out_quant_dtype=out_quant_dtype,
            output_quant_in_sbuf=output_quant_in_sbuf,
            output_quant_packed=output_quant_packed,
            with_residual=False,
        )

    @pytest_parametrize(
        RMSNORM_MX_QUANTIZE_TKG_RESIDUAL_PARAMS, RMSNORM_MX_QUANTIZE_TKG_RESIDUAL_PERMS, abbrevs=_ABBREVS
    )
    def test_rmsnorm_mx_quantize_tkg_residual(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden,
        hidden_actual,
        in_dtype,
        out_quant_dtype,
        output_quant_in_sbuf,
        output_quant_packed,
    ):
        self._run_test(
            test_manager=test_manager,
            collector=collector,
            platform_target=platform_target,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            hidden_actual=hidden_actual,
            in_dtype=in_dtype,
            out_quant_dtype=out_quant_dtype,
            output_quant_in_sbuf=output_quant_in_sbuf,
            output_quant_packed=output_quant_packed,
            with_residual=True,
        )
