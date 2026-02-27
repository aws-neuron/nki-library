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

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.integration.nkilib.utils.test_kernel_common import rms_norm
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.mx_utils import quantize_mx_golden
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import Any, final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.subkernels.rmsnorm_mx_quantize_tkg import rmsnorm_mx_quantize_tkg
from nkilib_src.nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info, kernel_assert


@nki.jit
def rmsnorm_mx_quantize_tkg_wrapper(
    inp,
    gamma,
    residual=None,
    hidden_actual=None,
):
    """Wrapper kernel that allocates SBUF buffers and spills outputs to HBM."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("rmsnorm_mx_quantize_tkg_wrapper", (0, 1))
    kernel_assert(n_prgs == 2, f"Expected LNC=2 for rmsnorm_mx_quantize_tkg_wrapper, got LNC={n_prgs}")

    B, S, H = inp.shape
    H_par = nl.tile_size.pmax
    H_free = H // H_par
    T = B * S
    kernel_assert(H_free % 4 == 0, f"H_free must be divisible by 4 for quantize_mx, got {H_free=}")

    with_residual = residual is not None

    output_shape = (H_par, T, H_free)
    quant_shape = (H_par, H_free // 4, T)
    output_sb = nl.ndarray(output_shape, dtype=inp.dtype, buffer=nl.sbuf)
    output_quant_sb = nl.ndarray(quant_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    output_scale_sb = nl.ndarray(quant_shape, dtype=nl.uint8, buffer=nl.sbuf)

    output_residual = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm) if with_residual else None

    rmsnorm_mx_quantize_tkg(
        input=inp,
        gamma=gamma,
        output=output_sb,
        output_quant=output_quant_sb,
        output_scale=output_scale_sb,
        residual=residual,
        output_residual=output_residual,
        hidden_actual=hidden_actual,
    )

    output = nl.ndarray(output_shape, dtype=inp.dtype, buffer=nl.shared_hbm)
    output_quant = nl.ndarray(quant_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.shared_hbm)
    output_scale = nl.ndarray(quant_shape, dtype=nl.uint8, buffer=nl.shared_hbm)

    T_local = T // n_prgs
    T_slice_output = nl.ds(T_local * prg_id, T_local)
    nisa.dma_copy(src=output_sb[:, T_slice_output, :], dst=output[:, T_slice_output, :])

    T_slice_quant = nl.ds(T_local * (1 - prg_id), T_local)
    nisa.dma_copy(src=output_quant_sb[:, :, T_slice_quant], dst=output_quant[:, :, T_slice_quant])

    for sb_quadrant in nl.affine_range(4):
        nisa.dma_copy(
            src=output_scale_sb[nl.ds(sb_quadrant * 32, 4), :, T_slice_quant],
            dst=output_scale[nl.ds(sb_quadrant * 32, 4), :, T_slice_quant],
        )

    if with_residual:
        return output, output_quant, output_scale, output_residual
    return output, output_quant, output_scale


def golden_output_validator(inp: dict[str, Any], hidden_actual: int, out_dtype, out_quant_dtype):
    """Compute golden output for rmsnorm_mx_quantize_tkg kernel."""
    hidden = inp["inp"]
    gamma = inp["gamma"]
    residual = inp.get("residual")
    B, S, H = hidden.shape
    BxS = B * S
    H0, H1 = 128, H // 128
    _q_width = 4
    scale_H0 = H0 // 8
    n_H512_tiles = H1 // 4

    # Residual add if provided
    if residual is not None:
        hidden = hidden + residual
        out_residual = hidden.reshape(BxS, H).astype(out_dtype)
    else:
        out_residual = None

    result = rms_norm(hidden, gamma, hidden_actual=hidden_actual)
    result = result.reshape((BxS, H1, H0)).transpose((2, 0, 1)).astype(out_dtype)

    qmx_input = result.copy().reshape(H0, BxS, _q_width, n_H512_tiles).transpose(0, 3, 1, 2)
    qmx_input_2D = qmx_input.reshape(H0, -1)

    out_quant_golden, out_scale_golden = quantize_mx_golden(qmx_input_2D, out_quant_dtype)
    out_quant_golden = out_quant_golden.reshape(H0, n_H512_tiles, BxS)
    out_scale_golden = out_scale_golden.reshape(scale_H0, n_H512_tiles, BxS)

    sb_quadrant_size, num_quadrants_in_sb = 32, 4
    scale_par_idx = (np.arange(_q_width)[:, None] * sb_quadrant_size + np.arange(num_quadrants_in_sb)).ravel()
    out_scale_golden_strided = np.zeros((H0, n_H512_tiles, BxS), dtype=np.uint8)
    out_scale_golden_strided[scale_par_idx, ...] = out_scale_golden[...]

    result_dict = {"out": result, "out_quant": out_quant_golden, "out_scale": out_scale_golden_strided}
    if out_residual is not None:
        result_dict["out_residual"] = out_residual
    return result_dict


def build_kernel_input(batch, seqlen, hidden, hidden_actual, in_dtype, out_dtype, out_quant_dtype, tensor_gen):
    inp = tensor_gen(shape=(batch, seqlen, hidden), dtype=in_dtype, name="inp")
    gamma = tensor_gen(shape=(1, hidden), dtype=in_dtype, name="gamma")

    return {
        "inp": inp,
        "gamma": gamma,
        "hidden_actual": hidden_actual,
    }


@pytest_test_metadata(name="RMSNorm Quantize MX TKG", pytest_marks=["rmsnorm", "quantize", "mx", "tkg"])
@final
class TestRmsNormQuantizeMxTKGKernel:
    def run_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        hidden_actual: int,
        in_dtype,
        out_dtype,
        out_quant_dtype,
        with_residual: bool = False,
        tensor_gen=gaussian_tensor_generator(),
    ):
        kernel_input = build_kernel_input(
            batch, seqlen, hidden, hidden_actual, in_dtype, out_dtype, out_quant_dtype, tensor_gen
        )
        if with_residual:
            kernel_input["residual"] = tensor_gen(shape=(batch, seqlen, hidden), dtype=in_dtype, name="residual")

        BxS = batch * seqlen
        H0 = 128
        H1 = hidden // H0
        n_H512_tiles = hidden // 512
        actual_hidden = hidden_actual if hidden_actual else hidden

        golden = golden_output_validator(kernel_input, actual_hidden, out_dtype, out_quant_dtype)

        def create_lazy_golden():
            result = {"out": golden["out"], "out_scale": golden["out_scale"]}
            if with_residual:
                result["out_residual"] = golden["out_residual"]
            return result

        output_placeholder = {
            "out": np.zeros((H0, BxS, H1), dtype=out_dtype),
            "out_quant": np.zeros((H0, n_H512_tiles, BxS), dtype=out_quant_dtype),
            "out_scale": np.zeros((H0, n_H512_tiles, BxS), dtype=np.uint8),
        }
        if with_residual:
            output_placeholder["out_residual"] = np.zeros((BxS, hidden), dtype=out_dtype)

        test_manager.execute(
            KernelArgs(
                kernel_func=rmsnorm_mx_quantize_tkg_wrapper,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_placeholder, create_lazy_golden),
                    relative_accuracy=1e-2,
                    absolute_accuracy=1e-3,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    rmsnorm_mx_quantize_tkg_params = "batch, seqlen, hidden, hidden_actual, in_dtype, out_dtype, out_quant_dtype"
    rmsnorm_mx_quantize_tkg_perms = [
        # Basic shapes
        pytest.param(32, 1, 512, None, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(64, 1, 512, None, np.float16, np.float16, nl.float8_e4m3fn_x4),
        # Sweep multiples of 128
        pytest.param(128, 1, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(256, 1, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(384, 1, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        # Speculation shapes
        pytest.param(16, 3, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(32, 3, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(64, 3, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(16, 5, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(32, 5, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(64, 5, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(128, 4, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(128, 5, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize(rmsnorm_mx_quantize_tkg_params, rmsnorm_mx_quantize_tkg_perms)
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
        out_dtype,
        out_quant_dtype,
    ):
        if platform_target is not Platforms.TRN3:
            pytest.skip("MX quantization is only supported on TRN3.")
        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
        self.run_test(
            test_manager,
            compiler_args,
            collector,
            batch,
            seqlen,
            hidden,
            hidden_actual,
            in_dtype,
            out_dtype,
            out_quant_dtype,
        )

    rmsnorm_mx_quantize_tkg_residual_params = (
        "batch, seqlen, hidden, hidden_actual, in_dtype, out_dtype, out_quant_dtype"
    )
    rmsnorm_mx_quantize_tkg_residual_perms = [
        # Residual add requires BxS >= 256 and H1 % 8 == 0 (H >= 1024)
        pytest.param(256, 1, 1024, None, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(256, 1, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        # Speculation with residual
        pytest.param(128, 3, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
        pytest.param(64, 5, 3072, 2880, np.float16, np.float16, nl.float8_e4m3fn_x4),
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize(rmsnorm_mx_quantize_tkg_residual_params, rmsnorm_mx_quantize_tkg_residual_perms)
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
        out_dtype,
        out_quant_dtype,
    ):
        if platform_target is not Platforms.TRN3:
            pytest.skip("MX quantization is only supported on TRN3.")
        compiler_args = CompilerArgs(logical_nc_config=2, platform_target=platform_target)
        self.run_test(
            test_manager,
            compiler_args,
            collector,
            batch,
            seqlen,
            hidden,
            hidden_actual,
            in_dtype,
            out_dtype,
            out_quant_dtype,
            with_residual=True,
        )
