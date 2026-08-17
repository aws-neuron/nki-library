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
import random
from typing import Any, Callable, Optional, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.rmsnorm.rmsnorm_quant import (
    RmsNormQuantKernelArgs,
    rmsnorm_quant_kernel,
)
from nkilib_src.nkilib.core.rmsnorm.rmsnorm_quant_torch import rmsnorm_quant_torch_ref
from nkilib_src.nkilib.core.utils.common_types import DtypeMode, NormType, QuantizationType
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from typing_extensions import override

try:
    from test.integration.nkilib.core.rmsnorm.test_rmsnorm_quant_cte_model_config import (
        rmsnorm_quant_cte_model_configs,
    )
except ImportError:
    rmsnorm_quant_cte_model_configs = {}

from test.integration.nkilib.utils.tensor_generators import (
    FP8_E4M3_MAX,
    duplicate_row_rmsnorm_inp_generator,
    gaussian_tensor_generator,
)
from test.integration.nkilib.utils.test_kernel_common import resolve_dtype_mode_for_torch_ref
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.comparators import maxAllClose
from test.utils.coverage_parametrized_tests import BoundedRange
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, mark_uncacheable, torch_ref_wrapper


def static_scale_wrapper(default_tensor_generator):
    """Wrapper that generates hidden from a standard normal distribution,
    derives the static dequant scale from absmax of the clean activations,
    then injects 1-2% outliers (10x magnitude) to stress static quantization.
    """
    rng = np.random.default_rng(0)
    cache = {}

    def tensor_generator(shape, dtype, name):
        if name == "hidden":
            hidden = default_tensor_generator(shape, dtype, name)
            # Compute scale from clean activations
            abs_max = float(np.max(np.abs(hidden)))
            ideal_scale = abs_max / FP8_E4M3_MAX if abs_max > 0 else 1.0 / FP8_E4M3_MAX
            jitter = rng.uniform(0.8, 1.2)
            cache["scale"] = np.full((128, 1), ideal_scale * jitter, nl.float32)
            # Inject 2% outliers at random indices
            total = hidden.size
            n_outliers = max(1, int(np.ceil(0.02 * total)))
            outlier_idx = rng.choice(total, size=n_outliers, replace=False)
            hidden_flat = hidden.reshape(-1)
            hidden_flat[outlier_idx] *= 10.0
            return hidden_flat.reshape(shape)
        elif name == "input_dequant_scale":
            return cache["scale"]
        return default_tensor_generator(shape, dtype, name)

    return tensor_generator


def quant_input_tensor_generator(use_rng: bool = True):
    """Generate quantized inputs."""
    rng = np.random.default_rng(0)

    def tensor_generator(shape, dtype, name=None):
        if use_rng:
            return rng.uniform(0, 10, shape).astype(dtype)
        else:
            return np.full(shape=shape, fill_value=rng.random(), dtype=dtype)

    return tensor_generator


def _build_kernel_input(
    lnc_degree: int,
    batch: int,
    seqlen: int,
    hidden_dim: int,
    dtype: str,
    lower_bound: float,
    tensor_gen: Callable,
    quant_type: QuantizationType,
    quant_only: bool = False,
    pre_norm_gamma_flag: bool = False,
    residual_flag: bool = False,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
):
    kernel_assert(batch > 0, f"Batch size must be positive but got {batch}")
    gamma_shape = [1, hidden_dim] if lnc_degree > 1 else [hidden_dim]
    hidden = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="hidden")
    if pre_norm_gamma_flag:
        pre_norm_gamma = tensor_gen(shape=gamma_shape, dtype=dtype, name="pre_norm_gamma")
    else:
        pre_norm_gamma = None
    if residual_flag:
        residual = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="residual")
    else:
        residual = None
    gamma = tensor_gen(shape=gamma_shape, dtype=dtype, name="gamma")
    if quant_type == QuantizationType.STATIC:
        scale = tensor_gen(shape=(128, 1), dtype=nl.float32, name="input_dequant_scale")
    else:
        scale = None
    norm_type = NormType.NO_NORM if quant_only else NormType.RMS_NORM
    kargs = RmsNormQuantKernelArgs(quantization_type=quant_type, lower_bound=lower_bound, norm_type=norm_type, eps=1e-6)
    return {
        "hidden": hidden,
        "pre_norm_gamma": pre_norm_gamma,
        "residual": residual,
        "ln_w": gamma,
        "kargs": kargs,
        "input_dequant_scale": scale,
        "dtype_mode": dtype_mode,
    }


def _build_output_tensor(
    batch,
    seqlen,
    hidden_dim,
    quant_type,
    quant_dtype,
    dtype=nl.bfloat16,
    pre_norm_gamma_flag=False,
    residual_flag=False,
):
    if quant_type == QuantizationType.ROW:
        dtype_size_scale = dt.sizeinbytes(np.float32) // dt.sizeinbytes(quant_dtype)
        out = {"out": np.ndarray(shape=[batch, seqlen, hidden_dim + dtype_size_scale], dtype=quant_dtype)}
    else:
        out = {"out": np.ndarray(shape=[batch, seqlen, hidden_dim], dtype=quant_dtype)}
    if residual_flag:
        out["out_residual"] = np.ndarray(shape=[batch, seqlen, hidden_dim], dtype=quant_dtype)
    return out


# fmt: off
_LNC_PARAM_NAMES = \
    "seqlen, quant_only, quant_type, tensor_gen, lower_bound, lnc_degree, batch, hidden"
_LNC_TEST_PARAMS = [
    (128 // 64, True, QuantizationType.ROW, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (128 // 64, True, QuantizationType.ROW, gaussian_tensor_generator(), 0.5, 2, 1, 16384),
    (128 // 64, False, QuantizationType.ROW, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (128 // 64, False, QuantizationType.ROW, gaussian_tensor_generator(), 0.5, 2, 1, 16384),
    (2048, False, QuantizationType.ROW, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (160, False, QuantizationType.ROW, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (160, False, QuantizationType.ROW, duplicate_row_rmsnorm_inp_generator(), 0.0, 2, 1, 16384),
    (160, False, QuantizationType.ROW, duplicate_row_rmsnorm_inp_generator(all_ones=True), 0.0, 2, 1, 16384),
    (160, True, QuantizationType.ROW, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (128 // 64, True, QuantizationType.STATIC, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (128 // 64, False, QuantizationType.STATIC, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (2048, False, QuantizationType.STATIC, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (160, False, QuantizationType.STATIC, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (160, False, QuantizationType.STATIC, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (160, False, QuantizationType.STATIC, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
    (160, True, QuantizationType.STATIC, gaussian_tensor_generator(), 0.0, 2, 1, 16384),
]


def _lnc_with_fast_keys(fast_keys):
    """Return _LNC_TEST_PARAMS with marks=fast on rows whose
    (seqlen, quant_only, quant_type) tuple matches one in fast_keys.
    """
    fk = frozenset(fast_keys)
    out = []
    for c in _LNC_TEST_PARAMS:
        if (c[0], c[1], c[2]) in fk:
            out.append(pytest.param(*c, marks=pytest.mark.fast))
        else:
            out.append(pytest.param(*c))
    return out


_LNC_TEST_PARAMS_LNC_UNIT_FAST = _lnc_with_fast_keys(
    {
        (128 // 64, True, QuantizationType.ROW),
    }
)
_LNC_TEST_PARAMS_FUSED_RESIDUAL_FAST = _lnc_with_fast_keys(
    {
        (160, False, QuantizationType.STATIC),
    }
)
_LNC_TEST_PARAMS_LNC_ONLY_FAST = _lnc_with_fast_keys(set())

_VNC_PARAM_NAMES = \
    "seqlen, lower_bound, lnc_degree, batch, hidden, quant_type"

_VNC_TEST_PARAMS = [
    (128, 0.0, 1, 1, 16384, QuantizationType.ROW),
    pytest.param(128, 0.5, 1, 1, 16384, QuantizationType.ROW, marks=pytest.mark.fast),
    (128, 0.0, 2, 1, 16384, QuantizationType.ROW),
    (128, 0.5, 2, 1, 16384, QuantizationType.ROW),
    (256, 0.0, 2, 1, 8192, QuantizationType.ROW),
    # Llama models
    (2048, 0.0, 2, 1, 8192, QuantizationType.ROW),
    (2048, 0.0, 2, 1, 16384, QuantizationType.ROW),
    (32768, 0.0, 2, 1, 8192, QuantizationType.ROW),
    (32768, 0.0, 2, 1, 16384, QuantizationType.ROW),
]
# fmt: on

_ABBREVS = {
    "seqlen": "s",
    "quant_only": "qo",
    "quant_type": "qt",
    "tensor_gen": "tg",
    "lower_bound": "lb",
    "lnc_degree": "lnc",
    "batch": "b",
    "hidden": "h",
}


@pytest_test_metadata(name="RMSNorm Quantization", tags=["model"])
@pytest_marks(["rmsnorm", "quantization", "mx"])
@final
class TestRmsNormQuantKernel:
    def _run_test(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        *,
        batch: int,
        seqlen: int,
        hidden: int,
        dtype: str,
        lnc_degree: int,
        lower_bound: float,
        tensor_gen: Callable,
        quant_type: QuantizationType,
        quant_only: bool = False,
        is_negative_test: bool = False,
        pre_norm_gamma_flag: bool = False,
        residual_flag: bool = False,
        dtype_mode: Optional[DtypeMode] = None,
    ):
        is_fused = residual_flag
        # Defaults to DtypeMode.NON_OCP so existing tests stay unchanged.
        # ``test_rmsnorm_quant_dtype_mode`` below sweeps OCP / AUTO / NON_OCP.
        if dtype_mode is None:
            dtype_mode = DtypeMode.NON_OCP
        # Pre-resolve AUTO here — the torch ref runs on CPU and needs the
        # concrete dtype. The kernel receives the original mode and resolves
        # at trace time.
        resolved_dtype_mode = resolve_dtype_mode_for_torch_ref(dtype_mode, platform_target)
        quant_np_dtype = dt.float8_e4m3fn if resolved_dtype_mode == DtypeMode.OCP else dt.float8_e4m3

        def input_generator(_):
            return _build_kernel_input(
                lnc_degree=lnc_degree,
                batch=batch,
                seqlen=seqlen,
                hidden_dim=hidden,
                dtype=dtype,
                lower_bound=lower_bound,
                tensor_gen=tensor_gen,
                quant_type=quant_type,
                quant_only=quant_only,
                pre_norm_gamma_flag=pre_norm_gamma_flag,
                residual_flag=residual_flag,
                dtype_mode=dtype_mode,
            )

        # Wrap torch_ref to return {"out": norm_quant} and stash dequant_scale for the comparator
        ref_extras: dict[str, Any] = {}

        # Uncacheable: the comparator reads dequant_scale from ref_extras, populated only as a
        # side effect of running this ref. A golden-cache HIT skips the ref, leaving ref_extras
        # empty -> comparator KeyError. mark_uncacheable forces a recompute so the stash fires.
        @mark_uncacheable
        @torch_ref_wrapper
        def _torch_ref_with_remap(
            hidden,
            ln_w,
            kargs,
            input_dequant_scale=None,
            pre_norm_gamma=None,
            residual=None,
            dtype_mode=DtypeMode.NON_OCP,  # noqa: ARG001 — accepted to satisfy framework signature check; dtype is picked from quant_nki_dtype above
        ):
            result = rmsnorm_quant_torch_ref(
                hidden=hidden,
                ln_w=ln_w,
                kargs=kargs,
                dtype_mode=resolved_dtype_mode,
                # The optional tensors are only present for the flags this case enables;
                # forward just the ones the input generator produced.
                **{
                    name: tensor
                    for name, tensor in (
                        ("input_dequant_scale", input_dequant_scale),
                        ("pre_norm_gamma", pre_norm_gamma),
                        ("residual", residual),
                    )
                    if tensor is not None
                },
            )
            ref_extras["dequant_scale"] = result["dequant_scale"]
            out = {"out": result["norm_quant"]}
            if result["residual_out"] is not None:
                out["out_residual"] = result["residual_out"]
            return out

        def _comparator(golden_dict, output_tensors):
            norm_out_golden = golden_dict["out"]

            class _Validator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    if quant_type == QuantizationType.ROW:
                        full_fp8 = np.frombuffer(
                            inference_output.view(dtype=nl.bfloat16), dtype=quant_np_dtype
                        ).reshape(batch, seqlen, hidden + 4)
                        norm_out_hw_fp8 = full_fp8[:, :, :hidden]
                        last_4_contiguous = np.ascontiguousarray(full_fp8[:, :, hidden : hidden + 4])
                        norm_deq_scale_hw = last_4_contiguous.view(dtype=np.float32).reshape(batch, seqlen, 1)
                    else:
                        norm_out_hw_fp8 = np.frombuffer(
                            inference_output.view(dtype=nl.bfloat16), dtype=quant_np_dtype
                        ).reshape(batch, seqlen, hidden)

                    passed = maxAllClose(
                        dt.static_cast(norm_out_hw_fp8, np.float32),
                        dt.static_cast(norm_out_golden, np.float32),
                        rtol=0.072,
                        verbose=1,
                    )
                    if quant_type == QuantizationType.ROW:
                        passed &= maxAllClose(
                            norm_deq_scale_hw,
                            ref_extras["dequant_scale"],
                            rtol=0.007,
                            verbose=1,
                        )
                    return passed

            validators = {
                "out": CustomValidatorWithOutputTensorData(
                    validator=_Validator,
                    output_ndarray=output_tensors["out"],
                )
            }

            if is_fused:
                out_residual_golden = golden_dict["out_residual"]

                class _ResidualValidator(CustomValidator):
                    @override
                    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                        hw_residual = np.frombuffer(inference_output.view(dtype=nl.bfloat16), dtype=dtype).reshape(
                            batch, seqlen, hidden
                        )
                        return maxAllClose(
                            dt.static_cast(hw_residual, np.float32),
                            out_residual_golden.astype(np.float32),
                            rtol=0.072,
                            verbose=1,
                        )

                validators["out_residual"] = CustomValidatorWithOutputTensorData(
                    validator=_ResidualValidator,
                    output_ndarray=output_tensors["out_residual"],
                )

            return validators

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_quant_kernel,
            torch_ref=_torch_ref_with_remap,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=lambda _: _build_output_tensor(
                batch,
                seqlen,
                hidden,
                quant_type,
                quant_np_dtype,
                dtype=dtype,
                pre_norm_gamma_flag=pre_norm_gamma_flag,
                residual_flag=residual_flag,
            ),
            collector=collector,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target),
            is_negative_test=is_negative_test,
            custom_comparator=_comparator,
        )

    @pytest_parametrize(_LNC_PARAM_NAMES, _LNC_TEST_PARAMS_LNC_UNIT_FAST, abbrevs=_ABBREVS)
    def test_rmsnorm_quant_lnc_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        seqlen: int,
        quant_only: bool,
        quant_type: QuantizationType,
        tensor_gen: Callable,
        lower_bound: float,
        lnc_degree: int,
        batch: int,
        hidden: int,
    ):
        effective_gen = static_scale_wrapper(tensor_gen) if quant_type == QuantizationType.STATIC else tensor_gen
        self._run_test(
            test_manager,
            platform_target,
            collector,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            dtype=nl.bfloat16,
            lnc_degree=lnc_degree,
            lower_bound=lower_bound,
            tensor_gen=effective_gen,
            quant_type=quant_type,
            quant_only=quant_only,
        )

    @pytest.mark.parametrize("dtype_mode", [DtypeMode.NON_OCP, DtypeMode.OCP, DtypeMode.AUTO])
    @pytest.mark.parametrize("quant_type", [QuantizationType.STATIC, QuantizationType.ROW])
    def test_rmsnorm_quant_by_dtype_mode(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        quant_type: QuantizationType,
        dtype_mode: DtypeMode,
    ):
        """Canary for the opt-in dtype_mode path.

        Sweeps every DtypeMode against the rmsnorm_quant kernel:
            NON_OCP → ``nl.float8_e4m3`` (240), any platform.
            OCP     → ``nl.float8_e4m3fn`` (448), TRN3 only.
            AUTO    → ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3`` elsewhere.
        """
        if dtype_mode == DtypeMode.OCP and not platform_target.is_trn3():
            pytest.skip("dtype_mode=DtypeMode.OCP only exercises the OCP path on TRN3")
        tensor_gen = (
            static_scale_wrapper(gaussian_tensor_generator())
            if quant_type == QuantizationType.STATIC
            else gaussian_tensor_generator()
        )
        self._run_test(
            test_manager,
            platform_target,
            collector,
            batch=1,
            seqlen=2,
            hidden=16384,
            dtype=nl.bfloat16,
            lnc_degree=2,
            lower_bound=0.0,
            tensor_gen=tensor_gen,
            quant_type=quant_type,
            quant_only=False,
            dtype_mode=dtype_mode,
        )

    @pytest_parametrize(_VNC_PARAM_NAMES, _VNC_TEST_PARAMS, abbrevs=_ABBREVS)
    def test_rmsnorm_quant_vnc_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        seqlen: int,
        lower_bound: float,
        lnc_degree: int,
        batch: int,
        hidden: int,
        quant_type: QuantizationType,
    ):
        self._run_test(
            test_manager,
            platform_target,
            collector,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            dtype=nl.float16,
            lnc_degree=lnc_degree,
            lower_bound=lower_bound,
            tensor_gen=quant_input_tensor_generator(use_rng=True),
            quant_type=quant_type,
        )

    _sweep_seqlen_values = sorted(set(random.sample(range(128, 32769, 128), 15) + [128, 32768]))
    _sweep_hidden_values = sorted(set(random.sample(range(1024, 16385, 512), 8) + [1024, 16384]))
    _sweep_lower_bound_values = sorted(set([round(x, 2) for x in [random.uniform(0.0, 1.0) for _ in range(5)]] + [0.0]))

    @pytest.mark.coverage_parametrize(
        # MAX_S=32768, MAX_H=16384, MAX_B=2
        seqlen=BoundedRange(_sweep_seqlen_values, boundary_values=[32769]),
        hidden=BoundedRange(_sweep_hidden_values, boundary_values=[16385]),
        batch=BoundedRange([1, 2], boundary_values=[3]),
        lower_bound=BoundedRange(_sweep_lower_bound_values, boundary_values=[-0.1]),
        coverage="pairs",
    )
    def test_rmsnorm_row_quant_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        seqlen: int,
        hidden: int,
        batch: int,
        lower_bound: float,
        is_negative_test_case: bool,
    ):
        """Sweep test for ROW quantization with pairwise coverage."""
        self._run_test(
            test_manager,
            platform_target,
            collector,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            dtype=nl.bfloat16,
            lnc_degree=1,
            lower_bound=lower_bound,
            tensor_gen=static_scale_wrapper(gaussian_tensor_generator()),
            quant_type=QuantizationType.ROW,
            quant_only=False,
            is_negative_test=is_negative_test_case,
        )

    @pytest_parametrize(_LNC_PARAM_NAMES, _LNC_TEST_PARAMS_FUSED_RESIDUAL_FAST, abbrevs=_ABBREVS)
    def test_fused_rmsnorm_residual_add_rmsnorm_quant(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        seqlen: int,
        quant_only: bool,
        quant_type: QuantizationType,
        tensor_gen: Callable,
        lower_bound: float,
        lnc_degree: int,
        batch: int,
        hidden: int,
    ):
        effective_gen = static_scale_wrapper(tensor_gen) if quant_type == QuantizationType.STATIC else tensor_gen
        self._run_test(
            test_manager,
            platform_target,
            collector,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            dtype=nl.bfloat16,
            lnc_degree=lnc_degree,
            lower_bound=lower_bound,
            tensor_gen=effective_gen,
            quant_type=quant_type,
            quant_only=quant_only,
            pre_norm_gamma_flag=True,
            residual_flag=True,
        )

    @pytest_parametrize(_LNC_PARAM_NAMES, _LNC_TEST_PARAMS_LNC_ONLY_FAST, abbrevs=_ABBREVS)
    def test_residual_add_rmsnorm_quant(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        seqlen: int,
        quant_only: bool,
        quant_type: QuantizationType,
        tensor_gen: Callable,
        lower_bound: float,
        lnc_degree: int,
        batch: int,
        hidden: int,
    ):
        effective_gen = static_scale_wrapper(tensor_gen) if quant_type == QuantizationType.STATIC else tensor_gen
        self._run_test(
            test_manager,
            platform_target,
            collector,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            dtype=nl.bfloat16,
            lnc_degree=lnc_degree,
            lower_bound=lower_bound,
            tensor_gen=effective_gen,
            quant_type=quant_type,
            quant_only=quant_only,
            pre_norm_gamma_flag=False,
            residual_flag=True,
        )


@pytest_marks(["rmsnorm", "quantization", "model", "mx"])
@final
class TestRmsNormQuantCteModel:
    """Model-driven tests for RMSNorm Quant CTE kernel, organized by tier."""

    _MODEL_PARAM_NAMES = "seqlen, quant_only, quant_type, lower_bound, lnc_degree, batch, hidden, fused_residual"

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize(
            {ModelTestType.OPTIMAL: rmsnorm_quant_cte_model_configs.get(ModelTestType.OPTIMAL, [])}
        )
        if rmsnorm_quant_cte_model_configs
        else ([], [])
    )

    def _run_model_test(self, **kwargs):
        """Common test logic for model tiers."""
        fused = kwargs.get("fused_residual", False)
        TestRmsNormQuantKernel()._run_test(
            kwargs["test_manager"],
            kwargs["platform_target"],
            kwargs["collector"],
            batch=kwargs["batch"],
            seqlen=kwargs["seqlen"],
            hidden=kwargs["hidden"],
            dtype=nl.bfloat16,
            lnc_degree=kwargs["lnc_degree"],
            lower_bound=kwargs["lower_bound"],
            tensor_gen=static_scale_wrapper(gaussian_tensor_generator()),
            quant_type=kwargs["quant_type"],
            quant_only=kwargs["quant_only"],
            pre_norm_gamma_flag=fused,
            residual_flag=fused,
        )

    @pytest.mark.optimal
    @pytest.mark.parametrize(_MODEL_PARAM_NAMES, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        seqlen: int,
        quant_only: bool,
        quant_type: QuantizationType,
        lower_bound: float,
        lnc_degree: int,
        batch: int,
        hidden: int,
        fused_residual: bool,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)
