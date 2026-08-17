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

from typing import final

import nki.language as nl
import nkilib_src.nkilib.experimental.attention.attention_const_max as kernel_module
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.attention.attention_const_max_torch import attention_const_max_torch_ref

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_inputs(N, Sq, Sk, d):
    # After RMSNorm, Q and K have std ≈ 1.0.
    # Normalized scores = (Q^T @ K) / sqrt(d) have std = std_q * std_k = 1.0.
    # softmax_max ≈ 4 sigma covers >99.99% of scores.
    gen = gaussian_tensor_generator(std=1.0)
    return {
        "q_hbm": gen(name="q_hbm", shape=(N, d, Sq), dtype=nl.bfloat16),
        "k_hbm": gen(name="k_hbm", shape=(N, d, Sk), dtype=nl.bfloat16),
        "v_hbm": gen(name="v_hbm", shape=(N, Sk, d), dtype=nl.bfloat16),
    }


_SOFTMAX_MAX = 4.0  # 4 sigma of normalized scores (Q,K after RMSNorm, scale=1/sqrt(d))


@pytest_test_metadata(name="AttentionConstMax")
@final
class TestAttentionConstMax:
    @staticmethod
    def _output_tensors(kernel_input):
        q = kernel_input["q_hbm"]
        N, d, Sq = q.shape
        return {"out": np.zeros((N, Sq, d), dtype=q.dtype)}

    @pytest_parametrize(
        "N, Sq, Sk, d, softmax_max, softmax_scale",
        [
            pytest.param(1, 512, 16384, 128, _SOFTMAX_MAX, 0.0884, id="512_16k", marks=pytest.mark.fast),
            pytest.param(5, 512, 43008, 128, _SOFTMAX_MAX, 0.0884, id="5h_512_42k"),
            pytest.param(5, 675, 43264, 128, _SOFTMAX_MAX, 0.0884, id="5h_675_43264", marks=pytest.mark.fast),
            pytest.param(2, 16384, 16384, 128, _SOFTMAX_MAX, 0.0884, id="2h_16k_16k"),
            pytest.param(1, 900, 2176, 128, _SOFTMAX_MAX, 0.0884, id="partial_blocks", marks=pytest.mark.fast),
            pytest.param(5, 1, 43264, 128, _SOFTMAX_MAX, 0.0884, id="5h_tkg_sq1", marks=pytest.mark.fast),
        ],
    )
    def test_attention_const_max(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        N,
        Sq,
        Sk,
        d,
        softmax_max,
        softmax_scale,
    ):
        def input_generator(test_config):
            inputs = generate_inputs(N, Sq, Sk, d)
            inputs["softmax_max"] = softmax_max
            inputs["softmax_scale"] = softmax_scale
            return inputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_module.attention_const_max,
            torch_ref=torch_ref_wrapper(attention_const_max_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self._output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            inference_args=InferenceArgs(),
            rtol=0.01,
            atol=1e-3,
        )

    @pytest_parametrize(
        "N, Sq, Sk, d, softmax_max, softmax_scale",
        [
            pytest.param(1, 512, 16384, 128, _SOFTMAX_MAX, 0.0884, id="tensor_max_512_16k", marks=pytest.mark.fast),
            pytest.param(
                1, 900, 2176, 128, _SOFTMAX_MAX, 0.0884, id="tensor_max_partial_blocks", marks=pytest.mark.fast
            ),
            pytest.param(5, 1, 43264, 128, _SOFTMAX_MAX, 0.0884, id="tensor_max_tkg_sq1"),
            pytest.param(5, 675, 43264, 128, _SOFTMAX_MAX, 0.0884, id="tensor_max_5h_675_43264"),
        ],
    )
    def test_attention_const_max_tensor(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        N,
        Sq,
        Sk,
        d,
        softmax_max,
        softmax_scale,
    ):
        """softmax_max supplied as a runtime [128, 1] HBM tensor instead of a scalar.

        The tensor holds the same global max replicated across all 128 partitions, so the
        result is identical to passing the scalar — this exercises the DMA-load path in
        _prepare_softmax_max.
        """

        def input_generator(test_config):
            inputs = generate_inputs(N, Sq, Sk, d)
            inputs["softmax_max"] = np.full((128, 1), softmax_max, dtype=np.float32)
            inputs["softmax_scale"] = softmax_scale
            return inputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_module.attention_const_max,
            torch_ref=torch_ref_wrapper(attention_const_max_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self._output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            inference_args=InferenceArgs(),
            rtol=0.01,
            atol=1e-3,
        )

    @pytest_parametrize(
        "N, Sq, Sk, d, softmax_max, softmax_scale",
        [
            pytest.param(6, 512, 32768, 128, _SOFTMAX_MAX, 0.0884, id="lnc2_6h_512_32k", marks=pytest.mark.fast),
            pytest.param(5, 1024, 16384, 128, _SOFTMAX_MAX, 0.0884, id="lnc2_5h_1024_16k"),
            pytest.param(5, 1023, 16384, 128, _SOFTMAX_MAX, 0.0884, id="lnc2_5h_1023_16k_uneven"),
        ],
    )
    def test_attention_const_max_lnc2(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        N,
        Sq,
        Sk,
        d,
        softmax_max,
        softmax_scale,
    ):
        def input_generator(test_config):
            inputs = generate_inputs(N, Sq, Sk, d)
            inputs["softmax_max"] = softmax_max
            inputs["softmax_scale"] = softmax_scale
            return inputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_module.attention_const_max,
            torch_ref=torch_ref_wrapper(attention_const_max_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self._output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            inference_args=InferenceArgs(),
            rtol=0.01,
            atol=1e-3,
        )

    @pytest_parametrize(
        "N, Sq, Sk, d, softmax_max, softmax_scale, std, mean",
        [
            # Higher std — scores have wider range, need larger max
            pytest.param(2, 512, 16384, 128, 10.0, 0.0884, 1.5, 0.0, id="stress_std1.5_max10"),
            pytest.param(2, 512, 16384, 128, 20.0, 0.0884, 2.0, 0.0, id="stress_std2.0_max20"),
            pytest.param(2, 512, 16384, 128, 30.0, 0.0884, 2.5, 0.0, id="stress_std2.5_max30"),
            # Non-zero mean — shifts score distribution
            pytest.param(2, 512, 16384, 128, 8.0, 0.0884, 1.0, 0.3, id="stress_mean0.3_max8"),
            pytest.param(2, 512, 16384, 128, 12.0, 0.0884, 1.2, 0.5, id="stress_mean0.5_std1.2_max12"),
            pytest.param(2, 512, 16384, 128, 15.0, 0.0884, 1.0, 1.0, id="stress_mean1.0_max15"),
            # Large Sk with higher std
            pytest.param(1, 675, 43264, 128, 12.0, 0.0884, 1.5, 0.0, id="stress_o2_std1.5_max12"),
            pytest.param(1, 675, 43264, 128, 20.0, 0.0884, 2.0, 0.0, id="stress_o2_std2.0_max20"),
            # Overly large max (tests epsilon path — many scores underflow)
            pytest.param(2, 512, 4096, 128, 15.0, 0.0884, 1.0, 0.0, id="stress_large_max15_std1"),
            pytest.param(2, 512, 4096, 128, 25.0, 0.0884, 1.0, 0.0, id="stress_large_max25_std1"),
            # Small Sq (TKG-like) with higher std
            pytest.param(5, 1, 16384, 128, 12.0, 0.0884, 1.5, 0.0, id="stress_tkg_std1.5_max12"),
            pytest.param(5, 1, 16384, 128, 20.0, 0.0884, 2.0, 0.2, id="stress_tkg_std2.0_mean0.2_max20"),
            # Combination: high std + mean + large Sk
            pytest.param(2, 512, 32768, 128, 25.0, 0.0884, 2.0, 0.3, id="stress_std2_mean0.3_sk32k_max25"),
        ],
    )
    def test_attention_const_max_stress(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        N,
        Sq,
        Sk,
        d,
        softmax_max,
        softmax_scale,
        std,
        mean,
    ):
        """Stress tests with non-ideal input distributions (higher std, non-zero mean).

        Uses relaxed tolerance (5% rtol) since bf16 precision limits accuracy when the
        score dynamic range is wide.
        """

        def input_generator(test_config):
            gen = gaussian_tensor_generator(std=std, mean=mean)
            inputs = {
                "q_hbm": gen(name="q_hbm", shape=(N, d, Sq), dtype=nl.bfloat16),
                "k_hbm": gen(name="k_hbm", shape=(N, d, Sk), dtype=nl.bfloat16),
                "v_hbm": gen(name="v_hbm", shape=(N, Sk, d), dtype=nl.bfloat16),
            }
            inputs["softmax_max"] = softmax_max
            inputs["softmax_scale"] = softmax_scale
            return inputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_module.attention_const_max,
            torch_ref=torch_ref_wrapper(attention_const_max_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self._output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            inference_args=InferenceArgs(),
            rtol=0.03,
            atol=1e-1,
        )
