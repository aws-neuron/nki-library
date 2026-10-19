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
"""Integration test for the flash_attention_2 kernel."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.attention import flash_attention2_trn3 as kernel_mod
from nkilib_src.nkilib.experimental.attention.flash_attention2_trn3 import flash_attention_2_torch as refs

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _make_inputs(batch, Bkv, Sq, Sk, D, sq_tiles_per_blk, sk_tiles_per_blk):
    def generator(_):
        np.random.seed(42)
        return {
            "q": np.random.randn(batch, Sq, D).astype(ml_dtypes.bfloat16),
            "k": np.random.randn(Bkv, D, Sk).astype(ml_dtypes.bfloat16),
            "v": np.random.randn(Bkv, Sk, D).astype(ml_dtypes.bfloat16),
            "scale": float(D**-0.5),
            "config": kernel_mod.FlashAttentionConfig(
                sq_tiles_per_blk=sq_tiles_per_blk, sk_tiles_per_blk=sk_tiles_per_blk
            ),
        }

    return generator


# Dynamic-range cases: every one of these drives |scaled score| far past the ~88.7 limit of
# exp() in fp32/bf16. They pass only because the softmax reference always includes the
# current KV block's own max, which pins every exponent at or below zero. A reference taken
# from earlier blocks alone would give Inf here (positive fills) or collapse the denominator
# to zero (negative fills).
_RANGE_B, _RANGE_BKV, _RANGE_SQ, _RANGE_SK, _RANGE_D = 1, 1, 512, 2048, 128


def _make_range_inputs(q_fill, k_fill, std):
    """Constant fills give a uniform softmax; ``std`` instead draws a wide random spread."""

    def generator(_):
        np.random.seed(42)
        shape_q = (_RANGE_B, _RANGE_SQ, _RANGE_D)
        shape_k = (_RANGE_BKV, _RANGE_D, _RANGE_SK)
        if std is None:
            q = np.full(shape_q, q_fill, dtype=np.float32)
            k = np.full(shape_k, k_fill, dtype=np.float32)
        else:
            q = np.random.randn(*shape_q).astype(np.float32) * std
            k = np.random.randn(*shape_k).astype(np.float32) * std
        return {
            "q": q.astype(ml_dtypes.bfloat16),
            "k": k.astype(ml_dtypes.bfloat16),
            "v": np.random.randn(_RANGE_BKV, _RANGE_SK, _RANGE_D).astype(ml_dtypes.bfloat16),
            "scale": float(_RANGE_D**-0.5),
            "config": kernel_mod.FlashAttentionConfig(),
        }

    return generator


@pytest_marks(["neurotile"])
class TestNeurotileFlashAttention2:
    def _run(self, test_manager, platform_target, input_generator, out_shape):
        """Compile and validate the kernel against the torch reference.

        Args:
            test_manager: pytest orchestrator fixture.
            platform_target: target platform fixture.
            input_generator: kernel input generator.
            out_shape: ``(B, Sq, D)`` of the output tensor.
        """
        # nisa.exponential needs Trn3 B1 (gen4 sub-version 1); the A0 variants compile as trn3pre
        # and lack the VectorE instruction. The kernel asserts this too.
        if platform_target in (Platforms.TRN3_A0, Platforms.TRN3_PDS_A0) or not platform_target.is_trn3():
            pytest.skip("flash_attention_2 requires trn3 B1 (NeuronCore-v4 sub-version 1)")
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_mod.flash_attention_2,
            torch_ref=torch_ref_wrapper(refs.flash_attention_2_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=lambda _: {"out": np.zeros(out_shape, dtype=ml_dtypes.bfloat16)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                logical_nc_config=1,
                # Required, not incidental: the kernel places PSUM explicitly and the backend
                # allocation pass re-places it, which measurably regresses the kernel.
                additional_cmd_args=["--internal-skip-backend-allocation-opt-nki"],
            ),
            # bf16 matmuls on both MM1 and MM2 plus a bf16 probability operand.
            rtol=0.02,
            atol=1e-2,
        )

    @pytest_parametrize(
        "batch, Bkv, Sq, Sk, D, sq_tiles_per_blk, sk_tiles_per_blk",
        [
            # Block-aligned (Sq % 512, Sk % 2048): the regression baseline.
            pytest.param(1, 1, 512, 2048, 128, 4, 4, id="1h_512_2k_d128", marks=pytest.mark.fast),
            pytest.param(4, 1, 1024, 4096, 128, 4, 4, id="gqa_4to1_1k_4k_d128"),
            pytest.param(2, 2, 512, 2048, 384, 4, 4, id="2h_512_2k_d384", marks=pytest.mark.fast),
            pytest.param(1, 1, 512, 10240, 128, 4, 4, id="1h_512_10k_d128"),
            # Tile-aligned only. Each exercises a different way a trailing block can be ragged.
            pytest.param(1, 1, 512, 2560, 128, 4, 4, id="unaligned_sk_short_block", marks=pytest.mark.fast),
            pytest.param(1, 1, 512, 2816, 128, 4, 4, id="unaligned_sk_partial_tile"),
            pytest.param(1, 1, 640, 2048, 128, 4, 4, id="unaligned_sq_short_block"),
            pytest.param(1, 1, 896, 2816, 128, 4, 4, id="unaligned_both"),
            pytest.param(2, 2, 640, 2816, 384, 4, 4, id="unaligned_both_d384"),
            pytest.param(4, 1, 768, 3328, 128, 4, 4, id="unaligned_gqa"),
            # Sub-tile: a single partial tile on one or both axes.
            pytest.param(1, 1, 64, 64, 128, 4, 4, id="tiny_sq64_sk64", marks=pytest.mark.fast),
            pytest.param(1, 1, 64, 128, 128, 4, 4, id="tiny_sq64_sk128"),
            pytest.param(1, 1, 128, 64, 128, 4, 4, id="tiny_sq128_sk64"),
            # Sq = 2 tiles: fewer Sq tiles than sq_tiles_per_blk, the small-Sq case.
            pytest.param(1, 1, 256, 512, 128, 4, 4, id="small_sq256_sk512"),
            pytest.param(1, 1, 100, 72, 128, 4, 4, id="tiny_odd"),
            pytest.param(1, 1, 384, 2100, 128, 4, 4, id="unaligned_sk_partial_v_alone"),
            # Sub-V-tile tail sharing a block with whole tiles: needs the split stream.
            pytest.param(1, 1, 512, 2880, 128, 4, 4, id="arbitrary_sk_2880", marks=pytest.mark.fast),
            pytest.param(1, 1, 256, 200, 128, 4, 4, id="arbitrary_sk_200"),
            pytest.param(1, 1, 640, 1000, 128, 4, 4, id="arbitrary_sk_1000"),
            pytest.param(2, 2, 384, 3000, 384, 4, 4, id="arbitrary_d384"),
            # Long-Sk target shape: sq_tiles_per_blk = ceil(675/128) = 6 holds all of Q, so Q loads
            # once, K/V stream once, and there is a single pipeline fill.
            pytest.param(1, 1, 675, 43200, 128, 6, 4, id="target_675_43200"),
            pytest.param(5, 5, 675, 43200, 128, 6, 4, id="target_675_43200_5h"),
            pytest.param(2, 2, 96, 80, 384, 4, 4, id="tiny_d384"),
        ],
    )
    def test_flash_attention_2(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        Bkv,
        Sq,
        Sk,
        D,
        sq_tiles_per_blk,
        sk_tiles_per_blk,
    ):
        self._run(
            test_manager,
            platform_target,
            _make_inputs(batch, Bkv, Sq, Sk, D, sq_tiles_per_blk, sk_tiles_per_blk),
            (batch, Sq, D),
        )

    @pytest_parametrize(
        "q_fill, k_fill, std",
        [
            # Simulator sums instead of maxing on activate2's chained reduce (NKI-2608). Passes on hardware.
            pytest.param(
                5.0,
                5.0,
                None,
                id="uniform_pos_exp283",
                marks=[pytest.mark.fast, pytest.mark.skip_simulation],
            ),
            pytest.param(5.0, -5.0, None, id="uniform_neg_exp283", marks=pytest.mark.skip_simulation),
            pytest.param(None, None, 2.0, id="random_std2"),
        ],
    )
    def test_flash_attention_2_dynamic_range(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        q_fill,
        k_fill,
        std,
    ):
        """Guard the softmax against exp overflow and denominator collapse.

        Each case pushes |scaled score| to roughly 200-280, far beyond exp()'s ~88.7 range.
        """
        self._run(
            test_manager,
            platform_target,
            _make_range_inputs(q_fill, k_fill, std),
            (_RANGE_B, _RANGE_SQ, _RANGE_D),
        )
