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

"""Tests for the batch_norm_bwd kernel."""

from typing import Any, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.batch_norm.batch_norm_bwd import batch_norm_bwd
from nkilib_src.nkilib.experimental.batch_norm.batch_norm_bwd_torch import batch_norm_bwd_torch_ref
from typing_extensions import override

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_inputs(
    batch,
    channels,
    depth,
    height,
    width,
    fuse_relu,
    use_relu_mask,
    dtype,
    x_mean=0.0,
    eps=1e-5,
    const_channel=False,
):
    """Generate batch_norm_bwd kernel inputs.

    means / variances are the statistics a batchnorm forward would have saved for x_in, so
    they are computed from x_in rather than drawn at random — that keeps the backward math
    on the same footing as a real forward/backward pair.

    x_mean offsets x_in away from zero. The default of 0 leaves dgamma's two terms
    (sum(dy * x_in) and mean * dbeta) the same order of magnitude; a large offset makes them
    nearly cancel, which is the regime that exposes precision loss in the product tile.

    use_relu_mask leaves relu_mask out of the inputs when False. With fuse_relu that selects the
    kernel's recompute path (the mask is rebuilt from x_in and the statistics), which the torch
    reference mirrors by deriving the mask from its own forward output.

    eps is returned in the inputs so the kernel and the reference normalize by the same scale.

    const_channel forces channel 0 to a single value, making its saved variance exactly zero so
    rstd collapses to 1/sqrt(eps) — the divide-by-zero regime eps exists to guard.
    """
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()
    # x_in gets its own seed, not just its own mean: one generator replays a single stream, so a
    # shared one would make x_in ~ dy + x_mean. dy would then lie almost entirely in the subspace
    # batchnorm-backward annihilates, collapsing dx ~100x and voiding its relative tolerance.
    x_generate_tensor = gaussian_tensor_generator(mean=x_mean, seed=7)

    shape = (batch, channels, depth, height, width)
    dy = generate_tensor(name="dy", shape=shape, dtype=dtype)
    x_in = x_generate_tensor(name="x_in", shape=shape, dtype=dtype)
    if const_channel:
        # Channel 0 constant -> var == 0 exactly, so rstd is decided entirely by eps.
        x_in[:, 0, ...] = x_mean + 1.0

    # Statistics are reduced over batch and spatial dims, always float32.
    x_f32 = x_in.astype(np.float32)
    means = x_f32.mean(axis=(0, 2, 3, 4)).astype(np.float32)
    variances = x_f32.var(axis=(0, 2, 3, 4)).astype(np.float32)

    # gamma / beta are the per-channel affine params, [C, 1] float32.
    gamma = generate_tensor(name="gamma", shape=(channels, 1), dtype=np.float32)
    beta = generate_tensor(name="beta", shape=(channels, 1), dtype=np.float32)

    inputs = {
        "dy": dy,
        "x_in": x_in,
        "means": means,
        "variances": variances,
        "gamma": gamma,
        "beta": beta,
        "eps": eps,
        "fuse_relu": fuse_relu,
    }

    if use_relu_mask:
        # Mask of the forward ReLU: uint8, 1 where it passed its input through, 0 where it clipped.
        # Thresholded from a normal draw rather than drawn as integers, which keeps it ~50% ones.
        mask = generate_tensor(name="relu_mask", shape=shape, dtype=np.float32)
        inputs["relu_mask"] = (mask > 0).astype(np.uint8)

    return inputs


# dx goes through the kernel's bf16 intermediates (direct_x, var_x), so it carries ~2^-8 rounding at
# any input dtype. dgamma / dbeta reduce in float32 and stay exact.
_BF16_PRECISION_OUTPUTS = ("dx",)


# Element budget for the recomputed-ReLU-mask path (see the kernel docstring's Notes). A pass RATE,
# not a looser bar: 2e-5 leaves ~1.9x slack over the measured 179-of-16.7M worst case (335 free
# elements) while still failing a 5x regression. Inert below ~50k elements, so the fast vectors get
# no allowance at all — only the large recompute config can spend it.
_RECOMPUTE_MASK_MIN_PASS_RATE = 1.0 - 2e-5


def per_output_comparator(bf16_rtol, bf16_atol, exact_rtol, exact_atol, min_pass_rate=1.0):
    """Build a custom_comparator giving the bf16-precision outputs a looser tolerance.

    A single rtol for the whole test would have to be the bf16 one, which would stop checking
    dbeta — the one output the kernel computes entirely in float32 — at its real precision.

    min_pass_rate below 1.0 additionally allows that fraction of elements to miss the bar
    outright, for the recomputed-ReLU-mask path — see _RECOMPUTE_MASK_MIN_PASS_RATE.
    """
    from test.utils.comparators import maxAllClose

    def comparator(golden_dict, output_tensors):
        validators = {}
        for key in output_tensors:
            golden = golden_dict[key]
            if hasattr(golden, "numpy"):
                golden = golden.numpy()
            golden = golden.astype(np.float32)
            numpy_dtype = output_tensors[key].dtype
            if key in _BF16_PRECISION_OUTPUTS:
                rtol, atol = bf16_rtol, bf16_atol
                # The ReLU-kink allowance applies only to dx. dgamma / dbeta reduce in float32 and
                # are exact, so granting them a pass rate would contradict that.
                key_min_pass_rate = min_pass_rate
            else:
                rtol, atol = exact_rtol, exact_atol
                key_min_pass_rate = 1.0

            # Bind the loop variables as defaults: the class body below is evaluated per
            # iteration, but validate() runs later, after the loop has moved on.
            class PerOutputValidator(CustomValidator):
                @override
                def validate(
                    self,
                    inference_output: npt.NDArray[Any],
                    _golden=golden,
                    _numpy_dtype=numpy_dtype,
                    _rtol=rtol,
                    _atol=atol,
                    _min_pass_rate=key_min_pass_rate,
                ) -> bool:
                    actual = np.frombuffer(inference_output, dtype=_numpy_dtype).reshape(_golden.shape)
                    return maxAllClose(
                        actual.astype(np.float32),
                        _golden,
                        rtol=_rtol,
                        atol=_atol,
                        verbose=1,
                        logfile=self.logfile,
                        min_pass_rate=_min_pass_rate,
                    )

            validators[key] = CustomValidatorWithOutputTensorData(
                validator=PerOutputValidator,
                output_ndarray=output_tensors[key],
            )
        return validators

    return comparator


PARAM_NAMES = "batch, channels, depth, height, width, fuse_relu, use_relu_mask, dtype, x_mean, eps, const_channel"

BATCH_NORM_BWD_PARAMS = [
    # Baseline: single batch, sub-partition channel count, one spatial tile.
    pytest.param(1, 32, 2, 8, 8, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Fused ReLU on the same shape, so the mask path is the only difference.
    pytest.param(1, 32, 2, 8, 8, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Fused ReLU with NO mask supplied: recomputed from x_in and the saved statistics. Same shape as
    # the two above, so the recomputation is the only difference.
    pytest.param(1, 32, 2, 8, 8, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(1, 32, 2, 8, 8, True, False, nl.float32, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # float32 inputs: exercises the non-bf16 load/store path at strict tolerance.
    pytest.param(1, 32, 2, 8, 8, False, False, nl.float32, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Multi-batch: dbeta / dgamma accumulate across the batch loop as well as spatial tiles.
    pytest.param(4, 64, 1, 8, 8, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(4, 64, 1, 8, 8, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Exactly one full partition tile of channels.
    pytest.param(2, 128, 1, 16, 16, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # More than one channel tile, with a partial trailing tile (192 = 128 + 64).
    pytest.param(1, 192, 1, 8, 8, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(1, 192, 1, 8, 8, True, True, nl.float32, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Recomputed mask across more than one channel tile: beta / true_beta are per-channel-tile
    # columns, so a shared column would make the second tile mask against the first tile's bias.
    pytest.param(1, 192, 1, 8, 8, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Non-power-of-two channels and spatial dims: partial channel tile + partial spatial tail.
    pytest.param(3, 37, 1, 5, 7, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Degenerate spatial extent (single element per channel per batch).
    pytest.param(8, 64, 1, 1, 1, False, False, nl.float32, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Partition replication at the maximum factor: C=32 fills 128 partitions with 4 bands and the
    # batch divides exactly. The other C=32 cases have B=1 and run unreplicated.
    pytest.param(8, 32, 1, 8, 8, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(8, 32, 1, 8, 8, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Recomputed mask under replication: true_beta must be broadcast to every band, exactly like
    # direct_x_scale. Without the broadcast the bands above 0 mask against a stale bias.
    pytest.param(8, 32, 1, 8, 8, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Offset x_in: dgamma's two terms cancel once x_in's mean dominates its spread, where a bf16
    # product tile loses most of its digits (~20% error at mean 5, ~69% at 20). Untested at mean 0.
    pytest.param(4, 64, 1, 8, 8, False, False, nl.bfloat16, 5.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(4, 64, 1, 8, 8, True, True, nl.bfloat16, 20.0, 1e-5, False, marks=pytest.mark.fast),
    # Offset x_in with a recomputed mask: true_beta subtracts means * true_gamma, so a large mean is
    # where dropping that term shows up as an all-ones mask. No zero-mean config can catch it.
    pytest.param(4, 64, 1, 8, 8, True, False, nl.bfloat16, 20.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(1, 32, 2, 8, 8, False, False, nl.float32, 20.0, 1e-5, False, marks=pytest.mark.fast),
    # Resident-tail split: work_units > interleave, so phase 1 skips the store for its last tiles and
    # phase 2 reads them from SBUF. Configs above are all-resident. The second also replicates.
    pytest.param(32, 128, 1, 16, 16, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(64, 64, 1, 16, 16, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Recomputed mask with a resident tail: x_in_post_bn slots rotate over the whole walk, so a stale
    # slot shows up here but not in the all-resident configs above. Replicates too (C=64 -> rep 2).
    pytest.param(64, 64, 1, 16, 16, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # n_sp_tiles > 1 with a PARTIAL trailing spatial tile: spatial 9216 caps f_tile at 8192, giving 2
    # tiles with the second 1024 wide. Elsewhere n_sp_tiles is 1. Second is batch-sharded with rep 4.
    pytest.param(1, 32, 1, 96, 96, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(8, 32, 1, 96, 96, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Recomputed mask with a partial trailing spatial tile: x_in_post_bn is sliced to sp_size like
    # every other per-tile buffer, so a full-width activation would read past the loaded x_in.
    pytest.param(1, 32, 1, 96, 96, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Both mechanisms at once: spatial 18432 gives 3 spatial tiles (last 2048 wide) AND work_units 3 >
    # interleave 2, so the resident-tail handoff composes with the multi-spatial-tile walk.
    pytest.param(2, 32, 2, 96, 96, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # Depth > 1 at a larger spatial extent.
    pytest.param(2, 64, 4, 32, 32, False, False, nl.bfloat16, 0.0, 1e-5, False),
    pytest.param(2, 64, 4, 32, 32, True, True, nl.bfloat16, 0.0, 1e-5, False),
    # Larger, closer to a real conv3d activation shape.
    pytest.param(4, 256, 2, 32, 32, True, True, nl.bfloat16, 0.0, 1e-5, False),
    # ConvFuser configs
    pytest.param(64, 64, 1, 64, 64, True, True, nl.bfloat16, 0.0, 1e-5, False),
    # Recomputed mask at the ConvFuser shape: the only config in the boundary regime, where one
    # channel's ReLU threshold falls within the engines' operand precision of a representable x_in.
    pytest.param(64, 64, 1, 64, 64, True, False, nl.bfloat16, 0.0, 1e-5, False),
    # eps sweep, one shape unless noted. At unit variance eps shifts rstd ~5.7% at 1e-1 and ~33% at
    # 1.0 but only ~0.06% at 1e-3, so only the large values catch a hardcoded eps (verified).
    pytest.param(1, 32, 2, 8, 8, False, False, nl.bfloat16, 0.0, 1.0, False, marks=pytest.mark.fast),
    pytest.param(1, 32, 2, 8, 8, False, False, nl.bfloat16, 0.0, 1e-1, False, marks=pytest.mark.fast),
    # 1e-3 / 1e-8 shift rstd < 0.05% at unit variance, below every tolerance here, so they cannot
    # detect a wrong eps; kept unmarked as range coverage without paying fast-suite time.
    pytest.param(1, 32, 2, 8, 8, False, False, nl.bfloat16, 0.0, 1e-3, False),
    pytest.param(1, 32, 2, 8, 8, False, False, nl.bfloat16, 0.0, 1e-8, False),
    # Zero variance via a constant channel (N > 1, so dx does not cancel): rstd collapses to
    # 1/sqrt(eps), the regime eps exists to guard. dx there scales as 1/sqrt(eps), ~100x between
    # these two rows, so it is also the strongest eps-detection vector in the file.
    pytest.param(2, 32, 1, 8, 8, False, False, nl.float32, 0.0, 1e-1, True, marks=pytest.mark.fast),
    pytest.param(2, 32, 1, 8, 8, True, False, nl.float32, 0.0, 1e-5, True, marks=pytest.mark.fast),
    # float32 holds dgamma / dbeta to the strict 1e-3 bar, catching eps applied at a wrong precision.
    pytest.param(1, 32, 2, 8, 8, False, False, nl.float32, 0.0, 1e-1, False, marks=pytest.mark.fast),
    # Recomputed mask: eps moves the mask threshold, flipping whole elements of dy' rather than
    # perturbing them.
    pytest.param(1, 32, 2, 8, 8, True, False, nl.bfloat16, 0.0, 1e-1, False, marks=pytest.mark.fast),
    # Batch sharding + replication, where each core derives rstd and they must agree.
    pytest.param(8, 32, 1, 8, 8, True, True, nl.bfloat16, 0.0, 1e-1, False, marks=pytest.mark.fast),
    # Two channel tiles, one per core at LNC=2, so each core's own rsqrt sees eps.
    pytest.param(1, 192, 1, 8, 8, False, False, nl.bfloat16, 0.0, 1e-1, False, marks=pytest.mark.fast),
    # Stress test: largest shape that fits a device, 134 MB per bf16 tensor and ~0.7 GB across the
    # five resident tensors. slow_simulation since the host float32 reference needs tens of GB.
    pytest.param(64, 128, 2, 64, 64, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow_simulation),
]


@pytest_test_metadata(name="BatchNorm Backward")
@pytest_marks(["batch_norm", "backward"])
@final
class TestBatchNormBwd:
    """Validation tests for the batch_norm_bwd kernel."""

    @pytest_parametrize(PARAM_NAMES, BATCH_NORM_BWD_PARAMS)
    def test_batch_norm_bwd(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        channels: int,
        depth: int,
        height: int,
        width: int,
        fuse_relu: bool,
        use_relu_mask: bool,
        dtype,
        x_mean: float,
        eps: float,
        const_channel: bool,
    ) -> None:
        """Run batch_norm_bwd against the torch reference for a single configuration."""

        def input_generator(test_config):
            return generate_inputs(
                batch,
                channels,
                depth,
                height,
                width,
                fuse_relu,
                use_relu_mask,
                dtype,
                x_mean,
                eps,
                const_channel,
            )

        def output_tensors(kernel_input):
            # Keys are matched positionally against the kernel's returned tuple
            # (dx, dgamma, dbeta), so their insertion order must match it.
            return {
                "dx": np.zeros((batch, channels, depth, height, width), dtype=dtype),
                "dgamma": np.zeros((channels, 1), dtype=np.float32),
                "dbeta": np.zeros((channels, 1), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=batch_norm_bwd,
            torch_ref=torch_ref_wrapper(batch_norm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        # dx gets the bf16 tolerance: direct_x and var_x are bf16, so it carries ~2^-8 rounding even
        # for float32 inputs (measured up to 0.33% on hardware). See _BF16_PRECISION_OUTPUTS.
        bf16_rtol, bf16_atol = 1e-2, 1e-2
        # dbeta is reduced straight from dy in the Scalar Engine's float32 accumulator, so it
        # keeps the input dtype's bar (measured exact on hardware for every config).
        exact_rtol = 1e-2 if dtype == nl.bfloat16 else 1e-3
        exact_atol = 1e-2 if dtype == nl.bfloat16 else 1e-5

        # A recomputed mask can disagree with the reference at the ReLU kink, so allow a tiny fraction
        # of such elements; every other element and config is held to the strict bar.
        min_pass_rate = _RECOMPUTE_MASK_MIN_PASS_RATE if (fuse_relu and not use_relu_mask) else 1.0

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            custom_comparator=per_output_comparator(
                bf16_rtol, bf16_atol, exact_rtol, exact_atol, min_pass_rate=min_pass_rate
            ),
        )

    @pytest.mark.fast
    def test_batch_norm_bwd_mask_dtype_raises(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ) -> None:
        """A non-uint8 relu_mask must be rejected by the kernel's validation."""
        batch, channels, depth, height, width = 1, 32, 2, 8, 8
        dtype = nl.bfloat16

        def input_generator(test_config):
            inputs = generate_inputs(batch, channels, depth, height, width, True, True, dtype)
            # The mask is validated on dtype whenever it is supplied, so a float32 mask is
            # rejected rather than silently reinterpreted.
            inputs["relu_mask"] = inputs["relu_mask"].astype(np.float32)
            return inputs

        def output_tensors(kernel_input):
            return {
                "dx": np.zeros((batch, channels, depth, height, width), dtype=dtype),
                "dgamma": np.zeros((channels, 1), dtype=np.float32),
                "dbeta": np.zeros((channels, 1), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=batch_norm_bwd,
            torch_ref=torch_ref_wrapper(batch_norm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            is_negative_test=True,
        )
