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

from types import SimpleNamespace
from typing import Any, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.batch_norm.batch_norm_bwd import _build_memory_config, batch_norm_bwd
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
    use_residual=False,
    compute_dresidual=False,
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

    use_residual adds a residual chosen to reverse the sign of the batchnorm-only output at every
    element. The final pre-ReLU value is driven to approximately +/-2, away from the ReLU kink, so
    the test proves the residual participates in the rematerialized predicate.
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

    if use_residual:
        channel_shape = (1, channels, 1, 1, 1)
        normalized = (x_f32 - means.reshape(channel_shape)) / np.sqrt(variances.reshape(channel_shape) + eps)
        batch_norm_output = normalized * gamma.reshape(channel_shape) + beta.reshape(channel_shape)
        opposite_target = np.where(batch_norm_output > 0, -2.0, 2.0)
        inputs["residuals_in"] = (opposite_target - batch_norm_output).astype(dtype)

    if compute_dresidual:
        inputs["compute_dresidual"] = True

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


# Bound on max |sum(dx)| per channel; measured 0.126 on trn2.
_CONV_BIAS_GRADIENT_ATOL = 0.5


def conv_bias_gradient_comparator(golden_dict, output_tensors, *, min_pass_rate):
    """Check per-channel sum(dx), a preceding conv's bias gradient, against its exact value of 0."""
    validators = per_output_comparator(
        1e-2,
        1e-2,
        1e-2,
        1e-2,
        min_pass_rate=min_pass_rate,
    )(golden_dict, output_tensors)
    golden_dx = golden_dict["dx"]
    if hasattr(golden_dx, "numpy"):
        golden_dx = golden_dx.numpy()
    golden_dx = golden_dx.astype(np.float32)
    reference_bias_gradient = golden_dx.sum(axis=(0, 2, 3, 4), dtype=np.float32)
    dx_dtype = output_tensors["dx"].dtype
    pointwise_validator = validators["dx"].validator

    class ConvBiasDxValidator(pointwise_validator):
        @override
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            pointwise_ok = super().validate(inference_output)
            actual_dx = np.frombuffer(inference_output, dtype=dx_dtype).reshape(golden_dx.shape)
            actual_bias_gradient = actual_dx.astype(np.float32).sum(axis=(0, 2, 3, 4), dtype=np.float32)
            max_actual = np.abs(actual_bias_gradient).max()
            bias_ok = max_actual <= _CONV_BIAS_GRADIENT_ATOL
            self._print_with_log(
                "Conv bias gradient from BN dx (exact value 0): "
                f"kernel max_abs={max_actual:.6g}, "
                f"reference max_abs={np.abs(reference_bias_gradient).max():.6g}, "
                f"atol={_CONV_BIAS_GRADIENT_ATOL}"
            )
            return pointwise_ok and bool(bias_ok)

    validators["dx"] = CustomValidatorWithOutputTensorData(
        validator=ConvBiasDxValidator,
        output_ndarray=output_tensors["dx"],
    )
    return validators


PARAM_NAMES = "batch, channels, depth, height, width, fuse_relu, use_relu_mask, dtype, x_mean, eps, const_channel"

BATCH_NORM_BWD_PARAMS = [
    # Small unreplicated explicit-mask path.
    pytest.param(1, 32, 2, 8, 8, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Float32 inputs on the recomputed-mask path exercise strict affine-gradient tolerances.
    pytest.param(1, 32, 2, 8, 8, True, False, nl.float32, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Multi-batch: dbeta / dgamma accumulate across the batch loop as well as spatial tiles.
    pytest.param(4, 64, 1, 8, 8, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    pytest.param(4, 64, 1, 8, 8, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Exactly one full partition tile of channels.
    pytest.param(2, 128, 1, 16, 16, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # More than one channel tile, with a partial trailing tile (192 = 128 + 64).
    pytest.param(1, 192, 1, 8, 8, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    pytest.param(1, 192, 1, 8, 8, True, True, nl.float32, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Recomputed mask across more than one channel tile: beta / true_beta are per-channel-tile
    # columns, so a shared column would make the second tile mask against the first tile's bias.
    pytest.param(1, 192, 1, 8, 8, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Non-power-of-two channels and spatial dims: partial channel tile + partial spatial tail.
    pytest.param(3, 37, 1, 5, 7, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Degenerate spatial extent (single element per channel per batch).
    pytest.param(8, 64, 1, 1, 1, False, False, nl.float32, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Partition replication at the maximum factor: C=32 fills 128 partitions with 4 bands and the
    # batch divides exactly. The other C=32 cases have B=1 and run unreplicated.
    pytest.param(8, 32, 1, 8, 8, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    pytest.param(8, 32, 1, 8, 8, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Recomputed mask under replication: true_beta must be broadcast to every band, exactly like
    # gamma_rstd. Without the broadcast the bands above 0 mask against a stale bias.
    pytest.param(8, 32, 1, 8, 8, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Offset x_in with a recomputed mask: true_beta subtracts means * true_gamma, so a large mean is
    # where dropping that term shows up as an all-ones mask. No zero-mean config can catch it.
    pytest.param(4, 64, 1, 8, 8, True, False, nl.bfloat16, 20.0, 1e-5, False, marks=pytest.mark.slow),
    # Float32 holds dgamma to a strict tolerance in the cancellation regime.
    pytest.param(1, 32, 2, 8, 8, False, False, nl.float32, 20.0, 1e-5, False, marks=pytest.mark.slow),
    # Resident-tail split: work_units > interleave, so phase 1 skips the store for its last tiles and
    # phase 2 reads them from SBUF. Configs above are all-resident. The second also replicates.
    pytest.param(32, 128, 1, 16, 16, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    pytest.param(64, 64, 1, 16, 16, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Recomputed mask with a resident tail: x_in_post_bn slots rotate over the whole walk, so a stale
    # slot shows up here but not in the all-resident configs above. Replicates too (C=64 -> rep 2).
    pytest.param(64, 64, 1, 16, 16, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.fast),
    # n_sp_tiles > 1 with a PARTIAL trailing spatial tile: spatial 9216 caps f_tile at 8192, giving 2
    # tiles with the second 1024 wide. Elsewhere n_sp_tiles is 1. Second is batch-sharded with rep 4.
    pytest.param(1, 32, 1, 96, 96, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    pytest.param(8, 32, 1, 96, 96, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Recomputed mask with a partial trailing spatial tile: x_in_post_bn is sliced to sp_size like
    # every other per-tile buffer, so a full-width activation would read past the loaded x_in.
    pytest.param(1, 32, 1, 96, 96, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Both mechanisms at once: spatial 18432 gives 3 spatial tiles (last 2048 wide) AND work_units 3 >
    # interleave 2, so the resident-tail handoff composes with the multi-spatial-tile walk.
    pytest.param(2, 32, 2, 96, 96, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Depth > 1 at a larger spatial extent.
    pytest.param(2, 64, 4, 32, 32, False, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    pytest.param(2, 64, 4, 32, 32, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Larger, closer to a real conv3d activation shape.
    pytest.param(4, 256, 2, 32, 32, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # ConvFuser configs
    pytest.param(64, 64, 1, 64, 64, True, True, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # Recomputed mask at the ConvFuser shape: the only config in the boundary regime, where one
    # channel's ReLU threshold falls within the engines' operand precision of a representable x_in.
    pytest.param(64, 64, 1, 64, 64, True, False, nl.bfloat16, 0.0, 1e-5, False, marks=pytest.mark.slow),
    # At unit variance eps=1.0 shifts rstd by ~33%, catching a hardcoded eps.
    pytest.param(1, 32, 2, 8, 8, False, False, nl.bfloat16, 0.0, 1.0, False, marks=pytest.mark.slow),
    # Zero variance via a constant channel (N > 1, so dx does not cancel): rstd collapses to
    # 1/sqrt(eps), the regime eps exists to guard. dx there scales as 1/sqrt(eps), ~100x between
    # these two rows, so it is also the strongest eps-detection vector in the file.
    pytest.param(2, 32, 1, 8, 8, False, False, nl.float32, 0.0, 1e-1, True, marks=pytest.mark.slow),
    pytest.param(2, 32, 1, 8, 8, True, False, nl.float32, 0.0, 1e-5, True, marks=pytest.mark.slow),
    # Recomputed mask: eps moves the mask threshold, flipping whole elements of dy' rather than
    # perturbing them.
    pytest.param(1, 32, 2, 8, 8, True, False, nl.bfloat16, 0.0, 1e-1, False, marks=pytest.mark.slow),
    # Stress test: largest shape that fits a device, 134 MB per bf16 tensor and ~0.7 GB across the
    # five resident tensors. slow_simulation since the host float32 reference needs tens of GB.
    pytest.param(
        64,
        128,
        2,
        64,
        64,
        True,
        True,
        nl.bfloat16,
        0.0,
        1e-5,
        False,
        marks=[pytest.mark.slow, pytest.mark.slow_simulation],
    ),
]

RESIDUAL_PARAM_NAMES = (
    "batch, channels, depth, height, width, use_relu_mask, dtype, logical_nc_config, compute_dresidual"
)

BATCH_NORM_BWD_RESIDUAL_PARAMS = [
    # The generated residual reverses every BN-only predicate and keeps the final value far from 0.
    # dx and dresidual have the same shape but different values, guarding positional output order.
    pytest.param(
        2,
        64,
        1,
        8,
        8,
        False,
        nl.bfloat16,
        1,
        True,
        marks=pytest.mark.slow,
        id="residual_sign_flip_output_order",
    ),
    # Two channel tiles, split across LNC=2.
    pytest.param(
        2,
        192,
        1,
        8,
        8,
        False,
        nl.bfloat16,
        2,
        True,
        marks=pytest.mark.fast,
        id="residual_multiple_channel_tiles_lnc2",
    ),
    # One channel tile forces LNC=2 to shard the batch and reduce the affine gradients.
    pytest.param(
        4,
        64,
        1,
        8,
        8,
        False,
        nl.bfloat16,
        2,
        True,
        marks=pytest.mark.slow,
        id="residual_batch_sharded_lnc2",
    ),
    pytest.param(2, 64, 1, 8, 8, False, nl.float32, 1, True, marks=pytest.mark.slow),
    # Preserve the explicit-mask API while still returning dresidual.
    pytest.param(
        2,
        64,
        1,
        8,
        8,
        True,
        nl.float32,
        1,
        True,
        marks=pytest.mark.fast,
        id="residual_explicit_mask",
    ),
    # Residual rematerialization must also preserve the original three-output ABI.
    pytest.param(
        2,
        64,
        1,
        8,
        8,
        False,
        nl.bfloat16,
        1,
        False,
        marks=pytest.mark.slow,
        id="residual_without_dresidual",
    ),
    # More work units than the maximum interleave forces direct_x through HBM between phases.
    pytest.param(
        64,
        64,
        1,
        16,
        16,
        False,
        nl.bfloat16,
        1,
        True,
        marks=pytest.mark.slow,
        id="residual_spill_reload",
    ),
    # Spatial 9216 exceeds _MAX_F_TILE and leaves a partial trailing residual/dresidual tile.
    pytest.param(
        1,
        32,
        1,
        96,
        96,
        False,
        nl.bfloat16,
        1,
        True,
        marks=pytest.mark.slow,
        id="residual_partial_spatial_tail",
    ),
]


@pytest.mark.fast
def test_batch_norm_bwd_memory_config_reduces_spatial_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Halve f_tile after every interleave depth fails at the initial size."""
    monkeypatch.setattr(nl, "tile_size", SimpleNamespace(total_available_sbuf_size=96 * 1024))

    mem_cfg = _build_memory_config(
        spatial=8192,
        batch=1,
        n_c_tiles=1,
        n_owned_c_tiles=1,
        replication=1,
        dtype_size=2,
        fuse_relu=True,
        recompute_relu_mask=True,
        recompute_with_residual=True,
    )

    assert mem_cfg.f_tile == 4096  # noqa: S101
    assert mem_cfg.n_sp_tiles == 2  # noqa: S101
    assert mem_cfg.interleave == 2  # noqa: S101


@pytest.mark.fast
def test_batch_norm_bwd_memory_config_raises_when_nothing_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a configuration after exhausting every possible spatial tile size."""
    monkeypatch.setattr(nl, "tile_size", SimpleNamespace(total_available_sbuf_size=1024))

    with pytest.raises(AssertionError, match="No SBUF-fitting memory configuration"):
        _build_memory_config(
            spatial=64,
            batch=1,
            n_c_tiles=1,
            n_owned_c_tiles=1,
            replication=1,
            dtype_size=2,
            fuse_relu=True,
            recompute_relu_mask=True,
            recompute_with_residual=True,
        )


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

    @pytest.mark.parametrize(
        "use_relu_mask",
        [False, True],
        ids=["recompute_relu_mask", "explicit_relu_mask"],
    )
    def test_conv_bias_gradient_after_batch_norm(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        use_relu_mask: bool,
    ) -> None:
        """Check sum(dx) for a large BF16 BN backward, with dy correlated with x_hat as real gradients are."""

        def input_generator(_test_config):
            inputs = generate_inputs(10, 64, 1, 23, 44, True, False, nl.bfloat16)
            # Match the affine parameters of a freshly initialized BatchNorm2d.
            inputs["gamma"] = np.ones((64, 1), dtype=np.float32)
            inputs["beta"] = np.zeros((64, 1), dtype=np.float32)
            channel_shape = (1, 64, 1, 1, 1)
            normalized = (inputs["x_in"].astype(np.float32) - inputs["means"].reshape(channel_shape)) / np.sqrt(
                inputs["variances"].reshape(channel_shape) + inputs["eps"]
            )
            noise = inputs["dy"].astype(np.float32)
            inputs["dy"] = (0.9 * normalized + 0.1 * noise).astype(nl.bfloat16)
            if use_relu_mask:
                inputs["relu_mask"] = (normalized > 0).astype(np.uint8)
            return inputs

        def output_tensors(kernel_input):
            return {
                "dx": np.zeros(kernel_input["dy"].shape, dtype=nl.bfloat16),
                "dgamma": np.zeros((64, 1), dtype=np.float32),
                "dbeta": np.zeros((64, 1), dtype=np.float32),
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
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            custom_comparator=lambda golden, outputs: conv_bias_gradient_comparator(
                golden,
                outputs,
                min_pass_rate=1.0 if use_relu_mask else _RECOMPUTE_MASK_MIN_PASS_RATE,
            ),
        )

    def _run_residual_case(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        channels: int,
        depth: int,
        height: int,
        width: int,
        use_relu_mask: bool,
        dtype,
        logical_nc_config: int,
        compute_dresidual: bool,
    ) -> None:
        def input_generator(test_config):
            return generate_inputs(
                batch,
                channels,
                depth,
                height,
                width,
                True,
                use_relu_mask,
                dtype,
                use_residual=True,
                compute_dresidual=compute_dresidual,
            )

        def output_tensors(kernel_input):
            # Insertion order is the kernel ABI: dx, dgamma, dbeta, then optional dresidual.
            outputs = {
                "dx": np.zeros((batch, channels, depth, height, width), dtype=dtype),
                "dgamma": np.zeros((channels, 1), dtype=np.float32),
                "dbeta": np.zeros((channels, 1), dtype=np.float32),
            }
            if compute_dresidual:
                outputs["dresidual"] = np.zeros(
                    (batch, channels, depth, height, width),
                    dtype=dtype,
                )
            return outputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=batch_norm_bwd,
            torch_ref=torch_ref_wrapper(batch_norm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        exact_rtol = 1e-2 if dtype == nl.bfloat16 else 1e-3
        exact_atol = 1e-2 if dtype == nl.bfloat16 else 1e-5
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                logical_nc_config=logical_nc_config,
            ),
            custom_comparator=per_output_comparator(
                1e-2,
                1e-2,
                exact_rtol,
                exact_atol,
            ),
        )

    @pytest_parametrize(RESIDUAL_PARAM_NAMES, BATCH_NORM_BWD_RESIDUAL_PARAMS)
    def test_batch_norm_bwd_residual(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        channels: int,
        depth: int,
        height: int,
        width: int,
        use_relu_mask: bool,
        dtype,
        logical_nc_config: int,
        compute_dresidual: bool,
    ) -> None:
        self._run_residual_case(
            test_manager,
            platform_target,
            batch,
            channels,
            depth,
            height,
            width,
            use_relu_mask,
            dtype,
            logical_nc_config,
            compute_dresidual,
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

    @pytest.mark.parametrize(
        "invalid_case, error_match",
        [
            pytest.param(
                "missing_residual",
                "compute_dresidual=True requires residuals_in",
                marks=pytest.mark.fast,
                id="compute_dresidual_requires_residual",
            ),
            pytest.param(
                "residual_without_relu",
                "residuals_in requires fuse_relu=True",
                marks=pytest.mark.fast,
                id="residual_requires_relu",
            ),
            pytest.param(
                "residual_shape",
                "residuals_in must have the same shape as dy",
                marks=pytest.mark.fast,
                id="residual_shape_mismatch",
            ),
            pytest.param(
                "residual_dtype",
                "residuals_in dtype",
                marks=pytest.mark.fast,
                id="residual_dtype_mismatch",
            ),
        ],
    )
    def test_batch_norm_bwd_invalid_residual_arguments_raise(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        invalid_case: str,
        error_match: str,
    ) -> None:
        batch, channels, depth, height, width = 2, 64, 1, 8, 8
        dtype = nl.bfloat16

        def input_generator(test_config):
            if invalid_case == "missing_residual":
                inputs = generate_inputs(
                    batch,
                    channels,
                    depth,
                    height,
                    width,
                    True,
                    False,
                    dtype,
                    compute_dresidual=True,
                )
            else:
                inputs = generate_inputs(
                    batch,
                    channels,
                    depth,
                    height,
                    width,
                    invalid_case != "residual_without_relu",
                    False,
                    dtype,
                    use_residual=True,
                    compute_dresidual=invalid_case != "residual_without_relu",
                )
            if invalid_case == "residual_shape":
                inputs["residuals_in"] = inputs["residuals_in"][:, :, :, :, :-1]
            elif invalid_case == "residual_dtype":
                inputs["residuals_in"] = inputs["residuals_in"].astype(np.float32)
            return inputs

        def output_tensors(kernel_input):
            outputs = {
                "dx": np.zeros((batch, channels, depth, height, width), dtype=dtype),
                "dgamma": np.zeros((channels, 1), dtype=np.float32),
                "dbeta": np.zeros((channels, 1), dtype=np.float32),
            }
            if kernel_input.get("compute_dresidual", False):
                outputs["dresidual"] = np.zeros(
                    (batch, channels, depth, height, width),
                    dtype=dtype,
                )
            return outputs

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=batch_norm_bwd,
            torch_ref=torch_ref_wrapper(batch_norm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        with pytest.raises(Exception, match=error_match):
            framework.run_test(
                test_config=None,
                compiler_args=CompilerArgs(
                    platform_target=platform_target,
                    logical_nc_config=1,
                ),
            )
