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

"""Integration tests for the Kimi Delta Attention (KDA) NKI kernels.

KDA is GatedDeltaNet with a **per-channel** (diagonal) forget gate instead of a
per-head scalar gate. The reference implementations follow FlashKDA
(https://github.com/MoonshotAI/FlashKDA) and flash-linear-attention's
``chunk_kda`` (https://github.com/fla-org/flash-linear-attention, MIT License).

Covers:
  - kda_prefill: KDA prefill (recurrent state + attention output)
  - kda_decode_step: Single-token per-channel-gated decode recurrence

The kernels take RAW pre-activation inputs: the QK L2-norm, the gate activation
``g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))`` and the beta
sigmoid are all fused on-chip, matching the upstream API.

Dimension constraints (from the NKI kernels):
  - dk, dv <= 128 (partition dimension); upstream requires dk == dv == 128
  - nv <= 128 (batched q/k transpose in decode)
  - T must be a multiple of C = 128 (prefill)
  - SUB = 16 sub-chunk, fixed: the prefill's k_inv = k * exp(-cumsum(g)) grows as
    exp(|lower_bound| * SUB), which overflows fp32 beyond 16
  - lower_bound in [-5, 0]
"""

from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_decode_batch_nki import kda_decode_batch
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_decode_batch_nki_torch import (
    kda_decode_batch_torch_ref,
)
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_decode_nki import kda_decode_step
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_decode_nki_torch import (
    kda_decode_step_torch_ref,
)
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_prefill_nki import (
    build_constant_masks,
    kda_prefill,
)
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_prefill_nki_torch import (
    kda_prefill_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Chunk tile size, must match the kernel.
_C = 128

# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def _generate_prefill_inputs(nv, T, dk, dv, seed_scale=0.0, lower_bound=-5.0, saturate_gate=False):
    """Generate inputs for kda_prefill.

    Args:
        nv: Number of value heads (GQA-expanded)
        T: Sequence length (must be a multiple of C = 128)
        dk: Key/query head dimension (<= 128)
        dv: Value head dimension (<= 128)
        seed_scale: 0.0 for fresh prefill, 1.0 for APC continuation
        lower_bound: Gate lower bound, in [-5, 0]
        saturate_gate: If True, drive the activated gate to ``lower_bound`` on
            every channel -- the worst case for the exp(-cumsum(g)) range.
    """
    np.random.seed(42)

    # q/k are L2-normalized in-kernel, so their input scale is irrelevant.
    q = np.random.randn(nv, T, dk).astype(np.float32)
    k = np.random.randn(nv, T, dk).astype(np.float32)
    v = (np.random.randn(nv, T, dv) * 0.1).astype(np.float32)

    if saturate_gate:
        # sigmoid(large) -> 1, so g -> lower_bound on every channel.
        g_raw = np.full((nv, T, dk), 20.0, dtype=np.float32)
        dt_bias = np.zeros((nv, dk), dtype=np.float32)
        a_log = np.zeros((nv, 1), dtype=np.float32)
    else:
        # Spread the activated gate across the full (lower_bound, 0) range.
        g_raw = (np.random.randn(nv, T, dk) * 3.0).astype(np.float32)
        dt_bias = (np.random.randn(nv, dk) * 0.5).astype(np.float32)
        a_log = np.random.rand(nv, 1).astype(np.float32)

    return {
        "q_ref": q,
        "k_ref": k,
        "v_ref": v,
        "g_ref": g_raw,
        "beta_ref": np.random.randn(nv, T, 1).astype(np.float32),
        "a_log_ref": a_log,
        "dt_bias_ref": dt_bias,
        **build_constant_masks(),
        "out.must_alias_input": np.zeros((nv, T, dv), dtype=np.float32),
        "state_ref.must_alias_input": (np.random.randn(nv, dk, dv) * 0.01).astype(np.float32),
        "seed_scale_ref": np.array([[seed_scale]], dtype=np.float32),
        "scale_ref": np.array([[1.0 / np.sqrt(dk)]], dtype=np.float32),
        "lower_bound_ref": np.array([[lower_bound]], dtype=np.float32),
    }


def _generate_decode_inputs(nv, dk, dv, lower_bound=-5.0, saturate_gate=False):
    """Generate inputs for kda_decode_step.

    Args:
        nv: Number of value heads
        dk: Key/query head dimension (<= 128)
        dv: Value head dimension (<= 128)
        lower_bound: Gate lower bound, in [-5, 0]
        saturate_gate: If True, drive g_raw very positive so the activated gate
            saturates at ``lower_bound`` on every channel -- the worst case for
            the exponential range of the decay.
    """
    np.random.seed(42)

    # q/k are L2-normalized in-kernel, so the input scale is irrelevant for them.
    q = np.random.randn(nv, dk).astype(np.float32)
    k = np.random.randn(nv, dk).astype(np.float32)
    v = np.random.randn(nv, dv).astype(np.float32) * 0.1

    if saturate_gate:
        # sigmoid(large) -> 1, so g -> lower_bound on every channel.
        g_raw = np.full((nv, dk), 20.0, dtype=np.float32)
        dt_bias = np.zeros((nv, dk), dtype=np.float32)
        a_log = np.zeros((nv, 1), dtype=np.float32)
    else:
        # Spread the activated gate across the full (lower_bound, 0) range.
        g_raw = (np.random.randn(nv, dk) * 3.0).astype(np.float32)
        dt_bias = (np.random.randn(nv, dk) * 0.5).astype(np.float32)
        a_log = (np.random.rand(nv, 1)).astype(np.float32)

    beta = np.random.randn(nv, 1).astype(np.float32)
    S_in = (np.random.randn(nv, dk, dv) * 0.01).astype(np.float32)

    return {
        "q_ref": q,
        "k_ref": k,
        "v_ref": v,
        "g_ref": g_raw,
        "beta_ref": beta,
        "a_log_ref": a_log,
        "dt_bias_ref": dt_bias,
        "out.must_alias_input": np.zeros((nv, dv), dtype=np.float32),
        "S_in_ref.must_alias_input": S_in,
        "scale_ref": np.array([[1.0 / np.sqrt(dk)]], dtype=np.float32),
        "lower_bound_ref": np.array([[lower_bound]], dtype=np.float32),
    }


def _generate_decode_batch_inputs(batch, nv, dk, dv, lower_bound=-5.0, saturate_gate=False):
    """Generate inputs for kda_decode_batch.

    Same distributions as _generate_decode_inputs with a leading batch axis;
    a_log and dt_bias stay unbatched since they are model parameters.

    Args:
        batch: Number of requests in the batch
        nv: Number of value heads
        dk: Key/query head dimension (<= 128)
        dv: Value head dimension (<= 128)
        lower_bound: Gate lower bound, in [-5, 0]
        saturate_gate: If True, drive g_raw very positive so the activated gate
            saturates at ``lower_bound`` on every channel -- the worst case for
            the exponential range of the decay.
    """
    np.random.seed(42)

    # q/k are L2-normalized in-kernel, so the input scale is irrelevant for them.
    q = np.random.randn(batch, nv, dk).astype(np.float32)
    k = np.random.randn(batch, nv, dk).astype(np.float32)
    v = (np.random.randn(batch, nv, dv) * 0.1).astype(np.float32)

    if saturate_gate:
        # sigmoid(large) -> 1, so g -> lower_bound on every channel.
        g_raw = np.full((batch, nv, dk), 20.0, dtype=np.float32)
        dt_bias = np.zeros((nv, dk), dtype=np.float32)
        a_log = np.zeros((nv, 1), dtype=np.float32)
    else:
        # Spread the activated gate across the full (lower_bound, 0) range.
        g_raw = (np.random.randn(batch, nv, dk) * 3.0).astype(np.float32)
        dt_bias = (np.random.randn(nv, dk) * 0.5).astype(np.float32)
        a_log = np.random.rand(nv, 1).astype(np.float32)

    return {
        "q_ref": q,
        "k_ref": k,
        "v_ref": v,
        "g_ref": g_raw,
        "beta_ref": np.random.randn(batch, nv, 1).astype(np.float32),
        "a_log_ref": a_log,
        "dt_bias_ref": dt_bias,
        "out.must_alias_input": np.zeros((batch, nv, dv), dtype=np.float32),
        "S_in_ref.must_alias_input": (np.random.randn(batch, nv, dk, dv) * 0.01).astype(np.float32),
        "scale_ref": np.array([[1.0 / np.sqrt(dk)]], dtype=np.float32),
        "lower_bound_ref": np.array([[lower_bound]], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Test class: KDA Prefill
# ---------------------------------------------------------------------------


@pytest_test_metadata(
    name="KDA Prefill",
    tags=["model"],
    pytest_marks=["attention", "kda", "prefill"],
)
@final
class TestKdaChunkPrefill:
    """Kimi Delta Attention prefill: verifies out and the in-place state update."""

    def _run(
        self,
        *,
        test_manager,
        platform_target,
        nv,
        T,
        dk,
        dv,
        seed_scale=0.0,
        lower_bound=-5.0,
        saturate_gate=False,
        rtol=0.2,
        atol=2e-3,
    ):
        # All matmul operands are bf16 with fp32 PSUM accumulation (~4x PE
        # throughput on trn2) while the torch_ref is pure fp32, so the two agree
        # only to bf16 mantissa precision. Measured worst case across the swept
        # configs is ~3e-4 absolute / ~5e-3 relative-to-max, and it does NOT grow
        # with chunk count (the state recurrence is stable). atol is the binding
        # constraint; rtol is loose because elementwise relative error is
        # meaningless on the near-zero entries. Same rationale as the sibling GDN
        # prefill test.
        def input_generator(test_config, input_tensor_def=None):
            return _generate_prefill_inputs(
                nv=nv,
                T=T,
                dk=dk,
                dv=dv,
                seed_scale=seed_scale,
                lower_bound=lower_bound,
                saturate_gate=saturate_gate,
            )

        def output_tensor_descriptor(kernel_input):
            return {
                "out": np.zeros((nv, T, dv), dtype=np.float32),
                "state_ref": np.zeros((nv, dk, dv), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kda_prefill,
            torch_ref=torch_ref_wrapper(kda_prefill_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )

    # --- Simulation tests (fast, no hardware required) ---

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize(
        "nv, dk, dv",
        [
            pytest.param(1, 128, 128, id="single_head"),
            pytest.param(2, 128, 128, id="two_heads"),
            # Odd nv > 1: under LNC2 both cores are non-empty AND the last core
            # absorbs the remainder head, unlike every evenly-divisible case.
            pytest.param(3, 128, 128, id="uneven_shard"),
            pytest.param(4, 128, 128, id="four_heads"),
            pytest.param(2, 64, 64, id="small_square"),
            # dk != dv: the state is non-square, so a dk/dv transposition in any
            # tile shape would surface here but not in the square cases.
            pytest.param(2, 128, 64, id="dk_gt_dv"),
            pytest.param(2, 64, 128, id="dv_gt_dk"),
        ],
    )
    @pytest.mark.parametrize("nchunks", [1, 2], ids=["1chunk", "2chunks"])
    def test_kda_prefill_sim(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        nv,
        dk,
        dv,
        nchunks,
    ):
        """Simulation-mode correctness test: sweeps head dims and chunk counts."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=nv,
            T=nchunks * _C,
            dk=dk,
            dv=dv,
        )

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize("lower_bound", [-5.0, -1.0], ids=["lb_neg5", "lb_neg1"])
    def test_kda_prefill_lower_bound(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        lower_bound,
    ):
        """Verifies the fused gate activation across the supported lower_bound range."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=2,
            T=_C,
            dk=128,
            dv=128,
            lower_bound=lower_bound,
        )

    @pytest.mark.fast
    @pytest.mark.simulation
    def test_kda_prefill_saturated_gate(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        """Worst case for the exp(-cumsum(g)) range: the gate saturates at lower_bound.

        This is the case that forces SUB = 16: at SUB = 32 the same inputs
        overflow fp32.
        """
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=2,
            T=2 * _C,
            dk=128,
            dv=128,
            saturate_gate=True,
        )

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize("seed_scale", [0.0, 1.0], ids=["fresh", "apc_continue"])
    def test_kda_prefill_seed_scale(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        seed_scale,
    ):
        """Verifies fresh prefill (zero state) vs APC continuation (seeded state)."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=2,
            T=2 * _C,
            dk=128,
            dv=128,
            seed_scale=seed_scale,
        )

    # --- Hardware tests (shared-fleet) ---

    @pytest_marks(["model"])
    @pytest.mark.parametrize(
        "nv, dk, dv, nchunks",
        [
            pytest.param(12, 128, 128, 4, id="tp8"),
            pytest.param(12, 128, 128, 16, id="tp8_seq2k"),
            pytest.param(96, 128, 128, 4, id="upstream_prod"),
            # Non-square state on hardware: the simulator does not model PSUM
            # bank geometry or partition-offset legality, so dk != dv needs its
            # own hardware check rather than relying on the simulation sweep.
            pytest.param(12, 128, 64, 2, id="dk_gt_dv"),
        ],
    )
    def test_kda_prefill_hw(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        nv,
        dk,
        dv,
        nchunks,
    ):
        """Hardware accuracy test: production-scale head counts and sequence lengths."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=nv,
            T=nchunks * _C,
            dk=dk,
            dv=dv,
        )


# ---------------------------------------------------------------------------
# Test class: KDA Decode Step
# ---------------------------------------------------------------------------


@pytest_marks(["attention", "kda", "decode"])
@final
class TestKdaDecodeStep:
    """Single-token Kimi Delta Attention decode step: verifies out and the in-place state update."""

    def _run(
        self,
        *,
        test_manager,
        platform_target,
        nv,
        dk,
        dv,
        lower_bound=-5.0,
        saturate_gate=False,
        rtol=1e-4,
        atol=1e-5,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_decode_inputs(
                nv=nv,
                dk=dk,
                dv=dv,
                lower_bound=lower_bound,
                saturate_gate=saturate_gate,
            )

        def output_tensor_descriptor(kernel_input):
            return {
                "out": np.zeros((nv, dv), dtype=np.float32),
                "S_in_ref": np.zeros((nv, dk, dv), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kda_decode_step,
            torch_ref=torch_ref_wrapper(kda_decode_step_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )

    # --- Simulation tests ---

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize(
        "nv, dk, dv",
        [
            pytest.param(1, 128, 128, id="single_head"),
            # Odd nv > 1: under LNC2 both cores are non-empty AND the last core
            # absorbs the remainder head, unlike every evenly-divisible case.
            pytest.param(3, 128, 128, id="uneven_shard"),
            pytest.param(4, 128, 128, id="four_heads"),
            pytest.param(12, 128, 128, id="tp8"),
            pytest.param(2, 64, 64, id="small_square"),
            # dk != dv: the state is non-square, so a dk/dv transposition in any
            # tile shape would surface here but not in the square cases.
            pytest.param(2, 128, 64, id="dk_gt_dv"),
            pytest.param(2, 64, 128, id="dv_gt_dk"),
            # nv == 128 is the partition-tile bound on the batched prologue.
            pytest.param(128, 128, 128, id="max_heads"),
        ],
    )
    def test_kda_decode_sim(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        nv,
        dk,
        dv,
    ):
        """Simulation-mode correctness test: sweeps head count and dimensions."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=nv,
            dk=dk,
            dv=dv,
        )

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize("lower_bound", [-5.0, -1.0], ids=["lb_neg5", "lb_neg1"])
    def test_kda_decode_lower_bound(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        lower_bound,
    ):
        """Verifies the fused gate activation across the supported lower_bound range."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=4,
            dk=128,
            dv=128,
            lower_bound=lower_bound,
        )

    @pytest.mark.fast
    @pytest.mark.simulation
    def test_kda_decode_saturated_gate(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        """Worst-case decay range: the activated gate saturates at lower_bound."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=4,
            dk=128,
            dv=128,
            saturate_gate=True,
        )

    # --- Hardware tests ---

    @pytest_marks(["model"])
    @pytest.mark.parametrize(
        "nv, dk, dv",
        [
            pytest.param(12, 128, 128, id="tp8"),
            pytest.param(96, 128, 128, id="upstream_prod"),
            # Non-square state on hardware: the simulator does not model PSUM
            # bank geometry or partition-offset legality, so dk != dv needs its
            # own hardware check rather than relying on the simulation sweep.
            pytest.param(12, 128, 64, id="dk_gt_dv"),
        ],
    )
    def test_kda_decode_hw(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        nv,
        dk,
        dv,
    ):
        """Hardware accuracy test: production head counts."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            nv=nv,
            dk=dk,
            dv=dv,
            rtol=1e-3,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Test class: KDA Batched Decode
# ---------------------------------------------------------------------------


@pytest_marks(["attention", "kda", "decode"])
@final
class TestKdaDecodeBatch:
    """Batched single-token KDA decode: batch requests per invocation."""

    def _run(
        self,
        *,
        test_manager,
        platform_target,
        batch,
        nv,
        dk,
        dv,
        lower_bound=-5.0,
        saturate_gate=False,
        rtol=1e-4,
        atol=1e-5,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_decode_batch_inputs(
                batch=batch,
                nv=nv,
                dk=dk,
                dv=dv,
                lower_bound=lower_bound,
                saturate_gate=saturate_gate,
            )

        def output_tensor_descriptor(kernel_input):
            return {
                "out": np.zeros((batch, nv, dv), dtype=np.float32),
                "S_in_ref": np.zeros((batch, nv, dk, dv), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kda_decode_batch,
            torch_ref=torch_ref_wrapper(kda_decode_batch_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=rtol,
            atol=atol,
        )

    # --- Simulation tests ---

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize(
        "batch, nv, dk, dv",
        [
            # batch == 1 must match the unbatched kernel exactly.
            pytest.param(1, 1, 128, 128, id="single_item"),
            pytest.param(2, 4, 128, 128, id="b2_four_heads"),
            # Odd batch*nv: under LNC2 both cores are non-empty AND the last core
            # absorbs the remainder item, unlike every evenly-divisible case.
            pytest.param(3, 3, 128, 128, id="uneven_shard"),
            pytest.param(4, 12, 128, 128, id="b4_tp8"),
            pytest.param(2, 2, 64, 64, id="small_square"),
            # dk != dv: the state is non-square, so a dk/dv transposition in any
            # tile shape would surface here but not in the square cases.
            pytest.param(2, 2, 128, 64, id="dk_gt_dv"),
            pytest.param(2, 2, 64, 128, id="dv_gt_dk"),
            # batch*nv > 128 forces the prologue to tile into multiple groups; 12
            # does not divide 128, so the second group starts mid-request and
            # exercises the run-coalescing path on a non-aligned boundary.
            pytest.param(16, 12, 128, 128, id="multi_group"),
        ],
    )
    def test_kda_decode_batch_sim(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        nv,
        dk,
        dv,
    ):
        """Simulation-mode correctness test: sweeps batch size, heads and dimensions."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            nv=nv,
            dk=dk,
            dv=dv,
        )

    @pytest.mark.fast
    @pytest.mark.simulation
    @pytest.mark.parametrize("lower_bound", [-5.0, -1.0], ids=["lb_neg5", "lb_neg1"])
    def test_kda_decode_batch_lower_bound(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        lower_bound,
    ):
        """Verifies the fused gate activation across the supported lower_bound range."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=2,
            nv=4,
            dk=128,
            dv=128,
            lower_bound=lower_bound,
        )

    @pytest.mark.fast
    @pytest.mark.simulation
    def test_kda_decode_batch_saturated_gate(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        """Worst-case decay range: the activated gate saturates at lower_bound."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=2,
            nv=4,
            dk=128,
            dv=128,
            saturate_gate=True,
        )

    # --- Hardware tests ---

    @pytest_marks(["model"])
    @pytest.mark.parametrize(
        "batch, nv, dk, dv",
        [
            # Batch sweep at the tp8 head count: shows the fixed per-launch cost
            # being amortized as batch grows.
            pytest.param(1, 12, 128, 128, id="b1_tp8"),
            pytest.param(8, 12, 128, 128, id="b8_tp8"),
            pytest.param(32, 12, 128, 128, id="b32_tp8"),
            pytest.param(4, 96, 128, 128, id="b4_upstream_prod"),
        ],
    )
    def test_kda_decode_batch_hw(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        nv,
        dk,
        dv,
    ):
        """Hardware accuracy test across the batch-size sweep."""
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            nv=nv,
            dk=dk,
            dv=dv,
            rtol=1e-3,
            atol=1e-4,
        )
