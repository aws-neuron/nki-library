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

"""MX scale-load DMA-shape microbenchmark: `[18432F, 4P] x4` vs `[4608F, 16P] x4`.

WHY THIS EXISTS
---------------
The gate/up (and down) MX scale load writes SBUF in the ISA-mandated 4-per-quadrant pattern
(partitions 0-3 / 32-35 / 64-67 / 96-99). A 128-rank HW profile measured that write at **16x
lower byte-throughput than its queue-mate weight DMA** on the same queue and in the same
20.8/20.9 us window (294,912 B vs 4,718,592 B), and that single scale load blocked the Tensor
engine 26.7 us against 0.8 us of actual matmul.

`VLLM_NEURON_MEGA_SCALE_1TRIG` already removed the I-tiling (24 -> 8 DMAs/layer, -576 static
triggers, -20,736 descriptors), leaving **4 DMAs per gate/up half -- the structural floor**,
because a 32-partition destination stride cannot be expressed (`reshape_dim` rejected on the
partition dim; `.ap()` asserts `Partition step must equal tensor free dimension size`).

The remaining idea is to reshape the scale so more PARTITIONS are used per DMA:
    HBM/SBUF  [16, 6, 3072]  ->  [64, 3, 1536]     (identical 294,912 B)
The ISA permits it: a scale must sit in the quadrant its scaling group came from, and
`mxmem1d_valid_scale_pidx` (NeuronArchIsaSource `common/assert_util.rs:2403`) allows
`scale_pidx % 4 == 0 && scale_pidx < 16`, i.e. offsets {0, 4, 8, 12} -- exactly 4 slots per
quadrant, so exactly a factor of 4 may leave the free dim. 64P is the ceiling; the 64 rows must
land 16-per-quadrant at 0-15 / 32-47 / 64-79 / 96-111, NOT densely at 0-63.

Same DMA count, same bytes, same descriptor-group count -- only the group SHAPE changes:
4x more partitions per group, 4x shorter rows. The two published data points differ in BOTH
dimensions (weight `[36864,128]`=9353 vs scale `[18432,4]`=587), so the sign of that trade is
NOT derivable from them. THIS TEST MEASURES IT.

It also demonstrates the second, unambiguous benefit: SBUF per-partition reservation drops
18,432 B -> 4,608 B (4x), because `scale_sb` is allocated with `weight_sb_shape` while only
16 of 128 partitions hold data.

WHAT IS AND IS NOT COVERED
--------------------------
Covered: the DMA half -- is `[4608F, 16P]` faster or slower than `[18432F, 4P]` for the same bytes.
NOT covered: whether NKI emits `start_scale_partition` (the `$MX[data,scale,start]` encoding field)
from a partition-offset `stationary_scale` AP. That is the separate consumer-side go/no-go and
needs an `nc_matmul_mx` test; it is pointless to write until the DMA answer is known.

Both arms are DMA-only and loop NUM_ITERS times with a distinct output slice per iteration so
nothing is dead-code-eliminated. `trace_only=True` skips golden comparison but still executes on
hardware, so the profile/wall time is real. Compare the two arms' captures; only the RATIO is
meaningful (absolute totals on this kernel class are not).

Run (single rank is enough -- no collectives):
    pytest -c test/pytest.ini \
      test/integration/nkilib/experimental/moe/test_mx_scale_load_microbench.py \
      --platform-target trn3 --nki-compilation-mode tracer --enable-dge-notifs -v -s
"""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest

from test.utils.common_dataclasses import (
    CompilerArgs,
    NKICompilationMode,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework

# Enough iterations that fixed startup is negligible next to the DMA stream.
NUM_ITERS = 32

P_MAX = 128  # nl.tile_size.pmax
QUAD = 32  # SBUF_QUADRANT_SIZE
NQ = 4  # quadrants spanned by a 128-partition data tile

# Arm A -- today's layout: 4 scale rows per quadrant, full (H/512, I) free dim.
A_SP, A_H512, A_I = 4, 6, 3072  # per-DMA [4 P, 6*3072 = 18,432 B]
# Arm B -- partition-fused: 16 scale rows per quadrant, free dim folded 4x.
B_SP, B_H512, B_I = 16, 3, 1536  # per-DMA [16 P, 3*1536 = 4,608 B]

# Byte-identical by construction; asserted at import so a future edit cannot silently skew it.
_BYTES_PER_DMA = A_SP * A_H512 * A_I
assert _BYTES_PER_DMA == B_SP * B_H512 * B_I == 73728, "arms must move identical bytes per DMA"


@nki.jit
def mx_scale_load_cur(scale: nl.NkiTensor) -> nl.NkiTensor:
    """Arm A: 4 DMAs/iter, each [4 P, 18,432 B] -- reproduces the shipping scale-load shape.

    scale: [NUM_ITERS, 16, 6, 3072] uint8 in HBM.
    """
    out = nl.ndarray((NUM_ITERS, 4), dtype=nl.uint8, buffer=nl.shared_hbm)
    for i in range(NUM_ITERS):
        sb = nl.ndarray((P_MAX, A_H512, A_I), dtype=nl.uint8, buffer=nl.sbuf)
        for q in range(NQ):
            src = scale.select(dim=0, index=i).slice(dim=0, start=A_SP * q, end=A_SP * (q + 1))
            nisa.dma_copy(src=src, dst=sb[nl.ds(QUAD * q, A_SP), :, :], dge_mode=nisa.dge_mode.none)
        # keep-alive: distinct output slice per iteration defeats dead-code elimination
        nisa.dma_copy(src=sb[0:1, 0, 0:4], dst=out[i : i + 1, :])
    return out


@nki.jit
def mx_scale_load_fused(scale: nl.NkiTensor) -> nl.NkiTensor:
    """Arm B: 4 DMAs/iter, each [16 P, 4,608 B] -- the partition-fused candidate.

    scale: [NUM_ITERS, 64, 3, 1536] uint8 in HBM. The 16 rows per quadrant land at
    partitions 16q..16q+15, i.e. offsets {0,4,8,12} x 4 rows -- the ISA-legal arrangement.
    """
    out = nl.ndarray((NUM_ITERS, 4), dtype=nl.uint8, buffer=nl.shared_hbm)
    for i in range(NUM_ITERS):
        sb = nl.ndarray((P_MAX, B_H512, B_I), dtype=nl.uint8, buffer=nl.sbuf)
        for q in range(NQ):
            src = scale.select(dim=0, index=i).slice(dim=0, start=B_SP * q, end=B_SP * (q + 1))
            nisa.dma_copy(src=src, dst=sb[nl.ds(QUAD * q, B_SP), :, :], dge_mode=nisa.dge_mode.none)
        nisa.dma_copy(src=sb[0:1, 0, 0:4], dst=out[i : i + 1, :])
    return out


def _rand_u8(shape):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


@pytest_test_metadata(name="MxScaleLoadMicrobench")
@pytest_marks(["moe"])
@final
@pytest.mark.skip_simulation
class TestMxScaleLoadMicrobench:
    """Isolated DMA-shape A/B for the MX scale load. Same bytes, same DMA count, different shape."""

    def _run(self, test_manager, platform_target, kernel, scale_shape):
        inputs = {"scale": _rand_u8(scale_shape)}
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            kernel_input_generator=lambda _cfg: inputs,
            trace_only=True,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=1,
                platform_target=platform_target,
                nki_compilation_mode=NKICompilationMode.tracer,
            ),
        )

    def test_scale_load_cur(self, test_manager: Orchestrator, platform_target: Platforms):
        """Arm A: 4 x [4 P, 18,432 B] per iteration (shipping shape). SBUF 18,432 B/partition."""
        self._run(
            test_manager,
            platform_target,
            mx_scale_load_cur,
            (NUM_ITERS, A_SP * NQ, A_H512, A_I),
        )

    def test_scale_load_fused(self, test_manager: Orchestrator, platform_target: Platforms):
        """Arm B: 4 x [16 P, 4,608 B] per iteration (fused shape). SBUF 4,608 B/partition."""
        self._run(
            test_manager,
            platform_target,
            mx_scale_load_fused,
            (NUM_ITERS, B_SP * NQ, B_H512, B_I),
        )
