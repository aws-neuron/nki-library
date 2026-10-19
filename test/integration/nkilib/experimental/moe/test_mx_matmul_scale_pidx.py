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

"""Consumer-side feasibility probe for the packed (64-partition) MX scale layout.

QUESTION
--------
The DMA half is settled: `[16 P, 4,608 B] x4` beats the shipping `[4 P, 18,432 B] x4` by **2.66x**
for byte-identical traffic (see test_mx_scale_load_microbench.py / the rev15_nki07 INDEX). Packing
4 scale groups per quadrant at partition offsets {0,4,8,12} is ISA-legal:
`mxmem1d_valid_scale_pidx` (NeuronArchIsaSource `common/assert_util.rs:2403`) requires
`scale_pidx % 4 == 0 && scale_pidx < 16`.

What is NOT known is whether **NKI can express it**. `nc_matmul_mx`'s encoding carries the field --
`src=<dtype>@$MX[<data_addr>,<scale_addr>,<start_scale_partition>]@[<step>][<num>]` -- but today
`_projection_matmul_mx:787` passes the scale with the *same free slice as the weight* and partition
dim `:`, letting hardware snake to the partition:
    stationary_scale=weight_scale_sb[:, tile_h, weight_I_slice]
The packed layout instead needs the group selected by the AP's **base partition** (0/4/8/12).

This probe compiles `nc_matmul_mx` with several candidate scale APs and then reads the emitted
`$MX[...]` encoding out of the artifacts to see which offset actually landed. Compile is enough --
no numerics claim is made here; this only answers "expressible, and does the offset propagate".

Note the scale tile has the SAME shape as the data tile by NKI convention (which is exactly why
`load_gate_up_weight_scale_bias` allocates `scale_sb` with `weight_sb_shape`), so all variants below
keep that shape and vary only the partition slice.

Run:
    pytest -c test/pytest.ini \
      test/integration/nkilib/experimental/moe/test_mx_matmul_scale_pidx.py \
      --platform-target trn3 --nki-compilation-mode tracer --test-mode compile-only -v -s
Then:  grep -oE '\\$MX\\[[^]]*\\]' <out>/artifacts/sg00/*.dbg | sort -u
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

H_P = 128  # contraction dim = partitions (must be a multiple of 32, <= 128)
I_SUB = 128  # stationary free dim
T = 32  # moving free dim

_DATA_DT = nl.float4_e2m1fn_x4  # MXFP4, 4-packed -- what the MoE experts use
_SCALE_DT = nl.float8_e8m0fnu  # MX scale dtype


def _mk(scale_variant: str, off: int):
    """Build a kernel whose ONLY difference is how `stationary_scale` is sliced."""

    @nki.jit
    def _k(seed: nl.NkiTensor) -> nl.NkiTensor:
        w = nl.ndarray((H_P, I_SUB), dtype=_DATA_DT, buffer=nl.sbuf)
        x = nl.ndarray((H_P, T), dtype=_DATA_DT, buffer=nl.sbuf)
        ws = nl.ndarray((H_P, I_SUB), dtype=_SCALE_DT, buffer=nl.sbuf)
        xs = nl.ndarray((H_P, T), dtype=_SCALE_DT, buffer=nl.sbuf)
        # Deterministic init so nothing reads uninitialised SBUF.
        nisa.memset(dst=w[...], value=0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=x[...], value=0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=ws[...], value=0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=xs[...], value=0, engine=nisa.gpsimd_engine)

        if scale_variant == "full":  # control: today's form (implicit offset 0)
            s_ap = ws[:, :]
        elif scale_variant == "off_rest":  # offset + remaining extent
            s_ap = ws[nl.ds(off, H_P - off), :]
        elif scale_variant == "off_4":  # offset + exactly 4 rows
            s_ap = ws[nl.ds(off, 4), :]
        elif scale_variant == "off_quadstride":  # offset + one full quadrant's worth
            s_ap = ws[nl.ds(off, 32), :]
        else:
            raise ValueError(scale_variant)

        psum = nl.ndarray((I_SUB, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul_mx(dst=psum, stationary=w, moving=x, stationary_scale=s_ap, moving_scale=xs)

        sb = nl.ndarray((I_SUB, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sb, src=psum)
        out = nl.ndarray((I_SUB, T), dtype=nl.float32, buffer=nl.shared_hbm)
        nl.store(out, value=sb)
        return out

    return _k


@pytest_test_metadata(name="MxMatmulScalePidx")
@pytest_marks(["moe"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@final
@pytest.mark.skip_simulation
class TestMxMatmulScalePidx:
    """Is a partition-offset `stationary_scale` AP expressible, and does the offset propagate?"""

    def _run(self, test_manager, platform_target, kernel):
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            kernel_input_generator=lambda _cfg: {"seed": np.zeros((1, 1), dtype=np.float32)},
            trace_only=True,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=1,
                platform_target=platform_target,
                nki_compilation_mode=NKICompilationMode.tracer,
            ),
        )

    def test_scale_full(self, test_manager: Orchestrator, platform_target: Platforms):
        """Control: `ws[:, :]` -- the shipping form. Expect start_scale_partition = 0."""
        self._run(test_manager, platform_target, _mk("full", 0))

    @pytest.mark.parametrize("off", [4, 8, 12])
    def test_scale_off_rest(self, test_manager: Orchestrator, platform_target: Platforms, off):
        """`ws[nl.ds(off, 128-off), :]` -- offset base partition, remaining extent."""
        self._run(test_manager, platform_target, _mk("off_rest", off))

    @pytest.mark.parametrize("off", [4, 8, 12])
    def test_scale_off_4(self, test_manager: Orchestrator, platform_target: Platforms, off):
        """`ws[nl.ds(off, 4), :]` -- offset base partition, exactly the 4 packed rows."""
        self._run(test_manager, platform_target, _mk("off_4", off))

    @pytest.mark.parametrize("off", [4])
    def test_scale_off_quadstride(self, test_manager: Orchestrator, platform_target: Platforms, off):
        """`ws[nl.ds(off, 32), :]` -- offset base partition, one quadrant of extent."""
        self._run(test_manager, platform_target, _mk("off_quadstride", off))
