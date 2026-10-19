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

"""Host-side unit tests for pure-Python helpers in projection_mx_constants.

These helpers (``pad_to_valid_qmx_partitions`` and the ``ProjConfig`` dataclass)
run without a kernel trace, so their branches are exercised directly rather than
through a kernel invocation.
"""

import pytest
from nkilib_src.nkilib.core.mlp.mlp_tkg.projection_mx_constants import (
    ProjConfig,
    pad_to_valid_qmx_partitions,
)


@pytest.mark.fast
def test_pad_to_valid_qmx_exact_boundary():
    # First valid size (32) already >= n, so the loop returns on the first iteration.
    assert pad_to_valid_qmx_partitions(32) == 32


@pytest.mark.fast
def test_pad_to_valid_qmx_overflow():
    # n exceeds every valid size, so the loop falls through to the trailing return 128.
    assert pad_to_valid_qmx_partitions(200) == 128


@pytest.mark.fast
def test_projconfig_out_p_offset():
    # Non-zero out_p_offset exercises the offset validation block in check_shapes.
    cfg = ProjConfig(H=512, I=512, BxS=4, n_prgs=1, prg_id=0, out_p_offset=32)
    assert cfg.out_p_offset == 32


@pytest.mark.fast
def test_projconfig_force_lnc1():
    # force_lnc1 forces single-program execution (n_prgs reset to 1, prg_id to 0).
    cfg = ProjConfig(H=512, I=512, BxS=4, n_prgs=1, prg_id=0, force_lnc1=True)
    assert cfg.n_prgs == 1
    assert cfg.prg_id == 0
