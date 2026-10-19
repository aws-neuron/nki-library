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

"""Host-side unit tests for projection_mx_constants.py.

Exercise the pure-Python quantize-partition padding helper and the ProjConfig
dataclass configuration branches directly, without a device or simulator.
"""

import pytest
from nkilib_src.nkilib.core.moe.moe_tkg.projection_mx_constants import (
    ProjConfig,
    pad_to_valid_qmx_partitions,
)


@pytest.mark.fast
def test_pad_to_valid_qmx_partitions_within_range():
    # n fits inside an allowed size -> returns the first size >= n.
    assert pad_to_valid_qmx_partitions(1) == 32
    assert pad_to_valid_qmx_partitions(64) == 64
    assert pad_to_valid_qmx_partitions(100) == 128


@pytest.mark.fast
def test_pad_to_valid_qmx_partitions_exhausts_to_max():
    # n larger than every allowed size -> loop falls through to the 128 fallback.
    assert pad_to_valid_qmx_partitions(200) == 128


def _proj_config(H=512, I=512, BxS=8, n_prgs=1, prg_id=0, out_p_offset=0, force_lnc1=False):
    # H % 128 == 0, n_prgs == 1 so H1/H1_sharded are divisible; I % 512 == 0.
    # Explicit kwargs (not **dict) so the type checker can verify each argument.
    return ProjConfig(H=H, I=I, BxS=BxS, n_prgs=n_prgs, prg_id=prg_id, out_p_offset=out_p_offset, force_lnc1=force_lnc1)


@pytest.mark.fast
def test_proj_config_default_zero_offset():
    cfg = _proj_config()
    assert cfg.out_p_offset == 0
    assert cfg.n_prgs == 1


@pytest.mark.fast
def test_proj_config_nonzero_out_p_offset():
    # out_p_offset != 0 (aligned to 32, offset + BxS <= 128) triggers the extra
    # partition-offset shape checks.
    cfg = _proj_config(out_p_offset=32)
    assert cfg.out_p_offset == 32


@pytest.mark.fast
def test_proj_config_force_lnc1_overrides_prgs():
    # force_lnc1 collapses to a single program regardless of the requested n_prgs.
    # H=1024 keeps H1_sharded (= H1 // n_prgs = 4) divisible by _q_width for n_prgs=2.
    cfg = _proj_config(H=1024, n_prgs=2, prg_id=1, force_lnc1=True)
    assert cfg.n_prgs == 1
    assert cfg.prg_id == 0
