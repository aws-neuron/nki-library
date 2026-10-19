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

"""Host-side unit tests for all_expert_mx_utils.py.

Exercise the pure-Python input-validation and block-size helpers directly,
without a device or simulator. The validation helper only reads attributes off
its config objects, so lightweight SimpleNamespace stubs stand in for the real
(device-constructed) dataclasses.
"""

from types import SimpleNamespace

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.moe.moe_tkg.all_expert_mx_utils import (
    _is_valid_block_size,
    validate_all_expert_mx_inputs,
)
from nkilib_src.nkilib.core.moe.moe_tkg.projection_mx_constants import MX_SCALE_DTYPE
from nkilib_src.nkilib.core.utils.common_types import (
    MoEAllToAllVStrategy,
    MoELNCShardingStrategy,
)

_PACKED_DTYPE = nl.float8_e4m3fn_x4  # in MX_PACKED_DTYPES


def _tensor(dtype=nl.bfloat16, shape=(8, 4)):
    return SimpleNamespace(dtype=dtype, shape=shape, buffer="hbm")


def _inputs(hidden_dtype=nl.bfloat16, scale=None, affinities_shape=(8, 4)):
    return SimpleNamespace(
        hidden_input=_tensor(dtype=hidden_dtype),
        hidden_input_scale=(SimpleNamespace(dtype=MX_SCALE_DTYPE) if scale else None),
        expert_affinities_masked=SimpleNamespace(shape=affinities_shape),
    )


def _kernel_cfg(is_static_quant=False, input_in_sbuf=False, output_in_sbuf=False):
    return SimpleNamespace(
        is_static_quant=is_static_quant,
        input_in_sbuf=input_in_sbuf,
        output_in_sbuf=output_in_sbuf,
    )


def _dims(sharding=MoELNCShardingStrategy.NO_SHARD, T=8, T_local=8, T_physical=8):
    return SimpleNamespace(sharding_strategy=sharding, T=T, T_local=T_local, T_physical=T_physical)


def _dynamism(strategy=MoEAllToAllVStrategy.DISABLED, is_all_expert_dynamic=False, block_size=None):
    return SimpleNamespace(
        all_to_all_v_strategy=strategy,
        is_all_expert_dynamic=is_all_expert_dynamic,
        block_size=block_size,
    )


# --- validate_all_expert_mx_inputs branches ---


@pytest.mark.fast
def test_validate_prequantized_no_a2av():
    # DISABLED + scale present -> the pre-quantized (packed dtype + uint8 scale) branch.
    validate_all_expert_mx_inputs(
        _inputs(hidden_dtype=_PACKED_DTYPE, scale=True),
        _kernel_cfg(),
        _dims(),
        _dynamism(),
    )


@pytest.mark.fast
def test_validate_quantized_shard_t():
    # Quantized input with SHARD_T triggers the T_local % 4 check.
    validate_all_expert_mx_inputs(
        _inputs(hidden_dtype=_PACKED_DTYPE, scale=True),
        _kernel_cfg(),
        _dims(sharding=MoELNCShardingStrategy.SHARD_T, T=8, T_local=4),
        _dynamism(),
    )


@pytest.mark.fast
def test_validate_static_mx_non_quantized_skips_t_alignment():
    # is_static_quant + non-quantized (bf16, no scale): the `not is_static_mx`
    # elif evaluates False, skipping the T % 32 alignment check.
    validate_all_expert_mx_inputs(
        _inputs(hidden_dtype=nl.bfloat16, scale=None),
        _kernel_cfg(is_static_quant=True),
        _dims(),
        _dynamism(),
    )


@pytest.mark.fast
def test_validate_input_in_sbuf_requires_prequant():
    # input_in_sbuf and not static -> asserts a pre-quantized (scale present) input.
    validate_all_expert_mx_inputs(
        _inputs(hidden_dtype=_PACKED_DTYPE, scale=True),
        _kernel_cfg(input_in_sbuf=True, is_static_quant=False),
        _dims(),
        _dynamism(),
    )


# --- _is_valid_block_size early-exit branches ---


@pytest.mark.fast
def test_is_valid_block_size_zero():
    assert _is_valid_block_size(128, 0) is False


@pytest.mark.fast
def test_is_valid_block_size_not_divisible():
    assert _is_valid_block_size(100, 32) is False


@pytest.mark.fast
def test_is_valid_block_size_too_large():
    assert _is_valid_block_size(128, 128) is False


@pytest.mark.fast
def test_is_valid_block_size_accepts_valid():
    # A valid block size (divides T, <= T/2, <= 32 and divisible by 4) returns True.
    assert _is_valid_block_size(128, 32) is True
