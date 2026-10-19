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

"""Read whole-kernel configs without overriding the caller's preset fields.

An exact key hit is usable only when every preset non-key field agrees with the
cached config. A conflict rejects the entire cached config. Misses and conflicts
return None so the kernel continues its normal default resolution.
"""

from dataclasses import fields, is_dataclass

from ..matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from ..mxfp_utils.mxfp8_utils.common_dataclasses import LncShardingMode
from . import _best_config_cache
from ._config_codec import decode_config
from ._config_helpers import make_moe_config_key
from .bwd.config import MXFP8MOEBwdConfig
from .fwd.config import MXFP8MOEFwdConfig


def _preset_fields_match(preset, cached):
    # Unset tuning choices are resolved after lookup; explicit False/enums remain presets.
    if preset == None:
        # Packing remains enabled unless the caller supplies a phase override.
        return not isinstance(cached, MatmulMxfp8KernelConfig) or cached.enable_scale_packing
    if is_dataclass(preset):
        if not isinstance(cached, type(preset)):
            return False
        if not isinstance(preset, (MXFP8MOEFwdConfig, MXFP8MOEBwdConfig, MatmulMxfp8KernelConfig)):
            return preset == cached
        for field in fields(preset):
            value = getattr(preset, field.name)
            if isinstance(preset, MatmulMxfp8KernelConfig):
                # Zero dimensions and AUTO sharding are unresolved phase facts.
                if field.name in ("M", "K", "N") and value == 0:
                    continue
                if field.name == "lnc_sharding" and value == LncShardingMode.AUTO:
                    continue
            if not _preset_fields_match(value, getattr(cached, field.name)):
                return False
        return True
    if isinstance(preset, tuple):
        if not isinstance(cached, tuple) or len(preset) != len(cached):
            return False
        for lhs, rhs in zip(preset, cached):
            if not _preset_fields_match(lhs, rhs):
                return False
        return True
    return preset == cached


def _lookup_config(key, user_config, cache, backward):
    entry = cache.get(key)
    if entry == None:
        return None
    cached = decode_config(entry[1], _best_config_cache._CONFIG_SCHEMAS, _best_config_cache._CONFIG_RECORDS)
    if make_moe_config_key(key.input_shape, key.weight_dtype, cached, backward) != key:
        return None
    if not _preset_fields_match(user_config, cached):
        return None
    return cached


def get_cached_moe_fwd_config(key, user_config):
    """Return a fresh compatible forward config, or None on miss/conflict."""
    return _lookup_config(key, user_config, _best_config_cache._MOE_FWD_CONFIG_CACHE, False)


def get_cached_moe_bwd_config(key, user_config):
    """Return a fresh compatible backward config, or None on miss/conflict."""
    return _lookup_config(key, user_config, _best_config_cache._MOE_BWD_CONFIG_CACHE, True)
