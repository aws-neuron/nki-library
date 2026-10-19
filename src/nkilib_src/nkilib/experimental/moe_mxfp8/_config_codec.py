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

"""Reconstruct data-free configs from shared measured fields and record tuples."""

from ...core.utils.kernel_assert import kernel_assert
from ..matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from ..moe.bwd.moe_bwd_parameters import ClampLimits, SkipMode
from ..mxfp_utils.mxfp8_utils.common_dataclasses import BlockDescriptor, TensorDescriptor
from .bwd.config import MXFP8MOEBwdConfig
from .fwd.config import MXFP8MOEFwdConfig
from .moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig

_CONFIG_TYPES = {
    cls.__name__: cls
    for cls in (
        MXFP8MOEFwdConfig,
        MXFP8MOEBwdConfig,
        MatmulMxfp8KernelConfig,
        TensorDescriptor,
        BlockDescriptor,
        ClampLimits,
        SkipMode,
        MXFP8MOECheckpointConfig,
    )
}


def _decode_value(value, schemas, records):
    if isinstance(value, tuple):
        if len(value) == 2 and value[0] == "ref":
            return decode_config(value[1], schemas, records)
        return tuple(_decode_value(item, schemas, records) for item in value)
    return value


def decode_config(index, schemas, records):
    """Construct a fresh record, including all nested configs and descriptors."""
    name, values = records[index]
    fixed, varying = schemas[name]
    kernel_assert(len(values) == len(varying), "Cached record does not match its field schema")
    kwargs = {field: _decode_value(value, schemas, records) for field, value in fixed.items()}
    for field, value in zip(varying, values):
        kwargs[field] = _decode_value(value, schemas, records)
    return _CONFIG_TYPES[name](**kwargs)
