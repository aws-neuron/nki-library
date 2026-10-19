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
"""Dispatch table mapping ``NormType`` to its torch reference implementation.

Used by torch_ref code in src/ that fuses normalization into another op
(QKV, MLP) and needs to call the matching torch reference for the chosen
``NormType``. Keeping the dispatch table in src/ avoids src/ → test/ imports.
"""

from ..utils.common_types import NormType
from .layernorm_torch import layer_norm_torch_ref
from .rmsnorm_torch import rms_norm_torch_ref

norm_name2func_torch = {
    NormType.NO_NORM: lambda *x, **_: x[0],
    NormType.RMS_NORM: rms_norm_torch_ref,
    NormType.LAYER_NORM: layer_norm_torch_ref,
    NormType.RMS_NORM_SKIP_GAMMA: rms_norm_torch_ref,
}
