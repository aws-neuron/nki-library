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
"""Torch references for 06_sliced_sources.py."""

_TILE_P = 128
_TILE_F = 128


def window_scale_torch_ref(src):
    return src[0:128, 0:512] * 2.0


def two_windows_torch_ref(src):
    out = src.clone()
    out[128:256, :] *= -1.0
    return out
