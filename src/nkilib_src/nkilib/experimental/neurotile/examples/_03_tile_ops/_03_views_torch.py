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
"""Torch references for _03_views.py."""


def view_chain_torch_ref(src):
    # src: [B, S, H]  ->  out: [H0, B*S*H1]
    B, S, H = src.shape
    H0 = 128
    H1 = H // H0
    return (src.reshape(B * S, H0, H1).permute(1, 0, 2) * 2.0).reshape(H0, B * S * H1)


def view_select_torch_ref(src):
    return src[0] * 3.0


def view_3d_direct_torch_ref(src):
    return src * 5.0


def nd_tile_alloc_sbuf_torch_ref(src):
    # src: [128, 4, 32]  ->  out: [128, 128] = (src * 3.0).reshape(128, 128)
    return (src * 3.0).reshape(128, 128)
