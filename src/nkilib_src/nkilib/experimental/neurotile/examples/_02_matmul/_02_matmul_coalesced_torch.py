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
"""Torch references for 02_matmul_coalesced.py."""


def matmul_coalesced_torch_ref(lhsT, rhs, TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=4):
    return lhsT.T @ rhs


def matmul_coalesced_streamed_torch_ref(lhsT, rhs, TILES_IN_BLOCK_M=2, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=4):
    return lhsT.T @ rhs
