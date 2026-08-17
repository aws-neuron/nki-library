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
"""Torch references for _04_transpose.py."""


def transpose_tiled_torch_ref(src):
    return src.t().contiguous()


def transpose_coalesced_torch_ref(src):
    return src.t().contiguous()


def transpose_coalesced_single_store_torch_ref(src):
    return src.t().contiguous()


def transpose_into_dst_torch_ref(src):
    return src.t().contiguous()


def transpose_into_dst_partial_torch_ref(src):
    return src.t().contiguous()


def transpose_streamed_torch_ref(src):
    return src.t().contiguous()


def transpose_block_streamed_torch_ref(src):
    return src.t().contiguous()


def transpose_block_streamed_sharded_torch_ref(src):
    # Shard 0 of 2 owns the even seq-tiles (0, 2, 4, ...); output is those transposed.
    import torch

    n_tiles = src.shape[0] // 128
    owned = torch.cat([src[t * 128 : (t + 1) * 128, :] for t in range(0, n_tiles, 2)], dim=0)
    return owned.t().contiguous()


def gather_transpose_torch_ref(data, indices):
    return data[indices.flatten().long(), :].t().contiguous()


def gather_transpose_3d_torch_ref(data, indices):
    return data[indices.flatten().long()].permute(2, 1, 0).contiguous()


def gather_transpose_4d_torch_ref(data, indices):
    rows, f_tiles, p = data.shape
    gathered = data[indices.flatten().long()]
    return gathered.reshape(rows, 1, f_tiles, p).permute(3, 1, 2, 0).contiguous()
