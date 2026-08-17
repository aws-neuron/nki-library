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

"""PyTorch reference for mxfp_load_performance_wrapper (swizzle + quantize from HBM)."""

import nki.language as nl
import numpy as np

from ...core.utils.mx_torch_common import quantize_mx_golden

P_MAX = 128
K_BLOCK_SIZE = 512
X4_PACK_FACTOR = 4


def mxfp_load_performance_wrapper_torch_ref(tensor: np.ndarray):
    """Golden reference: swizzle + quantize per 512-element K block.

    For each K block of 512 BF16 elements (= [M=128, 512]):
        Swizzle: [M, 512] -> reshape [M, 128, 4] -> transpose(1, 0, 2) -> [128, M*4]
        Quantize: [128, M*4] -> mx_data [128, M] x4, mx_scale [16, M] float8_e8m0fnu

    Scales are placed at partition offsets [0, 32, 64, 96] with 4 rows each,
    matching the hardware layout of quantize_mx. Remaining rows are zero-padded.

    Concatenate all K blocks along free dim -> [128, M * K_blocks]
    """
    M_dim, K_dim = tensor.shape
    k_block_count = K_dim // K_BLOCK_SIZE
    out_free_dim = M_dim * k_block_count

    all_data = []
    scale_buf = np.zeros((P_MAX, out_free_dim), dtype=np.uint8)

    for k_block_idx in range(k_block_count):
        block = tensor[:, k_block_idx * K_BLOCK_SIZE : (k_block_idx + 1) * K_BLOCK_SIZE].astype(np.float32)
        swizzled = block.reshape(M_dim, P_MAX, X4_PACK_FACTOR).transpose(1, 0, 2).reshape(P_MAX, M_dim * X4_PACK_FACTOR)

        out_data_dummy = np.empty((P_MAX, M_dim), dtype=nl.float8_e4m3fn_x4)
        out_scale_dummy = np.empty((P_MAX, M_dim), dtype=np.uint8)
        mx_result = quantize_mx_golden(swizzled, out_data_dummy, out_scale_dummy)
        all_data.append(mx_result["out_data_hbm"])
        free_slice = slice(k_block_idx * M_dim, (k_block_idx + 1) * M_dim)
        scale_buf[:, free_slice] = mx_result["out_scale_hbm"]

    mx_data_full = np.concatenate(all_data, axis=1)

    return {
        "out_data_hbm": mx_data_full.view(np.float32),
        "out_scale_hbm": scale_buf,
    }
