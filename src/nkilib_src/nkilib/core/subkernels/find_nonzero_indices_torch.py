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

"""PyTorch reference implementation for find_nonzero_indices kernel."""

import torch


def find_nonzero_indices_torch_ref(
    input_tensor: torch.Tensor,
    col_start_id: torch.Tensor = None,
    n_cols: int = None,
    chunk_size: int = None,
    index_dtype=torch.int32,
) -> dict[str, torch.Tensor]:
    """PyTorch reference for find_nonzero_indices kernel.

    Finds indices of nonzero elements along the T dimension for each column.
    This is a reference implementation for testing; chunk_size is a hardware
    optimization parameter and is unused here.

    Args:
        input_tensor (torch.Tensor): [T, C], Input tensor.
        col_start_id (torch.Tensor): [1], Optional starting column index.
        n_cols (int): Number of columns to process when col_start_id is set.
        chunk_size (int): Unused, accepted for signature compatibility.
        index_dtype: Data type for output indices tensor. Default is torch.int32.

    Returns:
        dict with 'indices' [C_out, T] and 'nonzero_counts' [C_out] tensors.
    """
    T, C_full = input_tensor.shape

    if col_start_id != None:
        start_col = col_start_id.item()
        C_out = n_cols
    else:
        start_col = 0
        C_out = C_full

    indices = torch.full((C_out, T), -1, dtype=index_dtype)
    nonzero_counts = torch.zeros(C_out, dtype=torch.int32)

    for col_idx in range(C_out):
        col = input_tensor[:, start_col + col_idx]
        nz = torch.nonzero(col, as_tuple=False).squeeze(-1)
        count = nz.shape[0]
        nonzero_counts[col_idx] = count
        if count > 0:
            indices[col_idx, :count] = nz.to(index_dtype)

    return {"indices": indices, "nonzero_counts": nonzero_counts}
