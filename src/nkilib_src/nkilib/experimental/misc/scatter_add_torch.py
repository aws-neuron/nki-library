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

"""PyTorch reference implementation for the scatter_add kernel."""

import torch


def scatter_add_torch_ref(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    unique_indices: bool = True,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of the scatter_add kernel.

    Equivalent to ``input.scatter_add(dim=0, index=index_2d, src=src)``, i.e.
    performs ``input[index[i], j] += src[i, j]`` for all ``i, j``.
    Called by ``torch_ref_wrapper``, which converts numpy arrays to torch
    tensors before invoking this function. The 1D ``index`` is expanded to
    match ``src``'s shape to satisfy ``torch.scatter_add``'s shape requirements.

    Args:
        input (torch.Tensor): [N, D], Destination tensor to accumulate into.
        dim (int): Dimension along which to scatter. Must be 0 to match the kernel contract.
        index (torch.Tensor): [K], 1D integer tensor of row indices into ``input``.
        src (torch.Tensor): [K, D], Source values to scatter-add.

    Returns:
        dict[str, torch.Tensor]: ``{"output": input_after}`` where ``input_after`` is
            ``input`` with ``src`` rows scatter-added at rows given by ``index``.

    Notes:
        - ``index`` is cast to int64 and broadcast to ``src``'s shape to satisfy
          ``torch.scatter_add``'s requirement that ``index`` and ``src`` share shape.
        - Only ``dim == 0`` is supported, matching the kernel's restriction.
    """
    if dim != 0:
        raise ValueError(f"scatter_add currently only supports dim=0, got dim={dim}")
    index_2d = index.long().unsqueeze(1).expand_as(src)
    return {"output": input.scatter_add(dim, index_2d, src)}
