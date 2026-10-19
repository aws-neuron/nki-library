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

"""PyTorch reference implementation for Multi-Scale Deformable Attention."""

import torch


def ms_deformable_attention_torch_ref(
    value: torch.Tensor,
    spatial_shapes,
    level_start_index,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
    max_indices_per_indirect=None,
    gather_method: str = "transpose",
) -> torch.Tensor:
    """
    PyTorch reference implementation of multi-scale deformable attention forward pass.

    Args:
        value: (B, L, N_h, C_h) or (B, N_h, L, C_h)
        spatial_shapes: list of (H_i, W_i) tuples or (N_l, 2) tensor
        level_start_index: list of ints or (N_l,) tensor
        sampling_locations: (B, N_q, N_h, N_l, N_p, 2) or (B, 2, N_q, N_h, N_l, N_p)
            Normalized (u, v) ∈ [0,1]
        attention_weights: (B, N_q, N_h, N_l, N_p)
        value_layout: "BLNC" or "BNLC"
        sampling_locations_layout: "BQHLP2" or "B2QHLP"
        align_corners: bool - If True, [0,1] maps to [0, H-1]. If False, [0,1] maps to [-0.5, H-0.5]
        padding_mode: str - One of "zeros", "border", "reflection"
            - "zeros": Out-of-bounds returns 0
            - "border": Clamps to edge pixels
            - "reflection": Mirrors at boundaries

    Returns:
        output: (B, N_q, N_h * C_h)
    """
    # Handle BNLC layout by transposing to BLNC
    if value_layout == "BNLC":
        value = value.permute(0, 2, 1, 3)  # (B, N_h, L, C_h) → (B, L, N_h, C_h)

    # Handle B2QHLP layout by transposing to BQHLP2
    if sampling_locations_layout == "B2QHLP":
        sampling_locations = sampling_locations.permute(
            0, 2, 3, 4, 5, 1
        )  # (B, 2, N_q, N_h, N_l, N_p) → (B, N_q, N_h, N_l, N_p, 2)

    B, L, N_h, C_h = value.shape
    _, N_q, _, N_l, N_p, _ = sampling_locations.shape

    # Convert to lists if tensors
    if isinstance(spatial_shapes, torch.Tensor):
        spatial_shapes = spatial_shapes.tolist()
    if isinstance(level_start_index, torch.Tensor):
        level_start_index = level_start_index.tolist()

    # Validate padding_mode
    if padding_mode not in ["zeros", "border", "reflection"]:
        raise ValueError(f"padding_mode must be one of ['zeros', 'border', 'reflection'], got {padding_mode}")

    # Initialize output
    output = torch.zeros(B, N_q, N_h, C_h, dtype=value.dtype, device=value.device)

    # Process queries
    for b in range(B):
        for q in range(N_q):
            for h in range(N_h):
                for l in range(N_l):
                    H_l, W_l = spatial_shapes[l]
                    level_start = level_start_index[l]

                    level_value = value[b, level_start : level_start + H_l * W_l, h, :].reshape(H_l, W_l, C_h)

                    for p in range(N_p):
                        # Get normalized sampling location
                        u = sampling_locations[b, q, h, l, p, 0].item()  # [0, 1]
                        v = sampling_locations[b, q, h, l, p, 1].item()  # [0, 1]

                        # Convert to pixel coordinates
                        if align_corners:
                            x = u * (W_l - 1)  # [0, 1] -> [0, W_l - 1]
                            y = v * (H_l - 1)  # [0, 1] -> [0, H_l - 1]
                        else:
                            x = u * W_l - 0.5  # [0, 1] -> [-0.5, W_l - 0.5]
                            y = v * H_l - 0.5  # [0, 1] -> [-0.5, H_l - 0.5]

                        # Get integer coordinates
                        x0 = int(torch.floor(torch.tensor(x)).item())
                        y0 = int(torch.floor(torch.tensor(y)).item())
                        x1 = x0 + 1
                        y1 = y0 + 1

                        # Calculate fractional parts
                        dx = x - x0
                        dy = y - y0

                        # Calculate bilinear weights
                        w_00 = (1 - dx) * (1 - dy)
                        w_01 = dx * (1 - dy)
                        w_10 = (1 - dx) * dy
                        w_11 = dx * dy

                        sampled = torch.zeros(C_h, dtype=value.dtype, device=value.device)

                        if padding_mode == "zeros":
                            # Zero padding: OOB accesses don't contribute
                            if 0 <= y0 < H_l and 0 <= x0 < W_l:
                                sampled += w_00 * level_value[y0, x0, :]

                            if 0 <= y0 < H_l and 0 <= x1 < W_l:
                                sampled += w_01 * level_value[y0, x1, :]

                            if 0 <= y1 < H_l and 0 <= x0 < W_l:
                                sampled += w_10 * level_value[y1, x0, :]

                            if 0 <= y1 < H_l and 0 <= x1 < W_l:
                                sampled += w_11 * level_value[y1, x1, :]

                        elif padding_mode == "border":
                            # Border padding: OOB accesses clamp coordinates to valid range
                            y0_clamped = max(0, min(y0, H_l - 1))
                            y1_clamped = max(0, min(y1, H_l - 1))
                            x0_clamped = max(0, min(x0, W_l - 1))
                            x1_clamped = max(0, min(x1, W_l - 1))

                            sampled += w_00 * level_value[y0_clamped, x0_clamped, :]
                            sampled += w_01 * level_value[y0_clamped, x1_clamped, :]
                            sampled += w_10 * level_value[y1_clamped, x0_clamped, :]
                            sampled += w_11 * level_value[y1_clamped, x1_clamped, :]

                        elif padding_mode == "reflection":
                            # Reflection padding: mirror coordinates at boundaries
                            def reflect_coord(coord, size):
                                if coord < 0:
                                    coord = -coord - 1
                                if coord >= size:
                                    coord = 2 * size - coord - 1
                                return max(0, min(coord, size - 1))

                            y0_ref = reflect_coord(y0, H_l)
                            y1_ref = reflect_coord(y1, H_l)
                            x0_ref = reflect_coord(x0, W_l)
                            x1_ref = reflect_coord(x1, W_l)

                            sampled += w_00 * level_value[y0_ref, x0_ref, :]
                            sampled += w_01 * level_value[y0_ref, x1_ref, :]
                            sampled += w_10 * level_value[y1_ref, x0_ref, :]
                            sampled += w_11 * level_value[y1_ref, x1_ref, :]

                        # Weighted accumulation
                        attn_w = attention_weights[b, q, h, l, p]
                        output[b, q, h, :] += attn_w * sampled

    output = output.reshape(B, N_q, N_h * C_h)  # (B, N_q, N_h * C_h)

    return output
