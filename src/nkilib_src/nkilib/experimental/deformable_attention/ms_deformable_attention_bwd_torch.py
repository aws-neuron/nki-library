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

"""PyTorch reference implementation for Multi-Scale Deformable Attention Backward Pass."""

from typing import Optional

import torch


def ms_deformable_attention_bwd_torch_ref(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    spatial_shapes,
    level_start_index,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
    max_gather_indices_per_indirect: Optional[int] = None,
    max_scatter_indices_per_indirect: Optional[int] = None,
    gather_method: str = "transpose",
    num_scatter_buffers: Optional[int] = None,
    compute_grad_value: bool = True,
    compute_grad_sampling_locations: bool = True,
    compute_grad_attention_weights: bool = True,
    iota_k_scale_in=None,
    grad_value_buffers_in=None,
    prev_reduced_grad_value=None,
    dump_iota_k_scale: bool = False,
    dump_grad_value_buffers: bool = False,
    dump_reduced_grad_value: bool = False,
) -> dict:
    """
    PyTorch reference implementation of multi-scale deformable attention backward pass.

    Args:
        grad_output: (B, N_q, N_h * C_h)
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

    Returns:
        dict with keys:
            - out_grad_value: (B, L, N_h, C_h) - gradient w.r.t. value (same layout as input)
            - out_grad_sampling_locations: (B, N_q, N_h, N_l, N_p, 2) - gradient w.r.t. sampling_locations
            - out_grad_attention_weights: (B, N_q, N_h, N_l, N_p) - gradient w.r.t. attention_weights
    """
    # Handle BNLC layout by transposing to BLNC
    original_value_layout = value_layout
    if value_layout == "BNLC":
        value = value.permute(0, 2, 1, 3)  # (B, N_h, L, C_h) -> (B, L, N_h, C_h)

    # Handle B2QHLP layout by transposing to BQHLP2
    original_sampling_layout = sampling_locations_layout
    if sampling_locations_layout == "B2QHLP":
        sampling_locations = sampling_locations.permute(
            0, 2, 3, 4, 5, 1
        )  # (B, 2, N_q, N_h, N_l, N_p) -> (B, N_q, N_h, N_l, N_p, 2)

    B, L, N_h, C_h = value.shape
    _, N_q, _, N_l, N_p, _ = sampling_locations.shape

    # Reshape grad_output from (B, N_q, N_h * C_h) to (B, N_q, N_h, C_h)
    grad_output = grad_output.reshape(B, N_q, N_h, C_h)

    # Convert to lists if tensors
    if isinstance(spatial_shapes, torch.Tensor):
        spatial_shapes = spatial_shapes.tolist()
    if isinstance(level_start_index, torch.Tensor):
        level_start_index = level_start_index.tolist()

    # Validate padding_mode
    if padding_mode not in ["zeros", "border", "reflection"]:
        raise ValueError(f"padding_mode must be one of ['zeros', 'border', 'reflection'], got {padding_mode}")

    # Initialize gradients
    grad_value = torch.zeros_like(value)
    grad_sampling_locations = torch.zeros_like(sampling_locations)
    grad_attention_weights = torch.zeros_like(attention_weights)

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

                        # Get attention weight and grad_output for this point
                        attn_w = attention_weights[b, q, h, l, p]
                        grad_out = grad_output[b, q, h, :]  # (C_h,)

                        # Sample values from the 4 corners
                        v_00 = torch.zeros(C_h, dtype=value.dtype, device=value.device)
                        v_01 = torch.zeros(C_h, dtype=value.dtype, device=value.device)
                        v_10 = torch.zeros(C_h, dtype=value.dtype, device=value.device)
                        v_11 = torch.zeros(C_h, dtype=value.dtype, device=value.device)

                        # Handle padding modes for sampling
                        if padding_mode == "zeros":
                            if 0 <= y0 < H_l and 0 <= x0 < W_l:
                                v_00 = level_value[y0, x0, :]
                            if 0 <= y0 < H_l and 0 <= x1 < W_l:
                                v_01 = level_value[y0, x1, :]
                            if 0 <= y1 < H_l and 0 <= x0 < W_l:
                                v_10 = level_value[y1, x0, :]
                            if 0 <= y1 < H_l and 0 <= x1 < W_l:
                                v_11 = level_value[y1, x1, :]

                        elif padding_mode == "border":
                            y0_clamped = max(0, min(y0, H_l - 1))
                            y1_clamped = max(0, min(y1, H_l - 1))
                            x0_clamped = max(0, min(x0, W_l - 1))
                            x1_clamped = max(0, min(x1, W_l - 1))

                            v_00 = level_value[y0_clamped, x0_clamped, :]
                            v_01 = level_value[y0_clamped, x1_clamped, :]
                            v_10 = level_value[y1_clamped, x0_clamped, :]
                            v_11 = level_value[y1_clamped, x1_clamped, :]

                        elif padding_mode == "reflection":

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

                            v_00 = level_value[y0_ref, x0_ref, :]
                            v_01 = level_value[y0_ref, x1_ref, :]
                            v_10 = level_value[y1_ref, x0_ref, :]
                            v_11 = level_value[y1_ref, x1_ref, :]

                        # Compute sampled value
                        sampled = w_00 * v_00 + w_01 * v_01 + w_10 * v_10 + w_11 * v_11

                        # grad_attn_weight = dot(grad_output, sampled)
                        grad_attention_weights[b, q, h, l, p] = torch.sum(grad_out * sampled)

                        # Scatter gradients back to the 4 corner pixels
                        # grad_value += w_corner * grad_output * attn_weight
                        top_grad_value = grad_out * attn_w  # (C_h,)

                        if padding_mode == "zeros":
                            if 0 <= y0 < H_l and 0 <= x0 < W_l:
                                grad_value[b, level_start + y0 * W_l + x0, h, :] += w_00 * top_grad_value
                            if 0 <= y0 < H_l and 0 <= x1 < W_l:
                                grad_value[b, level_start + y0 * W_l + x1, h, :] += w_01 * top_grad_value
                            if 0 <= y1 < H_l and 0 <= x0 < W_l:
                                grad_value[b, level_start + y1 * W_l + x0, h, :] += w_10 * top_grad_value
                            if 0 <= y1 < H_l and 0 <= x1 < W_l:
                                grad_value[b, level_start + y1 * W_l + x1, h, :] += w_11 * top_grad_value

                        elif padding_mode == "border":
                            y0_clamped = max(0, min(y0, H_l - 1))
                            y1_clamped = max(0, min(y1, H_l - 1))
                            x0_clamped = max(0, min(x0, W_l - 1))
                            x1_clamped = max(0, min(x1, W_l - 1))

                            grad_value[b, level_start + y0_clamped * W_l + x0_clamped, h, :] += w_00 * top_grad_value
                            grad_value[b, level_start + y0_clamped * W_l + x1_clamped, h, :] += w_01 * top_grad_value
                            grad_value[b, level_start + y1_clamped * W_l + x0_clamped, h, :] += w_10 * top_grad_value
                            grad_value[b, level_start + y1_clamped * W_l + x1_clamped, h, :] += w_11 * top_grad_value

                        elif padding_mode == "reflection":

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

                            grad_value[b, level_start + y0_ref * W_l + x0_ref, h, :] += w_00 * top_grad_value
                            grad_value[b, level_start + y0_ref * W_l + x1_ref, h, :] += w_01 * top_grad_value
                            grad_value[b, level_start + y1_ref * W_l + x0_ref, h, :] += w_10 * top_grad_value
                            grad_value[b, level_start + y1_ref * W_l + x1_ref, h, :] += w_11 * top_grad_value

                        # Compute derivatives of bilinear interpolation w.r.t. pixel coordinates
                        # ∂(sampled)/∂y = sum(top_grad * grad_h_weight) * H_l)
                        grad_h_weight = -(1 - dx) * v_00 - dx * v_01 + (1 - dx) * v_10 + dx * v_11

                        # ∂(sampled)/∂x = sum(top_grad * grad_w_weight) * W_l)
                        grad_w_weight = -(1 - dy) * v_00 + (1 - dy) * v_01 - dy * v_10 + dy * v_11

                        # grad_sampling_loc = sum_c [ grad_output[c] * attn_weight * ∂(sampled[c])/∂(u,v) ]
                        grad_u = torch.sum(top_grad_value * grad_w_weight) * W_l
                        grad_v = torch.sum(top_grad_value * grad_h_weight) * H_l

                        grad_sampling_locations[b, q, h, l, p, 0] = grad_u
                        grad_sampling_locations[b, q, h, l, p, 1] = grad_v

    # Restore original layouts if needed
    if original_value_layout == "BNLC":
        grad_value = grad_value.permute(0, 2, 1, 3)  # (B, L, N_h, C_h) -> (B, N_h, L, C_h)

    if original_sampling_layout == "B2QHLP":
        grad_sampling_locations = grad_sampling_locations.permute(
            0, 5, 1, 2, 3, 4
        )  # (B, N_q, N_h, N_l, N_p, 2) -> (B, 2, N_q, N_h, N_l, N_p)

    out = {}
    if compute_grad_value:
        out["out_grad_value"] = grad_value
    if compute_grad_sampling_locations:
        out["out_grad_sampling_locations"] = grad_sampling_locations
    if compute_grad_attention_weights:
        out["out_grad_attention_weights"] = grad_attention_weights
    return out
