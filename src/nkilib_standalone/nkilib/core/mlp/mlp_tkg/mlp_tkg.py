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


from typing import Optional, Tuple

# import logging
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ...utils.allocator import SbufManager

# common utils
from ...utils.common_types import ActFnType, NormType
from ...utils.logging import Logger

# MLP utils
from ..mlp_parameters import MLPParameters, mlpp_has_fused_add, mlpp_store_fused_add, mlpp_has_normalization
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_down_projection import process_down_projection
from .mlp_tkg_gate_up_projection import process_gate_up_projection
from .mlp_tkg_utils import input_fused_add, input_norm_load


def mlp_tkg(
    params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
) -> list[nl.ndarray]:
    """
    Allocated kernel that computes Norm(hidden) @ wMLP.
    """
    io_dtype = params.hidden_tensor.dtype

    # ---------------- Compute Kernel Dimensions & SBUF Manager ----------------
    dims = MLPTKGConstants.calculate_constants(params)

    sbm = SbufManager(0, 200 * 1024, Logger("mlp_tkg"))
    sbm.open_scope()  # Start SBUF allocation scope

    # ---------------- Fused Add ----------------
    # Apply fused add if present (hidden + attention output)
    hidden = params.hidden_tensor
    if mlpp_has_fused_add(params):
        if not params.fused_add_params.store_fused_add_result:
            H1_dim = dims.H1 if mlpp_has_normalization(params) else dims.H1_shard
            fused_add_output_sb = sbm.alloc_heap(
                (dims.H0, dims.T, H1_dim),
                dtype=io_dtype,
                buffer=nl.sbuf,
                name="fused_add_output_sb",
            )
            fused_add_output = fused_add_output_sb
        else:
            fused_add_output = output_stored_add_tensor_hbm

        input_fused_add(
            input=hidden,
            fused_add_tensor=params.fused_add_params.fused_add_tensor,
            fused_output=fused_add_output,
            normtype=params.norm_params.normalization_type,
            store_fused_add_result=params.fused_add_params.store_fused_add_result,
            sbm=sbm,
            dims=dims,
        )
        hidden = fused_add_output  # Use fused result as hidden input

    # ---------------- Norm / Input Load ----------------
    if mlpp_has_normalization(params) or (hidden.buffer != nl.sbuf):
        # Allocate heap tile for normalized or loaded input
        input_sb = sbm.alloc_heap(
            (dims.H0, dims.T, dims.H1_shard),
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="input_sbuf",
        )
        input_norm_load(hidden, input_sb, params, dims, sbm)  # Norm or direct load

        if hidden.buffer == nl.sbuf:
            sbm.pop_heap()  # dealloc fused_add_output_sb
    else:
        input_sb = hidden

    # ---------------- Allocate Gate/Up/Down Projection SBUF ----------------
    # Allocate SBUF tile for gate/up projection output
    gate_up_sb = sbm.alloc_stack(
        (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
        dtype=io_dtype,
        buffer=nl.sbuf,
        name="gate_up_sbuf",
    )

    # Heap tile for pre-transpose gate_up output if column tiling is enabled
    gate_up_sb_before_tp = None
    if params.use_tkg_gate_up_proj_column_tiling:
        gate_up_sb_before_tp = sbm.alloc_heap(
            (dims.T, dims.I),
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="gate_up_sbuf_before_tp",
        )

    # Allocate SBUF tile for down projection output
    if params.use_tkg_down_proj_column_tiling:
        down_sb = sbm.alloc_stack(
            (dims.T, dims.H_per_shard),
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="down_sbuf",
        )
    else:
        down_sb = sbm.alloc_stack(
            (dims.H0, dims.H1_shard, dims.T),
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="down_sbuf",
        )

    # ---------- Process gate/up projection, silu, gate/up multiplication ----------
    gate_output = gate_up_sb_before_tp if params.use_tkg_gate_up_proj_column_tiling else gate_up_sb
    gate_tile_info = process_gate_up_projection(
        hidden=input_sb,
        output=gate_output,
        params=params,
        dims=dims,
        sbm=sbm,
    )

    # dealloc input_sb
    sbm.pop_heap()

    # ---------- Transpose hidden if column tiling is enabled ----------
    if params.use_tkg_gate_up_proj_column_tiling:
        # Transpose hidden [T, I] to [I1, I0, T]
        for i1_tile in range(dims.num_total_128_tiles_per_I):
            i_tile_size = dims.I0 if dims.num_total_128_tiles_per_I - 1 != i1_tile else dims.I - dims.I0 * i1_tile
            psum_idx = i1_tile % dims._psum_bmax
            tp_psum = nl.ndarray(
                (i_tile_size, dims.T),
                dtype=io_dtype,
                buffer=nl.psum,
                name=f"transpose_psum_{i1_tile}",
                address=(0, psum_idx * dims._psum_fmax * 4),
            )
            nisa.nc_transpose(
                dst=tp_psum,
                data=gate_output[0 : dims.T, nl.ds(i1_tile * dims.I0, i_tile_size)],
            )
            nisa.tensor_copy(dst=gate_up_sb[0:i_tile_size, i1_tile, 0 : dims.T], src=tp_psum)

        # dealloc gate_up_sb_before_tp
        sbm.pop_heap()

    # ---------- Process down projection ----------
    process_down_projection(
        hidden=gate_up_sb,
        output=down_sb,
        params=params,
        dims=dims,
        gate_tile_info=gate_tile_info,
        sbm=sbm,
    )

    # ---------- Return output ----------
    if not params.store_output_in_sbuf:
        output_sb = sbm.alloc_stack(
            (dims.T, dims.H_per_shard),
            dtype=params.output_dtype,
            buffer=nl.sbuf,
            name="mlp_output_sb",
        )
        if params.use_tkg_down_proj_column_tiling:
            output_sb = down_sb
        else:
            # Transpose output[H0, H1, T] to [T, H]
            H0, H1, T = down_sb.shape
            for h1_tile in range(H1):
                psum_idx = h1_tile % dims._psum_bmax
                tp_psum = nl.ndarray(
                    (T, H0),
                    dtype=params.output_dtype,
                    buffer=nl.psum,
                    name=f"transpose_output_{h1_tile}",
                    address=(0, psum_idx * dims._psum_fmax * 4),
                )
                nisa.nc_transpose(dst=tp_psum[0:T, 0:H0], data=down_sb[0:H0, h1_tile, 0:T])
                if h1_tile % 2 == 0:
                    nisa.tensor_copy(
                        src=tp_psum[0:T, 0:H0],
                        dst=output_sb.ap(
                            pattern=[[dims.H_per_shard, T], [H1, H0]],
                            offset=h1_tile,
                        ),
                        engine=nisa.engine.vector,
                    )
                else:
                    nisa.tensor_copy(
                        src=tp_psum[0:T, 0:H0],
                        dst=output_sb.ap(
                            pattern=[[dims.H_per_shard, T], [H1, H0]],
                            offset=h1_tile,
                        ),
                        engine=nisa.engine.scalar,
                    )
        # reshape to 2D tensor
        B, S, H = output_tensor_hbm.shape
        output_tensor_hbm = output_tensor_hbm.reshape((B * S, H))
        nisa.dma_copy(
            dst=output_tensor_hbm[
                :,
                nl.ds(
                    dims.shard_id * dims.H_per_shard,
                    dims.H_per_shard,
                ),
            ],
            src=output_sb[:, 0 : dims.H_per_shard],
        )
        # reshape back to 3D tensor
        output_tensor_hbm = output_tensor_hbm.reshape((B, S, H))
        sbm.close_scope()  # Close SBUF allocation scope

        return (
            [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
        )

    else:
        sbm.close_scope()  # Close SBUF allocation scope

        return [down_sb, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [down_sb]
