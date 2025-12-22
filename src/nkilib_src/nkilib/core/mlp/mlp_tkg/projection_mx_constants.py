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

"""
Shared constants and configuration for MX projection sub-kernels.

These constants and ProjConfig are used by both H-sharding and I-sharding projection implementations.
"""

from dataclasses import dataclass

import nki.language as nl

from ...utils.kernel_assert import kernel_assert

# Hardware config
NUM_QUADRANTS_IN_SBUF = 4
SBUF_QUADRANT_SIZE = 32

# Trn3 quantization block shape for MXFP4/8
_q_height = 8
_q_width = 4

# QMX config
MX_DTYPES = [nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4, nl.float8_e5m2_x4]
SUPPORTED_QMX_INPUT_DTYPES = [nl.float16, nl.bfloat16]
SUPPORTED_QMX_OUTPUT_DTYPES = [nl.float8_e4m3fn_x4]
MX_SCALE_DTYPE = nl.uint8
SUPPORTED_QMX_OUTPUT_F_DIMS = [128, 512]
SCALE_P_ELEM_PER_QUADRANT = 4
ALLOWED_P_SCALE_IDX_OFFSETS = [
    0,
    4,
    8,
    12,
]  # Can fit 4x scales of width 4 in P into partitions 0:16 in each SBUF quadrant

# MMULMX config
# MatmultMX can accept P in [32, 64, 128], corresponding to unpacked P [128, 256, 512]
ALLOWED_P_DIM_MX = [32, 64, 128]
ALLOWED_UNPACKED_P_DIM_MX = [128, 256, 512]

# Gate/up indices
GATE_FUSED_IDX, UP_FUSED_IDX = 0, 1

# Constants for projections (the new NKI FE will implicitly set values to -1 if we use nl API, so hardcode here)
_pmax = 128  # sbuf max partition dim
_psum_fmax = 512  # psum max free dim (elts in fp32)
_psum_bmax = 8  # psum max bank dim


@dataclass
class ProjConfig(nl.NKIObject):
    """Configuration for MX projection sub-kernels with H-sharding."""

    # Tensor shapes / dims
    H: int
    I: int
    BxS: int

    # LNC info
    n_prgs: int
    prg_id: int
    force_lnc1: bool = False  # when set, force LNC1 and use n_prgs==1

    # Used when sharing one tensor for both gate & up proj
    bias_t_shared_between_gate_up: bool = False
    bias_t_shared_base_offset: int = 0

    ######### Down projection specifics #########
    # Used for tkg projections (BxS <= 128), this makes output SB tensor have 128
    # partitions, with the actual values starting on partitions [out_p_offset, out_p_offset+BxS).
    # This requires out_p_offset + BxS <= 128 and out_p_offset % 32 == 0
    out_p_offset: int = 0

    # Debug
    dbg_hidden: bool = False
    dbg_weight: bool = False

    def check_shapes(self):
        kernel_assert(self.H % _pmax == 0, f"H={self.H} must be divisible by num partitions ({_pmax})")
        kernel_assert(self.H1 % self.n_prgs == 0, f"H1={self.H1=} must be disible by num shards ({self.n_prgs})")
        kernel_assert(
            self.H1_sharded % _q_width == 0,
            f"We currently require H1_sharded={self.H1_sharded} to be divisible by {_q_width}",
        )

        kernel_assert(
            self.r_I512_tile % (_q_width * _q_height) == 0,
            f"MX4 MLP Proj requires I512 tile remainder ({self.r_I512_tile}) to be divisible by {_q_width*_q_height} for quantization",
        )

        if self.out_p_offset != 0:
            kernel_assert(0 <= self.out_p_offset < _pmax, f"MX4 Proj illegal {self.out_p_offset=}")
            kernel_assert(
                self.out_p_offset % 32 == 0,
                f"MX4 Proj requires output partition offset starting at SB quadrants, got {self.out_p_offset=}",
            )
            kernel_assert(
                self.out_p_offset + self.BxS <= 128,
                f"MX4 Proj requires output data to fit in 128 partitions when using non-zero partition offset",
            )

    # Compute intermediate shapes / dims
    def __post_init__(self):
        self.H0 = _pmax
        self.H1 = self.H // _pmax
        self.H1_sharded = self.H1 // self.n_prgs

        self.H_sharded = self.H // self.n_prgs

        # Get tiling info
        self.n_H512_tile_sharded = self.H1_sharded // _q_width

        self.n_H512_tile = self.n_H512_tile_sharded * self.n_prgs
        I512_tiling_info = divmod(self.I, _pmax * _q_width)
        self.n_I512_tile = I512_tiling_info[0]  # Deprecated: should not need this in the new FE, use n_total_I512_tile
        self.r_I512_tile = I512_tiling_info[1]  # Deprecated: should not need this in the new FE, use n_total_I512_tile
        self.n_total_I512_tile = self.n_I512_tile + (self.r_I512_tile > 0)
        self.n_par_r_I512_tile = (
            self.r_I512_tile // _q_width
        )  # num partitions for remainder I512 tile (set to fill consec. 4-elt seq on the free dim)

        self.BxS_tile_sz = min(self.BxS, _psum_fmax * 2 // _q_width)  # double psum elts because out is in bf16
        if self.force_lnc1:
            self.n_prgs = 1
            self.prg_id = 0

        self.check_shapes()
