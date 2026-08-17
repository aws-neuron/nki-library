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

"""MXFP8 utility functions for the attention TKG kernel."""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_assert import assert_shape, kernel_assert

"""
Hardware/tiling constants used as defaults by mm1_packing_geometry. These mirror
MXTileConstants / TileParams in attention_mxfp8_tkg.py, which passes its own values
explicitly; the defaults exist so the geometry can be derived standalone.
"""
_P_MAX = 128
_BLOCKS_PER_FOLD = 4


# ── Packed-Q Eviction Layout ─────────────────────────────────────────────────
def mm1_packing_geometry(q_head, p_max=_P_MAX, blocks_per_fold=_BLOCKS_PER_FOLD):
    """Derive the packed-Q eviction geometry for a given head count.

    A single MM1 PSUM tile is 128 partitions tall but each Q x K block matmul only
    fills q_head output rows, so several Q variants (each placing the real Q in a
    different free-column band) let multiple KV blocks share one PSUM tile.

    Returns:
        band_p: Partition rows one block's scores occupy (= q_head).
        variants_per_tile: Q variants packed into one PSUM tile (128 // q_head):
            2 for q_head=64, 4 for q_head=32.
        score_tiles_per_fold: Score tiles per fold (blocks_per_fold //
            variants_per_tile): 2 for q_head=64, 1 for q_head=32.
    """
    band_p = q_head
    variants_per_tile = p_max // q_head
    score_tiles_per_fold = blocks_per_fold // variants_per_tile
    return band_p, variants_per_tile, score_tiles_per_fold


# ── MXFP8 Swizzle + Quantize ─────────────────────────────────────────────────
def swizzle_quantize_mx(src: nl.NkiTensor, dst_data: nl.NkiTensor, dst_scale: nl.NkiTensor, sbm: SbufManager) -> None:
    """Swizzle and quantize a [M, K] BF16 tensor in SBUF to MXFP8.

    Stride-2 interleaved transpose into the layout expected by nc_matmul_mx,
    then quantize to float8_e4m3fn_x4.

    Layout Transformation
    =====================

    nc_matmul_mx requires a "group-of-4 transposed" operand layout: groups of
    4 adjacent bf16 elements along K stay contiguous, and the (M, K//4) grid
    of such groups is transposed so that K//4 lands on partitions.

    Shapes through the pipeline:

        src              [M, K]        bf16    (input)
            ↓ view
        src_fp32         [M, K//2]     fp32    (pairs adjacent bf16)
            ↓ stride-2 nc_transpose
        transposed_psum  [K//4, M*2]   fp32    (swizzled — see below)
            ↓ copy to SBUF
        swizzled_sbuf    [K//4, M*2]   fp32    (same data, SBUF resident)
            ↓ quantize_mx (src viewed as bf16)
        dst_data         [K//4, M]     fp8x4   (output, 4 fp8 per element)
        dst_scale        [K//4, M]     uint8   (output, MX scales)

    Element-level mapping (bf16 → bf16, before quantization):

        swizzled_bf16[g, 4m+0] = src[m, 4g+0]
        swizzled_bf16[g, 4m+1] = src[m, 4g+1]
        swizzled_bf16[g, 4m+2] = src[m, 4g+2]
        swizzled_bf16[g, 4m+3] = src[m, 4g+3]

    After quantize_mx, each fp8x4 element at dst_data[g, m] holds the four
    quantized values from src[m, 4g : 4g+4].

    Concrete example (M=4, K=8):

        src [4, 8] bf16:
        m=0: [ a  b  c  d | e  f  g  h ]
        m=1: [ i  j  k  l | m  n  o  p ]
        m=2: [ A  B  C  D | E  F  G  H ]
        m=3: [ I  J  K  L | M  N  O  P ]
              grp0(k=0..3)  grp1(k=4..7)

        swizzled [2, 16] bf16 (= [2, 8] fp32 physically):
        g=0: [ a  b  c  d   i  j  k  l   A  B  C  D   I  J  K  L ]
        g=1: [ e  f  g  h   m  n  o  p   E  F  G  H   M  N  O  P ]
              ←── m=0 ──→  ←── m=1 ──→  ←── m=2 ──→  ←── m=3 ──→

        dst_data [2, 4] fp8x4 (after quantize_mx):
        g=0: [q(a,b,c,d)  q(i,j,k,l)  q(A,B,C,D)  q(I,J,K,L)]
        g=1: [q(e,f,g,h)  q(m,n,o,p)  q(E,F,G,H)  q(M,N,O,P)]
              m=0           m=1          m=2          m=3

    Args:
        src: [M, K] bfloat16 in SBUF. M <= 128, K must be multiple of 128.
        dst_data: [K//4, M] float8_e4m3fn_x4 in SBUF. Pre-allocated output.
        dst_scale: [K//4, M] uint8 in SBUF. Pre-allocated output for scales.
        sbm: SbufManager for SBUF allocation.
    """
    M, K = src.shape
    kernel_assert(src.dtype == nl.bfloat16, f"src must be bfloat16, got {src.dtype}")
    kernel_assert(M <= 128, f"src partition dim must be <= 128, got {M=}")
    kernel_assert(K % 128 == 0, f"src free dim must be a multiple of 128, got {K=}")
    kernel_assert(dst_data.dtype == nl.float8_e4m3fn_x4, f"dst_data must be float8_e4m3fn_x4, got {dst_data.dtype}")
    kernel_assert(dst_scale.dtype == nl.uint8, f"dst_scale must be uint8, got {dst_scale.dtype}")
    assert_shape(dst_data, (K // 4, M), "dst_data")
    assert_shape(dst_scale, (K // 4, M), "dst_scale")

    K_fp32 = K // 2

    src_fp32 = src.view(nl.float32)
    transposed_psum = nl.ndarray((K // 4, M * 2), dtype=nl.float32, buffer=nl.psum)

    for stride_idx in nl.range(2):
        nisa.nc_transpose(
            dst=transposed_psum.slice(dim=1, start=stride_idx, end=M * 2, step=2),
            data=src_fp32.slice(dim=1, start=stride_idx, end=K_fp32, step=2),
        )

    swizzled_sbuf = sbm.alloc_stack((K // 4, M * 2), dtype=nl.float32)
    nisa.tensor_copy(dst=swizzled_sbuf, src=transposed_psum, engine=nisa.scalar_engine)
    nisa.quantize_mx(dst=dst_data, src=swizzled_sbuf.view(nl.bfloat16), dst_scale=dst_scale)
