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

"""Subkernels for performant loading of BF16 data with MXFP8 quantization. WARNING: Active development."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

P_MAX = 128
_BF16_TO_FP32_PACK = 2  # Two BF16 elements pack into one FP32 element
_K_BLOCK_SIZE = 512  # BF16 elements per K block (maps to 128P after swizzle)
_SWIZZLE_STRIDE = 2  # Stride-2 interleave for MX swizzle layout
_TRANSPOSED_FREE_DIM = 256  # Free dimension after FP32 transpose (_K_BLOCK_SIZE // _BF16_TO_FP32_PACK)
_QUANTIZED_FREE_DIM = 512  # Free dimension for quantize_mx input (_K_BLOCK_SIZE in BF16)


@nki.jit
def mxfp_load_performance_wrapper(tensor: nl.ndarray):
    """Load a BF16 [M, K] tensor from HBM, swizzle, and quantize to MXFP8.

    Wrapper for benchmarking the swizzled load + quantize pipeline across
    varying K dimensions.

    Dimensions:
        M: Number of rows (partition dimension), must be 128.
        K: Number of BF16 columns, must be a multiple of 512.

    Args:
        tensor (nl.ndarray): [M, K] bfloat16 in HBM. Input tensor to quantize.

    Returns:
        out_data_hbm (nl.ndarray): [P_MAX, M * K // 512] float8_e4m3fn_x4 in HBM.
            Quantized MXFP8 data with K blocks concatenated along free dimension.
        out_scale_hbm (nl.ndarray): [P_MAX, M * K // 512] uint8 in HBM.
            MX scales with 16 active rows at partition offsets [0, 32, 64, 96].

    Notes:
        - M must be exactly 128 (P_MAX). Smaller M support is TODO.
        - K must be a multiple of 512.

    Pseudocode:
        mx_data, mx_scale = allocate_sbuf(P_MAX, M * K // 512)
        load_and_quantize_mxfp_mk(tensor, mx_data, mx_scale)
        out_data_hbm, out_scale_hbm = dma_copy(mx_data), dma_copy(mx_scale)
        return out_data_hbm, out_scale_hbm
    """
    M, K = tensor.shape
    kernel_assert(M == P_MAX, f"M must be {P_MAX}, got {M=}")
    kernel_assert(K % _K_BLOCK_SIZE == 0, f"K must be a multiple of {_K_BLOCK_SIZE}, got {K=}")

    k_block_count = K // _K_BLOCK_SIZE
    out_free_dim = M * k_block_count

    mx_data_sbuf = nl.ndarray((P_MAX, out_free_dim), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    mx_scale_sbuf = nl.ndarray((P_MAX, out_free_dim), dtype=nl.uint8, buffer=nl.sbuf)

    load_and_quantize_mxfp_mk(tensor, mx_data_sbuf, mx_scale_sbuf)

    out_data_hbm = nl.ndarray(shape=mx_data_sbuf.shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.shared_hbm)
    out_scale_hbm = nl.ndarray(shape=mx_scale_sbuf.shape, dtype=nl.uint8, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out_data_hbm, src=mx_data_sbuf)
    nisa.dma_copy(dst=out_scale_hbm, src=mx_scale_sbuf)

    return out_data_hbm, out_scale_hbm


def load_and_quantize_mxfp_mk(
    tensor: nl.ndarray,
    mx_data_sbuf: nl.ndarray,
    mx_scales_sbuf: nl.ndarray,
) -> None:
    """Load a BF16 [M, K] tensor from HBM and quantize to MXFP8 in SBUF.

    Performs a swizzled load that establishes the stride-2 interleaved layout
    required by quantize_mx, then quantizes each 512-element K block.
    The input has contraction dimension K along the free axis (M-by-K layout).

    Args:
        tensor (nl.ndarray): [M, K] bfloat16 in HBM. Input tensor.
        mx_data_sbuf (nl.ndarray): [P_MAX, M * K // 512] float8_e4m3fn_x4 in SBUF.
            Pre-allocated output for quantized data.
        mx_scales_sbuf (nl.ndarray): [P_MAX, M * K // 512] uint8 in SBUF.
            Pre-allocated output for MX scales.

    Notes:
        - M must be exactly 128 (P_MAX). TODO: support smaller M with masking.
        - K must be a multiple of 512.
        - No-op if tensor is None.

    Pseudocode:
        sbuf_fp32 = dma_copy(tensor, reinterpret_as=fp32)  # [M, K // 2]
        for k_block_idx in range(K // 512):
            psum = strided_transpose(sbuf_fp32, block=k_block_idx)  # [128, 256] fp32
            swizzled = tensor_copy(psum)  # PSUM -> SBUF
            quantize_mx(swizzled.view(bf16))  # [128, 512] bf16 -> mx_data + mx_scale
    """
    if tensor is None:
        return

    M, K = tensor.shape
    k_fp32 = K // _BF16_TO_FP32_PACK
    k_block_count = K // _K_BLOCK_SIZE

    # Step 1: DMA copy HBM -> SBUF, reinterpreting BF16 pair as single FP32
    tensor_sbuf_fp32 = nl.ndarray((M, k_fp32), dtype=nl.float32, buffer=nl.sbuf)
    tensor_fp32_view = tensor.ap(pattern=[[k_fp32, M], [1, k_fp32]], dtype=nl.float32)
    nisa.dma_copy(dst=tensor_sbuf_fp32, src=tensor_fp32_view)

    for k_block_idx in nl.range(k_block_count):
        # Step 2: Stride-2 interleaved transpose (SBUF -> PSUM)
        transposed_psum = nl.ndarray(
            shape=(P_MAX, _TRANSPOSED_FREE_DIM),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for stride_idx in nl.range(_SWIZZLE_STRIDE):
            src_ap = tensor_sbuf_fp32.ap(
                pattern=[[k_fp32, P_MAX], [_SWIZZLE_STRIDE, P_MAX]],
                offset=stride_idx + (k_block_idx * _TRANSPOSED_FREE_DIM),
            )
            dst_ap = transposed_psum.ap(
                pattern=[[_TRANSPOSED_FREE_DIM, P_MAX], [_SWIZZLE_STRIDE, P_MAX]],
                offset=stride_idx,
            )
            nisa.nc_transpose(dst=dst_ap, data=src_ap)

        # Step 3: Copy PSUM -> SBUF as FP32
        swizzled_sbuf_fp32 = nl.ndarray(
            (P_MAX, _TRANSPOSED_FREE_DIM),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=swizzled_sbuf_fp32, src=transposed_psum, engine=nisa.scalar_engine)

        # Step 4: Reinterpret FP32 as BF16 via AP, then quantize to MXFP8
        quant_src = swizzled_sbuf_fp32.ap(
            pattern=[[_QUANTIZED_FREE_DIM, P_MAX], [1, _QUANTIZED_FREE_DIM]],
            dtype=nl.bfloat16,
        )
        nisa.quantize_mx(
            dst=mx_data_sbuf[:, nl.ds(k_block_idx * M, M)],
            src=quant_src,
            dst_scale=mx_scales_sbuf[:, nl.ds(k_block_idx * M, M)],
        )
