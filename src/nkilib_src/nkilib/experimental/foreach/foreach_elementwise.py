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

"""NKI kernels for foreach elementwise operations (add, sub, mul, div). These kernels implement memory-efficient elementwise operations using SPMD tiling across cores for both scalar and tensor operands."""

import nki
import nki.isa as nisa
import nki.language as nl

from .foreach_utils import get_spmd_tiling_info

# f_size: free-dim tile size so 2 tiles (input + output) fit in SBUF
_F_SIZE_BF16 = 2048
_F_SIZE_F32 = 1024


def _scalar_kernel_body(
    data: nl.NkiTensor,
    out_hbm: nl.NkiTensor,
    scalar_sb: nl.NkiTensor,
    numel: int,
    op,
) -> None:
    """
    Shared SPMD tiling body for scalar elementwise operations.

    Tiles input data across SPMD cores, applies scalar operation, and writes
    results to output tensor.

    Args:
        data (nl.NkiTensor): [N], Input tensor on HBM.
        out_hbm (nl.NkiTensor): [N], Output tensor on HBM.
        scalar_sb (nl.NkiTensor): [P_MAX, 1], Scalar value in SBUF.
        numel (int): Total number of elements in data.
        op: Operation to apply (nl.add, nl.subtract, nl.multiply).
    """
    P_MAX = nl.tile_size.pmax
    f_size = _F_SIZE_F32 if data.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        tile = nl.ndarray((P_MAX, masked_f), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(tile, data.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.tensor_scalar(tile, tile, op, scalar_sb)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), tile)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        tile = nl.ndarray((tail, 1), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(tile, data.ap([[1, tail], [1, 1]], tail_offset))
        nisa.tensor_scalar(tile, tile, op, scalar_sb[0:tail, 0:1])
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), tile)


def _tensor_kernel_body(
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    out_hbm: nl.NkiTensor,
    numel: int,
    op,
) -> None:
    """
    Shared SPMD tiling body for tensor-tensor elementwise operations.

    Tiles input tensors across SPMD cores, applies tensor operation, and writes
    results to output tensor.

    Args:
        data1 (nl.NkiTensor): [N], First input tensor on HBM.
        data2 (nl.NkiTensor): [N], Second input tensor on HBM.
        out_hbm (nl.NkiTensor): [N], Output tensor on HBM.
        numel (int): Total number of elements in data1.
        op: Operation to apply (nl.add, nl.subtract, nl.multiply).
    """
    P_MAX = nl.tile_size.pmax
    f_size = _F_SIZE_F32 if data1.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        t1 = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((P_MAX, masked_f), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t2, data2.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.tensor_tensor(t1, t1, t2, op)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), t1)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        t1 = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((tail, 1), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t2, data2.ap([[1, tail], [1, 1]], tail_offset))
        nisa.tensor_tensor(t1, t1, t2, op)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), t1)


def _tensor_alpha_kernel_body(
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    out_hbm: nl.NkiTensor,
    alpha_sb: nl.NkiTensor,
    numel: int,
    op,
) -> None:
    """
    Shared SPMD tiling body for tensor operations with alpha scaling.

    Tiles input tensors across SPMD cores, applies alpha * data2, then applies
    operation with data1, and writes results to output tensor.

    Args:
        data1 (nl.NkiTensor): [N], First input tensor on HBM.
        data2 (nl.NkiTensor): [N], Second input tensor on HBM.
        out_hbm (nl.NkiTensor): [N], Output tensor on HBM.
        alpha_sb (nl.NkiTensor): [P_MAX, 1], Alpha scalar in SBUF.
        numel (int): Total number of elements in data1.
        op: Operation to apply (nl.add, nl.subtract).
    """
    P_MAX = nl.tile_size.pmax
    f_size = _F_SIZE_F32 if data1.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        t1 = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((P_MAX, masked_f), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t2, data2.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.tensor_scalar(t2, t2, nl.multiply, alpha_sb)
        res = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(res, t1, t2, op)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), res)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        t1 = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((tail, 1), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t2, data2.ap([[1, tail], [1, 1]], tail_offset))
        nisa.tensor_scalar(t2, t2, nl.multiply, alpha_sb[0:tail, 0:1])
        res = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(res, t1, t2, op)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), res)


def _tensor_reciprocal_kernel_body(
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    out_hbm: nl.NkiTensor,
    numel: int,
) -> None:
    """
    Shared SPMD tiling body for tensor division using reciprocal.

    Tiles input tensors across SPMD cores, computes reciprocal of data2,
    multiplies with data1, and writes results to output tensor.

    Args:
        data1 (nl.NkiTensor): [N], First input tensor on HBM.
        data2 (nl.NkiTensor): [N], Second input tensor on HBM.
        out_hbm (nl.NkiTensor): [N], Output tensor on HBM.
        numel (int): Total number of elements in data1.
    """
    P_MAX = nl.tile_size.pmax
    f_size = _F_SIZE_F32 if data1.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        t1 = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((P_MAX, masked_f), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t2, data2.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.reciprocal(t2, t2)
        res = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(res, t1, t2, nl.multiply)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), res)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        t1 = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((tail, 1), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t1, data1.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t2, data2.ap([[1, tail], [1, 1]], tail_offset))
        nisa.reciprocal(t2, t2)
        res = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(res, t1, t2, nl.multiply)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), res)


@nki.jit
def add_scalar_kernel(
    data: nl.NkiTensor,
    scalar_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise add scalar to tensor.

    Computes out = data + scalar using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.NkiTensor): [N], Input tensor on HBM. Must have ndim >= 1.
        scalar_tensor (nl.NkiTensor): [P_MAX, 1], Scalar broadcast tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element in data:
            out[i] = data[i] + scalar
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    scalar_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(scalar_sb, scalar_tensor[0:P_MAX, 0:1])
    _scalar_kernel_body(data, out_hbm, scalar_sb, numel, nl.add)
    return out_hbm


@nki.jit
def sub_scalar_kernel(
    data: nl.NkiTensor,
    scalar_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise subtract scalar from tensor.

    Computes out = data - scalar using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.NkiTensor): [N], Input tensor on HBM. Must have ndim >= 1.
        scalar_tensor (nl.NkiTensor): [P_MAX, 1], Scalar broadcast tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element in data:
            out[i] = data[i] - scalar
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    scalar_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(scalar_sb, scalar_tensor[0:P_MAX, 0:1])
    _scalar_kernel_body(data, out_hbm, scalar_sb, numel, nl.subtract)
    return out_hbm


@nki.jit
def mul_scalar_kernel(
    data: nl.NkiTensor,
    scalar_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise multiply tensor by scalar.

    Computes out = data * scalar using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.NkiTensor): [N], Input tensor on HBM. Must have ndim >= 1.
        scalar_tensor (nl.NkiTensor): [P_MAX, 1], Scalar broadcast tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element in data:
            out[i] = data[i] * scalar
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    scalar_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(scalar_sb, scalar_tensor[0:P_MAX, 0:1])
    _scalar_kernel_body(data, out_hbm, scalar_sb, numel, nl.multiply)
    return out_hbm


@nki.jit
def div_scalar_kernel(
    data: nl.NkiTensor,
    scalar_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise divide tensor by scalar.

    Computes out = data / scalar using SPMD parallelization across cores.
    Implements division as multiplication by reciprocal for efficiency.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.NkiTensor): [N], Input tensor on HBM. Must have ndim >= 1.
        scalar_tensor (nl.NkiTensor): [P_MAX, 1], Scalar broadcast tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        reciprocal_scalar = 1.0 / scalar
        for each element in data:
            out[i] = data[i] * reciprocal_scalar
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    scalar_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(scalar_sb, scalar_tensor[0:P_MAX, 0:1])
    nisa.reciprocal(scalar_sb, scalar_sb)
    _scalar_kernel_body(data, out_hbm, scalar_sb, numel, nl.multiply)
    return out_hbm


@nki.jit
def add_tensor_kernel(
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    alpha_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise add tensors with alpha scaling.

    Computes out = data1 + alpha * data2 using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensors

    Args:
        data1 (nl.NkiTensor): [N], First input tensor on HBM. Must have ndim >= 1.
        data2 (nl.NkiTensor): [N], Second input tensor on HBM.
        alpha_tensor (nl.NkiTensor): [P_MAX, 1], Alpha scalar broadcast tensor on HBM.
        numel (int): Number of elements in data1.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            out[i] = data1[i] + alpha * data2[i]
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data1.shape, dtype=data1.dtype, buffer=nl.shared_hbm)
    alpha_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(alpha_sb, alpha_tensor[0:P_MAX, 0:1])
    _tensor_alpha_kernel_body(data1, data2, out_hbm, alpha_sb, numel, nl.add)
    return out_hbm


@nki.jit
def sub_tensor_kernel(
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    alpha_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise subtract tensors with alpha scaling.

    Computes out = data1 - alpha * data2 using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensors

    Args:
        data1 (nl.NkiTensor): [N], First input tensor on HBM. Must have ndim >= 1.
        data2 (nl.NkiTensor): [N], Second input tensor on HBM.
        alpha_tensor (nl.NkiTensor): [P_MAX, 1], Alpha scalar broadcast tensor on HBM.
        numel (int): Number of elements in data1.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            out[i] = data1[i] - alpha * data2[i]
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data1.shape, dtype=data1.dtype, buffer=nl.shared_hbm)
    alpha_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(alpha_sb, alpha_tensor[0:P_MAX, 0:1])
    _tensor_alpha_kernel_body(data1, data2, out_hbm, alpha_sb, numel, nl.subtract)
    return out_hbm


@nki.jit
def mul_tensor_kernel(
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise multiply tensors.

    Computes out = data1 * data2 using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensors

    Args:
        data1 (nl.NkiTensor): [N], First input tensor on HBM. Must have ndim >= 1.
        data2 (nl.NkiTensor): [N], Second input tensor on HBM.
        numel (int): Number of elements in data1.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            out[i] = data1[i] * data2[i]
    """
    out_hbm = nl.ndarray(data1.shape, dtype=data1.dtype, buffer=nl.shared_hbm)
    _tensor_kernel_body(data1, data2, out_hbm, numel, nl.multiply)
    return out_hbm


@nki.jit
def div_tensor_kernel(
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise divide tensors.

    Computes out = data1 / data2 using SPMD parallelization across cores.
    Implements division as multiplication by reciprocal for efficiency.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensors

    Args:
        data1 (nl.NkiTensor): [N], First input tensor on HBM. Must have ndim >= 1.
        data2 (nl.NkiTensor): [N], Second input tensor on HBM.
        numel (int): Number of elements in data1.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            reciprocal_data2 = 1.0 / data2[i]
            out[i] = data1[i] * reciprocal_data2
    """
    out_hbm = nl.ndarray(data1.shape, dtype=data1.dtype, buffer=nl.shared_hbm)
    _tensor_reciprocal_kernel_body(data1, data2, out_hbm, numel)
    return out_hbm


@nki.jit
def addcdiv_kernel(
    data: nl.NkiTensor,
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    value_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise addcdiv: data + value * (data1 / data2).

    Computes out = data + value * (data1 / data2) using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensors

    Args:
        data (nl.NkiTensor): [N], Base input tensor on HBM.
        data1 (nl.NkiTensor): [N], Numerator tensor on HBM.
        data2 (nl.NkiTensor): [N], Denominator tensor on HBM.
        value_tensor (nl.NkiTensor): [P_MAX, 1], Scalar value broadcast tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            out[i] = data[i] + value * (data1[i] / data2[i])
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    value_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(value_sb, value_tensor[0:P_MAX, 0:1])
    f_size = _F_SIZE_F32 if data.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        t0 = nl.ndarray((P_MAX, masked_f), dtype=data.dtype, buffer=nl.sbuf)
        t1 = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((P_MAX, masked_f), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t0, data.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t1, data1.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t2, data2.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.reciprocal(t2, t2)
        nisa.tensor_tensor(t1, t1, t2, nl.multiply)
        nisa.tensor_scalar(t1, t1, nl.multiply, value_sb)
        nisa.tensor_tensor(t0, t0, t1, nl.add)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), t0)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        t0 = nl.ndarray((tail, 1), dtype=data.dtype, buffer=nl.sbuf)
        t1 = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((tail, 1), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t0, data.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t1, data1.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t2, data2.ap([[1, tail], [1, 1]], tail_offset))
        nisa.reciprocal(t2, t2)
        nisa.tensor_tensor(t1, t1, t2, nl.multiply)
        nisa.tensor_scalar(t1, t1, nl.multiply, value_sb[0:tail, 0:1])
        nisa.tensor_tensor(t0, t0, t1, nl.add)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), t0)
    return out_hbm


@nki.jit
def addcmul_kernel(
    data: nl.NkiTensor,
    data1: nl.NkiTensor,
    data2: nl.NkiTensor,
    value_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise addcmul: data + value * (data1 * data2).

    Computes out = data + value * (data1 * data2) using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensors

    Args:
        data (nl.NkiTensor): [N], Base input tensor on HBM.
        data1 (nl.NkiTensor): [N], First multiplicand tensor on HBM.
        data2 (nl.NkiTensor): [N], Second multiplicand tensor on HBM.
        value_tensor (nl.NkiTensor): [P_MAX, 1], Scalar value broadcast tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            out[i] = data[i] + value * (data1[i] * data2[i])
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    value_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(value_sb, value_tensor[0:P_MAX, 0:1])
    f_size = _F_SIZE_F32 if data.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        t0 = nl.ndarray((P_MAX, masked_f), dtype=data.dtype, buffer=nl.sbuf)
        t1 = nl.ndarray((P_MAX, masked_f), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((P_MAX, masked_f), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t0, data.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t1, data1.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t2, data2.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.tensor_tensor(t1, t1, t2, nl.multiply)
        nisa.tensor_scalar(t1, t1, nl.multiply, value_sb)
        nisa.tensor_tensor(t0, t0, t1, nl.add)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), t0)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        t0 = nl.ndarray((tail, 1), dtype=data.dtype, buffer=nl.sbuf)
        t1 = nl.ndarray((tail, 1), dtype=data1.dtype, buffer=nl.sbuf)
        t2 = nl.ndarray((tail, 1), dtype=data2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t0, data.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t1, data1.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t2, data2.ap([[1, tail], [1, 1]], tail_offset))
        nisa.tensor_tensor(t1, t1, t2, nl.multiply)
        nisa.tensor_scalar(t1, t1, nl.multiply, value_sb[0:tail, 0:1])
        nisa.tensor_tensor(t0, t0, t1, nl.add)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), t0)
    return out_hbm


@nki.jit
def lerp_kernel(
    data: nl.NkiTensor,
    end: nl.NkiTensor,
    weight_tensor: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise linear interpolation: data + weight * (end - data).

    Computes out = data + weight * (end - data) using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensors

    Args:
        data (nl.NkiTensor): [N], Start tensor on HBM.
        end (nl.NkiTensor): [N], End tensor on HBM.
        weight_tensor (nl.NkiTensor): [P_MAX, 1], Interpolation weight broadcast tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            out[i] = data[i] + weight * (end[i] - data[i])
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    weight_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(weight_sb, weight_tensor[0:P_MAX, 0:1])
    f_size = _F_SIZE_F32 if data.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        t_start = nl.ndarray((P_MAX, masked_f), dtype=data.dtype, buffer=nl.sbuf)
        t_end = nl.ndarray((P_MAX, masked_f), dtype=end.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t_start, data.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.dma_copy(t_end, end.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        # diff = end - start
        nisa.tensor_tensor(t_end, t_end, t_start, nl.subtract)
        # diff = weight * diff
        nisa.tensor_scalar(t_end, t_end, nl.multiply, weight_sb)
        # out = start + diff
        nisa.tensor_tensor(t_start, t_start, t_end, nl.add)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), t_start)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        t_start = nl.ndarray((tail, 1), dtype=data.dtype, buffer=nl.sbuf)
        t_end = nl.ndarray((tail, 1), dtype=end.dtype, buffer=nl.sbuf)
        nisa.dma_copy(t_start, data.ap([[1, tail], [1, 1]], tail_offset))
        nisa.dma_copy(t_end, end.ap([[1, tail], [1, 1]], tail_offset))
        nisa.tensor_tensor(t_end, t_end, t_start, nl.subtract)
        nisa.tensor_scalar(t_end, t_end, nl.multiply, weight_sb[0:tail, 0:1])
        nisa.tensor_tensor(t_start, t_start, t_end, nl.add)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), t_start)
    return out_hbm


@nki.jit
def sqrt_kernel(
    data: nl.NkiTensor,
    numel: int,
) -> nl.NkiTensor:
    """
    Elementwise square root.

    Computes out = sqrt(data) using SPMD parallelization across cores.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.NkiTensor): [N], Input tensor on HBM. Elements must be non-negative.
        numel (int): Number of elements in data.

    Returns:
        out (nl.NkiTensor): [N], Output tensor on HBM.

    Pseudocode:
        for each element:
            out[i] = sqrt(data[i])
    """
    P_MAX = nl.tile_size.pmax
    out_hbm = nl.ndarray(data.shape, dtype=data.dtype, buffer=nl.shared_hbm)
    f_size = _F_SIZE_F32 if data.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, _ = get_spmd_tiling_info(numel, tile_size)
    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        tile = nl.ndarray((P_MAX, masked_f), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(tile, data.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        nisa.activation(tile, op=nl.sqrt, data=tile)
        nisa.dma_copy(out_hbm.ap([[masked_f, P_MAX], [1, masked_f]], offset), tile)
    if tail > 0:
        tail_offset = core_base + core_n_aligned
        tile = nl.ndarray((tail, 1), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(tile, data.ap([[1, tail], [1, 1]], tail_offset))
        nisa.activation(tile, op=nl.sqrt, data=tile)
        nisa.dma_copy(out_hbm.ap([[1, tail], [1, 1]], tail_offset), tile)
    return out_hbm
