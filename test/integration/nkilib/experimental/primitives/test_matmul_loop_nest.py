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

"""Tests for matmul_loop_nest abstraction."""

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.utils.tiled_tensor import TiledTensor
from nkilib_src.nkilib.experimental.primitives.matmul_loop_nest import matmul_loop_nest

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def alloc_tiled_sbuf(shape, tile_size, dtype, rotate_dim=None, num_rotation=None):
    """Allocate a grid of SBUF tiles."""
    grid = (shape[0] // tile_size[0], shape[1] // tile_size[1])
    rotate = (rotate_dim, num_rotation) if rotate_dim is not None else None
    return TiledTensor.alloc(grid=grid, tile_size=tile_size, dtype=dtype, buffer=nl.sbuf, rotate=rotate)


def alloc_tiled_psum(shape, tile_size, dtype=nl.float32, num_banks=None):
    """Allocate a grid of PSUM tiles."""
    grid = (shape[0] // tile_size[0], shape[1] // tile_size[1])
    n_banks = num_banks if num_banks else grid[0] * grid[1]
    return TiledTensor.alloc(grid=grid, tile_size=tile_size, dtype=dtype, buffer=nl.psum, num_banks=n_banks)


# ── NKIObject callback classes for parser mode ───────────────────────


@dataclass
class _LoadWeights(nl.NKIObject):
    mov_sb: object
    weights: object
    K_TILE: int

    def run(self, k, n, buf):
        nisa.dma_copy(dst=self.mov_sb[k, n], src=self.weights[k * self.K_TILE : (k + 1) * self.K_TILE, :])


@dataclass
class _DrainBias(nl.NKIObject):
    bias_sb: object

    def run(self, psum_tile, sbuf_tile, m, n):
        nisa.tensor_tensor(dst=sbuf_tile, data1=psum_tile, data2=self.bias_sb, op=nl.add)


@dataclass
class _PostMatmulScale(nl.NKIObject):
    dst_sb: object

    def run(self, psum_tile, k, m, n):
        nisa.activation(dst=self.dst_sb[m, n], op=nl.copy, data=psum_tile, scale=0.5)


def _drain_copy_sliced(psum_tile, sbuf_tile, m, n):
    nisa.tensor_copy(dst=sbuf_tile, src=psum_tile[: sbuf_tile.shape[0], : sbuf_tile.shape[1]])


# ── NKI kernel wrappers ──────────────────────────────────────────────


def kernel_basic_gemm(source, weights, out):
    """source[M,K] × weights[K,N] → out[M,N], K ≤ 128, M can exceed 128."""
    M, K = source.shape
    _, N = weights.shape
    M_TILE = 128

    # Stationary [K, M]: tile along M (free dim), each tile [K, M_TILE]
    stat_sb = alloc_tiled_sbuf(shape=(K, M), tile_size=(K, M_TILE), dtype=source.dtype)
    mov_sb = alloc_tiled_sbuf(shape=(K, N), tile_size=(K, N), dtype=weights.dtype)
    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M_TILE, N), dtype=out.dtype)

    # Load each M tile: dma_transpose source[m*128:(m+1)*128, :K] → SBUF [K, 128]
    NUM_M = M // M_TILE
    for m in range(NUM_M):
        nisa.dma_transpose(dst=stat_sb[0, m], src=source[m * M_TILE : (m + 1) * M_TILE, :])
    nisa.dma_copy(dst=mov_sb[0, 0], src=weights)

    matmul_loop_nest(stationary=stat_sb, moving=mov_sb, dst_sbuf=dst_sb)

    # Store each M tile
    for m in range(NUM_M):
        nisa.dma_copy(dst=out[m * M_TILE : (m + 1) * M_TILE, :], src=dst_sb[m, 0])
    return [out]


def kernel_k_accumulation(source, weights, out):
    """source[M,K] × weights[K,N] → out[M,N], K > 128."""
    M, K_total = source.shape
    _, N = weights.shape
    K_TILE = 128
    NUM_K = K_total // K_TILE
    NUM_BUFS = min(NUM_K, 4)

    # Stationary: NUM_K tiles of [K_TILE, M], each separately allocated
    stat_sb = alloc_tiled_sbuf(shape=(K_total, M), tile_size=(K_TILE, M), dtype=source.dtype)
    # Load each tile via dma_transpose: HBM source[:, k*128:(k+1)*128] → SBUF [128, M]
    for k in range(NUM_K):
        nisa.dma_transpose(dst=stat_sb[k, 0], src=source[:, k * K_TILE : (k + 1) * K_TILE])

    # Moving: rotating buffers
    mov_sb = alloc_tiled_sbuf(
        shape=(K_total, N), tile_size=(K_TILE, N), dtype=weights.dtype, rotate_dim=0, num_rotation=NUM_BUFS
    )
    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M, N), dtype=out.dtype)

    # HBM weights tiled for load_weights callback
    TiledTensor(weights, (K_TILE, N))

    matmul_loop_nest(
        stationary=stat_sb,
        moving=mov_sb,
        dst_sbuf=dst_sb,
        load_weights=_LoadWeights(mov_sb=mov_sb, weights=weights, K_TILE=K_TILE).run,
    )

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


def kernel_with_bias(source, weights, bias, out):
    """source[M,K] × weights[K,N] + bias[1,N] → out[M,N]."""
    M, K = source.shape
    _, N = weights.shape

    stat_sb = alloc_tiled_sbuf(shape=(K, M), tile_size=(K, M), dtype=source.dtype)
    mov_sb = alloc_tiled_sbuf(shape=(K, N), tile_size=(K, N), dtype=weights.dtype)
    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M, N), dtype=out.dtype)

    nisa.dma_transpose(dst=stat_sb[0, 0], src=source)
    nisa.dma_copy(dst=mov_sb[0, 0], src=weights)

    bias_sb = nl.ndarray((M, N), dtype=bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sb, src=bias)

    matmul_loop_nest(stationary=stat_sb, moving=mov_sb, dst_sbuf=dst_sb, on_drain=_DrainBias(bias_sb=bias_sb).run)

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


# ── Torch references ─────────────────────────────────────────────────


def torch_basic_gemm(source, weights, out):
    import torch

    out.copy_(torch.matmul(source, weights))
    return {"out": out}


def torch_k_accumulation(source, weights, out):
    import torch

    out.copy_(torch.matmul(source, weights))
    return {"out": out}


def torch_with_bias(source, weights, bias, out):
    import torch

    out.copy_(torch.matmul(source, weights) + bias)
    return {"out": out}


# ── Test parameters ──────────────────────────────────────────────────

BASIC_GEMM_PARAMS = [
    (128, 128, 512, nl.bfloat16),
    (128, 128, 128, nl.bfloat16),
    (128, 128, 256, nl.bfloat16),
    (256, 128, 512, nl.bfloat16),  # M > 128: 2 M tiles
    (512, 128, 512, nl.bfloat16),  # M > 128: 4 M tiles
]
K_ACCUM_PARAMS = [
    (128, 256, 512, nl.bfloat16),
    (128, 512, 512, nl.bfloat16),
    (128, 1024, 512, nl.bfloat16),
]
BIAS_PARAMS = [
    (128, 128, 512, nl.bfloat16),
    (128, 128, 128, nl.bfloat16),
]


# ── Test class ───────────────────────────────────────────────────────


class TestMatmulLoopNest:
    @pytest_parametrize("M, K, N, dtype", BASIC_GEMM_PARAMS, abbrevs={"dtype": "dt"})
    def test_basic_gemm(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_basic_gemm,
            torch_ref=torch_ref_wrapper(torch_basic_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)

    @pytest_parametrize("M, K, N, dtype", K_ACCUM_PARAMS, abbrevs={"dtype": "dt"})
    def test_k_accumulation(
        self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype
    ):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_k_accumulation,
            torch_ref=torch_ref_wrapper(torch_k_accumulation),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)

    @pytest_parametrize("M, K, N, dtype", BIAS_PARAMS, abbrevs={"dtype": "dt"})
    def test_with_bias(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "bias": np.broadcast_to(gen(name="bias", shape=(1, N), dtype=dtype), (M, N)).copy(),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_with_bias,
            torch_ref=torch_ref_wrapper(torch_with_bias),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── Additional kernel wrappers for doc examples ──────────────────────


def kernel_reduction(const_matrix, data, out):
    """Reduction: const[128,128] × data[128,N] → out[128,N].

    Models RMSNorm TKG: stationary is a constant 1/H matrix, K=1 tile.
    """
    K, M = const_matrix.shape
    _, N = data.shape

    stat_sb = alloc_tiled_sbuf(shape=(K, M), tile_size=(K, M), dtype=const_matrix.dtype)
    mov_sb = alloc_tiled_sbuf(shape=(K, N), tile_size=(K, N), dtype=data.dtype)
    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M, N), dtype=out.dtype)

    # const_matrix is already [K=128, M=128], no transpose needed
    nisa.dma_copy(dst=stat_sb[0, 0], src=const_matrix)
    nisa.dma_copy(dst=mov_sb[0, 0], src=data)

    matmul_loop_nest(stationary=stat_sb, moving=mov_sb, dst_sbuf=dst_sb)

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


def kernel_post_matmul(stationary_data, moving_data, out):
    """Q×K style: stationary[K,M] × moving_tiles[K,N_i] → PSUM, with on_post_matmul scaling.

    Models attention Q×K: scores stay in PSUM, on_post_matmul applies scale.
    dst_sbuf=None, results written via on_post_matmul callback.
    """
    K, M = stationary_data.shape
    _, N = moving_data.shape
    N_TILE = min(N, 512)
    NUM_N = N // N_TILE

    stat_sb = alloc_tiled_sbuf(shape=(K, M), tile_size=(K, M), dtype=stationary_data.dtype)
    mov_sb = alloc_tiled_sbuf(shape=(K, N), tile_size=(K, N_TILE), dtype=moving_data.dtype)
    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M, N_TILE), dtype=out.dtype)
    dst_psum = alloc_tiled_psum(shape=(M, N), tile_size=(M, N_TILE))

    nisa.dma_copy(dst=stat_sb[0, 0], src=stationary_data)
    for n in range(NUM_N):
        nisa.dma_copy(dst=mov_sb[0, n], src=moving_data[:, n * N_TILE : (n + 1) * N_TILE])

    # on_post_matmul: scale PSUM result and copy to SBUF
    matmul_loop_nest(
        stationary=stat_sb,
        moving=mov_sb,
        dst_psum=dst_psum,
        # dst_sbuf=None — results handled by on_post_matmul
        on_post_matmul=_PostMatmulScale(dst_sb=dst_sb).run,
    )

    for n in range(NUM_N):
        nisa.dma_copy(dst=out[:, n * N_TILE : (n + 1) * N_TILE], src=dst_sb[0, n])
    return [out]


# ── Additional torch references ──────────────────────────────────────


def torch_reduction(const_matrix, data, out):
    import torch

    out.copy_(torch.matmul(const_matrix.T.float(), data.float()).to(out.dtype))
    return {"out": out}


def torch_post_matmul(stationary_data, moving_data, out):
    import torch

    result = torch.matmul(stationary_data.T.float(), moving_data.float()) * 0.5
    out.copy_(result.to(out.dtype))
    return {"out": out}


# ── Additional test parameters ───────────────────────────────────────

REDUCTION_PARAMS = [
    (128, 128, 128, nl.bfloat16),  # const[128,128] × data[128,128]
    (128, 128, 512, nl.bfloat16),  # const[128,128] × data[128,512]
]

POST_MATMUL_PARAMS = [
    (128, 128, 512, nl.bfloat16),  # Single N tile
    (128, 128, 1024, nl.bfloat16),  # Two N tiles
]


# ── Additional test methods ──────────────────────────────────────────


class TestMatmulLoopNestExtended:
    @pytest_parametrize("K, M, N, dtype", REDUCTION_PARAMS, abbrevs={"dtype": "dt"})
    def test_reduction(self, test_manager: Orchestrator, platform_target: Platforms, K: int, M: int, N: int, dtype):
        """Reduction: const_matrix × data, models RMSNorm TKG."""
        np.random.seed(42)
        gen = gaussian_tensor_generator()
        H = 4096.0  # pretend hidden size for 1/H constant

        def input_generator(tc):
            const = np.full((K, M), 1.0 / H, dtype=dtype)
            return {
                "const_matrix": const,
                "data": gen(name="data", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_reduction,
            torch_ref=torch_ref_wrapper(torch_reduction),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)

    @pytest_parametrize("K, M, N, dtype", POST_MATMUL_PARAMS, abbrevs={"dtype": "dt"})
    def test_post_matmul(self, test_manager: Orchestrator, platform_target: Platforms, K: int, M: int, N: int, dtype):
        """Post-matmul callback: matmul + scale, models attention Q×K with masking."""
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "stationary_data": gen(name="stat", shape=(K, M), dtype=dtype),
                "moving_data": gen(name="mov", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_post_matmul,
            torch_ref=torch_ref_wrapper(torch_post_matmul),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── MN loop order kernel ─────────────────────────────────────────────


def kernel_mn_loop_order(source, weights, out):
    """Same as basic_gemm but with loop_order='MN'."""
    M, K = source.shape
    _, N = weights.shape
    M_TILE = 128

    stat_sb = alloc_tiled_sbuf(shape=(K, M), tile_size=(K, M_TILE), dtype=source.dtype)
    mov_sb = alloc_tiled_sbuf(shape=(K, N), tile_size=(K, N), dtype=weights.dtype)
    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M_TILE, N), dtype=out.dtype)

    NUM_M = M // M_TILE
    for m in range(NUM_M):
        nisa.dma_transpose(dst=stat_sb[0, m], src=source[m * M_TILE : (m + 1) * M_TILE, :])
    nisa.dma_copy(dst=mov_sb[0, 0], src=weights)

    matmul_loop_nest(stationary=stat_sb, moving=mov_sb, dst_sbuf=dst_sb, loop_order="MN")

    for m in range(NUM_M):
        nisa.dma_copy(dst=out[m * M_TILE : (m + 1) * M_TILE, :], src=dst_sb[m, 0])
    return [out]


def kernel_mn_k_accum(source, weights, out):
    """K accumulation with loop_order='MN'."""
    M, K_total = source.shape
    _, N = weights.shape
    K_TILE = 128
    NUM_K = K_total // K_TILE
    NUM_BUFS = min(NUM_K, 4)

    stat_sb = alloc_tiled_sbuf(shape=(K_total, M), tile_size=(K_TILE, M), dtype=source.dtype)
    for k in range(NUM_K):
        nisa.dma_transpose(dst=stat_sb[k, 0], src=source[:, k * K_TILE : (k + 1) * K_TILE])

    mov_sb = alloc_tiled_sbuf(
        shape=(K_total, N), tile_size=(K_TILE, N), dtype=weights.dtype, rotate_dim=0, num_rotation=NUM_BUFS
    )
    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M, N), dtype=out.dtype)

    matmul_loop_nest(
        stationary=stat_sb,
        moving=mov_sb,
        dst_sbuf=dst_sb,
        load_weights=_LoadWeights(mov_sb=mov_sb, weights=weights, K_TILE=K_TILE).run,
        loop_order="MN",
    )

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


# ── MN loop order tests ─────────────────────────────────────────────

MN_BASIC_PARAMS = [
    (128, 128, 512, nl.bfloat16),
    (256, 128, 512, nl.bfloat16),  # 2 M tiles
]

MN_K_ACCUM_PARAMS = [
    (128, 256, 512, nl.bfloat16),
    (128, 512, 512, nl.bfloat16),
]


class TestMatmulLoopNestMNOrder:
    @pytest_parametrize("M, K, N, dtype", MN_BASIC_PARAMS, abbrevs={"dtype": "dt"})
    def test_mn_basic(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_mn_loop_order,
            torch_ref=torch_ref_wrapper(torch_basic_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)

    @pytest_parametrize("M, K, N, dtype", MN_K_ACCUM_PARAMS, abbrevs={"dtype": "dt"})
    def test_mn_k_accumulation(
        self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype
    ):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_mn_k_accum,
            torch_ref=torch_ref_wrapper(torch_k_accumulation),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── Non-aligned tile size kernel ─────────────────────────────────────


def kernel_non_aligned_m(source, weights, out):
    """source[M,K] × weights[K,N] → out[M,N], M or N not a multiple of 128."""
    M, K = source.shape
    _, N = weights.shape
    M_TILE = 128
    N_TILE = 128
    NUM_M = (M + M_TILE - 1) // M_TILE
    NUM_N = (N + N_TILE - 1) // N_TILE

    stat_sb = alloc_tiled_sbuf(shape=(K, NUM_M * M_TILE), tile_size=(K, M_TILE), dtype=source.dtype)
    mov_sb = alloc_tiled_sbuf(shape=(K, NUM_N * N_TILE), tile_size=(K, N_TILE), dtype=weights.dtype)
    dst_sb = alloc_tiled_sbuf(shape=(NUM_M * M_TILE, NUM_N * N_TILE), tile_size=(M_TILE, N_TILE), dtype=out.dtype)

    for m in range(NUM_M):
        m_size = min(M_TILE, M - m * M_TILE)
        nisa.dma_transpose(dst=stat_sb[0, m][:K, :m_size], src=source[m * M_TILE : m * M_TILE + m_size, :])
    for n in range(NUM_N):
        n_size = min(N_TILE, N - n * N_TILE)
        nisa.dma_copy(dst=mov_sb[0, n][:K, :n_size], src=weights[:, n * N_TILE : n * N_TILE + n_size])

    matmul_loop_nest(stationary=stat_sb, moving=mov_sb, dst_sbuf=dst_sb)

    for m in range(NUM_M):
        m_size = min(M_TILE, M - m * M_TILE)
        for n in range(NUM_N):
            n_size = min(N_TILE, N - n * N_TILE)
            nisa.dma_copy(
                dst=out[m * M_TILE : m * M_TILE + m_size, n * N_TILE : n * N_TILE + n_size],
                src=dst_sb[m, n][:m_size, :n_size],
            )
    return [out]


NON_ALIGNED_PARAMS = [
    (66, 128, 128, nl.bfloat16),  # M < 128
    (200, 128, 256, nl.bfloat16),  # M = 128 + 72
    (128, 128, 66, nl.bfloat16),  # N < 128
    (128, 128, 200, nl.bfloat16),  # N = non-aligned, single M tile
    (200, 128, 66, nl.bfloat16),  # Both M and N non-aligned
]


class TestMatmulLoopNestNonAligned:
    @pytest_parametrize("M, K, N, dtype", NON_ALIGNED_PARAMS, abbrevs={"dtype": "dt"})
    def test_non_aligned_m(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_non_aligned_m,
            torch_ref=torch_ref_wrapper(torch_basic_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── PSUM auto-slicing regression test ────────────────────────────────
# The 3D double-row case (shape[-1] vs shape[1]) is covered by
# output_projection/test_output_proj_tkg.py qt-1 configs.
# This test verifies the fix doesn't break 2D operands with oversized PSUM.


def kernel_oversized_psum(source, weights, out):
    """GEMM where PSUM tile is larger than matmul output.

    Allocates PSUM at (128, 512) but matmul output is (M, N) where N < 512.
    Verifies PSUM auto-slicing with shape[-1].
    """
    M, K = source.shape
    _, N = weights.shape

    stat_sb = alloc_tiled_sbuf(shape=(K, M), tile_size=(K, M), dtype=source.dtype)
    nisa.dma_transpose(dst=stat_sb[0, 0], src=source)

    mov_sb = alloc_tiled_sbuf(shape=(K, N), tile_size=(K, N), dtype=weights.dtype)
    nisa.dma_copy(dst=mov_sb[0, 0], src=weights)

    # Oversized PSUM: 512 cols but matmul only produces N cols
    psum_big = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    psum_tiled = TiledTensor._make_tile_list([psum_big], (1, 1), psum_big.shape)

    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M, N), dtype=out.dtype)

    matmul_loop_nest(
        stationary=stat_sb, moving=mov_sb, dst_psum=psum_tiled, dst_sbuf=dst_sb, on_drain=_drain_copy_sliced
    )

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


OVERSIZED_PSUM_PARAMS = [
    (128, 128, 128, nl.bfloat16),  # N=128 < PSUM 512
    (128, 128, 256, nl.bfloat16),  # N=256 < PSUM 512
    (64, 128, 64, nl.bfloat16),  # Both M and N < PSUM dims
]


class TestMatmulLoopNestPsumSlicing:
    @pytest_parametrize("M, K, N, dtype", OVERSIZED_PSUM_PARAMS, abbrevs={"dtype": "dt"})
    def test_oversized_psum(
        self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype
    ):
        """Regression: PSUM auto-slicing uses shape[-1] for free dim."""
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_oversized_psum,
            torch_ref=torch_ref_wrapper(torch_basic_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── Over-allocated K dim test ────────────────────────────────────────


def kernel_overalloc_k(source, weights, out):
    """GEMM where moving buffer is over-allocated on K dim.

    source[M=128, K=96] × weights[K=96, N=128] → out[128, 128]
    But moving buffer allocated at [128, N] (max K_TILE), filled with 96 rows.
    Tests min(K_stat, K_mov) auto-clamping.
    """
    M, K = source.shape
    _, N = weights.shape
    K_TILE = 128  # over-allocated

    # Stationary: [K=96, M=128] — actual K size
    stat_sb = nl.ndarray((K, M), dtype=source.dtype, buffer=nl.sbuf)
    nisa.dma_transpose(dst=stat_sb, src=source)
    stat_tiled = TiledTensor._make_tile_list([stat_sb], (1, 1), stat_sb.shape)

    # Moving: over-allocated [128, N] but only [96, N] has valid data
    mov_sb = nl.ndarray((K_TILE, N), dtype=weights.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=mov_sb[:K, :], src=weights)
    mov_tiled = TiledTensor._make_tile_list([mov_sb], (1, 1), mov_sb.shape)

    dst_sb = alloc_tiled_sbuf(shape=(M, N), tile_size=(M, N), dtype=out.dtype)

    matmul_loop_nest(stationary=stat_tiled, moving=mov_tiled, dst_sbuf=dst_sb)

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


def torch_overalloc_k(source, weights, out):
    import torch

    out.copy_(torch.matmul(source, weights))
    return {"out": out}


OVERALLOC_K_PARAMS = [
    (128, 96, 128, nl.bfloat16),  # K=96 < K_TILE=128
    (128, 64, 256, nl.bfloat16),  # K=64 < K_TILE=128
]


class TestMatmulKAutoClamp:
    @pytest_parametrize("M, K, N, dtype", OVERALLOC_K_PARAMS, abbrevs={"dtype": "dt"})
    def test_overalloc_k(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        """GEMM with over-allocated moving K dim — tests min(K_stat, K_mov) clamping."""
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_overalloc_k,
            torch_ref=torch_ref_wrapper(torch_overalloc_k),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── TiledTensor.alloc() tests ────────────────────────────────────────


def kernel_alloc_basic(source, weights, out):
    """Basic GEMM using TiledTensor.alloc() for all buffers."""
    M, K = source.shape
    _, N = weights.shape

    stat_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(K, M), dtype=source.dtype, buffer=nl.sbuf)
    mov_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(K, N), dtype=weights.dtype, buffer=nl.sbuf)
    dst_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(M, N), dtype=out.dtype, buffer=nl.sbuf)

    nisa.dma_transpose(dst=stat_sb[0, 0], src=source)
    nisa.dma_copy(dst=mov_sb[0, 0], src=weights)

    matmul_loop_nest(stationary=stat_sb, moving=mov_sb, dst_sbuf=dst_sb)

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


@dataclass
class _AllocLoadWt(nl.NKIObject):
    mov_sb: object
    weights: object
    K_TILE: int

    def run(self, k, n, buf):
        nisa.dma_copy(dst=self.mov_sb[k, n], src=self.weights[k * self.K_TILE : (k + 1) * self.K_TILE, :])


def kernel_alloc_rotate(source, weights, out):
    """K accumulation using TiledTensor.alloc() with rotate for double-buffering."""
    M, K_total = source.shape
    _, N = weights.shape
    K_TILE = 128
    NUM_K = K_total // K_TILE

    stat_sb = TiledTensor.alloc(grid=(NUM_K, 1), tile_size=(K_TILE, M), dtype=source.dtype, buffer=nl.sbuf)
    for k in range(NUM_K):
        nisa.dma_transpose(dst=stat_sb[k, 0], src=source[:, k * K_TILE : (k + 1) * K_TILE])

    mov_sb = TiledTensor.alloc(
        grid=(NUM_K, 1), tile_size=(K_TILE, N), dtype=weights.dtype, buffer=nl.sbuf, rotate=(0, min(NUM_K, 2))
    )
    dst_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(M, N), dtype=out.dtype, buffer=nl.sbuf)

    matmul_loop_nest(
        stationary=stat_sb,
        moving=mov_sb,
        dst_sbuf=dst_sb,
        load_weights=_AllocLoadWt(mov_sb=mov_sb, weights=weights, K_TILE=K_TILE).run,
    )

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


def kernel_alloc_psum_banks(source, weights, out):
    """Multi-N GEMM using TiledTensor.alloc() with PSUM num_banks."""
    M, K = source.shape
    _, N = weights.shape
    N_TILE = min(N, 512)
    NUM_N = N // N_TILE

    stat_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(K, M), dtype=source.dtype, buffer=nl.sbuf)
    mov_sb = TiledTensor.alloc(grid=(1, NUM_N), tile_size=(K, N_TILE), dtype=weights.dtype, buffer=nl.sbuf)
    dst_sb = TiledTensor.alloc(grid=(1, NUM_N), tile_size=(M, N_TILE), dtype=out.dtype, buffer=nl.sbuf)
    psum = TiledTensor.alloc(
        grid=(1, NUM_N), tile_size=(M, N_TILE), dtype=nl.float32, buffer=nl.psum, num_banks=min(NUM_N, 4)
    )

    nisa.dma_transpose(dst=stat_sb[0, 0], src=source)
    for n in range(NUM_N):
        nisa.dma_copy(dst=mov_sb[0, n], src=weights[:, n * N_TILE : (n + 1) * N_TILE])

    matmul_loop_nest(stationary=stat_sb, moving=mov_sb, dst_psum=psum, dst_sbuf=dst_sb)

    for n in range(NUM_N):
        nisa.dma_copy(dst=out[:, n * N_TILE : (n + 1) * N_TILE], src=dst_sb[0, n])
    return [out]


ALLOC_BASIC_PARAMS = [
    (128, 128, 128, nl.bfloat16),
    (128, 128, 512, nl.bfloat16),
]

ALLOC_ROTATE_PARAMS = [
    (128, 256, 512, nl.bfloat16),
    (128, 512, 512, nl.bfloat16),
]

ALLOC_PSUM_PARAMS = [
    (128, 128, 512, nl.bfloat16),
    (128, 128, 1024, nl.bfloat16),
]


class TestTiledTensorAllocBasic:
    @pytest_parametrize("M, K, N, dtype", ALLOC_BASIC_PARAMS, abbrevs={"dtype": "dt"})
    def test_alloc_basic(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_alloc_basic,
            torch_ref=torch_ref_wrapper(torch_basic_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


class TestTiledTensorAllocRotate:
    @pytest_parametrize("M, K, N, dtype", ALLOC_ROTATE_PARAMS, abbrevs={"dtype": "dt"})
    def test_alloc_rotate(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_alloc_rotate,
            torch_ref=torch_ref_wrapper(torch_k_accumulation),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


class TestTiledTensorAllocPsumBanks:
    @pytest_parametrize("M, K, N, dtype", ALLOC_PSUM_PARAMS, abbrevs={"dtype": "dt"})
    def test_alloc_psum_banks(
        self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype
    ):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_alloc_psum_banks,
            torch_ref=torch_ref_wrapper(torch_basic_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── N-packing tests ─────────────────────────────────────────────────


def kernel_n_packing(source, weights, out):
    """GEMM with n_packing: multiple N tiles packed into one PSUM bank.

    source[M=128, K=128] × weights[K=128, N=512] → out[128, 512]
    N_TILE=128, n_packing=4 → 4 tiles packed per PSUM bank (128*4=512=F_MAX)
    """
    M, K = source.shape
    _, N = weights.shape
    N_TILE = 128
    NUM_N = N // N_TILE

    stat_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(K, M), dtype=source.dtype, buffer=nl.sbuf)
    mov_sb = TiledTensor.alloc(grid=(1, NUM_N), tile_size=(K, N_TILE), dtype=weights.dtype, buffer=nl.sbuf)
    dst_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(M, N), dtype=out.dtype, buffer=nl.sbuf)

    nisa.dma_transpose(dst=stat_sb[0, 0], src=source)
    for n in range(NUM_N):
        nisa.dma_copy(dst=mov_sb[0, n], src=weights[:, n * N_TILE : (n + 1) * N_TILE])

    # PSUM: one bank holds all 4 N tiles (128*4=512)
    psum = TiledTensor.alloc(grid=(1, 1), tile_size=(M, N), dtype=nl.float32, buffer=nl.psum)

    def drain(psum_tile, sbuf_tile, m, n):
        nisa.tensor_copy(dst=dst_sb[m, 0], src=psum_tile)

    matmul_loop_nest(
        stationary=stat_sb,
        moving=mov_sb,
        dst_psum=psum,
        dst_sbuf=dst_sb,
        n_packing=NUM_N,
        on_drain=drain,
    )

    nisa.dma_copy(dst=out, src=dst_sb[0, 0])
    return [out]


N_PACKING_PARAMS = [
    (128, 128, 512, nl.bfloat16),  # 4 tiles of 128 packed into 512
    (128, 128, 256, nl.bfloat16),  # 2 tiles of 128 packed into 256
]


class TestMatmulNPacking:
    @pytest_parametrize("M, K, N, dtype", N_PACKING_PARAMS, abbrevs={"dtype": "dt"})
    def test_n_packing(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_n_packing,
            torch_ref=torch_ref_wrapper(torch_basic_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── v2 n_packing + on_output test ────────────────────────────────────


def kernel_n_packing_v2(source, weights, scale, out):
    """GEMM with n_packing + v2 on_output: psum * scale → out.

    source[M=128, K=128] × weights[K=128, N=512] → out[128, 512]
    scale[M=128, N_tiles=4] broadcast per tile.
    n_packing=4, on_output fires once per bank with full packed tile.
    """
    M, K = source.shape
    _, N = weights.shape
    N_TILE = 128
    NUM_N = N // N_TILE

    stat_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(K, M), dtype=source.dtype, buffer=nl.sbuf)
    mov_sb = TiledTensor.alloc(grid=(1, NUM_N), tile_size=(K, N_TILE), dtype=weights.dtype, buffer=nl.sbuf)
    # Output and scale tiled at packed granularity: one tile = full N
    out_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(M, N), dtype=out.dtype, buffer=nl.sbuf)
    scale_sb = TiledTensor.alloc(grid=(1, 1), tile_size=(M, NUM_N), dtype=nl.float32, buffer=nl.sbuf)

    nisa.dma_transpose(dst=stat_sb[0, 0], src=source)
    for n in range(NUM_N):
        nisa.dma_copy(dst=mov_sb[0, n], src=weights[:, n * N_TILE : (n + 1) * N_TILE])
    nisa.dma_copy(dst=scale_sb[0, 0], src=scale)

    def on_output_scale(psum_tile, out_tile, scale_tile):
        scale_bc = scale_tile.expand_dim(2).broadcast(dim=2, size=N_TILE)
        nisa.tensor_tensor(dst=out_tile, data1=psum_tile, data2=scale_bc, op=nl.multiply)

    matmul_loop_nest(
        stationary=stat_sb,
        moving=mov_sb,
        output=out_sb,
        auxiliaries=[scale_sb],
        on_output=on_output_scale,
        n_packing=NUM_N,
    )

    nisa.dma_copy(dst=out, src=out_sb[0, 0])
    return [out]


def torch_scaled_gemm(source, weights, scale, out):
    import torch

    result = torch.matmul(source.float(), weights.float())
    N_TILE = 128
    scale_expanded = scale.float().repeat_interleave(N_TILE, dim=1)
    out.copy_((result * scale_expanded).to(source.dtype))
    return {"out": out}


N_PACKING_V2_PARAMS = [
    (128, 128, 512, nl.bfloat16),  # 4 tiles of 128
    (128, 128, 256, nl.bfloat16),  # 2 tiles of 128
]


class TestMatmulNPackingV2:
    @pytest_parametrize("M, K, N, dtype", N_PACKING_V2_PARAMS, abbrevs={"dtype": "dt"})
    def test_n_packing_v2(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()
        NUM_N = N // 128

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, 128), dtype=dtype),
                "weights": gen(name="weights", shape=(128, N), dtype=dtype),
                "scale": np.random.uniform(0.5, 2.0, size=(M, NUM_N)).astype(np.float32),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_n_packing_v2,
            torch_ref=torch_ref_wrapper(torch_scaled_gemm),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── K-outermost (KMN) test ───────────────────────────────────────────


def kernel_kmn_gemm(source, weights, out):
    """GEMM with K outermost: source[M,K] × weights[K,N] → out[M,N], K > 128."""
    M, K = source.shape
    _, N = weights.shape
    K_TILE = 128
    M_TILE = 128
    NUM_K = K // K_TILE
    NUM_M = M // M_TILE

    stat_sb = TiledTensor.alloc(grid=(NUM_K, NUM_M), tile_size=(K_TILE, M_TILE), dtype=source.dtype, buffer=nl.sbuf)
    mov_sb = TiledTensor.alloc(grid=(NUM_K, 1), tile_size=(K_TILE, N), dtype=weights.dtype, buffer=nl.sbuf)
    out_sb = TiledTensor.alloc(grid=(NUM_M, 1), tile_size=(M_TILE, N), dtype=out.dtype, buffer=nl.sbuf)
    psum = TiledTensor.alloc(grid=(NUM_M, 1), tile_size=(M_TILE, N), dtype=nl.float32, buffer=nl.psum, num_banks=NUM_M)

    # Load all source tiles
    for k in range(NUM_K):
        for m in range(NUM_M):
            nisa.dma_transpose(
                dst=stat_sb[k, m], src=source[m * M_TILE : (m + 1) * M_TILE, k * K_TILE : (k + 1) * K_TILE]
            )
    # Load all weight tiles
    for k in range(NUM_K):
        nisa.dma_copy(dst=mov_sb[k, 0], src=weights[k * K_TILE : (k + 1) * K_TILE, :])

    def on_output_copy(psum_tile, out_tile):
        nisa.tensor_copy(dst=out_tile, src=psum_tile)

    matmul_loop_nest(
        stationary=stat_sb,
        moving=mov_sb,
        dst_psum=psum,
        output=out_sb,
        on_output=on_output_copy,
        loop_order="KMN",
    )

    for m in range(NUM_M):
        nisa.dma_copy(dst=out[m * M_TILE : (m + 1) * M_TILE, :], src=out_sb[m, 0])
    return [out]


KMN_PARAMS = [
    (256, 256, 128, nl.bfloat16),  # 2 K tiles, 2 M tiles
    (128, 256, 128, nl.bfloat16),  # 2 K tiles, 1 M tile
]


class TestMatmulKMN:
    @pytest_parametrize("M, K, N, dtype", KMN_PARAMS, abbrevs={"dtype": "dt"})
    def test_kmn_gemm(self, test_manager: Orchestrator, platform_target: Platforms, M: int, K: int, N: int, dtype):
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(M, K), dtype=dtype),
                "weights": gen(name="weights", shape=(K, N), dtype=dtype),
                "out.must_alias_input": np.zeros((M, N), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        def torch_ref(source, weights, out):
            import torch

            out.copy_(torch.matmul(source.float(), weights.float()).to(source.dtype))
            return {"out": out}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_kmn_gemm,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)


# ── Bug regression tests ─────────────────────────────────────────────


def kernel_psum_bank_reuse_m_tiles(source, weights, out):
    """Regression: PSUM banks must not be reused across M tiles.

    With M_tiles=2, N_tiles=2, num_banks must be 4 (M*N), not 2.
    If banks are reused, the second M tile accumulates on stale data from the first.
    """
    K = 128
    M = 256
    N = 256
    P = 128

    # Stationary: (K, M) = (128, 256), tile (128, 128) -> grid (1, 2) -> K=1, M=2
    stat_sb = nl.ndarray((K, M), dtype=source.dtype, buffer=nl.sbuf)
    for m in range(2):
        nisa.dma_transpose(dst=stat_sb[:, nl.ds(m * P, P)], src=source[nl.ds(m * P, P), :K])

    # Moving: (K, N) = (128, 256), tile (128, 128) -> grid (1, 2) -> K=1, N=2
    mov_sb = nl.ndarray((K, N), dtype=weights.dtype, buffer=nl.sbuf)
    mov_sb[:, :] = nl.load(weights[:K, :N])

    stat_tiled = TiledTensor(stat_sb, tile_size=(K, P))
    mov_tiled = TiledTensor(mov_sb, tile_size=(K, P))
    psum_tiled = TiledTensor.alloc(grid=(2, 2), tile_size=(P, P), dtype=nl.float32, buffer=nl.psum, num_banks=4)
    out_tiled = TiledTensor.alloc(grid=(2, 2), tile_size=(P, P), dtype=source.dtype, buffer=nl.sbuf)

    matmul_loop_nest(
        stationary=stat_tiled,
        moving=mov_tiled,
        dst_psum=psum_tiled,
        output=out_tiled,
        on_output=lambda psum_tile, out_tile: nisa.tensor_copy(dst=out_tile, src=psum_tile),
    )

    for m in range(2):
        for n in range(2):
            nl.store(out[nl.ds(m * P, P), nl.ds(n * P, P)], value=out_tiled[m, n])
    return [out]


def kernel_degenerate_partition_dim(source, weights, out):
    """Regression: n_packing with partition dim = 1.

    When d_size=1 (partition=1), _DimsAccessor.tile_size must correctly report
    the free dim size for N_tile_sz_for_packing. Previously tile_size was (12, 1)
    instead of (1, 12), causing overlapping PSUM writes.
    """
    # Simulates d_size=1, n_heads=4, h1=128, h2=8, bxs=32
    d_size = 1
    n_heads = 4
    h1 = 128
    bxs = 32
    h2 = 8
    F_MAX = 512
    NUM_BS_PER_PSUM_BANK = F_MAX // bxs  # 16

    # Weight: (d_size=1, n_heads=4, h1*h2) in SBUF
    w_sb = nl.ndarray((d_size, n_heads, h1 * h2), dtype=weights.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(n_heads):
        w_sb[0:d_size, i, 0 : h1 * h2] = nl.load(weights[0:d_size, nl.ds(i * h1 * h2, h1 * h2)])

    # Attn: (d_size=1, n_heads*bxs) in SBUF
    attn_sb = nl.ndarray((d_size, n_heads * bxs), dtype=source.dtype, buffer=nl.sbuf)
    attn_sb[0:d_size, 0 : n_heads * bxs] = nl.load(source[0:d_size, 0 : n_heads * bxs])

    # Tile weight: reshape to (1, 4, 128, 8), tile_size=(1, 1, 128, 1)
    w_view = w_sb.reshape_dim(2, (h1, h2))
    w_tiled = TiledTensor(w_view, tile_size=(d_size, 1, h1, 1))
    # Grid: (1, 4, 1, 8). squeeze_dim(0).squeeze_dim(1) -> (4, 8)
    stat_tiled = w_tiled.squeeze_dim(0).squeeze_dim(1)

    # Tile attn: reshape to (1, 4, 32), tile_size=(1, 1, 32)
    attn_view = attn_sb.reshape_dim(1, (n_heads, bxs))
    attn_tiled = TiledTensor(attn_view, tile_size=(d_size, 1, bxs))
    mov_tiled = attn_tiled.squeeze_dim(0)

    # PSUM and output
    num_psum_groups = (h2 + NUM_BS_PER_PSUM_BANK - 1) // NUM_BS_PER_PSUM_BANK  # ceil(8/16) = 1
    packed_bxs = min(NUM_BS_PER_PSUM_BANK, h2) * bxs  # min(16, 8)*32 = 256
    psum_tiled = TiledTensor.alloc(
        grid=(1, num_psum_groups), tile_size=(h1, F_MAX), dtype=nl.float32, buffer=nl.psum, num_banks=num_psum_groups
    )

    out_sb = nl.ndarray((h1, h2 * bxs), dtype=source.dtype, buffer=nl.sbuf)
    out_tiled = TiledTensor(out_sb, tile_size=(h1, packed_bxs))

    matmul_loop_nest(
        stationary=stat_tiled,
        moving=mov_tiled,
        dst_psum=psum_tiled,
        stationary_dims={"K": 0, "N": 1},
        moving_dims={"K": 0},
        n_packing=NUM_BS_PER_PSUM_BANK,
        output=out_tiled,
        on_output=lambda psum_tile, out_tile: nisa.tensor_copy(dst=out_tile, src=psum_tile),
    )

    out_reshaped = out_sb.reshape((h1, h2, bxs))
    nl.store(out[0:h1, 0:h2, 0:bxs], value=out_reshaped)
    return [out]


def kernel_n_packing_partial_group(source, weights, out):
    """Regression: N_tiles must not be padded to n_packing multiples.

    When N_total < n_packing (e.g., h2=8, n_packing=16), the loop must skip
    indices >= N_total. Previously N_tiles was set to n_packing (padded),
    causing out-of-bounds stationary access.
    """
    K = 128
    h1 = 128
    h2 = 8  # < n_packing
    bxs = 32
    F_MAX = 512
    NUM_BS_PER_PSUM_BANK = F_MAX // bxs  # 16 > h2=8

    # Weight: (K, h1*h2) in SBUF
    w_sb = nl.ndarray((K, h1 * h2), dtype=weights.dtype, buffer=nl.sbuf)
    w_sb[:, :] = nl.load(weights[0:K, 0 : h1 * h2])

    # Source: (K, bxs) in SBUF
    src_sb = nl.ndarray((K, bxs), dtype=source.dtype, buffer=nl.sbuf)
    src_sb[:, :] = nl.load(source[0:K, 0:bxs])

    # Tile weight: reshape to (K, h1, h2), tile_size=(K, h1, 1)
    w_view = w_sb.reshape_dim(1, (h1, h2))
    w_tiled = TiledTensor(w_view, tile_size=(K, h1, 1))
    # Grid: (1, 1, h2). squeeze_dim(0) -> (1, h2)
    stat_tiled = w_tiled.squeeze_dim(0)

    # Tile source: (K, bxs), tile_size=(K, bxs). Grid: (1, 1)
    mov_tiled = TiledTensor(src_sb, tile_size=(K, bxs))

    # PSUM: 1 group (h2=8 < n_packing=16)
    psum_tiled = TiledTensor.alloc(grid=(1, 1), tile_size=(h1, F_MAX), dtype=nl.float32, buffer=nl.psum, num_banks=1)

    packed_bxs = min(NUM_BS_PER_PSUM_BANK, h2) * bxs  # min(16, 8)*32 = 256
    out_sb = nl.ndarray((h1, h2 * bxs), dtype=source.dtype, buffer=nl.sbuf)
    out_tiled = TiledTensor(out_sb, tile_size=(h1, packed_bxs))

    matmul_loop_nest(
        stationary=stat_tiled,
        moving=mov_tiled,
        dst_psum=psum_tiled,
        stationary_dims={"K": 0, "N": 1},
        moving_dims={"K": 0},
        n_packing=NUM_BS_PER_PSUM_BANK,
        output=out_tiled,
        on_output=lambda psum_tile, out_tile: nisa.tensor_copy(dst=out_tile, src=psum_tile),
    )

    out_reshaped = out_sb.reshape((h1, h2, bxs))
    nl.store(out[0:h1, 0:h2, 0:bxs], value=out_reshaped)
    return [out]


# ── Torch refs for bug regression tests ──────────────────────────────


def torch_psum_bank_reuse_m_tiles(source, weights, out):
    import torch

    # stat = source.T (transposed to K=128, M=256), mov = weights (K=128, N=256)
    # nc_matmul: stat.T @ mov = source @ weights
    out.copy_(torch.matmul(source[:256, :128].float(), weights[:128, :256].float()).to(source.dtype))
    return {"out": out}


def torch_degenerate_partition_dim(source, weights, out):
    """d_size=1, n_heads=4, h1=128, h2=8, bxs=32."""
    import torch

    d_size, n_heads, h1, h2, bxs = 1, 4, 128, 8, 32
    # weights: (1, n_heads*h1*h2) -> (1, 4, 128, 8)
    w = weights[:d_size, : n_heads * h1 * h2].float().reshape(d_size, n_heads, h1, h2)
    # source: (1, n_heads*bxs) -> (1, 4, 32)
    attn = source[:d_size, : n_heads * bxs].float().reshape(d_size, n_heads, bxs)
    # For each h2: sum over heads of stat.T @ mov
    # stat[head, :, h2_idx] is (1, 128), mov[head, :] is (1, 32)
    # stat.T @ mov = (128, 1) @ (1, 32) = (128, 32)
    result = torch.zeros(h1, h2, bxs)
    for n in range(h2):
        for head in range(n_heads):
            stat = w[0, head, :, n].reshape(1, h1)  # (1, 128)
            mov = attn[0, head, :].reshape(1, bxs)  # (1, 32)
            result[:, n, :] += stat.T @ mov  # (128, 32)
    out.copy_(result.to(source.dtype))
    return {"out": out}


def torch_n_packing_partial_group(source, weights, out):
    """K=128, h1=128, h2=8, bxs=32."""
    import torch

    K, h1, h2, bxs = 128, 128, 8, 32
    w = weights[:K, : h1 * h2].float().reshape(K, h1, h2)
    src = source[:K, :bxs].float()
    # For each h2: stat[:, :, h2_idx].T @ src = (h1, K) @ (K, bxs) = (h1, bxs)
    result = torch.zeros(h1, h2, bxs)
    for n in range(h2):
        stat = w[:, :, n]  # (K, h1)
        result[:, n, :] = stat.T @ src  # (h1, bxs)
    out.copy_(result.to(source.dtype))
    return {"out": out}


# ── Test class for bug regressions ───────────────────────────────────

REGRESSION_PARAMS = [
    nl.bfloat16,
]


class TestMatmulLoopNestRegressions:
    @pytest_parametrize("dtype", REGRESSION_PARAMS, abbrevs={"dtype": "dt"})
    def test_psum_bank_reuse_m_tiles(self, test_manager: Orchestrator, platform_target: Platforms, dtype):
        """PSUM banks reused across M tiles cause stale accumulation."""
        np.random.seed(42)
        gen = gaussian_tensor_generator()

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(256, 128), dtype=dtype),
                "weights": gen(name="weights", shape=(128, 256), dtype=dtype),
                "out.must_alias_input": np.zeros((256, 256), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_psum_bank_reuse_m_tiles,
            torch_ref=torch_ref_wrapper(torch_psum_bank_reuse_m_tiles),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)

    @pytest_parametrize("dtype", REGRESSION_PARAMS, abbrevs={"dtype": "dt"})
    def test_degenerate_partition_dim(self, test_manager: Orchestrator, platform_target: Platforms, dtype):
        """Partition dim = 1 with n_packing caused overlapping PSUM writes."""
        np.random.seed(42)
        gen = gaussian_tensor_generator()
        d_size, n_heads, h1, h2, bxs = 1, 4, 128, 8, 32

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(d_size, n_heads * bxs), dtype=dtype),
                "weights": gen(name="weights", shape=(d_size, n_heads * h1 * h2), dtype=dtype),
                "out.must_alias_input": np.zeros((h1, h2, bxs), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_degenerate_partition_dim,
            torch_ref=torch_ref_wrapper(torch_degenerate_partition_dim),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)

    @pytest_parametrize("dtype", REGRESSION_PARAMS, abbrevs={"dtype": "dt"})
    def test_n_packing_partial_group(self, test_manager: Orchestrator, platform_target: Platforms, dtype):
        """N_tiles padded to n_packing caused out-of-bounds access."""
        np.random.seed(42)
        gen = gaussian_tensor_generator()
        K, h1, h2, bxs = 128, 128, 8, 32

        def input_generator(tc):
            return {
                "source": gen(name="source", shape=(K, bxs), dtype=dtype),
                "weights": gen(name="weights", shape=(K, h1 * h2), dtype=dtype),
                "out.must_alias_input": np.zeros((h1, h2, bxs), dtype=dtype),
            }

        def output_tensors(ki):
            return {"out": ki["out.must_alias_input"].copy()}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_n_packing_partial_group,
            torch_ref=torch_ref_wrapper(torch_n_packing_partial_group),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=5e-2, atol=5e-2)
