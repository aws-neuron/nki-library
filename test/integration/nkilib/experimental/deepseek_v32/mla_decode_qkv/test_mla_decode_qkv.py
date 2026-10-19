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

"""Integration tests for the DeepSeek-V3.2 MLA decode QKV front-fold kernel."""

from typing import Any

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.deepseek_v32.mla_decode_qkv import (
    P_MAX,
    get_mla_decode_qkv_shape,
    mla_decode_qkv,
    mla_decode_qkv_torch_ref,
)

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# kv_lora_rank must equal F_MAX so the PSUM n-tile boundary falls on the cache row's field
# boundary, which is what lets the latent and k_pe fields be evicted independently.
_KV_LORA_RANK = 512

PARAM_NAMES = "n_tokens, hidden, q_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, block_size, n_blocks, dtype"
TEST_PARAMS = [
    pytest.param(
        1, 512, 256, 1, 128, 64, 16, 8, ml_dtypes.bfloat16, id="ds32_mla_qkv_t1_h512_nh1_bf16", marks=pytest.mark.fast
    ),
    pytest.param(
        4, 512, 256, 2, 128, 64, 16, 8, ml_dtypes.bfloat16, id="ds32_mla_qkv_t4_h512_nh2_bf16", marks=pytest.mark.fast
    ),
    pytest.param(
        2, 1024, 512, 4, 128, 64, 32, 16, np.float32, id="ds32_mla_qkv_t2_h1024_nh4_f32", marks=pytest.mark.fast
    ),
    pytest.param(8, 1024, 512, 8, 64, 64, 32, 16, ml_dtypes.bfloat16, id="ds32_mla_qkv_t8_h1024_nh8_bf16"),
    pytest.param(16, 7168, 1536, 1, 128, 64, 32, 1547, ml_dtypes.bfloat16, id="DS32_MLA_DECODE_QKV_PROD_CONFIG_B16"),
]


@nki.jit
def mla_decode_qkv_evicted(
    hidden_states: nl.NkiTensor,
    wq_a: nl.NkiTensor,
    q_norm_w: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    wkv_a: nl.NkiTensor,
    kv_norm_w: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    slot_mapping: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    norm_eps: float = 1e-6,
    input_norm_w: nl.NkiTensor | None = None,
):
    """
    Runs ``mla_decode_qkv`` and evicts its SBUF-resident query tiles to HBM so they can be validated.

    The kernel hands the two query tiles over in SBUF on purpose, since it is meant to be traced
    inline into an attention core. A standalone test therefore has to do the eviction the core would
    otherwise skip, which is what this wrapper exists for. It also reads the scattered cache rows back
    out at ``slot_mapping``, so that a scatter landing on the wrong slot fails the comparison rather
    than passing unnoticed.
    """

    result = mla_decode_qkv(
        hidden_states,
        wq_a,
        q_norm_w,
        wq_b,
        wkv_a,
        kv_norm_w,
        cos,
        sin,
        q_absorb_w,
        kv_cache,
        slot_mapping,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        norm_eps,
        input_norm_w,
    )
    shapes = get_mla_decode_qkv_shape(
        hidden_states, wq_a, wq_b, wkv_a, kv_cache, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank
    )
    n_tokens, n_kv_tiles = shapes.n_tokens, kv_lora_rank // P_MAX
    kv_row_dim, dtype = shapes.kv_row_dim, shapes.dtype

    # q_absorbed lives as [P_MAX, n_kv_tiles, n_tokens, n_heads]; flatten the tile axis onto the
    # feature axis so the result is directly comparable to [kv_lora_rank, n_tokens, n_heads].
    q_absorbed_hbm = nl.ndarray((kv_lora_rank, n_tokens, n_heads), dtype=dtype, buffer=nl.shared_hbm)
    for kv_tile in range(n_kv_tiles):
        nisa.dma_copy(
            dst=q_absorbed_hbm[kv_tile * P_MAX : (kv_tile + 1) * P_MAX, 0:n_tokens, 0:n_heads],
            src=result.q_absorbed_sbuf[0:P_MAX, kv_tile, 0:n_tokens, 0:n_heads],
        )

    q_pe_hbm = nl.ndarray((qk_rope_head_dim, n_tokens, n_heads), dtype=dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=q_pe_hbm[0:qk_rope_head_dim, 0:n_tokens, 0:n_heads],
        src=result.q_pe_sbuf[0:qk_rope_head_dim, 0:n_tokens, 0:n_heads],
    )

    # Read back the rows the kernel scattered, so the destination slot is covered too.
    slot_sbuf = nl.ndarray((n_tokens, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slot_sbuf[0:n_tokens, 0:1], src=slot_mapping[0:n_tokens].reshape((n_tokens, 1)))

    scattered_sbuf = nl.ndarray((P_MAX, kv_row_dim), dtype=dtype, buffer=nl.sbuf)
    kv_cache_rows = kv_cache.reshape((shapes.n_cache_rows, kv_row_dim))
    nisa.dma_copy(
        dst=scattered_sbuf[0:n_tokens, 0:kv_row_dim],
        src=kv_cache_rows.ap(
            pattern=[[kv_row_dim, n_tokens], [1, kv_row_dim]],
            offset=0,
            vector_offset=slot_sbuf.ap(pattern=[[1, n_tokens], [1, 1]], offset=0),
            indirect_dim=0,
        ),
    )
    scattered_hbm = nl.ndarray((n_tokens, kv_row_dim), dtype=dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=scattered_hbm[0:n_tokens, 0:kv_row_dim], src=scattered_sbuf[0:n_tokens, 0:kv_row_dim])

    return q_absorbed_hbm, q_pe_hbm, result.kv_row_cur_hbm, scattered_hbm


class TestMlaDecodeQkv:
    """Test executor for DeepSeek-V3.2 MLA decode QKV tests."""

    @pytest.mark.parametrize("lnc", [1])
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_mla_decode_qkv(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_tokens: int,
        hidden: int,
        q_lora_rank: int,
        n_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        block_size: int,
        n_blocks: int,
        dtype: type,
        lnc: int,
    ):
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return self._generate_inputs(
                n_tokens, hidden, q_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, block_size, n_blocks, dtype
            )

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            shapes = get_mla_decode_qkv_shape(
                kernel_input["hidden_states"],
                kernel_input["wq_a"],
                kernel_input["wq_b"],
                kernel_input["wkv_a"],
                kernel_input["kv_cache"],
                kernel_input["n_heads"],
                kernel_input["qk_nope_head_dim"],
                kernel_input["qk_rope_head_dim"],
                kernel_input["kv_lora_rank"],
            )
            return {
                "output_0": np.zeros((_KV_LORA_RANK, shapes.n_tokens, shapes.n_heads), dtype=dtype),
                "output_1": np.zeros((shapes.qk_rope_dim, shapes.n_tokens, shapes.n_heads), dtype=dtype),
                "output_2": np.zeros((shapes.n_tokens, shapes.kv_row_dim), dtype=dtype),
                "output_3": np.zeros((shapes.n_tokens, shapes.kv_row_dim), dtype=dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            torch_ref=torch_ref_wrapper(mla_decode_qkv_torch_ref),
            kernel_entry=mla_decode_qkv_evicted,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            atol=1e-2,
            rtol=3e-2 if is_bf16 else 1e-3,
        )

    @staticmethod
    def _generate_inputs(
        n_tokens: int,
        hidden: int,
        q_lora_rank: int,
        n_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        block_size: int,
        n_blocks: int,
        dtype: type,
    ) -> dict[str, Any]:
        """Wrapper around input generator for TestMlaDecodeQkv."""

        np.random.seed(42)
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_row_dim = qk_rope_head_dim + _KV_LORA_RANK

        def random_tensor(
            name: str, *shape, tensor_dtype: type = dtype, mean: float = 0.0, std: float = 0.2
        ) -> npt.NDArray[Any]:
            """Generates a random tensor for the provided name, shape, dtype and optionally mean and std."""
            generator = gaussian_tensor_generator(mean, std)
            return generator(shape, tensor_dtype, name)

        # Distinct slots in distinct blocks, so a scatter that lands on the wrong row is visible.
        slots = (np.arange(n_tokens, dtype=np.int32) * block_size + 1) % (n_blocks * block_size)

        return {
            "hidden_states": random_tensor("hidden_states", n_tokens, hidden),
            "wq_a": random_tensor("wq_a", hidden, q_lora_rank),
            "q_norm_w": random_tensor("q_norm_w", 1, q_lora_rank, tensor_dtype=np.float32, mean=1.0, std=0.1),
            "wq_b": random_tensor("wq_b", q_lora_rank, n_heads * qk_head_dim),
            "wkv_a": random_tensor("wkv_a", hidden, kv_row_dim),
            "kv_norm_w": random_tensor("kv_norm_w", 1, _KV_LORA_RANK, tensor_dtype=np.float32, mean=1.0, std=0.1),
            "cos": random_tensor("cos", n_tokens, qk_rope_head_dim // 2, tensor_dtype=np.float32),
            "sin": random_tensor("sin", n_tokens, qk_rope_head_dim // 2, tensor_dtype=np.float32),
            "q_absorb_w": random_tensor("q_absorb_w", n_heads, qk_nope_head_dim, _KV_LORA_RANK),
            "kv_cache": np.zeros((n_blocks, 1, block_size, kv_row_dim), dtype=dtype),
            "slot_mapping": slots,
            "n_heads": n_heads,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "kv_lora_rank": _KV_LORA_RANK,
            "norm_eps": 1e-6,
            "input_norm_w": random_tensor("input_norm_w", 1, hidden, tensor_dtype=np.float32, mean=1.0, std=0.1),
        }
