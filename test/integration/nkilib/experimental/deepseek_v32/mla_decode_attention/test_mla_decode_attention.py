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

"""Integration tests for the DeepSeek-V3.2 MLA decode attention core."""

from typing import Any

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.experimental.deepseek_v32.mla_decode_attention import (
    P_MAX,
    build_prior_index,
    get_mla_decode_attention_shape,
    mla_decode_attention,
    mla_decode_attention_torch_ref,
)

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# kv_lora_rank must be a multiple of P_MAX and fit one PSUM bank, and qk_rope_dim is whatever is left
# of a cache row. Both are model constants rather than free test parameters.
_KV_LORA_RANK = 512
_QK_ROPE_DIM = 64

# The model scales scores by the full per-head QK width, nope plus rope.
_SOFTMAX_SCALE = float((128 + _QK_ROPE_DIM) ** -0.5)

PARAM_NAMES = "n_tokens, n_heads, n_prior, v_head_dim, hidden, block_size, max_blocks_per_seq, dtype"
TEST_PARAMS = [
    pytest.param(
        1, 1, 128, 128, 512, 16, 16, ml_dtypes.bfloat16, id="ds32_mla_attn_t1_nh1_k128_bf16", marks=pytest.mark.fast
    ),
    pytest.param(
        4, 2, 256, 128, 512, 16, 32, ml_dtypes.bfloat16, id="ds32_mla_attn_t4_nh2_k256_bf16", marks=pytest.mark.fast
    ),
    # n_prior is ragged, so the current token's key shares the last tile with prior keys and the
    # causal predicate has to keep it despite the tile's unselected tail.
    pytest.param(
        2, 1, 96, 128, 256, 16, 16, ml_dtypes.bfloat16, id="ds32_mla_attn_t2_k96_ragged_bf16", marks=pytest.mark.fast
    ),
    # fp32 has no indirect dma_transpose, so every latent chunk transposes on the PE.
    pytest.param(2, 4, 128, 128, 512, 32, 16, np.float32, id="ds32_mla_attn_t2_nh4_k128_f32", marks=pytest.mark.fast),
    # Six tokens is one full softmax group plus a short one.
    pytest.param(6, 1, 256, 128, 512, 16, 32, ml_dtypes.bfloat16, id="ds32_mla_attn_t6_nh1_k256_bf16"),
    pytest.param(16, 1, 2048, 128, 7168, 32, 256, ml_dtypes.bfloat16, id="DS32_MLA_DECODE_ATTENTION_PROD_CONFIG_B16"),
]


@nki.jit
def mla_decode_attention_staged(
    q_absorbed: nl.NkiTensor,
    q_pe: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    kv_row_cur: nl.NkiTensor,
    gather_indices: nl.NkiTensor,
    block_table: nl.NkiTensor,
    positions: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    softmax_scale: float,
) -> nl.NkiTensor:
    """
    Stages the query tiles into SBUF and builds the prior index, then runs ``mla_decode_attention``.

    The kernel takes its query tiles in SBUF on purpose, since it is meant to be traced inline after
    the QKV front-fold, which already leaves them there feature-major. A standalone test therefore has
    to stage them itself, which is the mirror of the eviction the front-fold's own test does. Building
    the index here also puts ``build_prior_index`` under test, and in the order a real caller uses it.
    """

    shapes = get_mla_decode_attention_shape(kv_cache, positions, out_absorb_w, o_proj_w, gather_indices.shape[1])
    n_tokens, n_heads, dtype = shapes.n_tokens, shapes.n_heads, shapes.dtype

    # [kv_lora_rank, n_tokens, n_heads] -> [P_MAX, n_latent_tiles, n_tokens, n_heads]
    q_absorbed_sbuf = nl.ndarray((P_MAX, shapes.n_latent_tiles, n_tokens, n_heads), dtype=dtype, buffer=nl.sbuf)
    for latent_tile in range(shapes.n_latent_tiles):
        nisa.dma_copy(
            dst=q_absorbed_sbuf[0:P_MAX, latent_tile, 0:n_tokens, 0:n_heads],
            src=q_absorbed[latent_tile * P_MAX : (latent_tile + 1) * P_MAX, 0:n_tokens, 0:n_heads],
        )

    q_pe_sbuf = nl.ndarray((P_MAX, n_tokens, n_heads), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=q_pe_sbuf[0 : shapes.qk_rope_dim, 0:n_tokens, 0:n_heads],
        src=q_pe[0 : shapes.qk_rope_dim, 0:n_tokens, 0:n_heads],
    )

    prior = build_prior_index(gather_indices, block_table, kv_cache)

    return mla_decode_attention(
        q_absorbed_sbuf,
        q_pe_sbuf,
        kv_cache,
        kv_row_cur,
        prior.rows,
        prior.key_pos,
        prior.rows_natural,
        positions,
        out_absorb_w,
        o_proj_w,
        gather_indices.shape[1],
        softmax_scale,
    )


def mla_decode_attention_staged_torch_ref(
    q_absorbed: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_row_cur: torch.Tensor,
    gather_indices: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    out_absorb_w: torch.Tensor,
    o_proj_w: torch.Tensor,
    softmax_scale: float,
) -> dict[str, torch.Tensor]:
    """
    Oracle for ``mla_decode_attention_staged``, resolving the prior index itself before delegating.

    The kernel's own reference takes the index tables already resolved, because ``build_prior_index``
    resolves them on device. Resolving them here independently instead of reusing that device output is
    what puts the index build under test: a wrong block lookup, a wrong intra-block offset or a wrong
    key-tile permutation makes the two disagree.
    """

    n_tokens, n_prior = gather_indices.shape
    n_prior_tiles = (n_prior + P_MAX - 1) // P_MAX
    block_size, n_cache_rows = kv_cache.shape[2], kv_cache.shape[0] * kv_cache.shape[2]
    max_blocks_per_seq = block_table.shape[1]

    # cache_row = block_table[seq, pos // block_size] * block_size + pos % block_size, clamped into
    # range the same way the kernel clamps it, so an unmapped block resolves rather than being
    # undefined. Such keys are always causally masked, so which row they land on does not matter.
    selected = gather_indices.to(torch.int64)
    logical_block = torch.clamp(selected // block_size, 0, max_blocks_per_seq - 1)
    physical_block = torch.gather(block_table.to(torch.int64), 1, logical_block)
    rows = torch.clamp(physical_block * block_size + selected % block_size, 0, n_cache_rows - 1)

    prior_rows = _to_key_tiles(rows.to(torch.int32), n_prior_tiles)
    prior_key_pos = _to_key_tiles(selected.float(), n_prior_tiles)

    # [kv_lora_rank, ...] -> [P_MAX, n_latent_tiles, ...], the partition-tiled layout SBUF holds.
    n_latent_tiles = q_absorbed.shape[0] // P_MAX
    q_absorbed_sbuf = q_absorbed.reshape(n_latent_tiles, P_MAX, n_tokens, -1).permute(1, 0, 2, 3).contiguous()

    return mla_decode_attention_torch_ref(
        q_absorbed_sbuf,
        q_pe,
        kv_cache,
        kv_row_cur,
        # The gather-order table is immaterial to the reference, which reads the natural order only.
        prior_rows,
        prior_key_pos,
        prior_rows,
        positions,
        out_absorb_w,
        o_proj_w,
        n_prior,
        softmax_scale,
    )


def _to_key_tiles(table: torch.Tensor, n_prior_tiles: int) -> torch.Tensor:
    """
    Tiles a ``[n_tokens, n_prior]`` table onto partitions, key ``tile * P_MAX + p`` landing at ``p``.

    Returns ``[P_MAX, n_tokens, n_prior_tiles]``. A ragged last tile is zero-filled, which the
    reference never reads because it slices back to ``n_prior`` keys.
    """

    n_tokens, n_prior = table.shape
    padded = torch.zeros((n_tokens, n_prior_tiles * P_MAX), dtype=table.dtype)
    padded[:, 0:n_prior] = table
    return padded.reshape(n_tokens, n_prior_tiles, P_MAX).permute(2, 0, 1).contiguous()


class TestMlaDecodeAttention:
    """Test executor for DeepSeek-V3.2 MLA decode attention tests."""

    @pytest.mark.parametrize("lnc", [1])
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_mla_decode_attention(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_tokens: int,
        n_heads: int,
        n_prior: int,
        v_head_dim: int,
        hidden: int,
        block_size: int,
        max_blocks_per_seq: int,
        dtype: type,
        lnc: int,
    ):
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return self._generate_inputs(
                n_tokens, n_heads, n_prior, v_head_dim, hidden, block_size, max_blocks_per_seq, dtype
            )

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_tokens, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            torch_ref=torch_ref_wrapper(mla_decode_attention_staged_torch_ref),
            kernel_entry=mla_decode_attention_staged,
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
        n_heads: int,
        n_prior: int,
        v_head_dim: int,
        hidden: int,
        block_size: int,
        max_blocks_per_seq: int,
        dtype: type,
    ) -> dict[str, Any]:
        """Wrapper around input generator for TestMlaDecodeAttention."""

        np.random.seed(42)
        kv_row_dim = _QK_ROPE_DIM + _KV_LORA_RANK

        def random_tensor(
            name: str, *shape, tensor_dtype: type = dtype, mean: float = 0.0, std: float = 0.2
        ) -> npt.NDArray[Any]:
            """Generates a random tensor for the provided name, shape, dtype and optionally mean and std."""
            generator = gaussian_tensor_generator(mean, std)
            return generator(shape, tensor_dtype, name)

        # Each sequence owns a distinct run of blocks, with a few spare so a wrong physical block is
        # visible rather than aliasing onto a neighbour's data.
        n_blocks = n_tokens * max_blocks_per_seq + 4
        block_table = (np.arange(n_tokens * max_blocks_per_seq, dtype=np.int32) % n_blocks).reshape(
            n_tokens, max_blocks_per_seq
        )

        # Spread the decode positions so the causal mask runs at several densities against selections
        # drawn from the whole range. Token 0 sits at position 1, which keeps exactly one prior key and
        # so leaves the current token's own key carrying almost all of the softmax weight -- the case
        # that catches a mask which drops the current key along with the unselected tail.
        prior_capacity = max_blocks_per_seq * block_size
        positions = (1 + np.arange(n_tokens, dtype=np.int64) * (prior_capacity - 1) // n_tokens).astype(np.int32)

        # The indexer emits its selections unsorted, so shuffle rather than hand the kernel a
        # monotonic list that would hide an ordering mistake.
        gather_indices = np.empty((n_tokens, n_prior), dtype=np.int32)
        picks = np.linspace(0, prior_capacity - 1, n_prior).astype(np.int32)
        for token in range(n_tokens):
            shuffled = picks.copy()
            np.random.default_rng(1234 + token).shuffle(shuffled)
            gather_indices[token] = shuffled

        return {
            # Queries are drawn wider than the cache so the scores have real spread and the softmax
            # cannot pass by being near-uniform.
            "q_absorbed": random_tensor("q_absorbed", _KV_LORA_RANK, n_tokens, n_heads, std=1.0),
            "q_pe": random_tensor("q_pe", _QK_ROPE_DIM, n_tokens, n_heads, std=1.0),
            "kv_cache": random_tensor("kv_cache", n_blocks, 1, block_size, kv_row_dim),
            "kv_row_cur": random_tensor("kv_row_cur", n_tokens, kv_row_dim),
            "gather_indices": gather_indices,
            "block_table": block_table,
            "positions": positions,
            "out_absorb_w": random_tensor(
                "out_absorb_w", n_heads, v_head_dim, _KV_LORA_RANK, std=1.0 / np.sqrt(_KV_LORA_RANK)
            ),
            "o_proj_w": random_tensor(
                "o_proj_w", n_heads * v_head_dim, hidden, std=1.0 / np.sqrt(n_heads * v_head_dim)
            ),
            "softmax_scale": _SOFTMAX_SCALE,
        }
