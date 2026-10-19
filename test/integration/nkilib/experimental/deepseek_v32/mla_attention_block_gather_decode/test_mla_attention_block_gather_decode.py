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

"""Integration tests for the DeepSeek-V3.2 MLA decode attention block."""

from typing import Any

import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.deepseek_v32.mla_attention_block_gather_decode import (
    mla_attention_block_gather_decode,
    mla_attention_block_gather_decode_torch_ref,
)

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# kv_lora_rank must equal F_MAX for the front-fold's PSUM eviction and be a multiple of P_MAX for the
# core's score contraction. It is a model constant rather than a free test parameter.
_KV_LORA_RANK = 512

# Both composed kernels are traced into one NEFF, so the block needs no wrapper of its own: it takes
# HBM tensors and returns the block output, unlike its two halves which hand query tiles over in SBUF.
PARAM_NAMES = (
    "n_tokens, hidden, q_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, "
    "n_prior, v_head_dim, block_size, max_blocks_per_seq, dtype"
)
TEST_PARAMS = [
    pytest.param(
        1,
        512,
        256,
        1,
        128,
        64,
        128,
        128,
        16,
        16,
        ml_dtypes.bfloat16,
        id="ds32_mla_block_t1_nh1_k128_bf16",
        marks=pytest.mark.fast,
    ),
    pytest.param(
        4,
        512,
        256,
        2,
        128,
        64,
        256,
        128,
        16,
        32,
        ml_dtypes.bfloat16,
        id="ds32_mla_block_t4_nh2_k256_bf16",
        marks=pytest.mark.fast,
    ),
    # n_prior is ragged, so the current token's key shares the last key tile with prior keys.
    pytest.param(
        2,
        512,
        256,
        1,
        128,
        64,
        96,
        128,
        16,
        16,
        ml_dtypes.bfloat16,
        id="ds32_mla_block_t2_k96_ragged_bf16",
        marks=pytest.mark.fast,
    ),
    # fp32 has no indirect dma_transpose, so every latent chunk transposes on the PE.
    pytest.param(
        2,
        1024,
        512,
        4,
        128,
        64,
        128,
        128,
        32,
        16,
        np.float32,
        id="ds32_mla_block_t2_nh4_k128_f32",
        marks=pytest.mark.fast,
    ),
    pytest.param(
        16,
        7168,
        1536,
        1,
        128,
        64,
        2048,
        128,
        32,
        256,
        ml_dtypes.bfloat16,
        id="DS32_MLA_ATTENTION_BLOCK_GATHER_DECODE_PROD_CONFIG_B16",
    ),
]


class TestMlaAttentionBlockGatherDecode:
    """Test executor for DeepSeek-V3.2 MLA decode attention block tests."""

    @pytest.mark.parametrize("lnc", [1])
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_mla_attention_block_gather_decode(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_tokens: int,
        hidden: int,
        q_lora_rank: int,
        n_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        n_prior: int,
        v_head_dim: int,
        block_size: int,
        max_blocks_per_seq: int,
        dtype: type,
        lnc: int,
    ):
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return self._generate_inputs(
                n_tokens,
                hidden,
                q_lora_rank,
                n_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                n_prior,
                v_head_dim,
                block_size,
                max_blocks_per_seq,
                dtype,
            )

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_tokens, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            torch_ref=torch_ref_wrapper(mla_attention_block_gather_decode_torch_ref),
            kernel_entry=mla_attention_block_gather_decode,
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
        n_prior: int,
        v_head_dim: int,
        block_size: int,
        max_blocks_per_seq: int,
        dtype: type,
    ) -> dict[str, Any]:
        """Wrapper around input generator for TestMlaAttentionBlockGatherDecode."""

        np.random.seed(42)
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_row_dim = qk_rope_head_dim + _KV_LORA_RANK

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

        # Spread the decode positions so the causal mask runs at several densities. Token 0 sits at
        # position 1, which keeps exactly one prior key and so leaves the current token's own key
        # carrying almost all of the softmax weight.
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

        # Distinct slots in distinct blocks, so a scatter that lands on the wrong row is visible. The
        # cache carries real data because the core gathers prior rows out of it, unlike the front-fold's
        # own test where only the written row is read back.
        slots = (np.arange(n_tokens, dtype=np.int32) * block_size + 1) % (n_blocks * block_size)

        return {
            "hidden_states": random_tensor("hidden_states", n_tokens, hidden),
            "input_norm_w": random_tensor("input_norm_w", 1, hidden, tensor_dtype=np.float32, mean=1.0, std=0.1),
            "wq_a": random_tensor("wq_a", hidden, q_lora_rank),
            "q_norm_w": random_tensor("q_norm_w", 1, q_lora_rank, tensor_dtype=np.float32, mean=1.0, std=0.1),
            "wq_b": random_tensor("wq_b", q_lora_rank, n_heads * qk_head_dim),
            "wkv_a": random_tensor("wkv_a", hidden, kv_row_dim),
            "kv_norm_w": random_tensor("kv_norm_w", 1, _KV_LORA_RANK, tensor_dtype=np.float32, mean=1.0, std=0.1),
            "q_absorb_w": random_tensor("q_absorb_w", n_heads, qk_nope_head_dim, _KV_LORA_RANK),
            "out_absorb_w": random_tensor(
                "out_absorb_w", n_heads, v_head_dim, _KV_LORA_RANK, std=1.0 / np.sqrt(_KV_LORA_RANK)
            ),
            "o_proj_w": random_tensor(
                "o_proj_w", n_heads * v_head_dim, hidden, std=1.0 / np.sqrt(n_heads * v_head_dim)
            ),
            "cos": random_tensor("cos", n_tokens, qk_rope_head_dim // 2, tensor_dtype=np.float32),
            "sin": random_tensor("sin", n_tokens, qk_rope_head_dim // 2, tensor_dtype=np.float32),
            "kv_cache": random_tensor("kv_cache", n_blocks, 1, block_size, kv_row_dim),
            "slot_mapping": slots,
            "block_table": block_table,
            "positions": positions,
            "gather_indices": gather_indices,
            "n_heads": n_heads,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "kv_lora_rank": _KV_LORA_RANK,
            # The model scales scores by the full per-head QK width, nope plus rope.
            "softmax_scale": float(qk_head_dim**-0.5),
            "norm_eps": 1e-6,
        }
