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

"""Tests for MXFP8 attention TKG — separate KV blocks with pre-computed masking."""

from dataclasses import dataclass

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.attention_mxfp8.attention_mxfp8_tkg import attention_mxfp8_tkg
from nkilib_src.nkilib.experimental.attention_mxfp8.attention_mxfp8_tkg_torch import attention_mxfp8_tkg_torch_ref

from test.integration.nkilib.experimental.attention_mxfp8.mxfp8_attention_utils import (
    quantize_kv_cache_block,
    static_cast,
)
from test.integration.nkilib.experimental.matmul_mxfp8.utils import check_correctness
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.rng import NKITestsRNG
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework

_rng = NKITestsRNG()

# ── Constants matching kernel ─────────────────────────────────────────────────
D_HEAD = 128
BLOCK_LEN = 128  # tokens per block (the atomic KV input unit)
P_PER_BLOCK = 32
PACKED_COLS = 160
P_MAX = 128
HALF_P = 64
BLOCKS_PER_FOLD = 4
FOLDS_PER_CHUNK = 4
BLOCKS_PER_CHUNK = BLOCKS_PER_FOLD * FOLDS_PER_CHUNK  # 16
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_LEN  # 2048
SCORE_FREE = FOLDS_PER_CHUNK * BLOCK_LEN * 2  # 1024


# ── Test configurations ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mxfp8AttnTkgTestConfig:
    pos_id: int
    bs: int = 1
    q_head: int = 64
    lnc: int = 2

    def __repr__(self):
        return f"bs{self.bs}_qh{self.q_head}_pos{self.pos_id}_lnc{self.lnc}"


TEST_CONFIGS = [
    # Full pos_id sweep on the sharded path (bs=1, lnc=2)
    pytest.param(Mxfp8AttnTkgTestConfig(pos_id=2048), marks=pytest.mark.fast),
    Mxfp8AttnTkgTestConfig(pos_id=8192),
    Mxfp8AttnTkgTestConfig(pos_id=65536),
    Mxfp8AttnTkgTestConfig(pos_id=131072),
    Mxfp8AttnTkgTestConfig(pos_id=524288),
    # bs=1, lnc=1 with one short and one long pos_id
    pytest.param(Mxfp8AttnTkgTestConfig(pos_id=2048, lnc=1), marks=pytest.mark.fast),
    Mxfp8AttnTkgTestConfig(pos_id=524288, lnc=1),
    # bs=8 at 128k with both lnc values
    Mxfp8AttnTkgTestConfig(pos_id=131072, bs=8, lnc=1),
    Mxfp8AttnTkgTestConfig(pos_id=131072, bs=8),
]


def _make_mxfp8_comparator(label):
    """Build a custom_comparator validating output against the torch_ref golden via check_correctness."""

    def comparator(golden_dict, output_tensors):
        golden = np.asarray(golden_dict["out_hbm"]).astype(np.float32)

        class _Validator(CustomValidator):
            def validate(self, inference_output):
                actual = (
                    np.frombuffer(inference_output.tobytes(), dtype=ml_dtypes.bfloat16)
                    .astype(np.float32)
                    .reshape(golden.shape)
                )
                passed, metrics = check_correctness(actual, golden)
                self._print_with_log(f"--- {label} ---")
                self._print_with_log(f"  metrics: {metrics}")
                if not passed:
                    self._print_with_log(f"  kernel[0,0,:5]: {actual[0, 0, :5]}")
                    self._print_with_log(f"  golden[0,0,:5]: {golden[0, 0, :5]}")
                return passed

        return {
            "out_hbm": CustomValidatorWithOutputTensorData(
                validator=_Validator,
                output_ndarray=output_tensors["out_hbm"],
            ),
        }

    return comparator


def _build_active_blocks_table(batch, num_blocks):
    """Build active_blocks_table [B, num_blocks] with sequential block mapping."""
    table = np.arange(num_blocks, dtype=np.int32).reshape(1, num_blocks)
    return np.tile(table, (batch, 1))


def _build_chunk_masks(bs, num_blocks, pos_id_val):
    """Build chunk masks [B * num_chunks, P_MAX, SCORE_FREE] uint8 in the eviction layout.

    Block g of a chunk lands at partition half g%2 and free offset
    (pair * BLOCKS_PER_FOLD + fold) * BLOCK_LEN, where fold = (g % 16) // 4
    and pair = (g % 4) // 2. Tokens at positions >= pos_id_val are masked out.
    """
    num_chunks = num_blocks // BLOCKS_PER_CHUNK
    chunk_masks = np.zeros((bs * num_chunks, P_MAX, SCORE_FREE), dtype=np.uint8)
    for c in range(num_chunks):
        for blk_in_chunk in range(BLOCKS_PER_CHUNK):
            fold_idx, blk_in_fold = divmod(blk_in_chunk, BLOCKS_PER_FOLD)
            pair_idx, half = divmod(blk_in_fold, 2)
            f_off = (pair_idx * BLOCKS_PER_FOLD + fold_idx) * BLOCK_LEN
            tok_start = (c * BLOCKS_PER_CHUNK + blk_in_chunk) * BLOCK_LEN
            n_valid = int(np.clip(pos_id_val - tok_start, 0, BLOCK_LEN))
            if n_valid > 0:
                p_slice = slice(0, HALF_P) if half == 0 else slice(HALF_P, P_MAX)
                chunk_masks[c::num_chunks, p_slice, f_off : f_off + n_valid] = 1
    return chunk_masks


def _generate_inputs(bs, q_head, bucket_size, pos_id_val, seed=42):
    """Generate all kernel inputs for the given config."""
    _rng.reset(seed)

    num_blocks = bucket_size // BLOCK_LEN

    q_bf16 = static_cast(_rng.randn(bs, q_head, 1, D_HEAD).numpy(), nl.bfloat16)
    k_blocks_bf16 = static_cast(_rng.randn(num_blocks, BLOCK_LEN, D_HEAD).numpy(), nl.bfloat16)
    v_blocks_bf16 = static_cast(_rng.randn(num_blocks, BLOCK_LEN, D_HEAD).numpy(), nl.bfloat16)

    # Zero V beyond pos_id so masked positions contribute nothing even prior to masking
    for blk in range(num_blocks):
        blk_start = blk * BLOCK_LEN
        n_valid = int(np.clip(pos_id_val - blk_start, 0, BLOCK_LEN))
        v_blocks_bf16[blk, n_valid:, :] = 0.0

    # Pack separate K and V caches: each [num_blocks, 32, 160]
    k_cache_3d = np.stack([quantize_kv_cache_block(k_blocks_bf16[blk], "d_head") for blk in range(num_blocks)]).astype(
        np.float32
    )
    v_cache_3d = np.stack(
        [quantize_kv_cache_block(v_blocks_bf16[blk], "block_len") for blk in range(num_blocks)]
    ).astype(np.float32)

    return {
        "q": q_bf16,
        "k_active": static_cast(_rng.randn(bs, D_HEAD).numpy(), nl.bfloat16),
        "v_active": static_cast(_rng.randn(bs, D_HEAD).numpy(), nl.bfloat16),
        "k_prior": k_cache_3d,
        "v_prior": v_cache_3d,
        "mask": _build_chunk_masks(bs, num_blocks, pos_id_val),
        "identity_hbm": np.eye(P_MAX, dtype=ml_dtypes.bfloat16),
        "active_blocks_table": _build_active_blocks_table(bs, num_blocks),
    }


def _run_attention_mxfp8_tkg_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: Mxfp8AttnTkgTestConfig,
    is_negative_test_case: bool = False,
):
    """Run one MXFP8 attention TKG config vs the quantization-aware torch golden."""
    min_chunks = cfg.lnc  # LNC2 sharding needs at least one chunk per NC
    bucket_size = max(cfg.pos_id, min_chunks * CHUNK_TOKENS)
    bucket_size = (bucket_size + CHUNK_TOKENS - 1) // CHUNK_TOKENS * CHUNK_TOKENS

    kernel_input = _generate_inputs(cfg.bs, cfg.q_head, bucket_size, cfg.pos_id)

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=attention_mxfp8_tkg,
        kernel_input_generator=lambda _: kernel_input,
        torch_ref=attention_mxfp8_tkg_torch_ref,
        output_tensor_descriptor=lambda _: {
            "out_hbm": np.zeros((cfg.bs, cfg.q_head, D_HEAD), dtype=ml_dtypes.bfloat16)
        },
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=cfg.lnc),
        is_negative_test=is_negative_test_case,
        custom_comparator=_make_mxfp8_comparator(
            f"MXFP8 Accuracy (bucket={bucket_size}, pos_id={cfg.pos_id}, lnc={cfg.lnc})"
        ),
    )


# ── Test class ────────────────────────────────────────────────────────────────
@pytest_test_metadata(name="AttentionMxfp8Tkg", pytest_marks=["attention", "mxfp8"])
class TestAttentionMxfp8Tkg:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=repr)
    def test_attention_mxfp8_tkg(
        self, test_manager: Orchestrator, platform_target: Platforms, cfg: Mxfp8AttnTkgTestConfig
    ):
        """MXFP8 flash decode attention vs quantization-aware torch golden (curated configs)."""
        _run_attention_mxfp8_tkg_test(test_manager=test_manager, platform_target=platform_target, cfg=cfg)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.coverage_parametrize(
        bs=[1, 2, 4, 8],
        q_head=[64],  # kernel requires q_head == 64
        pos_id=[2048, 3000, 8192, 65536, 131072, 524288],
        lnc=[1, 2],
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_attention_mxfp8_tkg_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs: int,
        q_head: int,
        pos_id: int,
        lnc: int,
        is_negative_test_case: bool,
    ):
        """Pairwise sweep over bs/pos_id/lnc plus boundary negative tests for kernel asserts."""
        _run_attention_mxfp8_tkg_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=Mxfp8AttnTkgTestConfig(pos_id=pos_id, bs=bs, q_head=q_head, lnc=lnc),
            is_negative_test_case=is_negative_test_case,
        )
