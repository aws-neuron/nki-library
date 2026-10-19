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
"""Tests for the Gemma4 attention_tkg variant (partitioned block-K load + batch-shard override).

Mirrors the core attention_tkg test harness but drives nkilib.models.gemma4.attention.
"""

import math
from dataclasses import asdict
from typing import Any, Optional, final

import neuron_dtypes as dt
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.core.utils.allocator import SbufManager, create_auto_alloc_manager, sizeinbytes
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil
from nkilib_src.nkilib.core.utils.logging import Logger
from nkilib_src.nkilib.models.gemma4.attention.attention_tkg import (
    AttnTKGConfig,
    TileConstants,
    _compute_tile_params,
    attention_tkg,
    is_batch_sharded,
    is_s_prior_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
    uses_flash_attention,
)
from nkilib_src.nkilib.models.gemma4.attention.attention_tkg_torch import attention_tkg_torch_ref
from nkilib_src.nkilib.models.gemma4.attention.attention_tkg_utils import (
    is_qk_swapped,
    uses_batch_tiling,
)
from nkilib_src.nkilib.models.gemma4.attention.gen_mask_tkg_torch import build_full_attention_mask
from typing_extensions import override

from test.integration.nkilib.core.attention.test_attention_tkg_utils import (
    AttnTKGTestParams,
    build_active_attention_mask,
    build_swa_positions,
    gen_deterministic_active_block_table,
    generate_cache_lens,
)
from test.integration.nkilib.utils.tensor_generators import np_random_sample, np_random_sample_fp8
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.comparators import maxAllClose
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.tensor_histogram import TensorHistogram
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

P_MAX = 128
FP8_TEST_DTYPE = nl.float8_e4m3

_ABBREVS = {
    "batch_size": "bs",
    "block_len": "blen",
    "use_pos_id": "posid",
}


def cfg_repr(cfg: AttnTKGConfig | AttnTKGTestParams):
    if isinstance(cfg, AttnTKGTestParams):
        return "-".join(
            [f"dt_{cfg.dtype}", f"sink_{int(cfg.test_sink)}", f"fp8_{int(cfg.fp8_kv)}", f"sw_{cfg.sliding_window}"]
        )
    elif isinstance(cfg, AttnTKGConfig):
        return "-".join(
            [
                f"bs_{cfg.bs}",
                f"qh_{cfg.q_head}",
                f"sa_{cfg.s_active}",
                f"sp_{cfg.curr_sprior}",
                f"spf_{cfg.full_sprior}",
                f"d_{cfg.d_head}",
                f"bl_{cfg.block_len}",
                f"tpk_{int(cfg.tp_k_prior)}",
                f"pid_{int(cfg.use_pos_id)}",
                f"fsr_{int(cfg.fuse_rope)}",
                f"osb_{int(cfg.out_in_sb)}",
                f"kosb_{int(cfg.k_out_in_sb)}",
                f"qsb_{int(cfg.qk_in_sb)}",
                f"sm1_{int(cfg.strided_mm1)}",
            ]
            # Only appended when set, so IDs of pre-existing configs are unchanged.
            + ([f"bssm_{cfg.batch_shard_sprior_max}"] if cfg.batch_shard_sprior_max else [])
            + (["pkv_1"] if cfg.partitioned_kv_load else [])
        )
    else:
        raise ValueError(f"Unknown type for cfg_repr: {type(cfg)}")


def get_bqs_tile_parameters(p_max: int, cfg: AttnTKGConfig, lnc: int):
    bs_n_prgs = (
        lnc
        if is_batch_sharded(
            cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, p_max, cfg.fuse_rope, cfg.batch_shard_sprior_max
        )
        else 1
    )
    bqs_size = cfg.bs // bs_n_prgs * cfg.q_head * cfg.s_active
    bqs_tiles = div_ceil(bqs_size, p_max)
    bqs_tile_size = p_max if bqs_tiles > 1 else bqs_size

    return bs_n_prgs, bqs_size, bqs_tiles, bqs_tile_size


def get_debug_tensor_shapes(p_max: int, cfg: AttnTKGConfig, lnc: int):
    qk_shape = (p_max, cfg.curr_sprior // p_max, cfg.bs * cfg.q_head * cfg.s_active)
    bs_n_prgs, _, bqs_tiles, bqs_tile_size = get_bqs_tile_parameters(p_max, cfg, lnc)
    reduced_shape = (bs_n_prgs, bqs_tiles, bqs_tile_size)
    return qk_shape, reduced_shape


@nki.jit
def attn_tkg_wrapper(
    q,
    k_active,
    v_active,
    k_prior,
    v_prior,
    mask,
    inv_freqs,
    rope_pos_ids,
    start_pos_ids,
    sink,
    active_blocks_table,
    attn_out_shape,
    attn_out_sb_shape,
    k_out_shape,
    cfg: AttnTKGConfig,
    dtype,
    lnc,
    use_auto_alloc,
    DBG,
    max_context_len=None,
):
    # Dummy validation step so that invalid configurations are verified early in test
    _compute_tile_params(
        cfg, TileConstants.get_tile_constants(), q, k_prior, v_prior, k_active, v_active, active_blocks_table
    )

    is_block_kv = cfg.block_len > 0
    if not use_auto_alloc:
        # Leave some room for compiler allocated usage
        sbm = SbufManager(
            0,
            nl.tile_size.total_available_sbuf_size - 16 * 1024,
            Logger(
                "SBM",
            ),
        )  # LogLevel.DEBUG))
    else:
        sbm = create_auto_alloc_manager(logger=Logger("SBM"))
    sbm.open_scope()

    # Create output tensors
    d_tile_size = min(cfg.d_head, P_MAX)
    n_d_tiles = math.ceil(cfg.d_head / P_MAX)
    out_buffer = nl.sbuf if cfg.out_in_sb else nl.shared_hbm
    k_out_buffer = nl.sbuf if cfg.k_out_in_sb else nl.shared_hbm
    if cfg.out_in_sb:
        out = sbm.alloc_stack(attn_out_sb_shape, dtype=dtype, buffer=nl.sbuf)
    else:
        out = nl.ndarray(attn_out_shape, dtype=dtype, buffer=out_buffer)
    k_out = None
    if cfg.fuse_rope:
        # assert False
        if cfg.k_out_in_sb:
            k_out = sbm.alloc_stack(k_out_shape, dtype=dtype, buffer=k_out_buffer)
        else:
            k_out = nl.ndarray(k_out_shape, dtype=dtype, buffer=k_out_buffer)

    # Load QK if kernel wants SB inputs
    if cfg.qk_in_sb:
        q_free = cfg.bs * cfg.q_head * cfg.s_active
        q_input = sbm.alloc_stack((d_tile_size, n_d_tiles * q_free), dtype=q.dtype, buffer=nl.sbuf)
        for i_d in range(n_d_tiles):
            d_start = i_d * d_tile_size
            nisa.dma_copy(
                q_input[:, i_d * q_free : (i_d + 1) * q_free],
                q[d_start : d_start + d_tile_size, :],
            )
        k_free = cfg.bs * cfg.s_active
        k_active_input = sbm.alloc_stack((d_tile_size, n_d_tiles * k_free), dtype=k_active.dtype, buffer=nl.sbuf)
        for i_d in range(n_d_tiles):
            d_start = i_d * d_tile_size
            nisa.dma_copy(
                k_active_input[:, i_d * k_free : (i_d + 1) * k_free],
                k_active[d_start : d_start + d_tile_size, :],
            )
    else:
        q_input = q
        k_active_input = k_active

    DBG_TENSORS = None
    if DBG:
        TC = TileConstants.get_tile_constants()

        qk_shape, reduced_shape = get_debug_tensor_shapes(TC.p_max, cfg, lnc)
        DBG_QK = nl.ndarray(qk_shape, dtype=nl.float32, buffer=nl.shared_hbm)
        DBG_QK_MAX = nl.ndarray(reduced_shape, dtype=nl.float32, buffer=nl.shared_hbm)
        DBG_QK_EXP = nl.ndarray(qk_shape, dtype=dtype, buffer=nl.shared_hbm)
        DBG_EXP_SUM = nl.ndarray(reduced_shape, dtype=nl.float32, buffer=nl.shared_hbm)
        DBG_TENSORS = (DBG_QK, DBG_QK_MAX, DBG_QK_EXP, DBG_EXP_SUM)

        if is_block_kv:
            _, resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
                cfg.curr_sprior // cfg.block_len,
                cfg.block_len,
                lnc,
                TC.p_max,
                cfg.bs,
                cfg.q_head,
                cfg.s_active,
                full_sprior=cfg.full_sprior,
                enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
                fuse_rope=cfg.fuse_rope,
                batch_shard_sprior_max=cfg.batch_shard_sprior_max,
            )
            DBG_ACTIVE_TABLE = nl.ndarray(
                (TC.p_max, active_blocks_table.shape[1] * resize_factor // TC.p_max, cfg.bs),
                dtype=nl.int32,
                buffer=nl.shared_hbm,
            )
            DBG_TENSORS = DBG_TENSORS + (DBG_ACTIVE_TABLE,)

    attn_out, k_out = attention_tkg(
        q_input,
        k_active_input,
        v_active,
        k_prior,
        v_prior,
        mask,
        out,
        cfg,
        sbm,
        inv_freqs,
        rope_pos_ids,
        start_pos_ids,
        sink,
        active_blocks_table,
        k_out,
        DBG_TENSORS=DBG_TENSORS,
        max_context_len=max_context_len,
    )

    # Handle output if needed
    if cfg.out_in_sb:
        out_bqh = cfg.bs * cfg.q_head * cfg.s_active
        attn_out_hbm = nl.ndarray(attn_out_shape, dtype=attn_out.dtype, buffer=nl.shared_hbm, name="attn_out_hbm")
        for i_d in range(n_d_tiles):
            d_start = i_d * d_tile_size
            src_offset = i_d * out_bqh
            nisa.dma_copy(
                dst=attn_out_hbm[d_start : d_start + d_tile_size, :],
                src=attn_out[:, src_offset : src_offset + out_bqh],
            )
        attn_out = attn_out_hbm

    if cfg.k_out_in_sb and cfg.fuse_rope:
        k_out_hbm = nl.ndarray(k_out.shape, dtype=k_out.dtype, buffer=nl.shared_hbm, name='k_out_hbm')
        nisa.dma_copy(dst=k_out_hbm, src=k_out)
        k_out = k_out_hbm

    sbm.close_scope()

    out = (attn_out,)
    if cfg.fuse_rope:
        out = out + (k_out,)
    if DBG:
        out = out + (DBG_QK, DBG_QK_MAX, DBG_QK_EXP, DBG_EXP_SUM)
        if is_block_kv:
            out = out + (DBG_ACTIVE_TABLE,)

    return out


def run_attention_tkg_test(
    test_manager: Orchestrator,
    compiler_args: CompilerArgs,
    cfg: AttnTKGConfig,
    dtype: str,
    test_sink: bool,
    fp8_kv: bool,
    sliding_window: int,
    is_negative_test_case: bool = False,
    DBG: bool = False,
    max_context_len_value: Optional[int] = None,
    sequence_lengths: Optional[list[int]] = None,
    use_auto_alloc: bool = False,
):
    np.random.seed(42)

    lnc = compiler_args.logical_nc_config
    is_block_kv = cfg.block_len != 0

    if fp8_kv:
        relative_tolerance, absolute_tolerance = 3e-2, 3e-2
    else:
        relative_tolerance, absolute_tolerance = 1e-2, 1e-5

    # KV cache dtype for this config (used to derive the is_2byte_kv arg of is_qk_swapped).
    kv_dtype = FP8_TEST_DTYPE if fp8_kv else dtype

    # Shapes that differ base on config
    q_shape = (
        (cfg.d_head, cfg.bs * cfg.q_head * cfg.s_active)
        if cfg.qk_in_sb
        else (cfg.bs, cfg.q_head, cfg.s_active, cfg.d_head)
    )
    k_active_shape = (cfg.d_head, cfg.bs * cfg.s_active) if cfg.qk_in_sb else (cfg.bs, 1, cfg.s_active, cfg.d_head)
    if is_block_kv:
        assumed_num_cache_blocks = cfg.bs * cfg.curr_sprior // cfg.block_len
        if cfg.fp8_packed:
            k_prior_shape = (assumed_num_cache_blocks, cfg.block_len // 2, cfg.d_head, 2)
        else:
            k_prior_shape = (assumed_num_cache_blocks, cfg.block_len, cfg.d_head)
        v_prior_shape = (assumed_num_cache_blocks, cfg.block_len, cfg.d_head)
    else:
        k_prior_shape = (
            (cfg.bs, 1, cfg.full_sprior, cfg.d_head) if cfg.tp_k_prior else (cfg.bs, 1, cfg.d_head, cfg.full_sprior)
        )
        v_prior_shape = (cfg.bs, 1, cfg.full_sprior, cfg.d_head)

    n_d_tiles = math.ceil(cfg.d_head / P_MAX)
    d_tile_size = min(cfg.d_head, P_MAX)
    if cfg.out_in_sb:
        attn_out_shape = (cfg.d_head, cfg.bs * cfg.q_head * cfg.s_active)
        attn_out_sb_shape = (d_tile_size, n_d_tiles * cfg.bs * cfg.q_head * cfg.s_active)
    else:
        attn_out_shape = (cfg.bs, cfg.q_head, cfg.d_head, cfg.s_active)
        attn_out_sb_shape = None
    k_out_shape = (cfg.d_head, cfg.bs * cfg.s_active) if cfg.k_out_in_sb else (cfg.bs, 1, cfg.d_head, cfg.s_active)

    # Generate pos_id outside of tensor_gen so it stays the same for all inputs.
    if sequence_lengths is not None:
        lengths = np.asarray(sequence_lengths, dtype=np.int32)
        assert lengths.shape == (cfg.bs,)
        assert np.all(lengths >= cfg.s_active)
        assert np.all(lengths <= cfg.curr_sprior)
        pos_id = lengths[:, None] - cfg.s_active
    else:
        pos_id = generate_cache_lens(cfg.bs, cfg.curr_sprior, cfg.s_active, mode="normal")
    if max_context_len_value is not None:
        assert sequence_lengths is None
        max_pos = cfg.curr_sprior - cfg.s_active
        pos_id = generate_cache_lens(
            cfg.bs, cfg.curr_sprior, cfg.s_active, mean_frac=max_context_len_value / (2 * max_pos)
        )
        pos_id = np.clip(pos_id, 0, max_context_len_value - cfg.s_active)
        pos_id[0] = max_context_len_value - cfg.s_active  # ensure at least one batch hits the max

    def input_generator(test_config):
        random_gen = np_random_sample()

        if fp8_kv:
            kv_random_gen = np_random_sample_fp8()
            kv_dtype = FP8_TEST_DTYPE
        else:
            kv_random_gen = random_gen
            kv_dtype = dtype

        q = random_gen(shape=q_shape, dtype=dtype, name='q')
        k_active = kv_random_gen(shape=k_active_shape, dtype=kv_dtype, name='k_active').astype(kv_dtype)
        v_active = kv_random_gen(shape=(cfg.bs, 1, cfg.s_active, cfg.d_head), dtype=kv_dtype, name='v_active').astype(
            kv_dtype
        )
        k_prior = kv_random_gen(shape=k_prior_shape, dtype=kv_dtype, name='k_prior').astype(kv_dtype)
        v_prior = kv_random_gen(shape=v_prior_shape, dtype=kv_dtype, name='v_prior').astype(kv_dtype)

        if cfg.use_pos_id:
            active_mask = (
                build_active_attention_mask(batch=cfg.bs, num_heads=cfg.q_head, s_active=cfg.s_active, transposed=True)
                .numpy()
                .astype(np.bool_)
            )
        else:
            # TODO: migrate to gen_mask_tkg_hbm_torch_ref so this test shares the
            # same swap-aware mask reference as attention_block_tkg.
            # This requires flat-KV support for non-strided MM1 in the HBM reference.
            cache_lens_torch = torch.from_numpy(np.asarray(pos_id).flatten()).to(torch.float32)
            active_mask = (
                build_full_attention_mask(
                    cache_lens=cache_lens_torch,
                    batch=cfg.bs,
                    num_heads=cfg.q_head,
                    s_active=cfg.s_active,
                    s_ctx=cfg.curr_sprior,
                    lnc=lnc,
                    block_len=cfg.block_len,
                    include_active_mask=True,
                    transposed=True,
                    enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
                    fuse_rope=cfg.fuse_rope,
                    batch_shard_sprior_max=cfg.batch_shard_sprior_max,
                )
                .numpy()
                .astype(np.bool_)
            )
            if is_qk_swapped(
                bs=cfg.bs,
                q_head=cfg.q_head,
                d_head=cfg.d_head,
                s_active=cfg.s_active,
                curr_sprior=cfg.curr_sprior,
                lnc=lnc,
                p_max=P_MAX,
                is_block_kv=is_block_kv,
                is_2byte_kv=sizeinbytes(kv_dtype) == 2,
                fp8_packed=cfg.fp8_packed,
                fuse_rope=cfg.fuse_rope,
                batch_shard_sprior_max=cfg.batch_shard_sprior_max,
            ):
                active_mask = np.ascontiguousarray(active_mask.transpose(1, 2, 3, 0))  # [s_ctx,B,N,S]->[B,N,S,s_ctx]
        active_mask = np.ascontiguousarray(active_mask, dtype=np.uint8)

        inv_freqs = np.random.random(size=(cfg.d_head // 2, 1)).astype(np.float32) if cfg.fuse_rope else None
        start_pos_ids, rope_pos_ids = None, None
        if cfg.fuse_rope or cfg.use_pos_id:
            if sliding_window > 0:
                start_pos_ids, rope_pos_ids = build_swa_positions(
                    pos_id=pos_id,
                    bs=cfg.bs,
                    s_active=cfg.s_active,
                    sliding_window=sliding_window,
                    cache_len=cfg.curr_sprior,
                    block_len=cfg.block_len,
                )
            else:
                rope_pos_ids = (
                    np.broadcast_to(pos_id, (cfg.bs, cfg.s_active)).astype(np.float32)
                    + np.arange(cfg.s_active, dtype=np.float32)[np.newaxis, :]
                )
        active_blocks_table = (
            gen_deterministic_active_block_table(
                cfg.bs, cfg.curr_sprior, cfg.s_active, pos_id, cfg.block_len, cfg.bs * cfg.curr_sprior // cfg.block_len
            ).astype(np.int32)
            if is_block_kv
            else None
        )
        sink = random_gen(shape=(cfg.q_head, 1), dtype=np.float32, name='sink') if test_sink else None

        return {
            "q": q,
            "k_active": k_active,
            "v_active": v_active,
            "k_prior": k_prior,
            "v_prior": v_prior,
            "mask": active_mask,
            "inv_freqs": inv_freqs,
            "rope_pos_ids": rope_pos_ids,
            "start_pos_ids": start_pos_ids,
            "sink": sink,
            "active_blocks_table": active_blocks_table,
            "attn_out_shape": attn_out_shape,
            "attn_out_sb_shape": attn_out_sb_shape,
            "k_out_shape": k_out_shape,
            "cfg": cfg,
            "dtype": dtype,
            "lnc": lnc,
            "use_auto_alloc": use_auto_alloc,
            "DBG": DBG,
            "max_context_len": np.array([max_context_len_value], dtype=np.int32)
            if max_context_len_value is not None
            else None,
        }

    def attn_tkg_torch_wrapper(
        q,
        k_active,
        v_active,
        k_prior,
        v_prior,
        mask,
        inv_freqs,
        rope_pos_ids,
        start_pos_ids,
        sink,
        active_blocks_table,
        attn_out_shape,
        attn_out_sb_shape,
        k_out_shape,
        cfg: AttnTKGConfig,
        dtype,
        lnc,
        use_auto_alloc,
        DBG,
        max_context_len=None,
    ):
        out = torch.zeros(attn_out_shape, dtype=torch.float32)
        k_out = torch.zeros(k_out_shape, dtype=torch.float32) if cfg.fuse_rope else None

        DBG_TENSORS = None
        if DBG:
            qk_shape, _ = get_debug_tensor_shapes(P_MAX, cfg, lnc)
            reduced_shape = (cfg.bs, cfg.q_head, 1, cfg.s_active)
            DBG_QK = torch.zeros(qk_shape, dtype=torch.float32)
            DBG_QK_MAX = torch.zeros(reduced_shape, dtype=torch.float32)
            DBG_QK_EXP = torch.zeros(qk_shape, dtype=torch.float32)
            DBG_EXP_SUM = torch.zeros(reduced_shape, dtype=torch.float32)
            DBG_TENSORS = (DBG_QK, DBG_QK_MAX, DBG_QK_EXP, DBG_EXP_SUM)

            if is_block_kv:
                _, resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
                    cfg.curr_sprior // cfg.block_len,
                    cfg.block_len,
                    lnc,
                    P_MAX,
                    cfg.bs,
                    cfg.q_head,
                    cfg.s_active,
                    full_sprior=cfg.full_sprior,
                    enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
                    fuse_rope=cfg.fuse_rope,
                    batch_shard_sprior_max=cfg.batch_shard_sprior_max,
                )
                DBG_ACTIVE_TABLE = torch.zeros(
                    (P_MAX, active_blocks_table.shape[1] * resize_factor // P_MAX, cfg.bs),
                    dtype=torch.int32,
                )
                DBG_TENSORS = DBG_TENSORS + (DBG_ACTIVE_TABLE,)

        # input_generator hands the kernel the swap (bqh-major) mask [B, N, s_active, s_ctx] for swap
        # configs. attention_tkg_torch_ref only implements the canonical s_prior-major [s_ctx, B, N, s_active]
        # layout, so un-swap here.
        golden_mask = mask
        if not cfg.use_pos_id and is_qk_swapped(
            bs=cfg.bs,
            q_head=cfg.q_head,
            d_head=cfg.d_head,
            s_active=cfg.s_active,
            curr_sprior=cfg.curr_sprior,
            lnc=lnc,
            p_max=P_MAX,
            is_block_kv=is_block_kv,
            is_2byte_kv=sizeinbytes(kv_dtype) == 2,
            fp8_packed=cfg.fp8_packed,
            fuse_rope=cfg.fuse_rope,
            batch_shard_sprior_max=cfg.batch_shard_sprior_max,
        ):
            golden_mask = mask.permute(3, 0, 1, 2)  # [B, N, s_active, s_ctx] -> [s_ctx, B, N, s_active]

        out, k_out = attention_tkg_torch_ref[lnc](
            q=q,
            k_active=k_active,
            v_active=v_active,
            k_prior=k_prior,
            v_prior=v_prior,
            mask=golden_mask,
            out=out,
            cfg=cfg,
            sbm=None,
            inv_freqs=inv_freqs,
            rope_pos_ids=rope_pos_ids,
            start_pos_ids=start_pos_ids,
            sink=sink,
            active_blocks_table=active_blocks_table,
            k_out=k_out,
            DBG_TENSORS=DBG_TENSORS,
            max_context_len=max_context_len,
        )

        # These tensors are padded and have "don't care" values that are removed by this custom comparator
        def custom_debug_tensor_comparator(golden_tensor: npt.NDArray[Any], tensor_name: str):
            bs_n_prgs, bqs_size, bqs_tiles, bqs_tile_size = get_bqs_tile_parameters(P_MAX, cfg, lnc)

            class AttentionTkgValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    inference_output = inference_output.view(np.float32)
                    inference_output = inference_output.reshape(bs_n_prgs, bqs_tiles * bqs_tile_size)
                    inference_output = inference_output[:, :bqs_size]
                    inference_output = inference_output.reshape(golden_tensor.shape)
                    expected = golden_tensor.astype(np.float32)

                    self._print_with_log(f"Summary for {tensor_name}:")
                    passed = maxAllClose(
                        inference_output,
                        expected,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                        verbose=1,
                        logfile=self.logfile,
                    )

                    visualizer = TensorHistogram()
                    visualizer.print_full_comparison_report(
                        actual=inference_output,
                        expected=expected,
                        name=tensor_name,
                        atol=absolute_tolerance,
                        rtol=relative_tolerance,
                        passed=passed,
                        logfile=self.logfile,
                    )

                    return passed

            return CustomValidatorWithOutputTensorData(
                AttentionTkgValidator, np.ndarray((bs_n_prgs, bqs_tiles, bqs_tile_size), dtype=golden_tensor.dtype)
            )

        def skip_comparator(tensor_name: str, shape: tuple, dtype: np.dtype):
            """Create a comparator that always passes (skips validation)."""

            class SkipValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    self._print_with_log(f"SKIPPED: {tensor_name} (not validated for this configuration)")
                    return True

            return CustomValidatorWithOutputTensorData(SkipValidator, np.ndarray(shape, dtype=dtype))

        golds = {'golden_out': dt.static_cast(out.numpy(), dtype)}
        if cfg.fuse_rope:
            assert k_out is not None, "the fused-rope path produces a k_out tensor"
            golds['golden_k_out'] = dt.static_cast(k_out.numpy(), dtype)
        if DBG:
            DBG_QK_NP = DBG_QK.numpy()
            DBG_QK_MAX_NP = DBG_QK_MAX.numpy()
            DBG_QK_EXP_NP = DBG_QK_EXP.numpy()
            DBG_EXP_SUM_NP = DBG_EXP_SUM.numpy()

            # Determine if FA and batch tiling are used (affects which debug tensors are valid)
            _shard_args = (
                cfg.bs,
                cfg.q_head,
                cfg.s_active,
                cfg.curr_sprior,
                P_MAX,
                cfg.fuse_rope,
                cfg.batch_shard_sprior_max,
            )
            s_prior_n_prgs = lnc if is_s_prior_sharded(*_shard_args) else 1
            bs_n_prgs = lnc if is_batch_sharded(*_shard_args) else 1
            s_prior_per_prg = cfg.curr_sprior // s_prior_n_prgs
            bs_per_nc = cfg.bs // bs_n_prgs
            use_fa, fa_tile_size = uses_flash_attention(cfg.enable_fa_s_prior_tiling, s_prior_per_prg)
            fa_tile_s_prior = fa_tile_size if use_fa else s_prior_per_prg
            use_bt, _ = uses_batch_tiling(
                bs_per_nc, cfg.q_head, cfg.s_active, fa_tile_s_prior, use_auto_alloc, dtype_size=sizeinbytes(dtype)
            )

            # DBG_QK is skipped when strided_mm1 + FA (complex K column remapping)
            # or strided_mm1 + batch tiling (batch and sprior tiles interleaved in QK buffer)
            if cfg.strided_mm1 and (use_fa or use_bt):
                golds['DBG_QK'] = skip_comparator('DBG_QK', DBG_QK_NP.shape, DBG_QK_NP.dtype)
            else:
                golds['DBG_QK'] = dt.static_cast(DBG_QK_NP, DBG_QK_NP.dtype)

            # DBG_QK_MAX is skipped when batch tiling
            # (BSQ tile layout differs between full-batch and per-tile)
            if use_bt:
                golds['DBG_QK_MAX'] = skip_comparator('DBG_QK_MAX', DBG_QK_MAX_NP.shape, DBG_QK_MAX_NP.dtype)
            else:
                golds['DBG_QK_MAX'] = custom_debug_tensor_comparator(DBG_QK_MAX_NP, 'DBG_QK_MAX')

            # DBG_QK_EXP is skipped when the kernel takes the online-softmax path (use_fa OR
            # sharded), because qk_io_type is computed against a per-tile/local max rather than
            # the global max; also skipped when strided_mm1 + batch tiling (same interleaving
            # issue as DBG_QK).
            online_softmax_enabled = use_fa or s_prior_n_prgs > 1
            if online_softmax_enabled or (cfg.strided_mm1 and use_bt):
                golds['DBG_QK_EXP'] = skip_comparator('DBG_QK_EXP', DBG_QK_EXP_NP.shape, DBG_QK_EXP_NP.dtype)
            else:
                golds['DBG_QK_EXP'] = dt.static_cast(DBG_QK_EXP_NP, dtype)

            # DBG_EXP_SUM is skipped when batch tiling (same BSQ tile layout issue as DBG_QK_MAX)
            if use_bt:
                golds['DBG_EXP_SUM'] = skip_comparator('DBG_EXP_SUM', DBG_EXP_SUM_NP.shape, DBG_EXP_SUM_NP.dtype)
            else:
                golds['DBG_EXP_SUM'] = custom_debug_tensor_comparator(DBG_EXP_SUM_NP, 'DBG_EXP_SUM')

            if is_block_kv:
                DBG_ACTIVE_TABLE_NP = DBG_ACTIVE_TABLE.numpy()
                golds['DBG_ACTIVE_TABLE'] = dt.static_cast(DBG_ACTIVE_TABLE_NP, np.int32)

        return golds

    def output_tensors(kernel_input):
        golden_output = {
            'golden_out': np.zeros(attn_out_shape, dtype),
        }
        if cfg.fuse_rope:
            golden_output['golden_k_out'] = np.zeros(k_out_shape, dtype)
        if DBG:
            qk_shape, reduced_shape = get_debug_tensor_shapes(P_MAX, cfg, lnc)
            golden_output['DBG_QK'] = np.zeros(qk_shape, dtype=np.float32)
            golden_output['DBG_QK_MAX'] = np.zeros(reduced_shape, dtype=np.float32)
            golden_output['DBG_QK_EXP'] = np.zeros(qk_shape, dtype)
            golden_output['DBG_EXP_SUM'] = np.zeros(reduced_shape, dtype=np.float32)
            if is_block_kv:
                _, resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
                    cfg.curr_sprior // cfg.block_len,
                    cfg.block_len,
                    lnc,
                    P_MAX,
                    cfg.bs,
                    cfg.q_head,
                    cfg.s_active,
                    full_sprior=cfg.full_sprior,
                    enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
                    fuse_rope=cfg.fuse_rope,
                    batch_shard_sprior_max=cfg.batch_shard_sprior_max,
                )
                golden_output['DBG_ACTIVE_TABLE'] = np.zeros(
                    (P_MAX, cfg.bs * cfg.curr_sprior * resize_factor // cfg.block_len // P_MAX),
                    dtype=np.int32,
                )
        return golden_output

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=attn_tkg_wrapper,
        torch_ref=torch_ref_wrapper(attn_tkg_torch_wrapper),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensors,
    )
    framework.run_test(
        test_config=None,
        compiler_args=compiler_args,
        inference_args=TKG_INFERENCE_ARGS,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        is_negative_test=is_negative_test_case,
    )


# Partitioned block-K load (cfg.partitioned_kv_load) with the batch-shard override (cfg.batch_shard_sprior_max).
# Targets sliding-window priors whose block count is not a multiple of p_max (e.g. 1280 tokens / 16 = 80 blocks),
# where the stock path shrinks the block length and issues one DMA per resized block.


def _pkv_cfg(
    bs, q_head, s_active, s_prior, d_head, block_len, use_pos_id, fp8_packed=False, batch_shard_sprior_max=2048
):
    """AttnTKGConfig for the partitioned block-K load with the batch-shard override (block KV, SBUF I/O)."""
    return AttnTKGConfig(
        bs,
        q_head,
        s_active,
        s_prior,
        s_prior,
        d_head,
        block_len,
        tp_k_prior=True,
        strided_mm1=False,
        use_pos_id=use_pos_id,
        qk_in_sb=True,
        k_out_in_sb=True,
        out_in_sb=True,
        fp8_packed=fp8_packed,
        batch_shard_sprior_max=batch_shard_sprior_max,
        partitioned_kv_load=True,
    )


attention_tkg_partitioned_kv_configs = [
    # Sliding-window decode geometry: 80 blocks of 16, d_head=256 (2 d-tiles), fp8 and bf16 KV, s_active 1 and 4 (MTP)
    pytest.param(
        _pkv_cfg(32, 2, 1, 1280, 256, 16, use_pos_id=True), AttnTKGTestParams(fp8_kv=True), marks=pytest.mark.fast
    ),
    [_pkv_cfg(32, 2, 1, 1280, 256, 16, use_pos_id=True), AttnTKGTestParams()],
    [_pkv_cfg(32, 2, 4, 1280, 256, 16, use_pos_id=True), AttnTKGTestParams(fp8_kv=True)],
    [_pkv_cfg(8, 2, 4, 1280, 256, 16, use_pos_id=True), AttnTKGTestParams(fp8_kv=True, sliding_window=1024)],
    # Pre-generated HBM mask (use_pos_id=False) must follow the same resized ordering
    pytest.param(_pkv_cfg(2, 2, 1, 1280, 128, 16, use_pos_id=False), AttnTKGTestParams(), marks=pytest.mark.fast),
    [_pkv_cfg(4, 2, 4, 1280, 256, 16, use_pos_id=False), AttnTKGTestParams(fp8_kv=True)],
    # 160 blocks of 8: two gathers per batch (128 + 32 blocks), resized block length 2
    pytest.param(
        _pkv_cfg(4, 2, 4, 1280, 128, 8, use_pos_id=True), AttnTKGTestParams(fp8_kv=True), marks=pytest.mark.fast
    ),
    # 128 blocks (no resize): partitioned path degenerates to one full-fold gather
    [_pkv_cfg(8, 2, 1, 1024, 128, 8, use_pos_id=True), AttnTKGTestParams()],
    [_pkv_cfg(16, 2, 1, 2048, 256, 16, use_pos_id=True), AttnTKGTestParams(fp8_kv=True, sliding_window=1024)],
    # 256 blocks of 8: two full gathers per batch
    [_pkv_cfg(4, 2, 1, 2048, 256, 8, use_pos_id=True), AttnTKGTestParams()],
    # Large blocks: 20 blocks of 64 (resized to 2, 32 sub-blocks per block), partial-quadrant gather
    [_pkv_cfg(4, 2, 1, 1280, 128, 64, use_pos_id=True), AttnTKGTestParams(test_sink=True)],
    # Flash attention tiles (8192 + 1024) on the partitioned path; the override must cover the full prior on LNC2
    [
        AttnTKGConfig(
            2,
            2,
            1,
            9216,
            9216,
            128,
            16,
            use_pos_id=True,
            tp_k_prior=True,
            strided_mm1=False,
            qk_in_sb=True,
            k_out_in_sb=True,
            out_in_sb=True,
            batch_shard_sprior_max=10240,
            partitioned_kv_load=True,
        ),
        AttnTKGTestParams(fp8_kv=True),
    ],
    # Stock-load twins of the sliding-window geometry for A/B: same sharding without the partitioned load, and full defaults
    [
        AttnTKGConfig(
            32,
            2,
            1,
            1280,
            1280,
            256,
            16,
            use_pos_id=True,
            tp_k_prior=True,
            strided_mm1=False,
            qk_in_sb=True,
            k_out_in_sb=True,
            out_in_sb=True,
            batch_shard_sprior_max=2048,
        ),
        AttnTKGTestParams(fp8_kv=True),
    ],
    [
        AttnTKGConfig(
            32,
            2,
            1,
            1280,
            1280,
            256,
            16,
            use_pos_id=True,
            tp_k_prior=True,
            strided_mm1=False,
            qk_in_sb=True,
            k_out_in_sb=True,
            out_in_sb=True,
        ),
        AttnTKGTestParams(fp8_kv=True),
    ],
]

# Unsupported combinations must fail at trace time with a message naming the field.
# Entries: (config, test params, lnc values on which the config is invalid)
attention_tkg_partitioned_kv_invalid_configs = [
    # fp8_packed has no PE-transpose path
    (_pkv_cfg(4, 2, 1, 2048, 128, 16, use_pos_id=True, fp8_packed=True), AttnTKGTestParams(fp8_kv=True), (1, 2)),
    # flat KV cache
    (
        AttnTKGConfig(
            4,
            2,
            1,
            1024,
            1024,
            128,
            0,
            use_pos_id=True,
            tp_k_prior=True,
            strided_mm1=False,
            qk_in_sb=True,
            k_out_in_sb=True,
            out_in_sb=True,
            partitioned_kv_load=True,
        ),
        AttnTKGTestParams(),
        (1, 2),
    ),
    # s_prior sharding: without the override an even batch with bs*q_head*s_active < p_max shards s_prior on LNC2
    (
        AttnTKGConfig(
            2,
            2,
            1,
            1280,
            1280,
            128,
            16,
            use_pos_id=True,
            tp_k_prior=True,
            strided_mm1=False,
            qk_in_sb=True,
            k_out_in_sb=True,
            out_in_sb=True,
            partitioned_kv_load=True,
        ),
        AttnTKGTestParams(),
        (2,),
    ),
    # odd batch cannot batch-shard on LNC2 regardless of the override
    (_pkv_cfg(1, 2, 1, 1280, 128, 16, use_pos_id=True), AttnTKGTestParams(), (2,)),
]
# fmt: on


def filter_invalid_partitioned_kv_tests(
    batch_size: int | None = None,
    lnc: int | None = None,
    **_,
) -> FilterResult:
    """Partitioned K load requires batch sharding, which an odd batch cannot satisfy on LNC2."""
    if batch_size is not None and lnc is not None and lnc == 2 and batch_size % 2 != 0:
        return FilterResult.INVALID
    return FilterResult.VALID


@pytest_test_metadata(name="Attention TKG Gemma4")
@pytest_marks(["attention", "tkg", "gemma4"])
@final
class TestAttentionTkgGemma4:
    @pytest.mark.parametrize("lnc", [1, 2])
    @pytest.mark.parametrize("attn_cfg, test_cfg", attention_tkg_partitioned_kv_configs, ids=cfg_repr)
    def test_attn_tkg_gemma4_partitioned_kv_load(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_cfg: AttnTKGConfig,
        lnc: int,
        test_cfg: AttnTKGTestParams,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc, platform_target=platform_target)
        run_attention_tkg_test(test_manager=test_manager, compiler_args=compiler_args, cfg=attn_cfg, **asdict(test_cfg))

    @pytest.mark.parametrize(
        "attn_cfg, test_cfg, lnc",
        [
            pytest.param(
                cfg,
                params,
                lnc,
                id=f"{cfg_repr(cfg)}-{cfg_repr(params)}-lnc{lnc}",
                # Keep one LNC2 row per invalid geometry fast; the LNC1 and odd-batch
                # (bs == 1) rows add no branch coverage over them.
                marks=pytest.mark.fast if (lnc == 2 and cfg.bs != 1) else (),
            )
            for cfg, params, lncs in attention_tkg_partitioned_kv_invalid_configs
            for lnc in lncs
        ],
    )
    def test_attn_tkg_gemma4_partitioned_kv_load_invalid(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_cfg: AttnTKGConfig,
        test_cfg: AttnTKGTestParams,
        lnc: int,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc, platform_target=platform_target)
        run_attention_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            cfg=attn_cfg,
            is_negative_test_case=True,
            **asdict(test_cfg),
        )

    @pytest.mark.coverage_parametrize(
        batch_size=[1, 2, 4, 8, 16, 32],
        s_active=[1, 4],
        d_head=[128, 256],
        block_len=[8, 16],
        s_prior=[1024, 1280, 2048],
        fp8_kv=[True, False],
        use_pos_id=[True, False],
        lnc=[1, 2],
        coverage="pairs",
        filter=filter_invalid_partitioned_kv_tests,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=True,
        max_invalid_tests=4,  # every invalid combination is the same odd-batch LNC2 case
        abbrev=_ABBREVS,
    )
    def test_attn_tkg_gemma4_partitioned_kv_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch_size: int,
        s_active: int,
        d_head: int,
        block_len: int,
        s_prior: int,
        fp8_kv: bool,
        use_pos_id: bool,
        lnc: int,
        is_negative_test_case: bool,
    ):
        """Pairwise sweep of the partitioned K load over the framework's sliding-window decode grid."""
        cfg = _pkv_cfg(batch_size, 2, s_active, s_prior, d_head, block_len, use_pos_id=use_pos_id)
        compiler_args = CompilerArgs(logical_nc_config=lnc, platform_target=platform_target)
        run_attention_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            cfg=cfg,
            dtype=nl.bfloat16,
            test_sink=False,
            fp8_kv=fp8_kv,
            sliding_window=0,
            is_negative_test_case=is_negative_test_case,
        )
