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
import contextlib
import os
from dataclasses import asdict
from functools import lru_cache
from test.integration.nkilib.core.attention.test_attention_tkg_utils import (
    AttnTKGTestParams,
    build_active_attention_mask,
    build_swa_positions,
    cfg_repr,
    gen_deterministic_active_block_table,
    generate_pow_range,
    get_bqs_tile_parameters,
    get_debug_tensor_shapes,
    print_test_config,
)
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import nki

try:
    from test.integration.nkilib.core.attention.test_attention_tkg_model_config import (
        attention_tkg_model_configs,
    )
except ImportError:
    attention_tkg_model_configs = []

from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.tensor_generators import np_random_sample, np_random_sample_fp8
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.tensor_histogram import TensorHistogram
from test.utils.test_orchestrator import Orchestrator
from typing import Any, final

import neuron_dtypes as dt
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.core.attention.attention_tkg import (
    AttnTKGConfig,
    TileConstants,
    attention_tkg,
    is_batch_sharded,
    is_s_prior_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
    uses_flash_attention,
)
from nkilib_src.nkilib.core.attention.attention_tkg_torch import attention_tkg_torch_ref
from nkilib_src.nkilib.core.attention.attention_tkg_utils import get_total_n_prgs, uses_batch_tiling
from nkilib_src.nkilib.core.attention.gen_mask_tkg_torch import build_full_attention_mask
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.logging import Logger
from typing_extensions import override

P_MAX = 128
FP8_TEST_DTYPE = nl.float8_e4m3
# These parameters are chosen such that tests generated all take <1 min
MAX_B = 32
MAX_Q_HEAD = 16
MAX_S_ACTIVE = 8
MAX_S_PRIOR = 2**14


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
    k_out_shape,
    cfg: AttnTKGConfig,
    dtype,
    lnc,
    DBG,
):
    is_block_kv = cfg.block_len > 0
    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, Logger("SBM"))
    sbm.open_scope()

    # Create output tensors
    out_buffer = nl.sbuf if cfg.out_in_sb else nl.shared_hbm
    k_out_buffer = nl.sbuf if cfg.k_out_in_sb else nl.shared_hbm
    if cfg.out_in_sb:
        out = sbm.alloc_stack(attn_out_shape, dtype=dtype, buffer=nl.sbuf)
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
        q_input = sbm.alloc_stack(q.shape, dtype=q.dtype, buffer=nl.sbuf)
        k_active_input = sbm.alloc_stack(k_active.shape, dtype=k_active.dtype, buffer=nl.sbuf)
        nisa.dma_copy(q_input, q)
        nisa.dma_copy(k_active_input, k_active)
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
                cfg.curr_sprior // cfg.block_len, cfg.block_len, get_total_n_prgs(cfg, lnc, TC.p_max), TC.p_max
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
    )

    # Handle output if needed
    if cfg.out_in_sb:
        # This is a bug, the name generation logic is faulty.
        attn_out_hbm = nl.ndarray(attn_out.shape, dtype=attn_out.dtype, buffer=nl.shared_hbm, name="attn_out_hbm")
        nisa.dma_copy(dst=attn_out_hbm, src=attn_out)
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
):
    np.random.seed(42)

    lnc = compiler_args.logical_nc_config
    is_block_kv = cfg.block_len != 0

    if fp8_kv:
        relative_tolerance, absolute_tolerance = 3e-2, 3e-2
    else:
        relative_tolerance, absolute_tolerance = 1e-2, 1e-5

    # Shapes that differ base on config
    q_shape = (
        (cfg.d_head, cfg.bs * cfg.q_head * cfg.s_active)
        if cfg.qk_in_sb
        else (cfg.bs, cfg.q_head, cfg.s_active, cfg.d_head)
    )
    k_active_shape = (cfg.d_head, cfg.bs * cfg.s_active) if cfg.qk_in_sb else (cfg.bs, 1, cfg.s_active, cfg.d_head)
    resize_factor = None
    if is_block_kv:
        assumed_num_cache_blocks = cfg.bs * cfg.curr_sprior // cfg.block_len
        k_prior_shape = v_prior_shape = (assumed_num_cache_blocks, cfg.block_len, cfg.d_head)
    else:
        k_prior_shape = (
            (cfg.bs, 1, cfg.full_sprior, cfg.d_head) if cfg.tp_k_prior else (cfg.bs, 1, cfg.d_head, cfg.full_sprior)
        )
        v_prior_shape = (cfg.bs, 1, cfg.full_sprior, cfg.d_head)

    attn_out_shape = (
        (cfg.d_head, cfg.bs * cfg.q_head * cfg.s_active)
        if cfg.out_in_sb
        else (cfg.bs, cfg.q_head, cfg.d_head, cfg.s_active)
    )
    k_out_shape = (cfg.d_head, cfg.bs * cfg.s_active) if cfg.k_out_in_sb else (cfg.bs, 1, cfg.d_head, cfg.s_active)

    # Generate pos_id outside of tensor_gen so it stays the same for all inputs
    pos_id = ((np.arange(cfg.bs) * 3 + (cfg.curr_sprior // 4 * 3)) % (cfg.curr_sprior - cfg.s_active))[:, np.newaxis]

    def input_generator(test_config):
        random_gen = np_random_sample()

        if fp8_kv:
            kv_random_gen = np_random_sample_fp8()
            kv_dtype = FP8_TEST_DTYPE
        else:
            kv_random_gen = random_gen
            kv_dtype = dtype

        q = random_gen(shape=q_shape, dtype=dtype, name='q').astype(kv_dtype)
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
                )
                .numpy()
                .astype(np.bool_)
            )
        active_mask = active_mask.astype(np.uint8)

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
                rope_pos_ids = np.broadcast_to(pos_id, (cfg.bs, cfg.s_active)).astype(np.float32)
        active_blocks_table = (
            gen_deterministic_active_block_table(
                cfg.bs, cfg.curr_sprior, cfg.s_active, pos_id, cfg.block_len, cfg.bs * cfg.curr_sprior // cfg.block_len
            ).astype(np.uint32)
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
            "k_out_shape": k_out_shape,
            "cfg": cfg,
            "dtype": dtype,
            "lnc": lnc,
            "DBG": DBG,
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
        k_out_shape,
        cfg: AttnTKGConfig,
        dtype,
        lnc,
        DBG,
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
                    cfg.curr_sprior // cfg.block_len, cfg.block_len, lnc, P_MAX
                )
                DBG_ACTIVE_TABLE = torch.zeros(
                    (P_MAX, active_blocks_table.shape[1] * resize_factor // P_MAX, cfg.bs),
                    dtype=torch.int32,
                )
                DBG_TENSORS = DBG_TENSORS + (DBG_ACTIVE_TABLE,)

        out, k_out = attention_tkg_torch_ref[lnc](
            q=q,
            k_active=k_active,
            v_active=v_active,
            k_prior=k_prior,
            v_prior=v_prior,
            mask=mask,
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
            golds['golden_k_out'] = dt.static_cast(k_out.numpy(), dtype)
        if DBG:
            DBG_QK_NP = DBG_QK.numpy()
            DBG_QK_MAX_NP = DBG_QK_MAX.numpy()
            DBG_QK_EXP_NP = DBG_QK_EXP.numpy()
            DBG_EXP_SUM_NP = DBG_EXP_SUM.numpy()

            # Determine if FA and batch tiling are used (affects which debug tensors are valid)
            s_prior_n_prgs = lnc if is_s_prior_sharded(cfg, P_MAX) else 1
            bs_n_prgs = lnc if is_batch_sharded(cfg, P_MAX) else 1
            s_prior_per_prg = cfg.curr_sprior // s_prior_n_prgs
            bs_per_nc = cfg.bs // bs_n_prgs
            use_fa, fa_tile_size = uses_flash_attention(s_prior_per_prg)
            fa_tile_s_prior = fa_tile_size if use_fa else s_prior_per_prg
            use_bt, _ = uses_batch_tiling(bs_per_nc, cfg.q_head, cfg.s_active, fa_tile_s_prior)

            # DBG_QK is skipped when strided_mm1 + FA (complex K column remapping)
            # or strided_mm1 + batch tiling (batch and sprior tiles interleaved in QK buffer)
            if cfg.strided_mm1 and (use_fa or use_bt):
                golds['DBG_QK'] = skip_comparator('DBG_QK', DBG_QK_NP.shape, DBG_QK_NP.dtype)
            else:
                golds['DBG_QK'] = dt.static_cast(DBG_QK_NP, DBG_QK_NP.dtype)

            # DBG_QK_MAX is skipped when FA (running quantities) or batch tiling
            # (BSQ tile layout differs between full-batch and per-tile)
            if use_fa or use_bt:
                golds['DBG_QK_MAX'] = skip_comparator('DBG_QK_MAX', DBG_QK_MAX_NP.shape, DBG_QK_MAX_NP.dtype)
            else:
                golds['DBG_QK_MAX'] = custom_debug_tensor_comparator(DBG_QK_MAX_NP, 'DBG_QK_MAX')

            # DBG_QK_EXP is skipped when FA (running quantities)
            # or strided_mm1 + batch tiling (same interleaving issue as DBG_QK)
            if use_fa or (cfg.strided_mm1 and use_bt):
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
                    cfg.curr_sprior // cfg.block_len, cfg.block_len, lnc, P_MAX
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


def filter_invalid_tests(
    batch_size: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    s_prior_full_multiple: int,
    d_head: int,
    block_len: int,
    tp_k_prior: bool,
    strided_mm1: bool,
    use_pos_id: bool,
    fuse_rope: bool,
    out_in_sb: bool,
    k_out_in_sb: bool,
    qk_in_sb: bool,
    dtype: str,
    sink: bool,
    fp8_kv: bool,
    sliding_window: int,
    lnc: int,
) -> FilterResult:
    cfg = AttnTKGConfig(
        bs=batch_size,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=s_prior,
        full_sprior=s_prior_full_multiple * s_prior,
        d_head=d_head,
        block_len=block_len,
        tp_k_prior=tp_k_prior,
        use_pos_id=use_pos_id,
        fuse_rope=fuse_rope,
        out_in_sb=out_in_sb,
        k_out_in_sb=k_out_in_sb,
        qk_in_sb=qk_in_sb,
        strided_mm1=strided_mm1,
    )

    s_prior_n_prgs = lnc if is_s_prior_sharded(cfg, P_MAX) else 1
    use_fa, fa_tile_size = uses_flash_attention(s_prior)

    # Invalid Value Combinations
    if fuse_rope:
        if batch_size * q_head * s_active > P_MAX:
            return FilterResult.INVALID
        if s_prior > 2**17:
            return FilterResult.INVALID  # Accuracy drops from modulo on float32
        if d_head % 64 != 0:
            return FilterResult.INVALID
    else:
        if not qk_in_sb:
            return FilterResult.INVALID

    if fp8_kv:
        if fuse_rope:
            return FilterResult.INVALID
        if not qk_in_sb:
            return FilterResult.INVALID

    if (s_prior // s_prior_n_prgs) % P_MAX != 0:
        return FilterResult.INVALID

    if block_len != 0:
        if not qk_in_sb:
            return FilterResult.INVALID

        if not tp_k_prior:
            return FilterResult.INVALID

        if strided_mm1:
            return FilterResult.INVALID

        if s_prior % (P_MAX * lnc) != 0:
            return FilterResult.INVALID

        with open(os.devnull, 'w') as dev_null:
            with contextlib.redirect_stdout(dev_null):
                block_len, _ = resize_cache_block_len_for_attention_tkg_kernel(
                    num_blocks_per_batch=s_prior // block_len,
                    block_len=block_len,
                    n_prgs=lnc,
                    p_max=P_MAX,
                )

    if use_fa:
        last_tile_size = s_prior % fa_tile_size
        if last_tile_size and last_tile_size < s_active:
            return FilterResult.INVALID

        if block_len != 0:
            if fa_tile_size % (block_len * P_MAX) != 0:
                return FilterResult.INVALID

            if last_tile_size and last_tile_size % (block_len * P_MAX) != 0:
                return FilterResult.INVALID

    # Memory Concern Restrictions
    if s_prior_full_multiple * s_prior > MAX_S_PRIOR:
        return FilterResult.REDUNDANT

    return FilterResult.VALID


# fmt: off
attention_tkg_fast_configs = [
    # #################### K Transpose ####################
    # # NKI Llama 76B Target slot 1
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 1536, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 3072, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 4608, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 7168, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 8704, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Llama 76B Target slot2
    [AttnTKGConfig(4, 1, 5, 14336, 24576, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 18432, 24576, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 22528, 24576, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 24576, 24576, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Llama 3.1 70B
    [AttnTKGConfig(4, 1, 5, 16384, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Llama 3.1 405B
    [AttnTKGConfig(4, 2, 7, 16384, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Smaller bucket sizes/prior sequence length
    [AttnTKGConfig(4, 2, 7, 256, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 512, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 1024, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 2048, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 4096, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI d_head = 64
    [AttnTKGConfig(4, 2, 7, 512, 16384, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 3, 7, 16384, 16384, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 18432, 24576, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],

    #################### No K Transpose ####################
    # NKI Llama 76B Target slot 1
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 1536, 10240, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 3072, 10240, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 4608, 10240, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 7168, 10240, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 8704, 10240, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Llama 76B Target slot2
    [AttnTKGConfig(4, 1, 5, 14336, 24576, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 18432, 24576, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 22528, 24576, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 24576, 24576, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Llama 3.1 70B
    [AttnTKGConfig(4, 1, 5, 16384, 16384, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Llama 3.1 405B
    [AttnTKGConfig(4, 2, 7, 16384, 16384, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI Smaller bucket sizes/prior sequence length
    [AttnTKGConfig(4, 2, 7, 256, 16384, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 512, 16384, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 1024, 16384, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 2048, 16384, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 2, 7, 4096, 16384, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # NKI d_head = 64
    [AttnTKGConfig(4, 2, 7, 512, 16384, 64, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 3, 7, 16384, 16384, 64, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 64, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 18432, 24576, 64, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],

    #################### Input / output in SBUF ####################
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, qk_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, use_pos_id=True, fuse_rope=True, qk_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, fuse_rope=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],

    #################### Sink ####################
    [AttnTKGConfig(8, 8, 1, 1024, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(128, 1, 1, 1024, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 4, 5, 1024, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(test_sink=True)],
    # Sink with fp32
    [AttnTKGConfig(8, 8, 1, 1024, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(dtype=nl.float32, test_sink=True)],
    [AttnTKGConfig(128, 1, 1, 1024, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(dtype=nl.float32, test_sink=True)],
    [AttnTKGConfig(4, 4, 5, 1024, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(dtype=nl.float32, test_sink=True)],

    #################### LNC-shard on batch ####################
    # for s_prior < 256
    [AttnTKGConfig(8, 4, 4, 128, 256, 64, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(32, 8, 1, 128, 256, 64, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(128, 1, 1, 128, 256, 64, 0, qk_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(1, 1, 1, 128, 256, 64, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(16, 8, 1, 8192, 8192, 64, 0, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],
    # large batch (b*n*s>128)
    [AttnTKGConfig(4, 8, 7, 3072, 10240, 64, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 8, 7, 128, 10240, 64, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(64, 8, 1, 2048, 2048, 128, 0, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],

    #################### Block KV Tests ####################
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 8, 7, 4096, 10240, 64, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 8, 7, 3072, 10240, 64, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(64, 8, 1, 2048, 2048, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],

    #################### FP8 KV Cache Tests ####################
    # FP8 KV requires: fuse_rope=False, qk_in_sb=True
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 0, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 2, 7, 4096, 16384, 128, 0, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, tp_k_prior=True, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 0, tp_k_prior=True, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 2, 7, 4096, 16384, 128, 0, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    # FP8 KV with Block KV
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 8, 7, 4096, 10240, 64, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(1, 2, 5, 26624, 36896, 128, 32, tp_k_prior=True, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    # FP8 KV with Block KV large batch
    [AttnTKGConfig(128, 1, 1, 2048, 2048, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    # Block KV Tests with gen_mask_tkg mask generation (use_pos_id=True)
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 8, 7, 4096, 10240, 64, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 8, 7, 3072, 10240, 64, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(64, 8, 1, 2048, 2048, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],

    #################### Disable strided_mm1 tests ####################
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, tp_k_prior=True, strided_mm1=False, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 1536, 10240, 128, 0, tp_k_prior=True, strided_mm1=False, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 10, 5, 5376, 5376, 64, 0, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams()],
    
    #################### Test sprior_sharding when s_active_bqh > 128####################
    [AttnTKGConfig(5, 7, 7, 24576, 24576, 64, 0, use_pos_id=True, qk_in_sb=True), AttnTKGTestParams(test_sink=True)],

    #################### SWA (Sliding Window Attention) Tests ####################
    # SWA requires use_pos_id=True, sliding_window > 0
    # Flat KV strided
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=128)],
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=256)],
    [AttnTKGConfig(4, 2, 7, 2048, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=512)],
    [AttnTKGConfig(4, 1, 5, 4096, 16384, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=128)],
    # Flat KV non-strided
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, tp_k_prior=True, strided_mm1=False, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=128)],
    [AttnTKGConfig(4, 2, 7, 2048, 16384, 128, 0, tp_k_prior=True, strided_mm1=False, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=256)],
    # Block KV with SWA
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, sliding_window=128)],
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, sliding_window=256)],
    # SWA + sink
    [AttnTKGConfig(4, 8, 7, 4096, 10240, 64, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, sliding_window=128)],
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(test_sink=True, sliding_window=128)],
    # SWA + FP8 KV (requires fuse_rope=False, qk_in_sb=True)
    [AttnTKGConfig(4, 1, 5, 1024, 10240, 128, 0, strided_mm1=False, use_pos_id=True, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True, sliding_window=128)],
    [AttnTKGConfig(4, 1, 5, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True, sliding_window=128)],
    # SWA + d_head=64
    [AttnTKGConfig(4, 2, 7, 1024, 16384, 64, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=128)],
    # SWA + large batch (batch-sharded)
    [AttnTKGConfig(4, 8, 7, 3072, 10240, 64, 0, tp_k_prior=True, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, sliding_window=256)],
    # SWA + flash attention (s_prior > 8K triggers FA tiling)
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(sliding_window=256)],
    [AttnTKGConfig(4, 1, 5, 10240, 10240, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(sliding_window=256)],
]


# Long sequence length tests (test up to 512k to keep test time reasonable)
attention_tkg_slow_configs = [
    # Without block KV (b_l=0), b=4, q=1
    [AttnTKGConfig(4, 1, 5, 16640, 16640, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 20480, 20480, 128, 0, strided_mm1=False, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 20480, 20480, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 82432, 82432, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 1, 5, 82432, 82432, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    [AttnTKGConfig(4, 1, 5, 82432, 82432, 128, 0, use_pos_id=True, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 1, 5, 262144, 262144, 128, 0, tp_k_prior=True, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    # The RoPE modulo operation (`_modulo` function) loses precision for very large position IDs due to float32 catastrophic cancellation.
    # This leads to golden_k_out failing validation with ~5% error while golden_out still passes.
    # [AttnTKGConfig(4, 1, 5, 524288, 524288, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],
    # [AttnTKGConfig(4, 1, 5, 1048576, 1048576, 128, 0, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],

    # Without block KV (b_l=0), b=8, q=8 - includes 8192 (no FA) as baseline
    [AttnTKGConfig(8, 8, 5, 16384, 16384, 128, 0, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(8, 8, 5, 16384, 16384, 128, 0, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(8, 8, 5, 16640, 16640, 128, 0, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(8, 8, 5, 82432, 82432, 128, 0, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],
    [AttnTKGConfig(8, 8, 5, 82432, 82432, 128, 0, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(8, 8, 5, 262144, 262144, 128, 0, tp_k_prior=True, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams()],

    # With block KV (b_l=16), b=4, q=1
    [AttnTKGConfig(4, 1, 5, 16640, 16640, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 1, 5, 20480, 20480, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 1, 5, 82432, 82432, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 1, 5, 82432, 82432, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, fp8_kv=True)],
    [AttnTKGConfig(4, 1, 5, 82432, 82432, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, fuse_rope=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 1, 5, 524288, 524288, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, fp8_kv=True)],

    # With block KV (b_l=16), b=8, q=8
    [AttnTKGConfig(8, 8, 5, 16640, 16640, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, fp8_kv=True)],
    [AttnTKGConfig(8, 8, 5, 20480, 20480, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(8, 8, 5, 82432, 82432, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, fp8_kv=True)],
    [AttnTKGConfig(8, 8, 5, 82432, 82432, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(8, 8, 5, 262144, 262144, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],

    # few extra tests with s_p < s_p_f
    [AttnTKGConfig(4, 8, 7, 102400, 131072, 64, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 8, 7, 128, 131072, 64, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(4, 2, 7, 102400, 131072, 128, 0, tp_k_prior=True, use_pos_id=True, fuse_rope=True), AttnTKGTestParams()],

    # very large BQS testing (fuse_rope not supported)
    [AttnTKGConfig(128, 4, 1, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(1024, 4, 1, 8192, 16384, 64, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 128, 2, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(7, 128, 2, 8448, 8448, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(3, 59, 7, 4096, 4096, 64, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(256, 4, 1, 16640, 16640, 128, 16, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(128, 4, 1, 8192, 8192, 64, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(1024, 4, 1, 8192, 16384, 128, 0, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(4, 128, 2, 8192, 8192, 128, 0, tp_k_prior=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(7, 128, 2, 8448, 8448, 128, 0, strided_mm1=False, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True)],
    [AttnTKGConfig(3, 59, 7, 4096, 4096, 128, 0, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],
    [AttnTKGConfig(256, 4, 1, 16640, 16640, 64, 0, tp_k_prior=True, strided_mm1=False, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True)],

    # very large BQS + SWA (batch tiling + sliding window)
    [AttnTKGConfig(128, 4, 1, 8192, 8192, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(sliding_window=128)],
    [AttnTKGConfig(4, 128, 2, 8192, 8192, 128, 0, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(sliding_window=256)],
    [AttnTKGConfig(7, 128, 2, 8448, 8448, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True), AttnTKGTestParams(fp8_kv=True, sliding_window=128)],
    [AttnTKGConfig(3, 59, 7, 4096, 4096, 128, 0, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, sliding_window=256)],
    [AttnTKGConfig(256, 4, 1, 16640, 16640, 128, 16, tp_k_prior=True, strided_mm1=False, use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True), AttnTKGTestParams(test_sink=True, sliding_window=128)],
]
# fmt: on


# Load model metadata for matching test configs to model names
@lru_cache(maxsize=1)
def _get_attention_tkg_metadata():
    return load_model_configs("test_attention_tkg")


@pytest_test_metadata(
    name="Attention TKG",
    pytest_marks=["attention", "tkg"],
    tags=["model"],
)
@final
class TestAttentionTkgKernel:
    @pytest.mark.fast
    @pytest.mark.parametrize("lnc", [1, 2])
    @pytest.mark.parametrize("attn_cfg, test_cfg", attention_tkg_fast_configs, ids=cfg_repr)
    def test_attn_tkg_fast(
        self,
        test_manager: Orchestrator,
        attn_cfg: AttnTKGConfig,
        lnc: int,
        test_cfg: AttnTKGTestParams,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc)
        run_attention_tkg_test(test_manager=test_manager, compiler_args=compiler_args, cfg=attn_cfg, **asdict(test_cfg))

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "attn_cfg, test_cfg",
        [
            pytest.param(
                *params,
                id=f"{MODEL_TEST_TYPE}_{cfg_repr(params[0])}_{cfg_repr(params[1] if len(params) > 1 else AttnTKGTestParams())}",
            )
            for params in attention_tkg_model_configs
        ],
    )
    def test_attn_tkg_model(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        request: pytest.FixtureRequest,
        attn_cfg: AttnTKGConfig,
        test_cfg: AttnTKGTestParams,
    ):
        test_metadata_key = {
            "bs": attn_cfg.bs,
            "q_h": attn_cfg.q_head,
            "s_a": attn_cfg.s_active,
            "s_p": attn_cfg.curr_sprior,
            "s_p_full": attn_cfg.full_sprior,
            "d_h": attn_cfg.d_head,
        }
        collector.match_and_add_metadata_dimensions(test_metadata_key, _get_attention_tkg_metadata())

        compiler_args = CompilerArgs()
        run_attention_tkg_test(test_manager=test_manager, compiler_args=compiler_args, cfg=attn_cfg, **asdict(test_cfg))

    @pytest.mark.parametrize(
        "attn_cfg, test_cfg",
        attention_tkg_slow_configs,
        ids=cfg_repr,
    )
    def test_attn_tkg_slow(
        self,
        test_manager: Orchestrator,
        attn_cfg: AttnTKGConfig,
        test_cfg: AttnTKGTestParams,
    ):
        compiler_args = CompilerArgs()
        run_attention_tkg_test(test_manager=test_manager, compiler_args=compiler_args, cfg=attn_cfg, **asdict(test_cfg))

    @pytest.mark.coverage_parametrize(
        batch_size=BoundedRange(generate_pow_range(1, MAX_B, 3, pow_lag=2, base=3) + [MAX_B], boundary_values=[0]),
        q_head=BoundedRange(generate_pow_range(1, MAX_Q_HEAD, pow_lag=2), boundary_values=[0]),
        s_active=BoundedRange([i + 1 for i in range(MAX_S_ACTIVE)], boundary_values=[0]),
        s_prior=BoundedRange(
            [128, 256] + generate_pow_range(256, MAX_S_PRIOR, 256, pow_lag=3)[1::2], boundary_values=[0, 127]
        ),
        s_prior_full_multiple=BoundedRange([1, 1.5, 2], boundary_values=[]),
        d_head=BoundedRange([32, 64, 128], boundary_values=[0, 129]),
        block_len=BoundedRange([0, 32], boundary_values=[]),  # Do not test -1 because breaks reshapes
        tp_k_prior=[True, False],
        strided_mm1=[True, False],
        use_pos_id=[True, False],
        fuse_rope=[True, False],
        out_in_sb=[True, False],
        k_out_in_sb=[True, False],
        qk_in_sb=[True, False],
        dtype=[nl.bfloat16, nl.float32],
        sink=[True, False],
        fp8_kv=[True, False],
        sliding_window=BoundedRange([0, 128, 256], boundary_values=[]),
        lnc=BoundedRange([1, 2], boundary_values=[]),
        coverage="pairs",
        filter=filter_invalid_tests,
        enable_automatic_boundary_tests=True,
        enable_invalid_combination_tests=True,
    )
    def test_attn_tkg_sweep(
        self,
        test_manager: Orchestrator,
        batch_size: int,
        q_head: int,
        s_active: int,
        s_prior: int,
        s_prior_full_multiple: int,
        d_head: int,
        block_len: int,
        tp_k_prior: bool,
        strided_mm1: bool,
        use_pos_id: bool,
        fuse_rope: bool,
        out_in_sb: bool,
        k_out_in_sb: bool,
        qk_in_sb: bool,
        dtype: str,
        sink: bool,
        fp8_kv: bool,
        sliding_window: int,
        lnc: int,
        is_negative_test_case: bool,
    ):
        cfg = AttnTKGConfig(
            bs=batch_size,
            q_head=q_head,
            s_active=s_active,
            curr_sprior=s_prior,
            full_sprior=int(s_prior * s_prior_full_multiple),
            d_head=d_head,
            block_len=block_len,
            tp_k_prior=tp_k_prior,
            strided_mm1=strided_mm1,
            use_pos_id=use_pos_id,
            fuse_rope=fuse_rope,
            out_in_sb=out_in_sb,
            k_out_in_sb=k_out_in_sb,
            qk_in_sb=qk_in_sb,
        )

        compiler_args = CompilerArgs(logical_nc_config=lnc)
        try:
            run_attention_tkg_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                cfg=cfg,
                dtype=dtype,
                test_sink=sink,
                fp8_kv=fp8_kv,
                sliding_window=sliding_window,
                is_negative_test_case=is_negative_test_case,
            )
        except Exception:
            # Collecting sweep tests takes a long time.
            # During development, copy failing parametrizations into manual tests and disable sweep test collection.
            print(
                f"Failed test, manual test vector: {print_test_config(cfg, AttnTKGTestParams(dtype=dtype, test_sink=sink, fp8_kv=fp8_kv, sliding_window=sliding_window))}"
            )
            raise
