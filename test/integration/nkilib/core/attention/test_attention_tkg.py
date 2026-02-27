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
import enum
from copy import deepcopy
from functools import cache

try:
    from test.integration.nkilib.core.attention.test_attention_tkg_model_config import (
        attention_tkg_model_configs,
    )
except ImportError:
    attention_tkg_model_configs = []

from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import np_random_sample, np_random_sample_fp8
from test.integration.nkilib.utils.test_kernel_common import convert_to_torch
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.tensor_histogram import TensorHistogram
from test.utils.test_orchestrator import Orchestrator
from typing import Any, final

import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.core.attention.attention_tkg import (
    _ALLOWED_CONFIGURATIONS,
    _MAX_S_ACTIVE,
    AttnTKGConfig,
    TileConstants,
    attention_tkg,
    is_batch_sharded,
    is_s_prior_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
    uses_flash_attention,
)
from nkilib_src.nkilib.core.attention.attention_tkg_torch import attention_tkg_torch_ref
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil
from nkilib_src.nkilib.core.utils.logging import Logger, LogLevel
from typing_extensions import override

# the shape of q, k, v are interlinked, therefore
# we pass in the config as one object
ATTN_TKG_OPT = "attn_tkg"
BATCH_DIM_NAME = "bs"
Q_HEAD_DIM_NAME = "q_h"
SEQLEN_ACTIVE_DIM_NAME = "s_a"
SEQLEN_PRIOR_DIM_NAME = "s_p"
SEQLEN_PRIOR_FULL_DIM_NAME = "s_p_full"
D_HEAD_DIM_NAME = "d_h"
BLOCK_LEN_DIM_NAME = "blk_len"
COMBINED_BOOLEAN_DIM_NAME = "c_bool"
TP_K_PRIOR_DIM_NAME = "tp_k_prior"
USE_POS_ID_DIM_NAME = "use_pos_id"
FUSE_ROPE_DIM_NAME = "fuse_rope"
OUTS_IN_SB_DIM_NAME = "outs_in_sb"
QK_IN_SB_DIM_NAME = "qk_in_sb"
SINK_DIM_NAME = "sink"
STRIDED_MM1_DIM_NAME = "strided_mm1"
DTYPE_DIM_NAME = "dtype"
FP8_KV_DIM_NAME = "fp8_kv"

P_MAX = 128

DTYPE_TYPE_TO_INT = {nl.bfloat16: 0, nl.float32: 1}
DTYPE_INT_TO_TYPE = {0: nl.bfloat16, 1: nl.float32}
assert len(DTYPE_TYPE_TO_INT) == len(DTYPE_INT_TO_TYPE)
for k, v in DTYPE_TYPE_TO_INT.items():
    assert DTYPE_INT_TO_TYPE[v] == k


class DependentBooleanState(enum.IntFlag):
    FUSE_ROPE = 1 << 0
    QK_IN_SB = 1 << 1


ALLOWED_BOOLEAN_COMBINATIONS = [
    DependentBooleanState.FUSE_ROPE,
    DependentBooleanState.QK_IN_SB,
    DependentBooleanState.FUSE_ROPE | DependentBooleanState.QK_IN_SB,
]


# ----------------------------------------------------
# Configuration-based testing to avoid combinatorial explosion
# ----------------------------------------------------

# Self-attention configuration
SELF_ATTENTION_CONFIG = {
    "kernel_name": "AttentionMMSoftmaxMM",
    "tp_q": True,
    "tp_out": False,
    "softmax_scale": 0.125,
}

# Causal attention configuration
CAUSAL_ATTENTION_CONFIG = {
    "kernel_name": "CausalAttentionMMSoftmaxMMWithoutSwap",
    "tp_q": True,
    "tp_out": False,
    "softmax_scale": 1.0,
}

# LLM-optimized layout configuration
LLM_OPTIMIZED_CONFIG = {
    "kernel_name": "CausalAttentionMMSoftmaxMMWithoutSwap",
    "tp_q": False,
    "tp_out": True,
    "softmax_scale": 1.0,
}


def golden_attention_tkg_fwd(
    inp_np,
    cfg: AttnTKGConfig,
    is_block_kv,
    test_sink,
    dtype,
    lnc: int,
    attn_out_shape,
    k_out_shape,
    relative_tolerance,
    absolute_tolerance,
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
                (P_MAX, inp_np.get('active_blocks_table').shape[1] * resize_factor // P_MAX, cfg.bs),
                dtype=torch.int32,
            )
            DBG_TENSORS = DBG_TENSORS + (DBG_ACTIVE_TABLE,)

    out, k_out = attention_tkg_torch_ref[lnc](
        q=convert_to_torch(inp_np['q']),
        k_active=convert_to_torch(inp_np['k_active']),
        v_active=convert_to_torch(inp_np['v_active']),
        k_prior=convert_to_torch(inp_np['k_prior']),
        v_prior=convert_to_torch(inp_np['v_prior']),
        mask=convert_to_torch(inp_np['active_mask']) if 'active_mask' in inp_np else convert_to_torch(inp_np['mask']),
        out=out,
        cfg=cfg,
        sbm=None,
        inv_freqs=convert_to_torch(inp_np.get('inv_freqs')) if cfg.fuse_rope else None,
        rope_pos_ids=convert_to_torch(inp_np.get('rope_pos_ids')) if (cfg.fuse_rope or cfg.use_pos_id) else None,
        sink=convert_to_torch(inp_np.get('sink')) if test_sink else None,
        active_blocks_table=convert_to_torch(inp_np.get('active_blocks_table')) if is_block_kv else None,
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

        # Determine if FA is used (affects which debug tensors are valid)
        s_prior_n_prgs = lnc if is_s_prior_sharded(cfg, P_MAX) else 1
        s_prior_per_prg = cfg.curr_sprior // s_prior_n_prgs
        use_fa, _ = uses_flash_attention(s_prior_per_prg)

        # DBG_QK is skipped when strided_mm1 + FA (complex K column remapping)
        if cfg.strided_mm1 and use_fa:
            golds['DBG_QK'] = skip_comparator('DBG_QK', DBG_QK_NP.shape, DBG_QK_NP.dtype)
        else:
            golds['DBG_QK'] = dt.static_cast(DBG_QK_NP, DBG_QK_NP.dtype)

        # DBG_QK_MAX and DBG_QK_EXP are skipped when FA (running quantities)
        if use_fa:
            golds['DBG_QK_MAX'] = skip_comparator('DBG_QK_MAX', DBG_QK_MAX_NP.shape, DBG_QK_MAX_NP.dtype)
            golds['DBG_QK_EXP'] = skip_comparator('DBG_QK_EXP', DBG_QK_EXP_NP.shape, DBG_QK_EXP_NP.dtype)
        else:
            golds['DBG_QK_MAX'] = custom_debug_tensor_comparator(DBG_QK_MAX_NP, 'DBG_QK_MAX')
            golds['DBG_QK_EXP'] = dt.static_cast(DBG_QK_EXP_NP, dtype)

        golds['DBG_EXP_SUM'] = custom_debug_tensor_comparator(DBG_EXP_SUM_NP, 'DBG_EXP_SUM')
        if is_block_kv:
            DBG_ACTIVE_TABLE_NP = DBG_ACTIVE_TABLE.numpy()
            golds['DBG_ACTIVE_TABLE'] = dt.static_cast(DBG_ACTIVE_TABLE_NP, np.int32)

    return golds


@cache  # Cache results to ensure the same table is returned for the same parametrization (instead of redoing random selections)
def generate_active_blocks_array(batch: int, S_ctx: int, block_len: int, assumed_num_cache_blocks: int):
    # Make sure blocks are unique across batches,
    # otherwise cache update can write to same block and cause mismatch against golden function.
    table_shape = (batch, S_ctx // block_len)
    arr = (
        np.random.choice(assumed_num_cache_blocks, size=np.prod(table_shape), replace=False)
        .reshape(table_shape)
        .astype(np.uint32)
    )
    return arr


def gen_deterministic_active_block_table(batch, S_ctx, S_tkg, pos_id, block_len, assumed_num_cache_blocks):
    # The active blocks table needs to be used for position_ids and the table initialization itself.
    arr = generate_active_blocks_array(batch, S_ctx, block_len, assumed_num_cache_blocks).copy()

    assumed_actual_ctx_lens = pos_id.flatten()
    for b in range(batch):
        # Number of blocks covering active cache and active token.
        num_actual_active_blks = div_ceil(assumed_actual_ctx_lens[b] + S_tkg, block_len)
        arr[b, num_actual_active_blks:] = 0
    return arr


def get_bqs_tile_parameters(p_max: int, cfg: AttnTKGConfig, lnc: int):
    bs_n_prgs = lnc if is_batch_sharded(cfg, p_max) else 1
    bqs_size = cfg.bs // bs_n_prgs * cfg.q_head * cfg.s_active
    bqs_tiles = div_ceil(bqs_size, p_max)
    bqs_tile_size = p_max if bqs_tiles > 1 else bqs_size

    return bs_n_prgs, bqs_size, bqs_tiles, bqs_tile_size


def get_debug_tensor_shapes(p_max: int, cfg: AttnTKGConfig, lnc: int):
    qk_shape = (p_max, cfg.curr_sprior // p_max, cfg.bs * cfg.q_head * cfg.s_active)
    bs_n_prgs, _, bqs_tiles, bqs_tile_size = get_bqs_tile_parameters(p_max, cfg, lnc)
    reduced_shape = (bs_n_prgs, bqs_tiles, bqs_tile_size)

    return qk_shape, reduced_shape


def numpy_gen_attention_active_mask(shape, transposed=False):
    out = np.tril(np.ones(shape, dtype=np.bool_), k=0)
    return out.transpose((3, 0, 1, 2)) if transposed else out


def numpy_gen_attention_cache_mask(cache_len, batch, num_heads, S_tkg, S_ctx, lnc, block_len, unify_for_cascaded=False):
    '''
    Generates attention cache mask based on cache lengths, with shape (batch * num_heads * S_tkg, S_ctx).
    This function is used as golden for gen_cache_mask_for_attention_tkg_kernel unit test, also used
    to generate the cache mask input for attention kernel. So we can match the two kernels are compatible.
    '''

    def reshape_output(mask):
        return mask.reshape(batch, num_heads, S_tkg, S_ctx)

    out = reshape_output((np.arange(S_ctx)[None, :] < cache_len[:, None]).repeat(num_heads * S_tkg, axis=0))

    if unify_for_cascaded:
        mask_active = np.tril(np.ones((batch, num_heads, S_tkg, S_tkg), dtype=np.bool_), k=0)
        out[:, :, :, S_ctx - S_tkg :] = mask_active

    if block_len > 0:
        reduced_block_len, _ = resize_cache_block_len_for_attention_tkg_kernel(
            S_ctx // block_len, block_len, lnc, P_MAX
        )
        out = out.reshape((num_heads * S_tkg, batch, lnc, -1, P_MAX, reduced_block_len)).swapaxes(-1, -2)

    out = reshape_output(out)
    # Transpose for cascaded attention input, i.e. (S_ctx, batch, num_heads, S_tkg)
    return out.transpose((3, 0, 1, 2)) if unify_for_cascaded else out


def build_attention_tkg_input(
    cfg: AttnTKGConfig,
    q_shape,
    dtype,
    k_active_shape,
    k_prior_shape,
    v_prior_shape,
    test_sink,
    pos_id,
    attn_out_shape,
    k_out_shape,
    lnc,
    DBG,
    fp8_kv=False,
):
    is_block_kv = cfg.block_len != 0

    if is_block_kv:
        _, resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
            cfg.curr_sprior // cfg.block_len, cfg.block_len, lnc, P_MAX
        )

    random_gen = np_random_sample()

    if fp8_kv:
        kv_random_gen = np_random_sample_fp8()
        kv_dtype = nl.float8_e4m3
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
        active_mask = numpy_gen_attention_active_mask((cfg.bs, cfg.q_head, cfg.s_active, cfg.s_active), transposed=True)
    else:
        active_mask = numpy_gen_attention_cache_mask(
            pos_id, cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, lnc, cfg.block_len, unify_for_cascaded=True
        )
    active_mask = active_mask.astype(np.uint8)

    inv_freqs = np.random.random(size=(cfg.d_head // 2, 1)).astype(np.float32) if cfg.fuse_rope else None
    rope_pos_ids = (
        np.broadcast_to(pos_id, (cfg.bs, cfg.s_active)).astype(np.float32)
        if (cfg.fuse_rope or cfg.use_pos_id)
        else None
    )
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
        "sink": sink,
        "active_blocks_table": active_blocks_table,
        "attn_out_shape": attn_out_shape,
        "k_out_shape": k_out_shape,
        "cfg": cfg,
        "dtype": dtype,
        "lnc": lnc,
        "DBG": DBG,
    }


def attention_tkg_wrapper(
    q,
    k_active,
    v_active,
    k_prior,
    v_prior,
    mask,
    inv_freqs,
    rope_pos_ids,
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
    # Empirically found SBUF requirements (64KB base, 2MB to not overflow)
    estimated_sbm_usage = max((cfg.bs * cfg.q_head * cfg.s_active), 128) * max((cfg.curr_sprior // 8), 1024)
    sbm = SbufManager(
        0, min(estimated_sbm_usage, nl.tile_size.total_available_sbuf_size), Logger("SBM", LogLevel.DEBUG)
    )
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
                cfg.curr_sprior // cfg.block_len, cfg.block_len, lnc, TC.p_max
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


def get_configuration(s_prior: int, configuration_index: int) -> tuple[int, tuple[int, int]]:
    min_sprior = 1
    found_cfg: list[tuple[int, int]] = _ALLOWED_CONFIGURATIONS[0][1]
    for cur_s_prior, cfg in _ALLOWED_CONFIGURATIONS:
        found_cfg = cfg
        if s_prior <= cur_s_prior:
            break
        min_sprior = cur_s_prior

    assert -len(found_cfg) <= configuration_index < len(found_cfg)
    return min_sprior, found_cfg[configuration_index]


# Load model metadata for matching test configs to model names
attention_tkg_metadata_list = load_model_configs("test_attention_tkg")

# fmt: off
attention_tkg_manual_test_grid = [
    # batch, q_head, s_active, s_prior, s_prior_full, d_head, block_len, tp_k_prior, use_pos_id, fuse_rope, outsInSB, qkvInSB, test_sink, dtype
    # #################### K Transpose ####################
    # # NKI Llama 76B Target slot 1
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5,  1024, 10240, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  1536, 10240, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  3072, 10240, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  4608, 10240, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  7168, 10240, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  8704, 10240, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],

    # NKI Llama 76B Target slot2
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5, 14336, 24576, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 18432, 24576, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 22528, 24576, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 24576, 24576, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    # NKI Llama 3.1 70B
    [  4, 1,   5, 16384, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 16384, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    # NKI Llama 3.1 405B
    [  4, 2,   7, 16384, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],

    # NKI Smaller bucket sizes/prior sequence length
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 2,   7,   256, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,   512, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,  1024, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,  2048, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,  4096, 16384, 128,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],

    # NKI d_head = 64
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 2,   7,   512, 16384,  64,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 3,   7, 16384, 16384,  64,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240,  64,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 18432, 24576,  64,   0,  True,  True,  True, False, False, False, True, nl.bfloat16, False],

    #################### No K Transpose ####################
    # NKI Llama 76B Target slot 1
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5,  1024, 10240, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  1536, 10240, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  3072, 10240, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  4608, 10240, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  7168, 10240, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5,  8704, 10240, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],

    # NKI Llama 76B Target slot2
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5, 14336, 24576, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 18432, 24576, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 22528, 24576, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 24576, 24576, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    # NKI Llama 3.1 70B
    [  4, 1,   5, 16384, 16384, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    # NKI Llama 3.1 405B
    [  4, 2,   7, 16384, 16384, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],

    # NKI Smaller bucket sizes/prior sequence length
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 2,   7,   256, 16384, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,   512, 16384, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,  1024, 16384, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,  2048, 16384, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 2,   7,  4096, 16384, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],

    # NKI d_head = 64
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb, sink,  s_mm1,      dtype, fp8_kv
    [  4, 2,   7, 512,   16384,  64,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 3,   7, 16384, 16384,  64,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240,  64,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
    [  4, 1,   5, 18432, 24576,  64,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],

    # Input / output in SBUF
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5, 10240, 10240, 128,   0, False, False, False, False,  True, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240, 128,   0, False,  True,  True, False,  True, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240, 128,   0, False, False, False,  True,  True, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240, 128,   0, False,  True, False,  True,  True, False, True, nl.bfloat16, False],
    [  4, 1,   5, 10240, 10240, 128,   0, False, False,  True,  True,  True, False, True, nl.bfloat16, False],

    # Sink
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  8, 8,   1,  1024, 10240,  64,   0,  True,  True,  True, False, False,  True, True, nl.bfloat16, False],
    [128, 1,   1,  1024, 10240,  64,   0,  True,  True,  True, False, False,  True, True, nl.bfloat16, False],
    [  4, 4,   5,  1024, 10240,  64,   0,  True,  True,  True, False, False,  True, True, nl.bfloat16, False],
    # Sink with fp32
    [  8, 8,   1,  1024, 10240,  64,   0,  True,  True,  True, False, False,  True, True, nl.float32, False],
    [128, 1,   1,  1024, 10240,  64,   0,  True,  True,  True, False, False,  True, True, nl.float32, False],
    [  4, 4,   5,  1024, 10240,  64,   0,  True,  True,  True, False, False,  True, True, nl.float32, False],

    # LNC-shard on batch for s_prior < 256.
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  8, 4,   4,   128,   256,  64,   0,  True, False, False,  True,  True,  True, True, nl.bfloat16, False],
    [ 32, 8,   1,   128,   256,  64,   0,  True, False, False,  True,  True,  True, True, nl.bfloat16, False],
    [128, 1,   1,   128,   256,  64,   0, False, False, False, False,  True,  True, True, nl.bfloat16, False],
    [  1, 1,   1,   128,   256,  64,   0,  True, False, False,  True,  True, False, True, nl.bfloat16, False],
    [ 16, 8,   1,  8192,  8192,  64,   0, False, False, False,  True,  True, False, True, nl.bfloat16, False],

    # Liger large batch (b*n*s>128), SWA and non-SWA
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 8,   7,  3072, 10240,  64,   0,  True, False, False,  True,  True,  True, True, nl.bfloat16, False],
    [  4, 8,   7,   128, 10240,  64,   0,  True, False, False,  True,  True,  True, True, nl.bfloat16, False],
    [ 64, 8,   1,  2048,  2048, 128,   0, False, False, False,  True,  True, False, True, nl.bfloat16, False],

    # Block KV Tests
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5,  8192,  8192, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False], # S shard, resize factor =1
    [  4, 1,   5, 10240, 10240, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False], # S shard, resize factor >1
    [  4, 8,   7,  4096, 10240,  64,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False], # B shard, resize factor =1
    [  4, 8,   7,  3072, 10240,  64,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False], # B shard, resize factor >1
    [ 64, 8,   1,  2048,  2048, 128,  16,  True, False, False,  True,  True, False, False, nl.bfloat16, False], # Large B

    #################### FP8 KV Cache Tests ####################
    # FP8 KV requires: fuse_rope=False, qk_in_sb=True
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5,  1024, 10240, 128,   0, False, False, False, False,  True, False, False, nl.bfloat16, True],
    [  4, 1,   5,  8192,  8192, 128,   0, False, False, False, False,  True, False, False, nl.bfloat16, True],
    [  4, 2,   7,  4096, 16384, 128,   0, False, False, False,  True,  True, False, False, nl.bfloat16, True],
    [  4, 1,   5,  1024, 10240, 128,   0,  True, False, False, False,  True, False, False, nl.bfloat16, True],
    [  4, 1,   5,  8192,  8192, 128,   0,  True, False, False, False,  True, False, False, nl.bfloat16, True],
    [  4, 2,   7,  4096, 16384, 128,   0,  True, False, False,  True,  True, False, False, nl.bfloat16, True],
    # FP8 KV with Block KV
    [  4, 1,   5,  8192,  8192, 128,  16,  True, False, False,  True,  True, False, False, nl.bfloat16, True],
    [  4, 8,   7,  4096, 10240,  64,  16,  True, False, False,  True,  True, False, False, nl.bfloat16, True],
    # FP8 KV with Block KV large batch
    [128, 1,   1,  2048,  2048, 128,  16,  True, False, False,  True,  True, False, False, nl.bfloat16, True],

    # Block KV Tests with gen_mask_tkg mask generation (use_pos_id=True)
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5,  8192,  8192, 128,  16,  True, True, False,  True,  True,  True, False, nl.bfloat16, False], # S shard, resize factor =1
    [  4, 1,   5, 10240, 10240, 128,  16,  True, True, False,  True,  True,  True, False, nl.bfloat16, False], # S shard, resize factor >1
    [  4, 8,   7,  4096, 10240,  64,  16,  True, True, False,  True,  True,  True, False, nl.bfloat16, False], # B shard, resize factor =1
    [  4, 8,   7,  3072, 10240,  64,  16,  True, True, False,  True,  True,  True, False, nl.bfloat16, False], # B shard, resize factor >1
    [ 64, 8,   1,  2048,  2048, 128,  16,  True, True, False,  True,  True, False, False, nl.bfloat16, False], # Large B

    # Disable strided_mm1 tests
    #  b, q, s_a,   s_p, s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype, fp8_kv
    [  4, 1,   5,  1024, 10240, 128,   0,  True,  True,  True, False, False, False, False, nl.bfloat16, False],
    [  4, 1,   5,  1536, 10240, 128,   0,  True,  True,  True, False, False, False, False, nl.bfloat16, False],
    [  4,10,   5,  5376,  5376,  64,   0, False, False, False, False,  True, False, False, nl.bfloat16, False],
]
# fmt: on


def create_attention_tkg_test_config(test_grid, test_type: str = "manual") -> RangeTestConfig:
    """Create RangeTestConfig from test grid with test_type dimension."""
    test_cases = []
    for test_case in test_grid:
        test_case_params = {
            BATCH_DIM_NAME: test_case[0],
            Q_HEAD_DIM_NAME: test_case[1],
            SEQLEN_ACTIVE_DIM_NAME: test_case[2],
            SEQLEN_PRIOR_DIM_NAME: test_case[3],
            SEQLEN_PRIOR_FULL_DIM_NAME: test_case[4],
            D_HEAD_DIM_NAME: test_case[5],
            BLOCK_LEN_DIM_NAME: test_case[6],
            TP_K_PRIOR_DIM_NAME: 1 if test_case[7] else 0,
            USE_POS_ID_DIM_NAME: 1 if test_case[8] else 0,
            FUSE_ROPE_DIM_NAME: 1 if test_case[9] else 0,
            OUTS_IN_SB_DIM_NAME: 1 if test_case[10] else 0,
            QK_IN_SB_DIM_NAME: 1 if test_case[11] else 0,
            SINK_DIM_NAME: 1 if test_case[12] else 0,
            STRIDED_MM1_DIM_NAME: 1 if test_case[13] else 0,
            DTYPE_DIM_NAME: DTYPE_TYPE_TO_INT[test_case[14]],
            FP8_KV_DIM_NAME: 1 if test_case[15] else 0,
        }

        test_cases.append({ATTN_TKG_OPT: test_case_params})

    return RangeTestConfig(
        additional_params={},
        global_tensor_configs=TensorRangeConfig(
            tensor_configs={
                ATTN_TKG_OPT: TensorConfig(
                    [
                        DimensionRangeConfig(name=BATCH_DIM_NAME),
                        DimensionRangeConfig(name=Q_HEAD_DIM_NAME),
                        DimensionRangeConfig(name=SEQLEN_ACTIVE_DIM_NAME),
                        DimensionRangeConfig(name=SEQLEN_PRIOR_DIM_NAME),
                        DimensionRangeConfig(name=SEQLEN_PRIOR_FULL_DIM_NAME),
                        DimensionRangeConfig(name=D_HEAD_DIM_NAME),
                        DimensionRangeConfig(name=BLOCK_LEN_DIM_NAME),
                        DimensionRangeConfig(name=TP_K_PRIOR_DIM_NAME),
                        DimensionRangeConfig(name=USE_POS_ID_DIM_NAME),
                        DimensionRangeConfig(name=FUSE_ROPE_DIM_NAME),
                        DimensionRangeConfig(name=OUTS_IN_SB_DIM_NAME),
                        DimensionRangeConfig(name=QK_IN_SB_DIM_NAME),
                        DimensionRangeConfig(name=SINK_DIM_NAME),
                        DimensionRangeConfig(name=STRIDED_MM1_DIM_NAME),
                        DimensionRangeConfig(name=DTYPE_DIM_NAME),
                        DimensionRangeConfig(name=FP8_KV_DIM_NAME),
                    ]
                )
            },
            monotonic_step_size=1,
            custom_generators=[RangeManualGeneratorStrategy(test_cases=test_cases, test_type=test_type)],
        ),
    )


def attention_tkg_unit_config():
    """Create combined config with manual and model test cases."""
    manual_config = create_attention_tkg_test_config(attention_tkg_manual_test_grid, test_type="manual")
    model_config = create_attention_tkg_test_config(attention_tkg_model_configs, test_type=MODEL_TEST_TYPE)
    manual_config.global_tensor_configs.custom_generators.extend(model_config.global_tensor_configs.custom_generators)
    return manual_config


@pytest_test_metadata(
    name="Attention TKG",
    pytest_marks=["attention", "tkg"],
    tags=["model"],
)
@final
class TestAttentionTkgKernel:
    def run_range_attention_tkg_test(
        self,
        test_manager: Orchestrator,
        test_options: RangeTestCase,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        cfg: AttnTKGConfig,
        DBG: bool,
    ):
        np.random.seed(42)

        attn_opt = test_options.tensors[ATTN_TKG_OPT]
        lnc_degree = compiler_args.logical_nc_config

        def validate_bools_and_cfg(attn_opt: dict[str, int], cfg: AttnTKGConfig):
            bool_dims = {}
            for dim_name in [
                TP_K_PRIOR_DIM_NAME,
                USE_POS_ID_DIM_NAME,
                FUSE_ROPE_DIM_NAME,
                QK_IN_SB_DIM_NAME,
                OUTS_IN_SB_DIM_NAME,
                SINK_DIM_NAME,
                STRIDED_MM1_DIM_NAME,
                FP8_KV_DIM_NAME,
            ]:
                if attn_opt[dim_name] not in [0, 1]:
                    pytest.skip("Got non binary value for bool parameter.")
                bool_dims[dim_name] = attn_opt[dim_name] == 1

            cfg.tp_k_prior = bool_dims[TP_K_PRIOR_DIM_NAME]
            cfg.use_pos_id = bool_dims[USE_POS_ID_DIM_NAME]
            cfg.fuse_rope = bool_dims[FUSE_ROPE_DIM_NAME]
            cfg.out_in_sb = bool_dims[OUTS_IN_SB_DIM_NAME]
            cfg.k_out_in_sb = bool_dims[OUTS_IN_SB_DIM_NAME]
            cfg.qk_in_sb = bool_dims[QK_IN_SB_DIM_NAME]
            cfg.strided_mm1 = bool_dims[STRIDED_MM1_DIM_NAME]

        validate_bools_and_cfg(attn_opt, cfg)
        test_sink = attn_opt[SINK_DIM_NAME] == 1

        # handle dtype dim
        if attn_opt[DTYPE_DIM_NAME] not in [0, 1]:
            pytest.skip("Got invalid value for dtype parameter.")

        dtype = DTYPE_INT_TO_TYPE[attn_opt[DTYPE_DIM_NAME]]
        fp8_kv = attn_opt.get(FP8_KV_DIM_NAME, 0) == 1

        has_cache_buffer = False

        is_block_kv = cfg.block_len != 0
        if is_block_kv:
            assert not cfg.strided_mm1, "Unexpected strided_mm1 with block kv"

        # Model configs should never be marked as negative test cases
        is_model_config = test_options.test_type == MODEL_TEST_TYPE
        is_negative_test_case = False if is_model_config else test_options.is_negative_test_case

        if (not cfg.fuse_rope) and (not cfg.qk_in_sb) and not is_model_config:
            is_negative_test_case = True

        # FP8 KV configuration constraints - kernel validates these
        if fp8_kv and (cfg.fuse_rope or not cfg.qk_in_sb) and not is_model_config:
            is_negative_test_case = True

        if is_negative_test_case and cfg.curr_sprior < 256:
            pytest.skip("Boundary test for s_prior < 256 isn't supported")

        if (
            not is_model_config
            and cfg.fuse_rope
            and (is_batch_sharded(cfg, P_MAX) or cfg.bs * cfg.q_head * cfg.s_active > P_MAX)
        ):
            # Fuse RoPE is not supported when bqs doesn't fit on partition dimension
            is_negative_test_case = True

        # Shapes that differ base on config
        q_shape = (
            (cfg.d_head, cfg.bs * cfg.q_head * cfg.s_active)
            if cfg.qk_in_sb
            else (cfg.bs, cfg.q_head, cfg.s_active, cfg.d_head)
        )
        k_active_shape = (cfg.d_head, cfg.bs * cfg.s_active) if cfg.qk_in_sb else (cfg.bs, 1, cfg.s_active, cfg.d_head)
        resize_factor = None
        if is_block_kv:
            assert cfg.tp_k_prior
            assert cfg.curr_sprior % cfg.block_len == 0
            assumed_num_cache_blocks = cfg.bs * cfg.curr_sprior // cfg.block_len
            k_prior_shape = v_prior_shape = (assumed_num_cache_blocks, cfg.block_len, cfg.d_head)
            _, resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
                cfg.curr_sprior // cfg.block_len, cfg.block_len, lnc_degree, P_MAX
            )
        else:
            prior_batch = cfg.bs + 1 if has_cache_buffer else cfg.bs
            k_prior_shape = (
                (prior_batch, 1, cfg.full_sprior, cfg.d_head)
                if cfg.tp_k_prior
                else (prior_batch, 1, cfg.d_head, cfg.full_sprior)
            )
            v_prior_shape = (prior_batch, 1, cfg.full_sprior, cfg.d_head)

        attn_out_shape = (
            (cfg.d_head, cfg.bs * cfg.q_head * cfg.s_active)
            if cfg.out_in_sb
            else (cfg.bs, cfg.q_head, cfg.d_head, cfg.s_active)
        )
        k_out_shape = (cfg.d_head, cfg.bs * cfg.s_active) if cfg.k_out_in_sb else (cfg.bs, 1, cfg.d_head, cfg.s_active)

        # Generate pos_id outside of tensor_gen so it stays the same for all inputs
        pos_id = ((np.arange(cfg.bs) * 3 + (cfg.curr_sprior // 4 * 3)) % (cfg.curr_sprior - cfg.s_active))[
            :, np.newaxis
        ]

        with assert_negative_test_case(is_negative_test_case):
            if fp8_kv:
                relative_tolerance, absolute_tolerance = 3e-2, 3e-2
            else:
                relative_tolerance, absolute_tolerance = 1e-2, 1e-5

            kernel_input = build_attention_tkg_input(
                cfg=cfg,
                q_shape=q_shape,
                dtype=dtype,
                k_active_shape=k_active_shape,
                k_prior_shape=k_prior_shape,
                v_prior_shape=v_prior_shape,
                test_sink=test_sink,
                pos_id=pos_id,
                attn_out_shape=attn_out_shape,
                k_out_shape=k_out_shape,
                lnc=lnc_degree,
                DBG=DBG,
                fp8_kv=fp8_kv,
            )

            # Set up dummy arrays for negative test case
            golden_output = {
                'golden_out': np.zeros(attn_out_shape, dtype),
            }
            if cfg.fuse_rope:
                golden_output['golden_k_out'] = np.zeros(k_out_shape, dtype)
            if DBG:
                qk_shape, reduced_shape = get_debug_tensor_shapes(P_MAX, cfg, lnc_degree)
                golden_output['DBG_QK'] = np.zeros(qk_shape, dtype=np.float32)
                golden_output['DBG_QK_MAX'] = np.zeros(reduced_shape, dtype=np.float32)
                golden_output['DBG_QK_EXP'] = np.zeros(qk_shape, dtype)
                golden_output['DBG_EXP_SUM'] = np.zeros(reduced_shape, dtype=np.float32)
                if is_block_kv:
                    golden_output['DBG_ACTIVE_TABLE'] = np.zeros(
                        (P_MAX, cfg.bs * cfg.curr_sprior * resize_factor // cfg.block_len // P_MAX),
                        dtype=np.int32,
                    )

            golden_generator = lambda: golden_attention_tkg_fwd(
                inp_np=kernel_input,
                cfg=cfg,
                is_block_kv=is_block_kv,
                test_sink=test_sink,
                dtype=dtype,
                lnc=lnc_degree,
                attn_out_shape=attn_out_shape,
                k_out_shape=k_out_shape,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                DBG=DBG,
            )

            test_manager.execute(
                KernelArgs(
                    kernel_func=attention_tkg_wrapper,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            output_ndarray=golden_output,
                            lazy_golden_generator=golden_generator,
                        ),
                        relative_accuracy=relative_tolerance,
                        absolute_accuracy=absolute_tolerance,
                    ),
                    inference_args=TKG_INFERENCE_ARGS,
                )
            )

    def run_sweep_attention_tkg_test(
        self,
        test_manager: Orchestrator,
        test_options: RangeTestCase,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
    ):
        attn_opt = test_options.tensors[ATTN_TKG_OPT]
        batch = attn_opt[BATCH_DIM_NAME]
        q_head = attn_opt[Q_HEAD_DIM_NAME]
        s_active = attn_opt[SEQLEN_ACTIVE_DIM_NAME]
        if SEQLEN_PRIOR_DIM_NAME in attn_opt and SEQLEN_PRIOR_FULL_DIM_NAME in attn_opt:
            s_prior = attn_opt[SEQLEN_PRIOR_DIM_NAME]
            s_prior_full = attn_opt[SEQLEN_PRIOR_FULL_DIM_NAME]
        elif SEQLEN_PRIOR_DIM_NAME in attn_opt:
            s_prior = attn_opt[SEQLEN_PRIOR_DIM_NAME]
            s_prior_full = s_prior
        elif SEQLEN_PRIOR_FULL_DIM_NAME in attn_opt:
            s_prior_full = attn_opt[SEQLEN_PRIOR_FULL_DIM_NAME]
            s_prior = 256 * np.random.randint(max(s_prior_full // 256, 1))
        else:
            assert False, "s_prior or s_prior_full must be provided"

        # HACK: Reshape in build input fails if s_prior is odd, so we make it not divisible by 256 but even
        if test_options.is_negative_test_case and s_prior % 2 != 0:
            s_prior += 1 if s_prior % 256 < 128 else -1  # Move it further away from a multiple of 256

        d_head = attn_opt[D_HEAD_DIM_NAME]

        # handle bool dims
        if COMBINED_BOOLEAN_DIM_NAME in attn_opt:
            combined_index = attn_opt[COMBINED_BOOLEAN_DIM_NAME]
        if combined_index < 0 or combined_index >= len(ALLOWED_BOOLEAN_COMBINATIONS):
            pytest.skip("Got invalid value for combined boolean parameter.")
        combined_bool = ALLOWED_BOOLEAN_COMBINATIONS[combined_index]
        attn_opt[FUSE_ROPE_DIM_NAME] = 1 if combined_bool & DependentBooleanState.FUSE_ROPE else 0
        attn_opt[QK_IN_SB_DIM_NAME] = 1 if combined_bool & DependentBooleanState.QK_IN_SB else 0

        block_len = 0

        cfg = AttnTKGConfig(
            batch,
            q_head,
            s_active,
            s_prior,
            s_prior_full,
            d_head,
            block_len,
        )

        self.run_range_attention_tkg_test(test_manager, test_options, compiler_args, collector, cfg, DBG=True)

    @staticmethod
    def sweep_s_prior_config(
        max_s_prior: int, configuration_index: int, monotonic_step_percent: int = 5, random_sample_size: int = 30
    ):
        min_s_prior, (B, QS) = get_configuration(max_s_prior, configuration_index)
        S_A = min(QS, _MAX_S_ACTIVE)
        Q = QS // S_A

        def multiple_of_to_match_monotonic(min_val, max_val, base_multiple_of):
            result_multiple_of = (
                (max_val - min_val) * monotonic_step_percent // 100 // base_multiple_of * base_multiple_of
            )
            new_min = max_val - 100 // monotonic_step_percent * result_multiple_of
            if result_multiple_of == 0:
                result_multiple_of = None

            return result_multiple_of, new_min

        # Ensure batch size is even to avoid weird cases where we can't shard over batch
        bs_multiple_of, min_bs = multiple_of_to_match_monotonic(1, B, 2)
        s_prior_multiple_of, min_s_prior = multiple_of_to_match_monotonic(min_s_prior, max_s_prior, 256)

        tensor_configs = [
            DimensionRangeConfig(min=min_bs, max=B, multiple_of=bs_multiple_of, name=BATCH_DIM_NAME),
            DimensionRangeConfig(max=Q, name=Q_HEAD_DIM_NAME),
            DimensionRangeConfig(max=S_A, name=SEQLEN_ACTIVE_DIM_NAME),
            DimensionRangeConfig(
                min=min_s_prior, max=max_s_prior, multiple_of=s_prior_multiple_of, name=SEQLEN_PRIOR_DIM_NAME
            ),
            DimensionRangeConfig(
                max=128, multiple_of=64, name=D_HEAD_DIM_NAME
            ),  # This is a restriction of fuse_rope only, but simplifies logic
            # bool dims
            DimensionRangeConfig(min=0, max=1, name=TP_K_PRIOR_DIM_NAME),
            DimensionRangeConfig(min=0, max=1, name=USE_POS_ID_DIM_NAME),
            DimensionRangeConfig(min=0, max=len(ALLOWED_BOOLEAN_COMBINATIONS) - 1, name=COMBINED_BOOLEAN_DIM_NAME),
            DimensionRangeConfig(min=0, max=1, name=OUTS_IN_SB_DIM_NAME),
            DimensionRangeConfig(min=0, max=1, name=SINK_DIM_NAME),
            DimensionRangeConfig(min=0, max=1, name=STRIDED_MM1_DIM_NAME),
            # dtype dim (0,1 -> bfloat16, float32)
            DimensionRangeConfig(min=0, max=len(DTYPE_INT_TO_TYPE) - 1, name=DTYPE_DIM_NAME),
            # fp8 kv dim
            DimensionRangeConfig(min=0, max=1, name=FP8_KV_DIM_NAME),
        ]

        # Create test options using class-level constants
        # For static parameters set min=max
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    ATTN_TKG_OPT: TensorConfig(tensor_configs),
                },
                monotonic_step_percent=monotonic_step_percent,
                random_sample_size=random_sample_size,
                # custom_generators=[
                #   # RangeRandomGeneratorStrategy(random_sample_size),
                #   RangeMonotonicGeneratorStrategy(step_percent=monotonic_step_percent),
                # ]
            ),
        )

    @range_test_config(sweep_s_prior_config(8 * 1024, 0))
    @pytest.mark.parametrize("lnc_degree", [1, 2])
    def test_ranged_attn_tkg_nki_sweep_s_prior_8k(
        self, test_manager: Orchestrator, range_test_options: RangeTestCase, collector: IMetricsCollector, lnc_degree
    ):
        test_options = deepcopy(range_test_options)
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_sweep_attention_tkg_test(
            test_manager=test_manager,
            test_options=test_options,
            compiler_args=compiler_args,
            collector=collector,
        )

    @range_test_config(sweep_s_prior_config(256, 0, monotonic_step_percent=20))
    def test_ranged_attn_tkg_nki_sweep_s_prior_256(
        self, test_manager: Orchestrator, range_test_options: RangeTestCase, collector: IMetricsCollector
    ):
        test_options = deepcopy(range_test_options)
        compiler_args = CompilerArgs()
        self.run_sweep_attention_tkg_test(
            test_manager=test_manager,
            test_options=test_options,
            compiler_args=compiler_args,
            collector=collector,
        )

    @range_test_config(sweep_s_prior_config(32 * 1024, 1, monotonic_step_percent=20))
    def test_ranged_attn_tkg_nki_sweep_s_prior_32k(
        self, test_manager: Orchestrator, range_test_options: RangeTestCase, collector: IMetricsCollector
    ):
        test_options = deepcopy(range_test_options)
        compiler_args = CompilerArgs()
        self.run_sweep_attention_tkg_test(
            test_manager=test_manager,
            test_options=test_options,
            compiler_args=compiler_args,
            collector=collector,
        )

    def run_manual_attention_tkg_test(
        self,
        test_manager: Orchestrator,
        test_options: RangeTestCase,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
    ):
        attn_opt = test_options.tensors[ATTN_TKG_OPT]
        batch = attn_opt[BATCH_DIM_NAME]
        q_head = attn_opt[Q_HEAD_DIM_NAME]
        s_active = attn_opt[SEQLEN_ACTIVE_DIM_NAME]
        s_prior = attn_opt[SEQLEN_PRIOR_DIM_NAME]
        s_prior_full = attn_opt[SEQLEN_PRIOR_FULL_DIM_NAME]
        d_head = attn_opt[D_HEAD_DIM_NAME]
        block_len = attn_opt[BLOCK_LEN_DIM_NAME]

        cfg = AttnTKGConfig(
            batch,
            q_head,
            s_active,
            s_prior,
            s_prior_full,
            d_head,
            block_len,
        )

        self.run_range_attention_tkg_test(test_manager, test_options, compiler_args, collector, cfg, DBG=False)

    @pytest.mark.fast
    @range_test_config(attention_tkg_unit_config())
    @pytest.mark.parametrize("lnc_degree", [1, 2])
    def test_ranged_attn_tkg_nki_sweep_manual(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        request: pytest.FixtureRequest,
        lnc_degree: int,
    ):
        test_options = deepcopy(range_test_options)
        attn_opt = test_options.tensors[ATTN_TKG_OPT]

        # Apply xfail for model configs and add metadata dimensions
        if range_test_options.test_type == MODEL_TEST_TYPE:
            request.node.add_marker(pytest.mark.xfail(strict=False, reason="Model coverage test"))
            test_metadata_key = {
                "bs": attn_opt[BATCH_DIM_NAME],
                "q_h": attn_opt[Q_HEAD_DIM_NAME],
                "s_a": attn_opt[SEQLEN_ACTIVE_DIM_NAME],
                "s_p": attn_opt[SEQLEN_PRIOR_DIM_NAME],
                "s_p_full": attn_opt[SEQLEN_PRIOR_FULL_DIM_NAME],
                "d_h": attn_opt[D_HEAD_DIM_NAME],
            }
            collector.match_and_add_metadata_dimensions(test_metadata_key, attention_tkg_metadata_list)

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_manual_attention_tkg_test(
            test_manager=test_manager,
            test_options=test_options,
            compiler_args=compiler_args,
            collector=collector,
        )

    # fmt: off
    # Long sequence length tests (test up to 512k to keep test time reasonable)
    # Parameters: b, q, s_a, s_p, s_p_f, d_h, b_l, tp_k, p_id, fs_r, ot_sb, qk_sb, sink, s_mm1, dtype, fp8_kv
    TEST_GRID_LONG = [

        # Without block KV (b_l=0), b=4, q=1
        #  b, q, s_a,   s_p,    s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype,  fp8_kv
        [  4, 1,   5,   16640,   16640, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
        [  4, 1,   5,   20480,   20480, 128,   0, False,  True,  True, False, False, False, False,nl.bfloat16, False],
        [  4, 1,   5,   20480,   20480, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
        [  4, 1,   5,   82432,   82432, 128,   0, False,  True,  True, False, False, True,  True, nl.bfloat16, False],
        [  4, 1,   5,   82432,   82432, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
        [  4, 1,   5,   82432,   82432, 128,   0, False,  True,  False,False, True,  False, True, nl.bfloat16, True],
        [  4, 1,   5,  262144,  262144, 128,   0, True,   False, False,False, True,  False, True, nl.bfloat16, True],
        # The RoPE modulo operation (`_modulo` function) loses precision for very large position IDs due to float32 catastrophic cancellation.
        # This leads to golden_k_out failing validation with ~5% error while golden_out still passes.
        # [  4, 1,   5,  524288,  524288, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],
        # [  4, 1,   5, 1048576, 1048576, 128,   0, False,  True,  True, False, False, False, True, nl.bfloat16, False],

        # Without block KV (b_l=0), b=8, q=8 - includes 8192 (no FA) as baseline
        #  b, q, s_a,   s_p,    s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype
        [  8, 8,   5,   16384,   16384, 128,   0, False, False, False,  True,  True, False, False, nl.bfloat16, False],
        [  8, 8,   5,   16384,   16384, 128,   0, False, False, False,  True,  True, False, True,  nl.bfloat16, True],
        [  8, 8,   5,   16640,   16640, 128,   0, False, False, False,  True,  True, False, False, nl.bfloat16, False],
        [  8, 8,   5,   82432,   82432, 128,   0, False, False, False,  True,  True, False, True,  nl.bfloat16, False],
        [  8, 8,   5,   82432,   82432, 128,   0, False, False, False,  True,  True, True,  True,  nl.bfloat16, False],
        [  8, 8,   5,  262144,  262144, 128,   0, True,  True,  False,   True,  True, False,  True, nl.bfloat16, False],


        # With block KV (b_l=16), b=4, q=1
        #  b, q, s_a,   s_p,    s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype
        [  4, 1,   5,   16640,   16640, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False],
        [  4, 1,   5,   20480,   20480, 128,  16,  True, False, False,  True,  True, True, False, nl.bfloat16, False],
        [  4, 1,   5,   82432,   82432, 128,  16,  True, False, False,  False, True,  True, False, nl.bfloat16, False],
        [  4, 1,   5,   82432,   82432, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, True],
        [  4, 1,   5,   82432,   82432, 128,  16,  True, True,  True,   True,  True,  True, False, nl.bfloat16, False],
        [  4, 1,   5,  524288,  524288, 128,  16,  True, True,  False,  True,  True,  True, False, nl.bfloat16, True],

        # With block KV (b_l=16), b=8, q=8
        #  b, q, s_a,   s_p,    s_p_f, d_h, b_l,  tp_k,  p_id,  fs_r, ot_sb, qk_sb,  sink, s_mm1,      dtype
        [  8, 8,   5,   16640,   16640, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, True],
        [  8, 8,   5,   20480,   20480, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False],
        [  8, 8,   5,   82432,   82432, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, True],
        [  8, 8,   5,   82432,   82432, 128,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False],
        [  8, 8,   5,  262144,  262144, 128,  16,  True, True,  False,   True,  True,  True, False, nl.bfloat16,False],

        # few extra tests with s_p < s_p_f
        [  4, 8,   7,  102400, 131072,  64,  16,  True, False, False,  True,  True,  True, False, nl.bfloat16, False],
        [  4, 8,   7,   128,   131072,  64,   0,  True, False, False,  True,  True,  True, True,  nl.bfloat16, False],
        [  4, 2,   7,  102400, 131072, 128,   0,  True,  True,  True, False, False, False, True,  nl.bfloat16, False],
    ]
    # fmt: on

    @range_test_config(create_attention_tkg_test_config(TEST_GRID_LONG))
    def test_ranged_attn_tkg_nki_sweep_manual_long(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        request: pytest.FixtureRequest,
    ):
        test_options = deepcopy(range_test_options)
        attn_opt = test_options.tensors[ATTN_TKG_OPT]

        compiler_args = CompilerArgs()
        self.run_manual_attention_tkg_test(
            test_manager=test_manager,
            test_options=test_options,
            compiler_args=compiler_args,
            collector=collector,
        )
