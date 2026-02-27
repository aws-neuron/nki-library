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
from dataclasses import dataclass, fields
from functools import cache

import nki.language as nl
import numpy as np
import torch
from nkilib_src.nkilib.core.attention.attention_tkg_utils import AttnTKGConfig, is_batch_sharded
from nkilib_src.nkilib.core.utils.kernel_helpers import div_ceil


@dataclass
class AttnTKGTestParams:
    dtype: str = nl.bfloat16
    test_sink: bool = False
    fp8_kv: bool = False
    sliding_window: int = 0


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
        )
    else:
        raise ValueError(f"Unknown type for cfg_repr: {type(cfg)}")


def generate_pow_range(start: int, end: int, start_step: int = 1, pow_lag: int = 1, base: int = 2):
    def get_new_step(cur, step):
        while cur >= step * base * pow_lag:
            step *= base
        return step

    cur, step = start, get_new_step(start, start_step)

    out = []
    while cur <= end:
        out.append(cur)
        step = get_new_step(cur, step)
        cur += step
    return out


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


def build_active_attention_mask(batch, num_heads, s_active, transposed=False):
    """Generate causal active mask (lower triangular)."""
    mask = torch.tril(torch.ones(s_active, s_active, dtype=torch.float32))
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, num_heads, -1, -1)
    if transposed:
        mask = mask.permute(3, 0, 1, 2)
    return mask


def build_swa_positions(pos_id, bs, s_active, sliding_window, cache_len, block_len=0):
    """Compute per-query start_pos_ids and rope_pos_ids for SWA.

    Args:
        pos_id: [bs, 1] array of base position IDs.
        bs: Batch size.
        s_active: Number of active tokens.
        sliding_window: SWA window size.
        cache_len: Total cache length (s_prior) for circular buffer modular arithmetic.
        block_len: Block length (0 for flat cache).

    Returns:
        start_pos_ids: [bs, s_active] — inclusive lower bound per query.
        rope_pos_ids: [bs, s_active] — position IDs with per-s_active increments.
    """
    rope_pos_ids = np.zeros((bs, s_active), dtype=np.float32)
    start_pos_ids = np.zeros((bs, s_active), dtype=np.float32)

    for b in range(bs):
        for i in range(s_active):
            pos = pos_id[b, 0] + i
            rope_pos_ids[b, i] = pos

            if block_len > 0:
                # Block KV uses a linear block table (not circular), so clamp to 0
                start_pos_ids[b, i] = max(0, pos - sliding_window + 1)
            else:
                # Flat KV uses a circular buffer, so wrap around with modular arithmetic
                start_pos_ids[b, i] = (pos - sliding_window + 1) % cache_len

    return start_pos_ids, rope_pos_ids


def print_test_config(attn_cfg, test_cfg):
    def process_attn_cfg_str(cfg: AttnTKGConfig):
        in_str = repr(cfg)
        in_str = in_str.replace("bs=", "", 1)
        in_str = in_str.replace("q_head=", "", 1)
        in_str = in_str.replace("s_active=", "", 1)
        in_str = in_str.replace("curr_sprior=", "", 1)
        in_str = in_str.replace("full_sprior=", "", 1)
        in_str = in_str.replace("d_head=", "", 1)
        in_str = in_str.replace("block_len=", "", 1)
        for field in fields(cfg):
            if field.type == bool:
                val = getattr(cfg, field.name)
                if val == field.default:
                    in_str = in_str.replace(f", {field.name}={val}", "", 1)
        return in_str

    def procces_test_cfg_str(cfg: AttnTKGTestParams):
        in_str = repr(cfg)
        for field in fields(cfg):
            val = getattr(cfg, field.name)
            if val == field.default:
                if isinstance(val, str):
                    in_str = in_str.replace(f"{field.name}='{val}'", "", 1)
                else:
                    in_str = in_str.replace(f"{field.name}={val}", "", 1)
        in_str = in_str.replace("'float32'", "nl.float32")
        in_str = in_str.replace(" ,", " ")
        in_str = ' '.join(in_str.split())
        in_str = in_str.replace("(, ", "(")
        in_str = in_str.replace(", )", ")")
        return in_str

    out = "["
    out += process_attn_cfg_str(attn_cfg)
    out += ", "
    out += procces_test_cfg_str(test_cfg)

    out += "]"
    return out
