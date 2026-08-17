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
import math
from dataclasses import dataclass, fields
from functools import cache

import nki.language as nl
import numpy as np
import torch
from nkilib_src.nkilib.core.attention.attention_tkg import INACTIVE_BLOCK_IDX
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
        .astype(np.int32)
    )
    return arr


def gen_deterministic_active_block_table(batch, S_ctx, S_tkg, pos_id, block_len, assumed_num_cache_blocks):
    # The active blocks table needs to be used for position_ids and the table initialization itself.
    assumed_actual_ctx_lens = pos_id.flatten()
    # Number of blocks covering active cache and active token.
    active_blocks_per_batch = [div_ceil(assumed_actual_ctx_lens[b] + S_tkg, block_len) for b in range(batch)]
    table_shape = (batch, S_ctx // block_len)

    if assumed_num_cache_blocks < np.prod(table_shape):
        # This generator does not model prefix caching: every populated logical table entry gets a
        # distinct physical block. Immutable prior blocks could be shared across batches, but blocks
        # receiving active-token updates must remain private or use copy-on-write.
        num_active_blocks = sum(active_blocks_per_batch)
        assert num_active_blocks <= assumed_num_cache_blocks, (
            f"Physical cache has {assumed_num_cache_blocks} blocks, but active table entries require "
            f"{num_active_blocks} unique blocks"
        )
        arr = np.full(table_shape, INACTIVE_BLOCK_IDX, dtype=np.int32)
        active_block_indices = np.random.choice(assumed_num_cache_blocks, size=num_active_blocks, replace=False).astype(
            np.int32
        )
        offset = 0
        for b, num_active_blocks_for_batch in enumerate(active_blocks_per_batch):
            arr[b, :num_active_blocks_for_batch] = active_block_indices[offset : offset + num_active_blocks_for_batch]
            offset += num_active_blocks_for_batch
        return arr

    # Preserve the existing generated tables when the physical pool covers the logical table.
    arr = generate_active_blocks_array(batch, S_ctx, block_len, assumed_num_cache_blocks).copy()
    for b in range(batch):
        arr[b, active_blocks_per_batch[b] :] = INACTIVE_BLOCK_IDX
    return arr


def get_bqs_tile_parameters(p_max: int, cfg: AttnTKGConfig, lnc: int):
    bs_n_prgs = lnc if is_batch_sharded(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, p_max, cfg.fuse_rope) else 1
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
        cache_len: Total cache length (s_prior).
        block_len: Block length (0 for flat cache).

    Returns:
        start_pos_ids: [bs, s_active] — inclusive lower bound per query.
        rope_pos_ids: [bs, s_active] — position IDs with per-s_active increments.
    """
    rope_pos_ids = np.zeros((bs, s_active), dtype=np.float32)
    start_pos_ids = np.zeros((bs, s_active), dtype=np.float32)

    # Flat KV circular buffer uses [0, cache_len - s_active) as usable slots.
    # pos_ids (rope_pos_ids) is the raw position, which equals the modded slot
    # index when cache_lens <= cache_len - s_active (the current test constraint).
    circular_size = cache_len - s_active

    for b in range(bs):
        for i in range(s_active):
            pos = pos_id[b, 0] + i
            rope_pos_ids[b, i] = pos

            if block_len > 0:
                # Block KV uses a linear block table (not circular), so clamp to 0
                start_pos_ids[b, i] = max(0, pos - sliding_window + 1)
            else:
                # Flat KV circular buffer: mod by usable region size
                start_pos_ids[b, i] = (pos - sliding_window + 1) % circular_size

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
            if field.type is bool:
                val = getattr(cfg, field.name)
                if val == field.default:
                    in_str = in_str.replace(f", {field.name}={val}", "", 1)
        return in_str

    def proccess_test_cfg_str(cfg: AttnTKGTestParams):
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
    out += proccess_test_cfg_str(test_cfg)

    out += "]"
    return out


def generate_cache_lens(bs, s_ctx, s_tkg, mode="normal", mean_frac=0.5, stddev_frac=0.1):
    """Generate per-batch cache lengths (pos_ids) for TKG attention tests.

    Args:
        bs: Batch size.
        s_ctx: Context length (s_prior).
        s_tkg: Number of active tokens.
        mode: "normal" for values drawn from a normal distribution around a mean,
                  simulating realistic traffic routing patterns.
              "spread" for evenly spreading batches from 1 to s_prior (shuffled).
                  E.g. bs=2 → [max_pos/3, 2*max_pos/3],
                       bs=4 → [max_pos/5, 2*max_pos/5, 3*max_pos/5, 4*max_pos/5].
              "random" for uniform random in [1, s_ctx - s_tkg].
              "three_quarter" for the legacy formula that places tokens at ~75% of s_ctx.
        mean_frac: Fraction of max_pos for the mean (used by "normal" mode). Default 0.5.
        stddev_frac: Fraction of max_pos for the standard deviation (used by "normal" mode).
            Default 0.1.

    Returns:
        cache_lens: np.ndarray of shape (bs, 1) with values in [1, s_ctx - s_tkg].
    """
    max_pos = s_ctx - s_tkg
    assert max_pos > 0, f"s_ctx ({s_ctx}) must be > s_tkg ({s_tkg})"

    if mode == "three_quarter":
        # Note that this legacy mode was relevant for performance profiling when we had power
        # of 2 bucketing along with DMA skipping.
        # Now we use "spread" to get good coverage since we do not have bucketing.
        cache_lens = ((np.arange(bs) * 3 + (s_ctx // 4 * 3)) % (max_pos + 1))[:, np.newaxis]
        cache_lens = np.clip(cache_lens, 1, max_pos)
    elif mode == "random":
        cache_lens = np.random.randint(1, max_pos + 1, size=(bs, 1))
    elif mode == "spread":
        # Evenly space cache lens across [1, max_pos]: batch i gets ceil(i+1/(bs+1) * max_pos).
        cache_lens = np.array([math.ceil(max_pos * (i + 1) / (bs + 1)) for i in range(bs)])
        np.random.shuffle(cache_lens)
        cache_lens = cache_lens[:, np.newaxis]
    elif mode == "normal":
        mean = max_pos * mean_frac
        stddev = max_pos * stddev_frac
        cache_lens = np.round(np.random.normal(mean, stddev, size=(bs, 1))).astype(int)
        cache_lens = np.clip(cache_lens, 1, max_pos)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'spread', 'normal', 'random', or 'three_quarter'.")

    assert cache_lens.min() >= 1
    assert cache_lens.max() <= max_pos
    return cache_lens
