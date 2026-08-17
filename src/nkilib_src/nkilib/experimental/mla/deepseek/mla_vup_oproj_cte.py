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

"""Standalone MX V-up + MX output projection (CTE).

KERNEL B of the split DeepSeek-V3.2 sparse-MLA forward. Reads the latent attention
output out_attn_hbm[B=1, S, H*L] (from kernel A, mla_sparse_attention_cte_kernel), V-ups each
head's latent with W_uv (MX fp8x4), then projects the H*d_v activation with W_o (MX) into
out_hbm[B=1, S, HID].

S-sharded (each core does its own queries over the full HID). Budget-aware o_proj weight
residency: the full W_o is loaded once when it fits SBUF (TIER 1), else K-slab-streamed
double-buffered (TIER 3). Consumes the cross-kernel MX 4-pack column layout kernel A
writes into out_attn (each (k512, sub)'s latent groups contiguous in HBM), so the latent
transpose is a contiguous swdge dma_transpose off the Tensor Engine.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.qkv.qkv_cte import _get_psum_bank_size
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .mla_common_cte import (
    _H_PACK,
    _MM1_TILE,
    _NUM_HW_PSUM_BANKS,
    _P_MAX,
    _dma_transpose_latent_for_mx,
    _load_mx_weights,
    _load_mx_weights_k_slab,
    _new_sbm,
    _transpose_preswizzled_for_mx,
)
from .mla_validate_params import _validate_mla_vupmx_oproj_inputs

_OPROJ_S_TILE = 128
_OPROJ_HID_TILE = 1024
_OPROJ_K_SLAB_512 = 8
_OPROJ_STAGE_512 = 8


@nki.jit
def mla_vupmx_oproj_cte_kernel(
    out_attn_hbm: nl.NkiTensor,
    wuv_qtz_hbm: nl.NkiTensor,
    wuv_scale_hbm: nl.NkiTensor,
    wo_qtz_hbm: nl.NkiTensor,
    wo_scale_hbm: nl.NkiTensor,
    compact_scales: bool = True,
) -> nl.NkiTensor:
    """Standalone MX V-up + MX output projection (S-sharded across cores).

    KERNEL B of the split DeepSeek-V3.2 sparse-MLA forward. Reads the latent attention
    output from kernel A (mla_sparse_attention_cte_kernel), V-ups each head's latent with W_uv
    (MX fp8x4), then projects the H*d_v activation with W_o (MX) into out_hbm[B=1, S, HID].
    Each core projects its own queries over the full HID. Intended for Context Encoding with
    DeepSeek-V3.2 dims (L == 512 kv_lora_rank, d_v == 128, H a multiple of 4 up to 128);
    requires B == 1 and S divisible by the number of cores. Budget-aware o_proj weight
    residency: full W_o loaded once when it fits SBUF, else K-slab-streamed double-buffered.

    Dimensions:
        B: Batch size (must be 1)
        S: Sequence length (this rank's S-shard)
        H: Number of attention heads (a multiple of 4)
        L: Latent (kv_lora_rank) dimension (fixed at 512 = P_MAX * 4)
        d_v: Per-head V dimension (must be 128 = P_MAX)
        HID: Output projection (hidden) dimension
        Hdv: H * d_v (the o_proj contraction)

    Args:
        out_attn_hbm (nl.NkiTensor): [B, S, H*L] bf16 latent attention output from kernel A,
            carrying the cross-kernel MX 4-pack column layout.
        wuv_qtz_hbm (nl.NkiTensor): [H*L // 4, d_v] fp8x4 packed MX V-up weight.
        wuv_scale_hbm (nl.NkiTensor): [H*L // 128, ceil(d_v / 128)] uint8 compact block-128 scales.
        wo_qtz_hbm (nl.NkiTensor): [H*d_v // 4, HID] fp8x4 packed MX o_proj weight.
        wo_scale_hbm (nl.NkiTensor): [H*d_v // 128, ceil(HID / 128)] uint8 compact block-128 scales.

    Returns:
        out (nl.NkiTensor): [B, S, HID] bf16 output projection result.

    Notes:
        - H and L are recovered from tensor shapes (L fixed at 512): wuv_qtz_hbm is
          [H*L // 4, d_v]. Do NOT pass H as a runtime scalar — the framework materializes
          it as an HBM tensor, not a trace-time int, so shape math derived from it is garbage.
        - Consumes the cross-kernel MX 4-pack column layout kernel A writes into out_attn
          (each (k512, sub)'s latent groups contiguous in HBM), so the latent transpose is a
          contiguous swdge dma_transpose off the Tensor Engine.

    Pseudocode:
        attn = out_attn[q]                       # de-permute 4-pack layout -> natural
        for h in range(H):
            attn_v[..., h, :] = attn[..., h, :] @ W_uv[h]   # MX V-up per head
        out[q] = attn_v.reshape(S, H*d_v) @ W_o             # MX o_proj
        return out
    """
    _validate_mla_vupmx_oproj_inputs(
        out_attn_hbm, wuv_qtz_hbm, wuv_scale_hbm, wo_qtz_hbm, wo_scale_hbm, compact_scales=compact_scales
    )

    B, S, HL = out_attn_hbm.shape
    L = _P_MAX * _H_PACK  # 512
    H = HL // L
    d_v = wuv_qtz_hbm.shape[1]
    HID = wo_qtz_hbm.shape[1]
    Hdv = H * d_v

    _, n_prgs, prg_id = get_verified_program_sharding_info("mla_vupmx_oproj_cte_kernel", (0, 1))
    kernel_assert(S % n_prgs == 0, f"S={S} must be divisible by n_prgs={n_prgs}")
    s_per_core = S // n_prgs
    s_start = prg_id * s_per_core

    out_hbm = nl.ndarray((B, S, HID), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sbm = _new_sbm("sparse_mla_vupmx_oproj")
    # S-sharded, full HID.
    _vupmx_oproj_stage(
        out_attn_hbm,
        wuv_qtz_hbm,
        wuv_scale_hbm,
        wo_qtz_hbm,
        wo_scale_hbm,
        out_hbm,
        H,
        L,
        sbm,
        s_start,
        s_per_core,
        0,
        HID,
        compact_scales=compact_scales,
    )
    return out_hbm


def _vupmx_oproj_stage(
    out_attn_hbm,
    wuv_qtz_hbm,
    wuv_scale_hbm,
    wo_qtz_hbm,
    wo_scale_hbm,
    out_hbm,
    H,
    L,
    sbm,
    s_start,
    s_per_core,
    hid_start,
    hid_per_core,
    s_tile_cap=_OPROJ_S_TILE,
    compact_scales: bool = True,
):
    """MX V-up + MX o_proj.

    Reads attn_latent from out_attn_hbm[B, S, H*L] for queries [s_start, s_start+
    s_per_core), V-ups to attn_v[s_tile, H*d_v], projects with W_o, and writes
    out_hbm[B, S, HID] for the HID slice [hid_start, hid_start+hid_per_core). Single
    source of truth for the standalone o_proj kernel (and reusable by a fused parent).

    Sharding: S is sharded via (s_start, s_per_core); the o_proj OUTPUT (HID) is
    sharded via (hid_start, hid_per_core) — kernel A is S-sharded and kernel B may
    instead shard HID (each core does all S, its HID slice), which only changes these
    two pairs. The full H*d_v contraction is always local.
    """
    HID = wo_qtz_hbm.shape[1]
    d_v = wuv_qtz_hbm.shape[1]
    HL = H * L
    Hdv = H * d_v

    kernel_assert(
        hid_per_core % _MM1_TILE == 0,
        f"_vupmx_oproj_stage: hid_per_core ({hid_per_core}) must be a multiple of {_MM1_TILE}",
    )
    # d_v == P_MAX and H % H_PACK == 0 so each swizzled 512-tile spans exactly H_PACK heads
    # (the 4-pack sub axis crosses head boundaries; see the Phase-V1 write AP comment).
    kernel_assert(d_v == _P_MAX, f"_vupmx_oproj_stage: pre-swizzled transpose requires d_v == {_P_MAX}, got {d_v}")
    # H = per-rank head count. MX o_proj tiles H*d_v in 512-wide blocks of H_PACK=4 heads, so
    # H must be a multiple of H_PACK (>= 4); fewer heads/rank (tp=64 -> 2) is unsupported.
    kernel_assert(0 < H <= _P_MAX, f"_vupmx_oproj_stage: H (heads/rank) must be in (0, {_P_MAX}], got {H}")
    kernel_assert(
        H % _H_PACK == 0,
        f"_vupmx_oproj_stage: H (heads/rank) must be a multiple of {_H_PACK} (>= {_H_PACK} heads); "
        f"MX o_proj tiles {_H_PACK} heads per 512-block. got H={H}",
    )
    oproj_s_tile = min(s_tile_cap, s_per_core)
    kernel_assert(
        s_per_core % oproj_s_tile == 0, f"_vupmx_oproj_stage: s_per_core must be a multiple of {oproj_s_tile}"
    )
    num_s_batches = s_per_core // oproj_s_tile
    """
    Two s-tile widths, mirroring the qkv stage's ``s_tile_sz`` / ``s_tile_pad`` split:

      * ``oproj_s_tile``     - the REAL rows this s-batch owns. Used for every HBM access
                               (latent read, output write) and for stepping s-batches, so a
                               shard never reads/writes outside its own slice.
      * ``oproj_s_tile_pad`` - the PADDED compute width. SBUF/PSUM buffers, transposes,
                               quantize and both matmuls run on this; the extra columns are
                               don't-care.

    Padding is needed because the V-up nc_matmul_mx puts the s-tile on its stationary FREE
    axis: a free dim of 1 (s_per_core == 1, e.g. S=2 over 2 cores) fails MLIR verification.
    Rounding UP to a multiple of _S_TILE_PAD_MULT also keeps the XBar dma_transpose output
    32B-aligned, so the tiny-tile latent load stays on the fast DMA-transpose path.
    """
    _S_TILE_PAD_MULT = 16  # 32B / bf16: XBar transpose output alignment
    oproj_s_tile_pad = div_ceil(oproj_s_tile, _S_TILE_PAD_MULT) * _S_TILE_PAD_MULT

    # HID PSUM-tile: only one oproj_hid_tile slice's PSUM is live at a time (Phase B is
    # K-slab-outer), so shrink it until num_n_tiles + the H_PACK transpose banks fit 8.
    oproj_hid_tile = _OPROJ_HID_TILE
    while hid_per_core % oproj_hid_tile != 0 or _H_PACK + (oproj_hid_tile // _MM1_TILE) > _NUM_HW_PSUM_BANKS:
        oproj_hid_tile -= _MM1_TILE
    kernel_assert(
        oproj_hid_tile >= _MM1_TILE, f"_vupmx_oproj_stage: cannot fit PSUM banks for hid_per_core={hid_per_core}"
    )

    num_mx_k_tiles = Hdv // (_P_MAX * _H_PACK)
    """
    KNOWN LIMIT: multi-s-batch (s_per_core > oproj_s_tile) is only correct when
    num_mx_k_tiles == 1 (H=4, Hdv=512, the validated head-sharded long-seq path). With
    num_mx_k_tiles > 1 the per-s_batch loop mis-handles the activation -> NaN. Fail early.
    """
    kernel_assert(
        num_s_batches == 1 or num_mx_k_tiles == 1,
        f"_vupmx_oproj_stage: multi-s-batch (s_per_core={s_per_core} > oproj_s_tile={oproj_s_tile}, "
        f"num_s_batches={num_s_batches}) is only supported when num_mx_k_tiles==1 (H={_H_PACK}, "
        f"Hdv=512); got H={H} -> num_mx_k_tiles={num_mx_k_tiles} (KNOWN-BROKEN regime).",
    )
    num_hid_tiles = hid_per_core // oproj_hid_tile
    num_n_tiles = oproj_hid_tile // _MM1_TILE  # PSUM banks per HID-slice.
    kernel_assert(
        _H_PACK + num_n_tiles <= _NUM_HW_PSUM_BANKS,
        f"stage-2 PSUM banks {_H_PACK + num_n_tiles} exceed {_NUM_HW_PSUM_BANKS}",
    )
    PSUM_BANK_SIZE = _get_psum_bank_size()

    sbm.open_scope()

    stage_512 = min(_OPROJ_STAGE_512, num_mx_k_tiles)
    kernel_assert(
        num_mx_k_tiles % stage_512 == 0,
        f"num_mx_k_tiles ({num_mx_k_tiles}) must be a multiple of stage tiles ({stage_512})",
    )
    num_stage_chunks = num_mx_k_tiles // stage_512

    """
    K-slab depth for the contiguous full-width weight load: pick the largest slab dividing
    num_mx_k_tiles whose double-buffered [P_MAX, slab_512, hid_per_core] fp8x4+scale buffer
    fits the budget (both buffers + out_acc + in_qtz must coexist).
    """
    _SLAB_BUDGET_BYTES = 70 * 1024  # per single buffer (fp8x4 4B + uint8 1B per elem)
    slab_512 = min(_OPROJ_K_SLAB_512, num_mx_k_tiles)
    while slab_512 > 1 and (num_mx_k_tiles % slab_512 != 0 or slab_512 * hid_per_core * (4 + 1) > _SLAB_BUDGET_BYTES):
        slab_512 -= 1
    kernel_assert(
        num_mx_k_tiles % slab_512 == 0,
        f"num_mx_k_tiles ({num_mx_k_tiles}) must be a multiple of K-slab tiles ({slab_512})",
    )
    num_k_slabs = num_mx_k_tiles // slab_512

    # V-up MX. L=512 => exactly H MX 512-tiles in the latent: head h == k-tile h.
    num_lat_k_tiles = HL // (_P_MAX * _H_PACK)  # == H
    lat_stage_512 = min(_OPROJ_STAGE_512, num_lat_k_tiles)
    kernel_assert(
        num_lat_k_tiles % lat_stage_512 == 0,
        f"num_lat_k_tiles ({num_lat_k_tiles}) must be a multiple of {lat_stage_512}",
    )
    num_lat_stage_chunks = num_lat_k_tiles // lat_stage_512

    # o_proj activation (quantized H*d_v contraction), resident for ALL s_per_core across
    # BOTH phases (small, ~10KB/part). Lives in the outer scope so it survives Phase B.
    # Covers the PADDED write of the last s-batch: it writes oproj_s_tile_pad columns at
    # offset (num_s_batches-1)*oproj_s_tile, which can exceed s_per_core when padding is on.
    s_qtz_width = (num_s_batches - 1) * oproj_s_tile + oproj_s_tile_pad
    in_qtz = sbm.alloc_stack(
        (_P_MAX, num_mx_k_tiles, s_qtz_width), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name="oproj_in_qtz"
    )
    in_scale = sbm.alloc_stack(
        (_P_MAX, num_mx_k_tiles, s_qtz_width), dtype=nl.uint8, buffer=nl.sbuf, name="oproj_in_scale"
    )

    # V-up scratch in its OWN scope so attn_v/lat_*/wuv_* free before Phase B allocates the
    # weight slab + accumulator.
    sbm.open_scope()
    attn_v = sbm.alloc_stack((oproj_s_tile_pad, Hdv), dtype=nl.bfloat16, buffer=nl.sbuf, name="attn_v")
    # Double-buffer the V0 latent-load (load N+1 overlaps N's V-up + Phase A). Only pays off
    # with >1 s-batch (head-sharded long-seq, where H is small so these buffers are tiny).
    n_lat_buf = 2 if num_s_batches > 1 else 1
    lat_qtz, lat_scale, lat_in_t, lat_planes = [], [], [], []
    for buf_idx in range(n_lat_buf):
        lat_qtz.append(
            sbm.alloc_stack(
                (_P_MAX, num_lat_k_tiles, oproj_s_tile_pad),
                dtype=nl.float8_e4m3fn_x4,
                buffer=nl.sbuf,
                name=f"vup_lat_qtz_{buf_idx}",
            )
        )
        lat_scale.append(
            sbm.alloc_stack(
                (_P_MAX, num_lat_k_tiles, oproj_s_tile_pad),
                dtype=nl.uint8,
                buffer=nl.sbuf,
                name=f"vup_lat_scale_{buf_idx}",
            )
        )
        lat_in_t.append(
            sbm.alloc_stack(
                (_P_MAX, lat_stage_512, oproj_s_tile_pad * _H_PACK),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"vup_lat_t_{buf_idx}",
            )
        )
        # Contiguous sub-outer planes [P_MAX, lat_stage_512, H_PACK, s_tile] written by the
        # swdge dma_transpose, then vector-permuted into lat_in_t (sub innermost) for quantize.
        lat_planes.append(
            sbm.alloc_stack(
                (_P_MAX, lat_stage_512, _H_PACK, oproj_s_tile_pad),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"vup_lat_planes_{buf_idx}",
            )
        )
    in_t = sbm.alloc_stack(
        (_P_MAX, stage_512, oproj_s_tile_pad * _H_PACK), dtype=nl.bfloat16, buffer=nl.sbuf, name="oproj_t"
    )
    wuv_qtz, wuv_scale = _load_mx_weights(
        wuv_qtz_hbm, wuv_scale_hbm, HL, d_v, sbm, name="vup_wuv", compact_scales=compact_scales
    )

    # Transpose scratch: Phase-A transpose uses all 8 PSUM banks. Phase-V1 reuses bank H_PACK
    # for vup_psum, but V1 fully precedes Phase A per s-batch so they don't overlap in time.
    tpsum = []
    for bank_idx in range(_NUM_HW_PSUM_BANKS):
        tpsum.append(
            nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, bank_idx * PSUM_BANK_SIZE))
        )

    """
    Phase V (per s-batch): build the quantized o_proj activation for all S into in_qtz.
    V0 loads the latent via _dma_transpose_latent_for_mx — the latent->partition transpose
    runs on the DMA (swdge) engine straight from HBM (no nc_transpose), enabled by the
    cross-kernel 4-pack column order the attention stage writes (each (k512, sub)'s groups
    contiguous). Prologue loads s-batch 0; each iteration prefetches the next s-batch first.
    """
    attn_row_base = s_start * HL
    for chunk in range(num_lat_stage_chunks):
        k512_start = chunk * lat_stage_512
        _dma_transpose_latent_for_mx(
            out_attn_hbm,
            HL,
            oproj_s_tile_pad,
            lat_stage_512,
            lat_planes[0],
            lat_in_t[0],
            attn_row_base + k512_start * _MM1_TILE,
            sbm_scratch=sbm,
            tpsum_scratch=tpsum,
            s_valid=oproj_s_tile,
        )
        nisa.quantize_mx(
            src=lat_in_t[0][0:_P_MAX, 0:lat_stage_512, 0 : oproj_s_tile_pad * _H_PACK],
            dst=lat_qtz[0][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
            dst_scale=lat_scale[0][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
        )
    for s_batch in range(num_s_batches):
        buf = s_batch % n_lat_buf
        s_off = s_batch * oproj_s_tile
        # Prefetch next s-batch's latent into the alternate buffer (overlaps this V-up + A).
        if n_lat_buf > 1 and s_batch + 1 < num_s_batches:
            ld_buf = (s_batch + 1) % n_lat_buf
            ld_row_base = (s_start + (s_batch + 1) * oproj_s_tile) * HL
            for chunk in range(num_lat_stage_chunks):
                k512_start = chunk * lat_stage_512
                _dma_transpose_latent_for_mx(
                    out_attn_hbm,
                    HL,
                    oproj_s_tile_pad,
                    lat_stage_512,
                    lat_planes[ld_buf],
                    lat_in_t[ld_buf],
                    ld_row_base + k512_start * _MM1_TILE,
                    sbm_scratch=sbm,
                    tpsum_scratch=tpsum,
                    s_valid=oproj_s_tile,
                )
                nisa.quantize_mx(
                    src=lat_in_t[ld_buf][0:_P_MAX, 0:lat_stage_512, 0 : oproj_s_tile_pad * _H_PACK],
                    dst=lat_qtz[ld_buf][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
                    dst_scale=lat_scale[ld_buf][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
                )
        lat_qtz_b, lat_scale_b = lat_qtz[buf], lat_scale[buf]

        """
        Phase V1: per head, V-up MX-matmul -> vup_psum[s, d_v], evicted into attn_v at
        PRE-SWIZZLED Hdv column positions so Phase A can read contiguous [s,128] slices.
        The Hdv swizzle (_swizzle_mla_cols) crosses head boundaries (a 512-tile spans 4
        heads), so it can't be applied per-head; instead the per-head evict uses a strided
        write AP that lands head h's output at the swizzled positions a 512-tile expects:
        base = (h//H_PACK)*P_MAX + (h%H_PACK)*(P_MAX//H_PACK); free pattern
        [[1, P_MAX//H_PACK], [num_mx_k_tiles*P_MAX, H_PACK]]. Result: assembled attn_v ==
        _swizzle_mla_cols(natural), so in_qtz is bit-identical to the natural path (ref unchanged).
        """
        sub_stride = num_mx_k_tiles * _P_MAX
        for h in range(H):
            vup_psum = nl.ndarray(
                (oproj_s_tile_pad, d_v), dtype=nl.bfloat16, buffer=nl.psum, address=(0, _H_PACK * PSUM_BANK_SIZE)
            )
            nisa.nc_matmul_mx(
                dst=vup_psum[0:oproj_s_tile_pad, 0:d_v],
                stationary=lat_qtz_b[0:_P_MAX, h, nl.ds(0, oproj_s_tile_pad)],
                moving=wuv_qtz[0:_P_MAX, h, nl.ds(0, d_v)],
                stationary_scale=lat_scale_b[0:_P_MAX, h, nl.ds(0, oproj_s_tile_pad)],
                moving_scale=wuv_scale[0:_P_MAX, h, nl.ds(0, d_v)],
            )
            base = (h // _H_PACK) * _P_MAX + (h % _H_PACK) * (_P_MAX // _H_PACK)
            nisa.tensor_scalar(
                attn_v.ap(
                    pattern=[[Hdv, oproj_s_tile_pad], [1, _P_MAX // _H_PACK], [sub_stride, _H_PACK]],
                    offset=base,
                ),
                vup_psum,
                op0=nl.multiply,
                operand0=1.0,
                engine=nisa.engine.scalar,
            )

        """
        Phase A: pre-swizzled-transpose + quantize attn_v -> in_qtz[:, :, s_off:] (this
        s-batch's slice of the resident, all-S quantized activation). attn_v is now in
        swizzled Hdv column order, so the contiguous-read 8-bank transpose applies.
        """
        for chunk in range(num_stage_chunks):
            k512_start = chunk * stage_512
            _transpose_preswizzled_for_mx(
                attn_v, oproj_s_tile_pad, num_mx_k_tiles, stage_512, in_t, tpsum, base_col_512=k512_start
            )
            nisa.quantize_mx(
                src=in_t[0:_P_MAX, 0:stage_512, 0 : oproj_s_tile_pad * _H_PACK],
                dst=in_qtz[0:_P_MAX, nl.ds(k512_start, stage_512), nl.ds(s_off, oproj_s_tile_pad)],
                dst_scale=in_scale[0:_P_MAX, nl.ds(k512_start, stage_512), nl.ds(s_off, oproj_s_tile_pad)],
            )

    sbm.close_scope()  # free V-up scratch (attn_v, lat_*, wuv_*, in_t) before Phase B.

    """
    ---- Phase B (K-slab outer, SBUF-accumulated): per s-batch, accumulate the H*d_v
    contraction across K-slabs into out_acc. Each slab loads the full hid_per_core weight
    width contiguously (high MBU), one matmul per HID PSUM-slice, one PSUM slice live at a time.
    """
    sbm.open_scope()
    """
    Weight residency (budget-aware). TIER 1: if the full W_o fits free SBUF, load it ONCE
    and reuse across s-batches — kills the per-s_batch reload that made o_proj DMA-bound at
    large s_per_core (streamed W_o num_s_batches times). TIER 3 (W_o too big, e.g. 128-head
    Hdv=16384 ~1.1MB/part): per-s_batch double-buffered K-slab streaming (slab N+1 DMA
    overlaps slab N compute). fp8x4 = 4B/elem + uint8 scale = 1B/elem, per partition.
    """
    wo_full_bytes = num_mx_k_tiles * hid_per_core * (4 + 1)
    budget = int(0.95 * sbm.get_free_space())
    weights_resident = wo_full_bytes <= budget
    slab_w, slab_scale = [], []
    if weights_resident:
        # Single resident buffer holding ALL K-tiles (loaded once below, before s_batch loop).
        slab_w.append(
            sbm.alloc_stack(
                (_P_MAX, num_mx_k_tiles, hid_per_core), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name="oproj_w_full"
            )
        )
        slab_scale.append(
            sbm.alloc_stack(
                (_P_MAX, num_mx_k_tiles, hid_per_core), dtype=nl.uint8, buffer=nl.sbuf, name="oproj_scale_full"
            )
        )
    else:
        # Weight slab spans the FULL hid_per_core width (contiguous HBM read), DOUBLE-BUFFERED
        # so slab N+1's weight DMA overlaps slab N's matmul+accumulate.
        for buf_idx in range(2):
            slab_w.append(
                sbm.alloc_stack(
                    (_P_MAX, slab_512, hid_per_core),
                    dtype=nl.float8_e4m3fn_x4,
                    buffer=nl.sbuf,
                    name=f"oproj_slab_w_{buf_idx}",
                )
            )
            slab_scale.append(
                sbm.alloc_stack(
                    (_P_MAX, slab_512, hid_per_core), dtype=nl.uint8, buffer=nl.sbuf, name=f"oproj_slab_scale_{buf_idx}"
                )
            )
    """
    bf16 accumulator: the within-slab K-contraction accumulates in the fp32 PSUM
    register (independent of the bank's declared dtype), so out_acc only sums the few
    per-slab partials (num_k_slabs = Hdv/512/8, e.g. 4 for full DeepSeek). A handful of
    bf16 adds of already-fp32-summed partials stays within tolerance, and keeping the
    accumulator bf16 removes the bf16<->fp32 casts on every slab eviction and the drain.
    """
    out_acc = sbm.alloc_stack((oproj_s_tile_pad, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name="oproj_out_acc")
    out_sb = sbm.alloc_stack((oproj_s_tile_pad, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name="oproj_out")
    """
    PSUM accumulators DOUBLE-BUFFERED (banks H_PACK..7) so the matmul of the next
    (slab,hid) tile can proceed while the SBUF add of the previous tile still reads its
    bank — breaks the matmul->add->matmul serialization. n_psum_buf pairs cycle.
    """
    n_psum_banks = _NUM_HW_PSUM_BANKS - _H_PACK
    n_psum_buf = max(1, n_psum_banks // num_n_tiles)
    psum_bufs = []
    for psum_buf_idx in range(n_psum_buf):
        row = []
        for i_n in range(num_n_tiles):
            row.append(
                nl.ndarray(
                    (_P_MAX, _MM1_TILE),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, (_H_PACK + psum_buf_idx * num_n_tiles + i_n) * PSUM_BANK_SIZE),
                )
            )
        psum_bufs.append(row)
    # TIER 1: load the FULL W_o (all K-tiles) ONCE, before the s_batch loop, so every
    # s-batch reuses the resident weights (no per-s_batch reload).
    if weights_resident:
        _load_mx_weights_k_slab(
            wo_qtz_hbm,
            wo_scale_hbm,
            slab_w[0],
            slab_scale[0],
            in_dim_full=Hdv,
            out_dim=hid_per_core,
            k_tile_start=0,
            k_tile_count=num_mx_k_tiles,
            sbm=sbm,
            name="oproj_w_full_load",
            full_out_dim=HID,
            out_col_offset=hid_start,
            weights_dge_mode=dge_mode.hwdge,
            compact_scales=compact_scales,
        )
    for s_batch in range(num_s_batches):
        s_off = s_batch * oproj_s_tile
        pbuf = 0  # ping-pong index across (slab,hid,i_n) tiles
        # TIER 3 prologue: prefetch slab 0's weight into buffer 0 (resident path skips this).
        if not weights_resident:
            _load_mx_weights_k_slab(
                wo_qtz_hbm,
                wo_scale_hbm,
                slab_w[0],
                slab_scale[0],
                in_dim_full=Hdv,
                out_dim=hid_per_core,
                k_tile_start=0,
                k_tile_count=slab_512,
                sbm=sbm,
                name=f"oproj_slab_{s_batch}_0",
                full_out_dim=HID,
                out_col_offset=hid_start,
                weights_dge_mode=dge_mode.hwdge,
                compact_scales=compact_scales,
            )
        for slab_id in range(num_k_slabs):
            wb = slab_id % 2
            # TIER 3: prefetch the NEXT slab's weight into the alternate buffer so its DMA
            # overlaps this slab's matmul+accumulate below. (resident path: weights already in.)
            if not weights_resident and slab_id + 1 < num_k_slabs:
                _load_mx_weights_k_slab(
                    wo_qtz_hbm,
                    wo_scale_hbm,
                    slab_w[(slab_id + 1) % 2],
                    slab_scale[(slab_id + 1) % 2],
                    in_dim_full=Hdv,
                    out_dim=hid_per_core,
                    k_tile_start=(slab_id + 1) * slab_512,
                    k_tile_count=slab_512,
                    sbm=sbm,
                    name=f"oproj_slab_{s_batch}_{slab_id + 1}",
                    full_out_dim=HID,
                    out_col_offset=hid_start,
                    weights_dge_mode=dge_mode.hwdge,
                    compact_scales=compact_scales,
                )
            for hid_idx in range(num_hid_tiles):
                hid_off = hid_idx * oproj_hid_tile
                for i_n in range(num_n_tiles):
                    n_sz = min(_MM1_TILE, oproj_hid_tile - i_n * _MM1_TILE)
                    n_base = hid_off + i_n * _MM1_TILE
                    ps = psum_bufs[pbuf][i_n]
                    for k_local in range(slab_512):
                        k_global = slab_id * slab_512 + k_local
                        # Resident path: index the single full buffer at the GLOBAL k-tile;
                        # streamed path: index the active double-buffer at the LOCAL k-tile.
                        w_buf = slab_w[0] if weights_resident else slab_w[wb]
                        sc_buf = slab_scale[0] if weights_resident else slab_scale[wb]
                        w_k = k_global if weights_resident else k_local
                        nisa.nc_matmul_mx(
                            dst=ps[0:oproj_s_tile_pad, 0:n_sz],
                            stationary=in_qtz[0:_P_MAX, k_global, nl.ds(s_off, oproj_s_tile_pad)],
                            moving=w_buf[0:_P_MAX, w_k, nl.ds(n_base, n_sz)],
                            stationary_scale=in_scale[0:_P_MAX, k_global, nl.ds(s_off, oproj_s_tile_pad)],
                            moving_scale=sc_buf[0:_P_MAX, w_k, nl.ds(n_base, n_sz)],
                            accumulate=(k_local > 0),
                        )
                    # First slab initializes out_acc (copy, no read-dep); later slabs add.
                    if slab_id == 0:
                        nisa.tensor_copy(
                            dst=out_acc[0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                            src=ps[0:oproj_s_tile_pad, 0:n_sz],
                            engine=nisa.engine.vector,
                        )
                    else:
                        nisa.tensor_tensor(
                            dst=out_acc[0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                            data1=out_acc[0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                            data2=ps[0:oproj_s_tile_pad, 0:n_sz],
                            op=nl.add,
                            engine=nisa.engine.vector,
                        )
                    pbuf = (pbuf + 1) % n_psum_buf
        # Drain fp32 accumulator -> bf16 -> HBM.
        out_row_off = (s_start + s_batch * oproj_s_tile) * HID + hid_start
        nisa.tensor_copy(dst=out_sb[0:oproj_s_tile, :], src=out_acc[0:oproj_s_tile, :], engine=nisa.engine.scalar)
        nisa.dma_copy(
            dst=out_hbm.ap(pattern=[[HID, oproj_s_tile], [1, hid_per_core]], offset=out_row_off),
            src=out_sb[0:oproj_s_tile, :],
        )

    sbm.close_scope()  # Phase-B scope
    sbm.close_scope()  # outer scope (in_qtz/in_scale)
