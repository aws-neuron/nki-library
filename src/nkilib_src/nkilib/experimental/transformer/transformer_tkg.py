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

"""Transformer forward pass optimized for token generation (TKG), composing attention_block_tkg and MLP kernels."""

from typing import List, Optional

import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from ...core.mlp.mlp import mlp
from ...core.utils.allocator import BufferManager, Logger
from ...core.utils.common_types import ActFnType, NormType, QuantizationType
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.tensor_view import TensorView
from .attention_block_tkg import attention_block_tkg

_SBM_SIZE_BYTES = 200 * 1024  # Buffer manager size in bytes

# ==================== Helper Functions (module-level for compiler compatibility) ====================


def _load_input_to_sbuf(dst_sb, src_hbm, BxS: int, H0: int, H1: int, H1_shard: int, n_prgs: int):
    """Load [B, S_tkg, H] HBM tensor to [H0, BxS*H1] SBUF layout."""
    src_view = TensorView(src_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    dst_reshaped = dst_sb.reshape((H0, BxS, n_prgs, H1_shard))
    for lnc_idx in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_view.slice(dim=2, start=lnc_idx, end=lnc_idx + 1).get_view(),
            dst=dst_reshaped[:, :, lnc_idx : lnc_idx + 1, :],
        )


def _store_output_to_hbm(out_hbm, in_sb, BxS: int, H0: int, H1: int, H1_shard: int, n_prgs: int):
    """Store [H0, BxS*H1] SBUF tensor to [B, S_tkg, H] HBM layout."""
    src_reshaped = in_sb.reshape((H0, BxS, n_prgs, H1_shard))
    dst_view = TensorView(out_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    for lnc_idx in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_reshaped[:, :, lnc_idx : lnc_idx + 1, :],
            dst=dst_view.slice(dim=2, start=lnc_idx, end=lnc_idx + 1).get_view(),
        )


def _sb2sb_all_reduce_gather(
    sharded_sb, dtype, replica_group, prg_id: int, n_prgs: int, H0: int, H1: int, H1_shard: int, BxS: int
):
    """SB2SB all-reduce with local gather, returns (output_sb, sharded_AR_sb)."""
    sharded_AR_sb = nl.ndarray(sharded_sb.shape, dtype=dtype, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[sharded_AR_sb], srcs=[sharded_sb], op=nl.add, replica_group=replica_group)

    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf)
    f_shard = nl.ds(start=prg_id * BxS * H1_shard, size=BxS * H1_shard)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=sharded_AR_sb)

    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other_shard = nl.ds(start=other_lnc * BxS * H1_shard, size=BxS * H1_shard)
        nisa.sendrecv(
            src=sharded_AR_sb,
            dst=gathered_sb[:, f_other_shard],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )

    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
    src_view = TensorView(gathered_sb).rearrange(('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1})
    nisa.tensor_copy(dst=output_sb.reshape((H0, BxS, H1)), src=src_view.get_view())

    return output_sb, sharded_AR_sb


# @nki.jit  # Commented out - use nki.jit() at call site to avoid double-jit stack overflow
def transformer_tkg(
    X: nl.ndarray,
    W_qkvs: List[nl.ndarray],
    W_outs: List[nl.ndarray],
    W_gates: List[nl.ndarray],
    W_ups: List[nl.ndarray],
    W_downs: List[nl.ndarray],
    W_gamma_qkvs: List[nl.ndarray],
    W_gamma_mlps: List[nl.ndarray],
    K_caches: List[nl.ndarray],
    V_caches: List[nl.ndarray],
    RoPE_cos: nl.ndarray,
    RoPE_sin: nl.ndarray,
    mask_cache: nl.ndarray,
    mask_active: nl.ndarray,
    position_ids: Optional[nl.ndarray],
    # Config parameters (replacing dataclass)
    num_layers: int,
    eps: float = 1e-6,
    replica_groups: Optional[List[List[int]]] = None,
    sbuf_residual_and_cc: bool = False,
    clamp_bound: float = 0.0,
    # FP8 scales (optional, per layer)
    W_gate_scales: Optional[List[nl.ndarray]] = None,
    W_up_scales: Optional[List[nl.ndarray]] = None,
    W_down_scales: Optional[List[nl.ndarray]] = None,
):
    """
    Transformer token generation forward pass megakernel.

    Performs num_layers transformer layers of the token-generation model.
    Within each layer: attention block, all-reduce CC, MLP, all-reduce CC.
    TODO: Specify intended usage range (e.g., sequence length, batch size)

    Dimensions:
        B: Batch size
        S_tkg: Token generation sequence length (number of new tokens)
        H: Hidden dimension (must be multiple of 128)
        H0: Partition tile size (pmax = 128)
        H1: H // H0
        H1_shard: H1 // n_prgs (per-core shard of hidden dimension)

    Args:
        X (nl.ndarray): [B, S_tkg, H], Input hidden states on HBM
        W_qkvs (List[nl.ndarray]): Per-layer QKV projection weights
        W_outs (List[nl.ndarray]): Per-layer output projection weights
        W_gates (List[nl.ndarray]): Per-layer MLP gate projection weights
        W_ups (List[nl.ndarray]): Per-layer MLP up projection weights
        W_downs (List[nl.ndarray]): Per-layer MLP down projection weights
        W_gamma_qkvs (List[nl.ndarray]): Per-layer RMSNorm gamma for QKV
        W_gamma_mlps (List[nl.ndarray]): Per-layer RMSNorm gamma for MLP
        K_caches (List[nl.ndarray]): Per-layer K caches on HBM
        V_caches (List[nl.ndarray]): Per-layer V caches on HBM
        RoPE_cos (nl.ndarray): [d_head//2, B, S_tkg], RoPE cosine embeddings
        RoPE_sin (nl.ndarray): [d_head//2, B, S_tkg], RoPE sine embeddings
        mask_cache (nl.ndarray): Attention mask for cached KV context
        mask_active (nl.ndarray): Attention mask for active tokens
        position_ids (Optional[nl.ndarray]): [B, 1], KV cache write positions (None = skip cache update)
        num_layers (int): Number of transformer layers to execute
        eps (float): RMSNorm epsilon (default 1e-6)
        replica_groups (Optional[List[List[int]]]): Replica groups for collective communication
        sbuf_residual_and_cc (bool): Use SBUF residual path with SB2SB all-reduce (default False)
        clamp_bound (float): FP8 quantization clipping boundary (default 0.0, 0 = no clipping)
        W_gate_scales (Optional[List[nl.ndarray]]): Per-layer FP8 gate weight scales
        W_up_scales (Optional[List[nl.ndarray]]): Per-layer FP8 up weight scales
        W_down_scales (Optional[List[nl.ndarray]]): Per-layer FP8 down weight scales

    Returns:
        output (nl.ndarray): [B, S_tkg, H], Final hidden states after all transformer layers

    Pseudocode:
        current = X
        for layer_idx in range(num_layers):
            # Step 1: Attention block (RMSNorm + QKV + RoPE + Attention + Output Projection)
            attn_out = attention_block_tkg(current, W_qkv[layer_idx], ...)

            # Step 2: All-reduce across tensor-parallel ranks
            attn_out = all_reduce(attn_out)

            # Step 3: Residual connection
            current = current + attn_out

            # Step 4: MLP block (RMSNorm + Gate/Up projection + SiLU + Down projection)
            mlp_out = mlp(current, W_gate[layer_idx], W_up[layer_idx], W_down[layer_idx], ...)

            # Step 5: All-reduce across tensor-parallel ranks
            mlp_out = all_reduce(mlp_out)

            # Step 6: Residual connection
            current = current + mlp_out
        return current
    """
    B, S_tkg, H = X.shape
    dtype = X.dtype

    # ========== LNC2 Initialization ==========
    _, n_prgs, prg_id = get_verified_program_sharding_info("transformer_tkg", (0, 1), 2)

    # Dimension constants
    H0 = nl.tile_size.pmax
    H1 = H // H0
    H1_shard = H1 // n_prgs
    BxS = B * S_tkg

    # Determine quantization type
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups != None else None

    sbm = BufferManager(0, _SBM_SIZE_BYTES, Logger("transformer_tkg"))
    sbm.set_auto_alloc(False)

    # ==================== Main Loop ====================
    current = X

    for layer_idx in range(num_layers):
        W_qkv = W_qkvs[layer_idx]
        W_out = W_outs[layer_idx]
        W_gate = W_gates[layer_idx]
        W_up = W_ups[layer_idx]
        W_down = W_downs[layer_idx]
        W_gamma_qkv = W_gamma_qkvs[layer_idx]
        W_gamma_mlp = W_gamma_mlps[layer_idx]
        K_cache = K_caches[layer_idx]
        V_cache = V_caches[layer_idx]
        W_gate_scale = W_gate_scales[layer_idx] if W_gate_scales else None
        W_up_scale = W_up_scales[layer_idx] if W_up_scales else None
        W_down_scale = W_down_scales[layer_idx] if W_down_scales else None

        quant_type = QuantizationType.ROW if W_gate_scale else QuantizationType.NONE

        if sbuf_residual_and_cc:
            # ========== SBUF Residual Path ==========
            sbm.set_name_prefix(f"L{layer_idx}_attn_")
            attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            _load_input_to_sbuf(attn_in_sb, current, BxS, H0, H1, H1_shard, n_prgs)

            residual_attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=residual_attn_in_sb, src=attn_in_sb)

            X_sb = attn_in_sb.reshape((H0, BxS, H1))
            sbm.set_auto_alloc(True)
            attn_result = attention_block_tkg(
                X=X_sb,
                X_hidden_dim_actual=H,
                rmsnorm_X_enabled=True,
                rmsnorm_X_eps=eps,
                rmsnorm_X_gamma=W_gamma_qkv,
                W_qkv=W_qkv,
                bias_qkv=None,
                quantization_type_qkv=QuantizationType.NONE,
                weight_dequant_scale_qkv=None,
                input_dequant_scale_qkv=None,
                rmsnorm_QK_pre_rope_enabled=False,
                rmsnorm_QK_pre_rope_eps=eps,
                rmsnorm_QK_pre_rope_W_Q=None,
                rmsnorm_QK_pre_rope_W_K=None,
                cos=RoPE_cos,
                sin=RoPE_sin,
                rope_contiguous_layout=True,
                rmsnorm_QK_post_rope_enabled=False,
                rmsnorm_QK_post_rope_eps=eps,
                rmsnorm_QK_post_rope_W_Q=None,
                rmsnorm_QK_post_rope_W_K=None,
                K_cache_transposed=True,
                active_blocks_table=None,
                K_cache=K_cache,
                V_cache=V_cache,
                attention_mask=mask,
                sink=None,
                update_cache=position_ids != None,
                kv_cache_update_idx=position_ids,
                W_out=W_out,
                bias_out=None,
                quantization_type_out=QuantizationType.NONE,
                weight_dequant_scale_out=None,
                input_dequant_scale_out=None,
                transposed_out=True,
                out_in_sb=True,
                sbm=sbm,
            )
            attn_kernel_out_sb = attn_result[0]

            # Free attention block's heap allocations so MLP has full SBM space
            while sbm.heap:
                sbm.pop_heap()

            attn_transformed_sb = attn_kernel_out_sb.reshape((H0, H1_shard * BxS))
            attn_layer_out_sb = _sb2sb_all_reduce_gather(
                attn_transformed_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_shard, BxS
            )[0]

            mlp_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mlp_in_sb, data1=residual_attn_in_sb, data2=attn_layer_out_sb, op=nl.add)

            residual_mlp_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=residual_mlp_in_sb, src=mlp_in_sb)

            sbm.set_name_prefix(f"L{layer_idx}_mlp_")
            mlp_in_reshaped = mlp_in_sb.reshape((H0, BxS, H1))
            sbm.set_auto_alloc(False)
            mlp_outputs = mlp(
                hidden_tensor=mlp_in_reshaped,
                gate_proj_weights_tensor=W_gate,
                up_proj_weights_tensor=W_up,
                down_proj_weights_tensor=W_down,
                normalization_weights_tensor=W_gamma_mlp,
                normalization_type=NormType.RMS_NORM,
                activation_fn=ActFnType.SiLU,
                eps=eps,
                quantization_type=quant_type,
                gate_w_scale=W_gate_scale,
                up_w_scale=W_up_scale,
                down_w_scale=W_down_scale,
                quant_clipping_bound=clamp_bound,
                store_output_in_sbuf=True,
                use_tkg_down_proj_column_tiling=False,
                sbm=sbm,
            )
            mlp_result = mlp_outputs[0]

            mlp_kernel_out_sb = mlp_result.reshape((H0, H1_shard * BxS))
            mlp_layer_out_sb = _sb2sb_all_reduce_gather(
                mlp_kernel_out_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_shard, BxS
            )[0]

            output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=output_sb, data1=residual_mlp_in_sb, data2=mlp_layer_out_sb, op=nl.add)

            layer_output = sbm.alloc((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="layer_output")
            if prg_id == 0:
                _store_output_to_hbm(layer_output, output_sb, BxS, H0, H1, H1_shard, n_prgs)

            if n_prgs > 1:
                nisa.core_barrier(data=layer_output, cores=(0, 1))

        else:
            # ========== HBM Path ==========
            sbm.set_name_prefix(f"L{layer_idx}_attn_")
            sbm.set_auto_alloc(True)
            attn_result = attention_block_tkg(
                X=current,
                X_hidden_dim_actual=H,
                rmsnorm_X_enabled=True,
                rmsnorm_X_eps=eps,
                rmsnorm_X_gamma=W_gamma_qkv,
                W_qkv=W_qkv,
                bias_qkv=None,
                quantization_type_qkv=QuantizationType.NONE,
                weight_dequant_scale_qkv=None,
                input_dequant_scale_qkv=None,
                rmsnorm_QK_pre_rope_enabled=False,
                rmsnorm_QK_pre_rope_eps=eps,
                rmsnorm_QK_pre_rope_W_Q=None,
                rmsnorm_QK_pre_rope_W_K=None,
                cos=RoPE_cos,
                sin=RoPE_sin,
                rope_contiguous_layout=True,
                rmsnorm_QK_post_rope_enabled=False,
                rmsnorm_QK_post_rope_eps=eps,
                rmsnorm_QK_post_rope_W_Q=None,
                rmsnorm_QK_post_rope_W_K=None,
                K_cache_transposed=True,
                active_blocks_table=None,
                K_cache=K_cache,
                V_cache=V_cache,
                attention_mask=mask_cache,
                sink=None,
                update_cache=position_ids != None,
                kv_cache_update_idx=position_ids,
                W_out=W_out,
                bias_out=None,
                quantization_type_out=QuantizationType.NONE,
                weight_dequant_scale_out=None,
                input_dequant_scale_out=None,
                transposed_out=False,
                out_in_sb=False,
                sbm=sbm,
            )
            attn_out = attn_result[0]

            # Free attention block's heap allocations so MLP has full SBM space
            while sbm.heap:
                sbm.pop_heap()

            if n_prgs > 1:
                nisa.core_barrier(data=attn_out, cores=(0, 1))

            # Get the attention output (either reduced or original)
            if rg != None:
                attn_reduced = sbm.alloc(attn_out.shape, dtype=dtype, buffer=nl.shared_hbm, name="attn_reduced")
                nccl.all_reduce(dsts=[attn_reduced], srcs=[attn_out], op=nl.add, replica_group=rg)
                attn_for_residual = attn_reduced
            else:
                attn_for_residual = attn_out

            # Residual add: current + attn_for_residual
            # Load to SBUF, add, store back
            attn_residual = sbm.alloc(current.shape, dtype=dtype, buffer=nl.shared_hbm, name="attn_residual")
            current_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            attn_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            result_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=current_sb, src=current.reshape((BxS, H)))
            nisa.dma_copy(dst=attn_sb, src=attn_for_residual)
            nisa.tensor_tensor(dst=result_sb, data1=current_sb, data2=attn_sb, op=nl.add)
            nisa.dma_copy(dst=attn_residual.reshape((BxS, H)), src=result_sb)

            # Use attn_residual for MLP input
            mlp_input = attn_residual

            sbm.set_name_prefix(f"L{layer_idx}_mlp_")
            sbm.set_auto_alloc(False)
            mlp_outputs = mlp(
                hidden_tensor=mlp_input,
                gate_proj_weights_tensor=W_gate,
                up_proj_weights_tensor=W_up,
                down_proj_weights_tensor=W_down,
                normalization_weights_tensor=W_gamma_mlp,
                normalization_type=NormType.RMS_NORM,
                activation_fn=ActFnType.SiLU,
                eps=eps,
                quantization_type=quant_type,
                gate_w_scale=W_gate_scale,
                up_w_scale=W_up_scale,
                down_w_scale=W_down_scale,
                quant_clipping_bound=clamp_bound,
                use_tkg_down_proj_column_tiling=False,
                sbm=sbm,
            )
            mlp_out = mlp_outputs[0]

            if n_prgs > 1:
                nisa.core_barrier(data=mlp_out, cores=(0, 1))

            # Get the MLP output (either reduced or original)
            if rg != None:
                mlp_reduced = sbm.alloc(mlp_out.shape, dtype=dtype, buffer=nl.shared_hbm, name="mlp_reduced")
                nccl.all_reduce(dsts=[mlp_reduced], srcs=[mlp_out], op=nl.add, replica_group=rg)
                mlp_for_residual = mlp_reduced
            else:
                mlp_for_residual = mlp_out

            # Residual add: attn_residual + mlp_for_residual
            # Load to SBUF, add, store back
            layer_output = sbm.alloc(attn_residual.shape, dtype=dtype, buffer=nl.shared_hbm, name="layer_output")
            attn_res_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            mlp_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            mlp_result_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=attn_res_sb, src=attn_residual.reshape((BxS, H)))
            nisa.dma_copy(dst=mlp_sb, src=mlp_for_residual.reshape((BxS, H)))
            nisa.tensor_tensor(dst=mlp_result_sb, data1=attn_res_sb, data2=mlp_sb, op=nl.add)
            nisa.dma_copy(dst=layer_output.reshape((BxS, H)), src=mlp_result_sb)

        current = layer_output

    return current
