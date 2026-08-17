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

"""Model configuration reader for generating test shapes from TorchTitan model configs."""

import json
import os
from typing import List

from test.integration.nkilib.experimental.matmul_mxfp8.config_helper import TestConfig


class TorchTitanModelConfig:
    # TorchTitan model-shape required params
    REQUIRED_KEYS = [
        "max_seq_len",  # S
        "dim",  # H
        "n_layers",  # L
        "n_heads",  # A
        "n_kv_heads",  # KvA
        "head_dim",  # per-head size
        "hidden_dim",  # FFN inner dim
    ]

    @staticmethod
    def from_json(path: str):
        """Load config from JSON file and validate required TorchTitan keys."""
        with open(path, "r") as f:
            cfg = json.load(f)

        missing = [key for key in TorchTitanModelConfig.REQUIRED_KEYS if key not in cfg]
        if missing:
            raise ValueError(f"Missing required TorchTitan config keys: {missing}")

        return cfg


def generate_attention_shapes(config_path: str, TP=1, CP=1):
    """
    Generate attention matmul shapes for one transformer block.

    Covers every large matmul in the attention portion of a layer (FWD and BWD):
      - Fused QKV projection
      - Attention QK^T and Attn*V (per-head shapes)
      - Output (O) projection

    Note: The following operations are NOT included because they are
    element-wise, or not matmuls:
      - RMSNorm / LayerNorm (element-wise)
      - RoPE (element-wise rotary embedding)
      - Residual additions
      - SiLU / softmax activations

    Args:
        config_path (str): Path to model configuration JSON file.
        TP (int): Tensor parallelism degree.
        CP (int): Context parallelism degree.

    Returns:
        list: List of [name, M, K, N] shape tuples.
    """
    cfg = TorchTitanModelConfig.from_json(config_path)

    S = cfg["max_seq_len"]
    H = cfg["dim"]
    Q = cfg["n_heads"]
    KV = cfg["n_kv_heads"]
    h = cfg["head_dim"]

    seq = S / CP

    all_shapes = []

    # ================================================================
    # 1. Fused QKV Projection: [S/CP, H] → Q,K,V heads.
    #    Weight: [H, (Q*h + 2*KV*h)/TP]
    # ================================================================

    qkv_out = (Q * h + 2 * KV * h) / TP

    # FWD: [S/CP, H] @ [H, qkv_out]
    all_shapes.append(["FWD-FuseQKV", seq, H, qkv_out])

    # BWD input grad: [S/CP, qkv_out] @ [qkv_out, H]
    all_shapes.append(["BWD-FuseQKV-IPGrad", seq, qkv_out, H])

    # BWD weight grad: [qkv_out, S/CP] @ [S/CP, H]
    all_shapes.append(["BWD-FuseQKV-WTGrad", qkv_out, seq, H])

    # ================================================================
    # 2. Attention Computation (per-head matmuls)
    #    These run once per head (Q/TP heads total per device).
    #
    #    QK^T:  [S/CP, h] @ [h, S/CP] → [S/CP, S/CP]
    #    Attn*V: [S/CP, S/CP] @ [S/CP, h] → [S/CP, h]
    # ================================================================

    # FWD — QK^T : score computation
    all_shapes.append(["FWD-Attn-QKt", seq, h, seq])

    # FWD — Attn * V : weighted value aggregation
    all_shapes.append(["FWD-Attn-AV", seq, seq, h])

    # BWD — dScores = dOut @ V^T : [S/CP, h] @ [h, S/CP] → [S/CP, S/CP]
    all_shapes.append(["BWD-Attn-dScores", seq, h, seq])

    # BWD — dV = Scores^T @ dOut : [S/CP, S/CP] @ [S/CP, h] → [S/CP, h]
    all_shapes.append(["BWD-Attn-dV", seq, seq, h])

    # BWD — dQ = dScores_softmax @ K : [S/CP, S/CP] @ [S/CP, h] → [S/CP, h]
    all_shapes.append(["BWD-Attn-dQ", seq, seq, h])

    # BWD — dK = dScores_softmax^T @ Q : [S/CP, S/CP] @ [S/CP, h] → [S/CP, h]
    all_shapes.append(["BWD-Attn-dK", seq, seq, h])

    # ================================================================
    # 3. Output (O) Projection
    #    Projects concatenated attention heads back to model dim.
    #    Input:  [S/CP, Q*h/TP]   (attention output, sharded across TP)
    #    Weight: [Q*h/TP, H]
    # ================================================================

    attn_out = Q * h / TP

    # FWD: [S/CP, Q*h/TP] @ [Q*h/TP, H]
    all_shapes.append(["FWD-OProj", seq, attn_out, H])

    # BWD input grad: [S/CP, H] @ [H, Q*h/TP]
    all_shapes.append(["BWD-OProj-IPGrad", seq, H, attn_out])

    # BWD weight grad: [H, S/CP] @ [S/CP, Q*h/TP]
    all_shapes.append(["BWD-OProj-WTGrad", H, seq, attn_out])

    return all_shapes


def generate_dense_mlp_shapes(config_path: str, TP=1, CP=1):
    """
    Generate dense MLP matmul shapes (SwiGLU gate+up and down projections).

    For dense transformer blocks only. MoE models should use
    generate_moe_expert_mlp_shapes instead.

    Args:
        config_path (str): Path to model configuration JSON file.
        TP (int): Tensor parallelism degree.
        CP (int): Context parallelism degree.

    Returns:
        list: List of [name, M, K, N] shape tuples.
    """
    cfg = TorchTitanModelConfig.from_json(config_path)

    S = cfg["max_seq_len"]
    H = cfg["dim"]
    FFN = cfg["hidden_dim"]

    seq = S / CP

    all_shapes = []

    # ================================================================
    # MLP — Fused Gate + Up Projection  (SwiGLU)
    #    gate_proj and up_proj are fused into one matmul.
    #    Input:  [S/CP, H]
    #    Weight: [H, 2*FFN/TP]   (gate and up concatenated)
    # ================================================================

    gate_up_out = 2 * FFN // TP

    # FWD: [S/CP, H] @ [H, 2*FFN/TP]
    all_shapes.append(["FWD-MLP-FuseGateUp", seq, H, gate_up_out])

    # BWD input grad: [S/CP, 2*FFN/TP] @ [2*FFN/TP, H]
    all_shapes.append(["BWD-MLP-FuseGateUp-IPGrad", seq, gate_up_out, H])

    # BWD weight grad: [2*FFN/TP, S/CP] @ [S/CP, H]
    all_shapes.append(["BWD-MLP-FuseGateUp-WTGrad", gate_up_out, seq, H])

    # ================================================================
    # MLP — Down Projection
    #    Projects FFN intermediate back to model dim.
    #    Input:  [S/CP, FFN/TP]
    #    Weight: [FFN/TP, H]
    # ================================================================

    down_in = FFN // TP

    # FWD: [S/CP, FFN/TP] @ [FFN/TP, H]
    all_shapes.append(["FWD-MLP-Down", seq, down_in, H])

    # BWD input grad: [S/CP, H] @ [H, FFN/TP]
    all_shapes.append(["BWD-MLP-Down-IPGrad", seq, H, down_in])

    # BWD weight grad: [H, S/CP] @ [S/CP, FFN/TP]
    all_shapes.append(["BWD-MLP-Down-WTGrad", H, seq, down_in])

    return all_shapes


def generate_transformer_block_shapes(config_path: str, TP=1, CP=1):
    """
    Generate all matmul shapes for one dense transformer block (attention + MLP).

    For MoE models, use generate_attention_shapes + generate_moe_expert_mlp_shapes instead.

    Args:
        config_path (str): Path to model configuration JSON file.
        TP (int): Tensor parallelism degree.
        CP (int): Context parallelism degree.

    Returns:
        list: List of [name, M, K, N] shape tuples.
    """
    return generate_attention_shapes(config_path, TP, CP) + generate_dense_mlp_shapes(config_path, TP, CP)


def generate_moe_expert_mlp_shapes(config_path: str, TP=1, moe_block_sizes=None):
    """
    Generate per-expert MLP matmul shapes for MoE models.

    MoE models replace the dense MLP with multiple experts. Each expert runs
    a SwiGLU MLP on a subset of tokens (block size). TP shards expert
    weights regardless of expert parallelism.

    Parallelism assumption:
        This function hardcodes Megatron-style column-row TP parallelism:
        - Gate+Up projection: column-parallel (TP shards output/N dim)
        - Down projection: row-parallel (TP shards input/K dim)
        LNC2 sharding is assumed blockwise (each core gets a different block),
        so block size B is not affected by LNC2.

    TODO: Add a parallelism_config parameter to support alternative sharding
    strategies (e.g., double column-parallel, all-blockwise) instead of
    hardcoding column-row.

    Args:
        config_path (str): Path to model configuration JSON file.
        TP (int): Tensor parallelism degree.
        moe_block_sizes (list): Block sizes (tokens per expert block). Defaults to [128, 256, 512].

    Returns:
        list: List of [name, M, K, N] shape tuples.
    """
    if moe_block_sizes is None:
        moe_block_sizes = [128, 256, 512]

    cfg = TorchTitanModelConfig.from_json(config_path)

    H = cfg["dim"]
    moe_inter_dim = cfg["moe_inter_dim"]

    all_shapes = []
    for moe_block_size in moe_block_sizes:
        # Column-row parallelism: gate+up shards N, down shards K
        gate_up_out = 2 * moe_inter_dim // TP  # column-parallel: full fused output per rank
        down_in = moe_inter_dim // TP  # row-parallel: per-rank slice of FFN input

        # FWD GateUp (column-parallel): Y = X @ W_gate_up
        # [B, H] @ [H, 2*moe_inter_dim/TP] — TP shards output (N) dim
        all_shapes.append([f"FWD-MoE-GateUp(B={moe_block_size})", moe_block_size, H, gate_up_out])
        # FWD Down (row-parallel): Y = X @ W_down
        # [B, moe_inter_dim/TP] @ [moe_inter_dim/TP, H] — TP shards input (K) dim
        all_shapes.append([f"FWD-MoE-Down(B={moe_block_size})", moe_block_size, down_in, H])
        # BWD GateUp IPGrad (row-parallel): dX = dY_gate_up @ W_gate_up^T
        # [B, 2*moe_inter_dim/TP] @ [2*moe_inter_dim/TP, H] — TP shards input (K) dim
        all_shapes.append([f"BWD-MoE-GateUp-IPGrad(B={moe_block_size})", moe_block_size, gate_up_out, H])
        # BWD GateUp WTGrad — unfused (column-parallel): dW_gate = X^T @ dY_gate
        # [H, B] @ [B, moe_inter_dim/TP] — gate and up computed as separate matmuls
        all_shapes.append([f"BWD-MoE-GateUp-WTGrad-Unfused(B={moe_block_size})", H, moe_block_size, down_in])
        # BWD GateUp WTGrad — fused (column-parallel): dW_gate_up = X^T @ [dY_gate | dY_up]
        # [H, B] @ [B, 2*moe_inter_dim/TP] — gate and up concatenated into one wider matmul
        all_shapes.append([f"BWD-MoE-GateUp-WTGrad-Fused(B={moe_block_size})", H, moe_block_size, gate_up_out])
        # BWD Down IPGrad (column-parallel): dX = dY_down @ W_down^T
        # [B, H] @ [H, moe_inter_dim/TP] — TP shards output (N) dim
        all_shapes.append([f"BWD-MoE-Down-IPGrad(B={moe_block_size})", moe_block_size, H, down_in])
        # BWD Down WTGrad (row-parallel): dW_down = X^T @ dY_down
        # [moe_inter_dim/TP, B] @ [B, H] — TP shards input (M) dim
        all_shapes.append([f"BWD-MoE-Down-WTGrad(B={moe_block_size})", down_in, moe_block_size, H])

    return all_shapes


# Keep the old name as an alias for backward compatibility
def generate_qkv_shapes(config_path: str, TP=1, CP=1):
    """Deprecated: use generate_transformer_block_shapes instead."""
    return generate_transformer_block_shapes(config_path, TP=TP, CP=CP)


def shapes_to_test_configs(shapes: List, model_name: str = "") -> List[TestConfig]:
    """
    Convert shape tuples to TestConfig objects.

    Args:
        shapes (List): List of [name, M, K, N] shape tuples.
        model_name (str): Model name prefix for descriptions.

    Returns:
        List[TestConfig]: List of test configuration objects.
    """
    configs = []
    for shape in shapes:
        name, M, K, N = shape
        # Convert to int in case they're floats
        M, K, N = int(M), int(K), int(N)

        description = f"{model_name} - {name}" if model_name else name
        config = TestConfig(M=M, K=K, N=N, description=description)
        configs.append(config)
    return configs


def load_model_configs(config_dir: str | None = None) -> dict:
    """
    Load all model configurations from the config directory.

    Args:
        config_dir (str): Directory containing model config JSON files.

    Returns:
        dict: Dictionary mapping config names to TestConfig lists.
    """
    if config_dir is None:
        # Default to the directory containing this file
        config_dir = os.path.dirname(os.path.abspath(__file__))

    configs = {}

    # Qwen3 8B
    qwen3_8b_path = os.path.join(config_dir, "qwen3_8B.json")
    if os.path.exists(qwen3_8b_path):
        shapes = generate_transformer_block_shapes(qwen3_8b_path, TP=1, CP=1)
        configs['qwen3_8b_tp1'] = shapes_to_test_configs(shapes, "Qwen3-8B-TP1")

        shapes = generate_transformer_block_shapes(qwen3_8b_path, TP=4, CP=1)
        configs['qwen3_8b_tp4'] = shapes_to_test_configs(shapes, "Qwen3-8B-TP4")

        shapes = generate_transformer_block_shapes(qwen3_8b_path, TP=16, CP=1)
        configs['qwen3_8b_tp16'] = shapes_to_test_configs(shapes, "Qwen3-8B-TP16")

    # Qwen3 32B (dense)
    qwen3_32b_path = os.path.join(config_dir, "qwen3_32B.json")
    if os.path.exists(qwen3_32b_path):
        shapes = generate_transformer_block_shapes(qwen3_32b_path, TP=1, CP=1)
        configs['qwen3_32b_tp1'] = shapes_to_test_configs(shapes, "Qwen3-32B-TP1")

        shapes = generate_transformer_block_shapes(qwen3_32b_path, TP=4, CP=1)
        configs['qwen3_32b_tp4'] = shapes_to_test_configs(shapes, "Qwen3-32B-TP4")

    # GPT-OSS-20B (MoE): attention shapes + expert MLP shapes
    gpt_oss_path = os.path.join(config_dir, "gpt_oss_20B.json")
    if os.path.exists(gpt_oss_path):
        attn_shapes = generate_attention_shapes(gpt_oss_path, TP=4, CP=1)
        moe_shapes = generate_moe_expert_mlp_shapes(gpt_oss_path, TP=4)
        configs['gpt_oss_20b_tp4'] = shapes_to_test_configs(attn_shapes + moe_shapes, "GPT-OSS-20B-TP4")

        attn_shapes = generate_attention_shapes(gpt_oss_path, TP=2, CP=1)
        moe_shapes = generate_moe_expert_mlp_shapes(gpt_oss_path, TP=2)
        configs['gpt_oss_20b_tp2'] = shapes_to_test_configs(attn_shapes + moe_shapes, "GPT-OSS-20B-TP2")

    # Qwen3 235B (MoE): attention shapes + expert MLP shapes
    qwen3_235b_path = os.path.join(config_dir, "qwen3_235B-A22B.json")
    if os.path.exists(qwen3_235b_path):
        attn_shapes = generate_attention_shapes(qwen3_235b_path, TP=4, CP=1)
        moe_shapes = generate_moe_expert_mlp_shapes(qwen3_235b_path, TP=4)
        configs['qwen3_235b_tp4'] = shapes_to_test_configs(attn_shapes + moe_shapes, "Qwen3-235B-TP4")

        attn_shapes = generate_attention_shapes(qwen3_235b_path, TP=2, CP=1)
        moe_shapes = generate_moe_expert_mlp_shapes(qwen3_235b_path, TP=2, moe_block_sizes=[512, 1024, 2048, 4096])
        configs['qwen3_235b_tp2'] = shapes_to_test_configs(attn_shapes + moe_shapes, "Qwen3-235B-TP2")

        attn_shapes = generate_attention_shapes(qwen3_235b_path, TP=1, CP=1)
        moe_shapes = generate_moe_expert_mlp_shapes(qwen3_235b_path, TP=1, moe_block_sizes=[512, 1024, 2048, 4096])
        configs['qwen3_235b_tp1'] = shapes_to_test_configs(attn_shapes + moe_shapes, "Qwen3-235B-TP1")

        shapes = generate_transformer_block_shapes(qwen3_235b_path, TP=4, CP=16)
        configs['qwen3_235b_cp16_tp4'] = shapes_to_test_configs(shapes, "Qwen3-235B-CP16-TP4")

        shapes = generate_transformer_block_shapes(qwen3_235b_path, TP=4, CP=4)
        configs['qwen3_235b_cp4_tp4'] = shapes_to_test_configs(shapes, "Qwen3-235B-CP4-TP4")

    return configs


# Load all configs at module import time
_all_configs = load_model_configs()

# Export individual config lists
qwen3_8b_tp1 = _all_configs.get('qwen3_8b_tp1', [])
qwen3_8b_tp4 = _all_configs.get('qwen3_8b_tp4', [])
qwen3_8b_tp16 = _all_configs.get('qwen3_8b_tp16', [])
qwen3_32b_tp1 = _all_configs.get('qwen3_32b_tp1', [])
qwen3_32b_tp4 = _all_configs.get('qwen3_32b_tp4', [])
gpt_oss_20b_tp4 = _all_configs.get('gpt_oss_20b_tp4', [])
gpt_oss_20b_tp2 = _all_configs.get('gpt_oss_20b_tp2', [])
qwen3_235b_tp4 = _all_configs.get('qwen3_235b_tp4', [])
qwen3_235b_tp2 = _all_configs.get('qwen3_235b_tp2', [])
qwen3_235b_tp1 = _all_configs.get('qwen3_235b_tp1', [])
qwen3_235b_cp16_tp4 = _all_configs.get('qwen3_235b_cp16_tp4', [])
qwen3_235b_cp4_tp4 = _all_configs.get('qwen3_235b_cp4_tp4', [])
