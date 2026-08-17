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

# HBM-safe kernel test specs.
# Each entry: (module_path, name, {arg_name: shape_tuple (tensor) or value (non-tensor)})

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from nkilib_src.nkilib.core.rmsnorm.rmsnorm_quant import RmsNormQuantKernelArgs
from nkilib_src.nkilib.core.utils.common_types import RouterActFnType
from nkilib_src.nkilib.core.utils.logging import Logger, LogLevel


@dataclass
class ProofTensorSpec:
    """Tensor argument spec for HBM-safe proofs."""

    shape: Tuple[int, ...]
    dtype: type = np.float32
    randint_max: int = 1  # upper bound for randint (integer dtypes only)


@dataclass
class HbmProof:
    """Proof spec for a single HBM-safe kernel."""

    module_path: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


_SILENT_LOGGER = Logger("SBM", LogLevel.ERROR)

HBM_SAFE_PROOF = [
    HbmProof(
        "nkilib_src.nkilib.core.cumsum.cumsum",
        "cumsum",
        {"x": ProofTensorSpec((128, 512))},
    ),
    HbmProof(
        "nkilib_src.nkilib.core.attention.attention_cte",
        "attention_cte",
        {
            "q": ProofTensorSpec((1, 128, 128)),
            "k": ProofTensorSpec((1, 128, 128)),
            "v": ProofTensorSpec((1, 128, 128)),
            "tp_k": True,
        },
    ),
    HbmProof(
        "nkilib_src.nkilib.core.embeddings.rope",
        "RoPE",
        {
            "x_in": ProofTensorSpec((128, 1, 1, 1)),
            "cos": ProofTensorSpec((64, 1, 1)),
            "sin": ProofTensorSpec((64, 1, 1)),
        },
    ),
    HbmProof(
        "nkilib_src.nkilib.core.attention.gen_mask_tkg",
        "gen_mask_tkg_hbm",
        {"pos_ids_hbm": ProofTensorSpec((1, 1)), "bs": 1, "q_head": 1, "s_active": 1, "s_prior": 128},
    ),
    HbmProof(
        "nkilib_src.nkilib.core.qkv.qkv",
        "qkv",
        {
            "input": ProofTensorSpec((1, 128, 512)),
            "fused_qkv_weights": ProofTensorSpec((512, 384)),
            "d_head": 128,
            "num_q_heads": 1,
            "num_kv_heads": 1,
        },
    ),
    HbmProof(
        "nkilib_src.nkilib.core.output_projection.output_projection_cte.output_projection_cte",
        "output_projection_cte",
        {"attention": ProofTensorSpec((1, 1, 128, 512)), "weight": ProofTensorSpec((128, 128))},
    ),
    HbmProof(
        "nkilib_src.nkilib.core.output_projection.output_projection_tkg",
        "output_projection_tkg",
        {"attention": ProofTensorSpec((128, 1, 1, 1)), "weight": ProofTensorSpec((128, 128))},
    ),
    HbmProof(
        "nkilib_src.nkilib.core.moe.moe_tkg.moe_tkg",
        "moe_tkg",
        {
            "hidden_input": ProofTensorSpec((1, 512)),
            "expert_gate_up_weights": ProofTensorSpec((2, 512, 2, 512)),
            "expert_down_weights": ProofTensorSpec((2, 512, 512)),
            "expert_affinities": ProofTensorSpec((1, 2)),
            "expert_index": ProofTensorSpec((1, 1), dtype=np.int32, randint_max=2),
            "is_all_expert": False,
        },
    ),
    HbmProof(
        "nkilib_src.nkilib.core.rmsnorm.rmsnorm_quant",
        "rmsnorm_quant_kernel",
        {
            "hidden": ProofTensorSpec((1, 128, 512), dtype=np.float16),
            "ln_w": ProofTensorSpec((1, 512), dtype=np.float16),
            "kargs": RmsNormQuantKernelArgs(),
        },
    ),
    HbmProof(
        "nkilib_src.nkilib.core.router_topk.router_topk",
        "router_topk",
        {
            "x": ProofTensorSpec((512, 128)),
            "w": ProofTensorSpec((512, 8)),
            "w_bias": ProofTensorSpec((1, 8)),
            "router_logits": ProofTensorSpec((128, 8)),
            "expert_affinities": ProofTensorSpec((128, 8)),
            "expert_index": ProofTensorSpec((128, 2), dtype=np.int32),
            "act_fn": RouterActFnType.SOFTMAX,
            "k": 2,
            "x_hbm_layout": 0,
            "x_sb_layout": 0,
            "_lnc": 2,
        },
    ),
    HbmProof(
        "nkilib_src.nkilib.core.mlp.mlp",
        "mlp",
        {
            "hidden_tensor": ProofTensorSpec((1, 1, 128)),
            "gate_proj_weights_tensor": ProofTensorSpec((128, 256)),
            "up_proj_weights_tensor": ProofTensorSpec((128, 256)),
            "down_proj_weights_tensor": ProofTensorSpec((256, 128)),
        },
    ),
]
