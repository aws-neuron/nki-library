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

"""Test config wrapper for the fused MXFP8 matmul backward kernel.

matmul_mxfp8_backward computes BOTH gradients in a single kernel invocation from
the FORWARD shape (M, K, N), where the forward pass is Y = X @ W^T with
X=[M, K], W=[N, K], Y=[M, N]:

    dX = dY @ W    -> logical matmul (M, N, K)   "input grad"
    dW = dY^T @ X  -> logical matmul (N, M, K)   "weight grad"

Because one kernel call produces two matmuls, a single TestConfig cannot describe
it: the two phases have different logical shapes and therefore different optimal
tiling. BackwardTestConfig holds exactly the three things the caller needs:

    1. a BF16 baseline for the fused backward pass
    2. one TestConfig for the input-grad (dX) phase
    3. one TestConfig for the weight-grad (dW) phase

The baseline must come from an actual measured BF16 backward run, which is why it
lives in its own cache (bf16_backward_baseline_cache.json) keyed by forward shape
rather than being derived from the forward cache.

Summing the two component entries out of bf16_baseline_cache.json would be wrong.
Those entries are standalone `torch.matmul` calls measured in isolation, and a
real BF16 backward pass is not two isolated matmuls: the two phases share the dY
operand, are scheduled together, and pay a different mix of DMA and transpose
costs than either matmul does on its own. A sum of isolated measurements is a
synthetic number that no BF16 implementation actually achieves, so any speedup
computed against it would be unfounded in either direction.

The cache therefore starts EMPTY. get_bf16_baseline returns None until real
backward baselines are measured and committed, and callers report "no baseline"
rather than a fabricated ratio. See cache_bf16_backward_baselines.py, which
measures a BF16 backward pass end to end and populates this cache.
"""

import json
import os
from typing import Any, Dict, List, Optional

from test.integration.nkilib.experimental.matmul_mxfp8.config_helper import TestConfig


def _load_bf16_backward_baseline_cache() -> Dict[str, Any]:
    """Load the measured BF16 BACKWARD baselines, keyed by forward shape "MxKxN".

    Separate from the forward cache on purpose: these are whole-backward-pass
    measurements, not per-matmul ones, so mixing the two key spaces would let a
    forward entry silently answer a backward lookup.
    """
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bf16_backward_baseline_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as cache_file:
                return json.load(cache_file)
        except Exception:
            pass
    return {}


_BF16_BACKWARD_BASELINE_CACHE = _load_bf16_backward_baseline_cache()


class BackwardTestConfig:
    """Bundles the two matmul configs and the BF16 baseline for one backward call.

    Args:
        M, K, N: the FORWARD shape (X=[M, K], W=[N, K], Y=[M, N]). The dX and dW
            logical shapes are derived from it, never passed in directly, so the
            two phases cannot drift out of sync with the forward shape.
        input_grad_config: TestConfig for the dX phase, logical shape (M, N, K).
        weight_grad_config: TestConfig for the dW phase, logical shape (N, M, K).
        description: human-readable label, reused in metrics.
    """

    def __init__(
        self,
        M: int,
        K: int,
        N: int,
        input_grad_config: TestConfig,
        weight_grad_config: TestConfig,
        description: str = "",
    ) -> None:
        self.M = M
        self.K = K
        self.N = N
        self.input_grad_config = input_grad_config
        self.weight_grad_config = weight_grad_config
        self.description = description

        # Guard against a mismatch between the forward shape and the per-phase
        # configs. A silent mismatch here would measure the wrong shapes while
        # still passing, which is the worst possible failure mode for a perf test.
        assert (input_grad_config.M, input_grad_config.K, input_grad_config.N) == (M, N, K), (
            f"input_grad_config shape {(input_grad_config.M, input_grad_config.K, input_grad_config.N)} "
            f"does not match expected dX shape {(M, N, K)} for forward shape {(M, K, N)}"
        )
        assert (weight_grad_config.M, weight_grad_config.K, weight_grad_config.N) == (N, M, K), (
            f"weight_grad_config shape {(weight_grad_config.M, weight_grad_config.K, weight_grad_config.N)} "
            f"does not match expected dW shape {(N, M, K)} for forward shape {(M, K, N)}"
        )

    # ------------------------------------------------------------------
    # Logical shapes of the two phases
    # ------------------------------------------------------------------

    @property
    def input_grad_shape(self) -> tuple:
        """Logical (M, K, N) of the dX matmul: dY[M, N] @ W[N, K]."""
        return (self.M, self.N, self.K)

    @property
    def weight_grad_shape(self) -> tuple:
        """Logical (M, K, N) of the dW matmul: dY^T[N, M] @ X[M, K]."""
        return (self.N, self.M, self.K)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def get_bf16_baseline(self) -> Optional[Dict[str, Any]]:
        """Return the MEASURED BF16 backward baseline for this forward shape.

        Looked up by forward shape in bf16_backward_baseline_cache.json, which
        holds real end-to-end BF16 backward measurements. Returns None when the
        shape has not been measured -- which is currently every shape, since that
        cache ships empty.

        Deliberately does NOT fall back to summing the two component entries in
        the forward cache. See the module docstring: a sum of two independently
        measured matmuls is not the cost of a real BF16 backward pass, and
        reporting a speedup against it would be unfounded. No baseline is the
        honest answer until the measurements exist.
        """
        entry = _BF16_BACKWARD_BASELINE_CACHE.get(f"{self.M}x{self.K}x{self.N}")
        if not entry:
            return None
        if entry.get("inference_time", -1.0) <= 0:
            return None
        return entry

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Flat metrics dict: forward shape, both phase configs, fused baseline."""
        metrics: Dict[str, Any] = {
            "shape_name": self.description,
            # Forward shape this backward pass belongs to.
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "input_grad_shape": "x".join(str(d) for d in self.input_grad_shape),
            "weight_grad_shape": "x".join(str(d) for d in self.weight_grad_shape),
            "kernel_mode": "backward_fused",
        }

        # Per-phase tiling, prefixed so the two phases stay distinguishable.
        for prefix, conf in (("dx", self.input_grad_config), ("dw", self.weight_grad_config)):
            for field in (
                "tile_m",
                "tile_k",
                "tile_n",
                "TILES_IN_BLOCK_M",
                "TILES_IN_BLOCK_N",
                "TILES_IN_BLOCK_K",
                "TILES_IN_LOAD_M",
                "TILES_IN_LOAD_N",
            ):
                metrics[f"{prefix}_{field}"] = getattr(conf, field)

        # Shared execution params come off the dX config; the two phases are
        # always built with the same dtypes/flags by the generators below.
        conf = self.input_grad_config
        for field in (
            "run_with_lnc2",
            "float8_dtype",
            "lhs_dtype",
            "rhs_dtype",
            "output_dtype",
            "lhs_is_swizzled",
            "rhs_is_swizzled",
            "load_with_PE_swizzle",
            "fast_dma_transpose",
            "quant_scheme",
            "disable_dma_transpose",
            "spill_reload",
            "enable_scale_packing",
            "lnc_2_shard_rhs",
            "seed",
        ):
            metrics[field] = getattr(conf, field, None)

        # Absent until real BF16 backward baselines are measured; the keys are
        # simply omitted rather than emitted as sentinels, so a missing baseline
        # cannot be mistaken for a measured zero.
        baseline = self.get_bf16_baseline()
        if baseline:
            metrics["BF16BaselineInferenceTime"] = baseline.get("inference_time", -1.0)
            metrics["BF16BaselineActiveInferenceTime"] = baseline.get("active_inference_time", -1.0)
            metrics["BF16BaselineMFU"] = baseline.get("mfu_percent", -1.0)
            metrics["BF16BaselineMBU"] = baseline.get("mbu_percent", -1.0)

        return metrics

    def __str__(self) -> str:
        baseline = self.get_bf16_baseline()
        lines = [
            f"BackwardTestConfig: {self.description}",
            f"  Forward Shape: M={self.M}, K={self.K}, N={self.N}",
            f"  dX (input grad):  M={self.input_grad_config.M}, "
            f"K={self.input_grad_config.K}, N={self.input_grad_config.N}",
            f"  dW (weight grad): M={self.weight_grad_config.M}, "
            f"K={self.weight_grad_config.K}, N={self.weight_grad_config.N}",
        ]
        if baseline:
            lines.append(
                f"  BF16 Backward Baseline: InferenceTime={baseline.get('inference_time')}, "
                f"MFU={baseline.get('mfu_percent')}"
            )
        else:
            lines.append("  BF16 Backward Baseline: not measured")
        return "\n".join(lines)


def make_backward_configs(
    shapes: List[tuple],
    config_factory,
    label: str,
    weight_grad_config_factory=None,
) -> List[BackwardTestConfig]:
    """Build BackwardTestConfigs from forward shapes using per-phase factories.

    Args:
        shapes: list of (M, K, N, description) FORWARD shapes.
        config_factory: callable taking a [(M, K, N, desc)] list and returning a
            list of TestConfig, used for the dX phase. The single-arm generators
            in test_input_microbenchmarks.py and test_qwen235b_perf.py have
            exactly this signature, so their tiling choices are reused rather
            than reimplemented per phase.
        label: suffix appended to each description.
        weight_grad_config_factory: optional separate factory for the dW phase.
            The two phases can have different operand dtypes -- e.g. in the
            prequantized-weight arm W is MXFP8 (a dX operand) while dW's operands
            are both BF16 -- and tiling is auto-tuned per dtype, so forcing one
            factory on both phases would tile dW for dtypes it does not use.
            Defaults to config_factory.

    Shapes whose phase configs a factory declines to generate (e.g. an alignment
    constraint filters them out) are skipped rather than half-built.
    """
    dw_factory = weight_grad_config_factory or config_factory
    configs = []
    for M, K, N, desc in shapes:
        dx = config_factory([(M, N, K, f"{desc} dX")])
        dw = dw_factory([(N, M, K, f"{desc} dW")])
        if not dx or not dw:
            continue
        configs.append(
            BackwardTestConfig(
                M=M,
                K=K,
                N=N,
                input_grad_config=dx[0],
                weight_grad_config=dw[0],
                description=f"{desc} [{label}]",
            )
        )
    return configs
