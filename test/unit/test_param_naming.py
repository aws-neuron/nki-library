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
"""
Lint test: detect abbreviation collisions and non-canonical param names.

Scans all _ABBREVS dicts in integration test files and flags cases where
the same short abbreviation maps to different full parameter names.
Also blocks usage of known aliases (e.g., "batch_size") in new test files.
"""

import ast
import os
from collections import defaultdict
from pathlib import Path

from ..utils.param_extractor import CANONICAL_PARAM_NAMES

INTEGRATION_TEST_DIR = Path(__file__).parent.parent / "integration" / "nkilib"

# Existing files that use non-canonical param names — grandfathered.
# The metrics emission layer normalizes these for OpenSearch automatically.
# Do NOT add new files here. Use canonical names in new tests.
_GRANDFATHERED_FILES = {
    "core/attention/test_attention_bwd.py",
    "core/attention/test_attention_cte.py",
    "core/attention/test_attention_segmented_cte.py",
    "core/attention/test_gen_mask_tkg.py",
    "core/attention/test_kv_parallel_segmented_prefill.py",
    "core/cumsum/test_cumsum.py",
    "core/embeddings/test_rope.py",
    "core/embeddings/test_rope_hf.py",
    "core/mlp/test_mlp_proj_mxfp4.py",
    "core/moe/moe_tkg/test_moe_tkg.py",
    "core/output_projection/test_output_proj_tkg.py",
    "core/qkv/test_qkv_cte.py",
    "core/qkv/test_qkv_tkg.py",
    "experimental/primitives/output_projection/test_output_proj_tkg.py",
    "core/router_topk/test_router_topk.py",
    "core/subkernels/test_find_nonzero_indices.py",
    "core/subkernels/test_indexed_flatten.py",
    "experimental/attention/test_ring_attention_fwd.py",
    "experimental/benchmark/test_find_nonzero_indices_with_count.py",
    "experimental/collectives/test_collectives.py",
    "experimental/conv/test_conv1d.py",
    "experimental/loss/test_cross_entropy_backward.py",
    "experimental/loss/test_cross_entropy_forward.py",
    "experimental/moe/moe_tkg/test_moe_tkg_mx_selective_primitives.py",
    "experimental/output_projection/test_output_projection_tkg_primitives.py",
    "experimental/qkv/test_qkv_cte_primitives.py",
    "experimental/subkernels/test_build_all_to_all_v_metadata.py",
    "experimental/subkernels/test_permute_routed_tokens.py",
    "experimental/subkernels/test_topk_reduce.py",
    "experimental/transformer/test_transformer_tkg.py",
}


def _extract_abbrevs_from_file(filepath: Path) -> dict[str, str]:
    """Extract abbreviation mappings from a Python file's _ABBREVS-like dicts."""
    try:
        tree = ast.parse(filepath.read_text())
    except SyntaxError:
        return {}

    abbrevs = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if "ABBREV" not in target.id.upper():
                continue
            if isinstance(node.value, ast.Dict):
                for key, val in zip(node.value.keys, node.value.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(val, ast.Constant):
                        abbrevs[str(key.value)] = str(val.value)
    return abbrevs


def test_no_abbreviation_collisions():
    """Ensure no two different param names share the same abbreviation across test files."""
    abbrev_to_sources: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for root, _, files in os.walk(INTEGRATION_TEST_DIR):
        for fname in files:
            if not fname.startswith("test_") or not fname.endswith(".py"):
                continue
            filepath = Path(root) / fname
            abbrevs = _extract_abbrevs_from_file(filepath)
            for full_name, short in abbrevs.items():
                rel = str(filepath.relative_to(INTEGRATION_TEST_DIR))
                abbrev_to_sources[short][full_name].append(rel)

    collisions = []
    for short, name_map in abbrev_to_sources.items():
        if len(name_map) > 1:
            details = "; ".join(f'"{name}" in {files}' for name, files in name_map.items())
            collisions.append(f'  "{short}" → {details}')

    if collisions:
        msg = "Abbreviation collisions detected (same short form, different meanings):\n"
        msg += "\n".join(sorted(collisions))
        msg += "\n\nFix: ensure each abbreviation maps to only one parameter name."
        import warnings

        warnings.warn(msg, stacklevel=2)


def test_no_new_alias_usage():
    """Block new test files from using known non-canonical param names.

    If you need batch size, use 'batch' not 'batch_size'.
    If you need sequence length, use 'seqlen' not 'seq_len'.
    See test/utils/param_extractor.py for the full mapping.
    """
    violations = []

    for root, _, files in os.walk(INTEGRATION_TEST_DIR):
        for fname in files:
            if not fname.startswith("test_") or not fname.endswith(".py"):
                continue
            filepath = Path(root) / fname
            rel = str(filepath.relative_to(INTEGRATION_TEST_DIR))

            if rel in _GRANDFATHERED_FILES:
                continue

            try:
                tree = ast.parse(filepath.read_text())
            except SyntaxError:
                continue

            # Scan all string constants that look like comma-separated param name lists
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                parts = [p.strip() for p in node.value.split(",")]
                if len(parts) < 2 or any(" " in p for p in parts if p):
                    continue
                for p in parts:
                    if p in CANONICAL_PARAM_NAMES:
                        canonical = CANONICAL_PARAM_NAMES[p]
                        violations.append(f'  {rel}: use "{canonical}" instead of "{p}"')

    assert not violations, (
        "Non-canonical param names in parametrize calls (use canonical names for metrics consistency):\n"
        + "\n".join(sorted(set(violations)))
        + "\n\nSee CANONICAL_PARAM_NAMES in test/utils/param_extractor.py"
    )
