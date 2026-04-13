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
Metadata loader for model-specific test configurations.

Loads JSON metadata files that map test configurations to model settings.
"""

import functools
import hashlib
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_config_version_id(config: dict) -> str:
    """Compute stable version ID from config content (SHA256 hash)."""
    content = json.dumps(config, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def load_model_configs(test_key: str) -> list[dict[str, Any]]:
    """
    Load model metadata for a given test key.

    Args:
        test_key: Test identifier (e.g., "test_moe_tkg")

    Returns:
        List of metadata entries, each containing:
        - kernel_key: Kernel identifier
        - test_settings: Test parameters (vnc, tokens, h, i, expert, etc.)
        - model_settings: Model information (model_name, dp_degree, tp_degree, etc.)

    Example:
        metadata = load_model_configs("test_moe_tkg")
        # Returns: [
        #   {
        #     "kernel_key": "moe_tkg",
        #     "test_settings": {"vnc": 2, "tokens": 1024, ...},
        #     "model_settings": {"model_name": "qwen3_235b", ...}
        #   },
        #   ...
        # ]
    """
    metadata_dir = Path(__file__).parent.parent.parent / "configuration" / "test_vector_metadata"

    # Use glob to find all matching files (e.g., test_moe_tkg_*.json)
    pattern = f"{test_key}_*.json"
    metadata_files = list(metadata_dir.glob(pattern))

    if not metadata_files:
        logger.warning(f"No metadata files found matching: {metadata_dir / pattern}")
        return []

    all_metadata = []
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)

            # Handle both single object and list of objects
            if isinstance(data, list):
                all_metadata.extend(data)
                logger.info(f"Loaded {len(data)} metadata entries from {metadata_file.name}")
            else:
                all_metadata.append(data)
                logger.info(f"Loaded 1 metadata entry from {metadata_file.name}")

        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_file}: {e}")

    logger.info(f"Total: {len(all_metadata)} metadata entries for {test_key}")
    return all_metadata


def _coerce_json_value(json_val: Any, test_val: Any) -> Any:
    """Convert a JSON-stored value to the Python type of test_val for comparison.

    Handles:
      - Enum: "RouterActFnType.SOFTMAX" -> RouterActFnType.SOFTMAX (via type(test_val)[name])
      - neuronlib dtype: "nl.bfloat16" -> "bfloat16" (strip prefix to match str(dtype))
    """
    if json_val == test_val:
        return json_val
    if isinstance(test_val, Enum) and isinstance(json_val, str) and "." in json_val:
        member_name = json_val.rsplit(".", 1)[1]
        try:
            return type(test_val)[member_name]
        except KeyError:
            return json_val
    if isinstance(json_val, str) and json_val.startswith("nl."):
        return json_val[3:]
    return json_val


def match_model_config_id(test_metadata_key: dict[str, Any], metadata_list: list[dict[str, Any]]) -> str | None:
    """Match test params against metadata entries and return the ModelConfigId hash.

    Args:
        test_metadata_key: Dict of abbreviated param names to Python values.
        metadata_list: List of metadata entries from load_model_configs().

    Returns:
        SHA256 hash of the matched entry, or None if no match found.
    """
    mismatches = None
    for entry in metadata_list:
        test_settings = entry.get("test_settings", {})
        mismatches = {
            k: (v, test_settings.get(k))
            for k, v in test_metadata_key.items()
            if _coerce_json_value(test_settings.get(k), v) != v
        }
        if not mismatches:
            return compute_config_version_id(entry)
    if mismatches:
        logger.info(f"No model config match found. Last entry mismatches: {mismatches}")
    return None
