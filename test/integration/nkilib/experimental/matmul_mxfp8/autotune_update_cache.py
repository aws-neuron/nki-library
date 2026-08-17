#!/usr/bin/env python3
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

"""Update the _AUTOTUNE_CACHE dict in matmul_mxfp8_config.py with best
configs from a sweep run.

Reads metrics from the test output directory, finds the best config per
(shape, dtype) pair, and updates the inline Python dict. Only updates
entries where the new result is faster than the existing one.

"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

CONFIG_KEYS = [
    "tile_m",
    "tile_k",
    "tile_n",
    "TILES_IN_BLOCK_M",
    "TILES_IN_BLOCK_N",
    "TILES_IN_BLOCK_K",
    "TILES_IN_LOAD_M",
    "TILES_IN_LOAD_N",
    "spill_reload",
]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PY = os.path.join(
    _THIS_DIR,
    "..",
    "..",
    "..",
    "..",
    "..",
    "src",
    "nkilib_src",
    "nkilib",
    "experimental",
    "matmul_mxfp8",
    "matmul_mxfp8_config.py",
)


def _cache_key_for(data):
    """Build the autotune cache key for a metrics record using production logic.

    Reuses the kernel's autotune_cache_key (from the NKI-free constants module)
    so the sweep-writer and the kernel reader share one definition of the key
    format. Requires the fields emitted by
    config_helper.TestConfig.to_metrics_dict(); raises KeyError if any are
    missing (e.g. a record from an older sweep).
    """
    from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_constants import autotune_cache_key

    return autotune_cache_key(
        data["M"],
        data["K"],
        data["N"],
        data["lhs_dtype"],
        data["rhs_dtype"],
        data["lhs_is_swizzled"],
        data["rhs_is_swizzled"],
        data.get("load_with_PE_swizzle", False),
        data.get("quant_scheme", "wrapX"),
        data["run_with_lnc2"],
        data["lnc_2_shard_rhs"],
    )


def _parse_existing_cache():
    """Parse the existing _AUTOTUNE_CACHE dict from matmul_mxfp8_config.py."""
    with open(_CONFIG_PY) as f:
        content = f.read()
    match = re.search(r'# fmt: off\n_AUTOTUNE_CACHE = (\{.*?\})\n# fmt: on', content, re.DOTALL)
    if not match:
        return {}
    # Safe eval: the dict only contains string keys and int values
    import ast

    return ast.literal_eval(match.group(1))


def collect_best_configs(output_dir):
    """Scan metrics JSONs and return best config per (shape_key, dtype)."""
    by_key = defaultdict(list)

    for out_dir in sorted(glob.glob(os.path.join(output_dir, "out-*"))):
        metrics_dir = os.path.join(out_dir, "metrics")
        if not os.path.isdir(metrics_dir):
            continue
        for jf in glob.glob(os.path.join(metrics_dir, "*.json")):
            with open(jf) as f:
                data = json.load(f)
            if data.get("Status") != "SUCCESS":
                continue

            inference_s = data.get("InferenceTime")
            if inference_s is None:
                continue
            inference_us = float(inference_s) * 1e6
            # Some runs report Status=SUCCESS (validation passed) but no captured
            # timing, sentinel-encoded as InferenceTime=-1.0. Skip those so a
            # non-positive time is never sorted to the front as the "fastest".
            if inference_us <= 0:
                continue

            # Build the cache key from the emitted metrics fields via production
            # logic; skip records missing the required fields (e.g. older sweeps).
            try:
                shape_key = _cache_key_for(data)
            except KeyError:
                continue

            config = {k: data[k] for k in CONFIG_KEYS if k in data}
            sr = data.get("spill_reload")
            if sr is not None:
                config["spill_reload"] = sr in (True, "True")
            by_key[shape_key].append((inference_us, config))

    best = {}
    for key, results in by_key.items():
        results.sort(key=lambda x: x[0])
        best[key] = (results[0][0], results[0][1])  # (time_us, config)
    return best


def update_config_py(existing, new_best, dry_run=False):
    """Merge new best configs into existing cache and rewrite _AUTOTUNE_CACHE."""
    merged = dict(existing)
    updated = added = 0

    for key, (new_time, new_config) in sorted(new_best.items()):
        old = merged.get(key)
        if old:
            # No timing in the Python dict, so always update from sweep results
            print(f"  UPDATE {key}: {new_time:.1f} µs")
            updated += 1
        else:
            print(f"  ADD    {key}: {new_time:.1f} µs")
            added += 1
        merged[key] = new_config

    # Rebuild the dict in config.py
    with open(_CONFIG_PY) as f:
        content = f.read()

    py_keys = [
        "tile_m",
        "tile_k",
        "tile_n",
        "TILES_IN_BLOCK_M",
        "TILES_IN_BLOCK_N",
        "TILES_IN_BLOCK_K",
        "TILES_IN_LOAD_M",
        "TILES_IN_LOAD_N",
        "spill_reload",
    ]
    lines = ["_AUTOTUNE_CACHE = {"]
    for key in sorted(merged.keys()):
        entry = merged[key]
        parts = ", ".join(f"'{k}': {entry[k]}" for k in py_keys if k in entry)
        lines.append(f"    \"{key}\": {{{parts}}},")
    lines.append("}")
    new_dict = "\n".join(lines)

    pattern = r'# fmt: off\n_AUTOTUNE_CACHE = \{.*?\}\n# fmt: on'
    replacement = f"# fmt: off\n{new_dict}\n# fmt: on"
    new_content, n_subs = re.subn(pattern, replacement, content, flags=re.DOTALL)

    if n_subs == 0:
        print("  WARNING: Could not find _AUTOTUNE_CACHE block to replace in config.py")
        return False
    if new_content == content:
        print("  Cache already up to date; no changes written.")
        return True

    if not dry_run:
        with open(_CONFIG_PY, "w") as f:
            f.write(new_content)
        print(f"\n  Config.py updated: {_CONFIG_PY}")

    print(f"  {updated} updated, {added} added, {len(merged)} total entries")
    return True


def main():
    parser = argparse.ArgumentParser(description="Update autotune cache with sweep results.")
    parser.add_argument("output_dir", help="Path to test output directory containing out-* dirs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    print(f"\nScanning metrics in: {args.output_dir}")
    new_best = collect_best_configs(args.output_dir)
    if not new_best:
        print("  No successful results found.")
        return

    print(f"  Found best configs for {len(new_best)} (shape, dtype) pairs\n")

    existing = _parse_existing_cache()
    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Updating _AUTOTUNE_CACHE in config.py:")
    update_config_py(existing, new_best, dry_run=args.dry_run)

    if args.dry_run:
        print("\n  (dry run — no files modified)")
    print()


if __name__ == "__main__":
    main()
