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

"""Find all NKI utils in nkilib/core (NKI functions that never call nisa, directly or indirectly)."""

import ast
import re
from pathlib import Path

from .get_nki_functions import CORE_DIR, ISA_PATTERN, KernelLocation, get_nki_functions, statement_source_end

# Allowed nisa calls that don't make something a kernel
ALLOWED_ISA_PATTERNS = [
    r'\bnisa\.get_nc_version\b',
    r'\bnisa\.nc_version\b',
    r'\bnki\.isa\.nc_version\b',
]


def _has_isa_call(source: str):
    """Check if source has any nki.isa or nisa call (excluding allowed calls)."""
    # Remove allowed patterns
    filtered = source
    for pattern in ALLOWED_ISA_PATTERNS:
        filtered = re.sub(pattern + r'[\w.]*', '', filtered)
    return bool(ISA_PATTERN.search(filtered))


def _is_valid_util_location(filepath: str, name: str):
    """Check if a util is in a valid location: file-private, *_utils file,
    or under a utils/ folder."""
    if name.startswith('_'):
        return True
    p = Path(filepath)
    if p.stem.endswith('_utils'):
        return True
    if 'utils' in p.parts:
        return True
    return False


_cache: dict[str, tuple[list[KernelLocation], list[KernelLocation]]] = {}


def get_nki_utils():
    """Return list of (filepath, line_number, function_name) for NKI utils."""
    funcs, _ = get_nki_utils_with_violators()
    return funcs


def get_nki_utils_with_violators():
    """Return (nki_utils, violators)."""
    if "utils" not in _cache:
        _cache["utils"] = _compute_nki_utils_with_violators()
    return _cache["utils"]


def _compute_nki_utils_with_violators():
    # Get all NKI functions (topologically sorted - dependencies come first)
    nki_funcs = get_nki_functions()

    # Build func AST nodes/bodies and collect module-level dict constants mapping to functions
    func_nodes: dict[tuple[str, str], ast.FunctionDef] = {}
    func_sources: dict[tuple[str, str], str] = {}
    module_dict_funcs: dict[str, set[str]] = {}
    all_nki_names: set[str] = {loc.name for loc in nki_funcs}

    for filepath in sorted(CORE_DIR.rglob("*.py")):
        if "__pycache__" in filepath.parts:
            continue
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)

        lines = source.splitlines()
        fpath = str(filepath)
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef):
                func_nodes[(fpath, node.name)] = node
                # Extract source text for regex-based ISA check
                start = node.lineno - 1
                end = statement_source_end(tree.body, i, len(lines))
                func_sources[(fpath, node.name)] = '\n'.join(lines[start:end])
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                    refs: set[str] = set()
                    for v in node.value.values:
                        if isinstance(v, ast.Name) and v.id in all_nki_names:
                            refs.add(v.id)
                    if refs:
                        module_dict_funcs[target.id] = refs

    # Track which functions use isa (directly or indirectly)
    uses_isa: set[str] = set()

    # Process in topological order (dependencies first)
    for loc in nki_funcs:
        key = (loc.filepath, loc.name)
        source = func_sources.get(key, "")

        # Check direct isa call (regex on source text)
        if _has_isa_call(source):
            uses_isa.add(loc.name)
            continue

        # Check if calls any function that uses isa (walk AST node)
        node = func_nodes.get(key)
        if node is None:
            continue

        calls_isa_func = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                called_name = None
                if isinstance(child.func, ast.Name):
                    called_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    called_name = child.func.attr
                if called_name and called_name in uses_isa:
                    calls_isa_func = True
                    break
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                dict_name = child.value.id
                if dict_name in module_dict_funcs:
                    if module_dict_funcs[dict_name] & uses_isa:
                        calls_isa_func = True
                        break

        if calls_isa_func:
            uses_isa.add(loc.name)

    # Fixup: propagate ISA usage through module-level dict constants
    changed = True
    while changed:
        changed = False
        for loc in nki_funcs:
            if loc.name in uses_isa:
                continue
            key = (loc.filepath, loc.name)
            node = func_nodes.get(key)
            if node is None:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                    dict_name = child.value.id
                    if dict_name in module_dict_funcs and module_dict_funcs[dict_name] & uses_isa:
                        uses_isa.add(loc.name)
                        changed = True
                        break

    # Return functions that don't use isa
    utils = [loc for loc in nki_funcs if loc.name not in uses_isa]
    violators = [loc for loc in utils if not _is_valid_util_location(loc.filepath, loc.name)]
    return utils, violators
