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

"""Find all NKI functions in nkilib/core (functions that don't use torch or np calls)."""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import nkilib_src.nkilib.core as _core_pkg

CORE_DIR = Path(_core_pkg.__file__).resolve().parent
_SRC_DIR = CORE_DIR.parent.parent.parent

ISA_PATTERN = re.compile(r'\b(nki\.isa|nisa)\.')


@dataclass(frozen=True, order=True)
class KernelLocation:
    filepath: str
    line: int
    name: str


def rel_path(filepath: str):
    """Return path relative to src/ (e.g. nkilib_src/nkilib/core/...)."""
    return str(Path(filepath).relative_to(_SRC_DIR))


def statement_source_end(body: list[ast.stmt], index: int, num_lines: int):
    """Return the exclusive 0-based end line of the statement at ``index`` in ``body``.

    The statement's source runs up to the start of the following statement, or up to that
    statement's first decorator when it has one, so decorators are attributed to the
    definition they belong to. ``num_lines`` is used for the last statement in the module.
    """
    if index + 1 >= len(body):
        return num_lines
    next_node = body[index + 1]
    if isinstance(next_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and next_node.decorator_list:
        return next_node.decorator_list[0].lineno - 1
    return next_node.lineno - 1


def _uses_external_lib(node: ast.AST):
    """Check if AST node calls torch.X or np.X() (np constants like np.float32 are allowed)."""
    called_attrs: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            called_attrs.add(id(child.func))

    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            curr = child.value
            while isinstance(curr, ast.Attribute):
                curr = curr.value
            if not isinstance(curr, ast.Name):
                continue
            if curr.id == "torch":
                return True
            if curr.id == "np" and id(child) in called_attrs:
                return True
    return False


def _get_called_functions(node: ast.AST, all_func_names: set[str]):
    """Return set of function names called within an AST node."""
    called: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id in all_func_names:
                called.add(child.func.id)
            elif isinstance(child.func, ast.Attribute) and child.func.attr in all_func_names:
                called.add(child.func.attr)
    return called


_FuncKey = tuple[str, str]  # (filepath, function_name)
_FuncInfo = dict[_FuncKey, tuple[int, ast.FunctionDef]]


def _topological_sort(func_info: _FuncInfo, dependencies: dict[_FuncKey, set[_FuncKey]]):
    """Topologically sort functions. Returns sorted list or raises error with cycle."""
    in_degree: dict[_FuncKey, int] = dict.fromkeys(func_info, 0)
    dependents: dict[_FuncKey, set[_FuncKey]] = {name: set() for name in func_info}

    for name, deps in dependencies.items():
        for dep in deps:
            if dep in func_info and dep != name:
                in_degree[name] += 1
                dependents[dep].add(name)

    queue: list[_FuncKey] = [name for name, deg in in_degree.items() if deg == 0]
    result: list[_FuncKey] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(func_info):
        remaining: set[_FuncKey] = {n for n, d in in_degree.items() if d > 0}
        cycle: list[_FuncKey] = []
        start = next(iter(remaining))
        curr = start
        while True:
            cycle.append(curr)
            next_node = None
            for dep in dependencies.get(curr, set()):
                if dep in remaining:
                    next_node = dep
                    break
            if next_node is None or next_node == start:
                cycle.append(start)
                break
            curr = next_node
        cycle_desc = " -> ".join(f"{fpath}:{name}" for fpath, name in cycle)
        raise ValueError(f"Cycle detected: {cycle_desc}")

    return result


_cache: dict[str, tuple[list[KernelLocation], list[KernelLocation]]] = {}


def get_nki_functions():
    """Return topologically sorted list of (filepath, line_number, function_name)."""
    funcs, _ = get_nki_functions_with_violators()
    return funcs


def get_nki_functions_with_violators():
    """Return (nki_functions, violators)."""
    if "functions" not in _cache:
        _cache["functions"] = _compute_nki_functions_with_violators()
    return _cache["functions"]


def _compute_nki_functions_with_violators():
    # First pass: collect all top-level functions and their AST nodes
    all_funcs: _FuncInfo = {}  # (filepath, name) -> (line, ast_node)

    for filepath in sorted(CORE_DIR.rglob("*.py")):
        if "__pycache__" in filepath.parts or filepath.stem.endswith("_torch"):
            continue
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)

        fpath = str(filepath)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                all_funcs[(fpath, node.name)] = (node.lineno, node)

    # Mark functions that directly use torch/np
    uses_external: set[_FuncKey] = set()
    for key, (_line, node) in all_funcs.items():
        if _uses_external_lib(node):
            uses_external.add(key)

    # Build call graph: callee key -> set of caller keys
    all_names: set[str] = {k[1] for k in all_funcs}
    name_to_keys: dict[str, list[_FuncKey]] = {}
    for key in all_funcs:
        name_to_keys.setdefault(key[1], []).append(key)

    callers_of: dict[_FuncKey, set[_FuncKey]] = {key: set() for key in all_funcs}
    for key, (_line, node) in all_funcs.items():
        for called_name in _get_called_functions(node, all_names):
            candidates = name_to_keys.get(called_name, [])
            same_file = [k for k in candidates if k[0] == key[0] and k != key]
            for callee_key in same_file or [k for k in candidates if k != key]:
                callers_of[callee_key].add(key)

    # Propagate: if a function uses external libs, all its transitive callers do too
    queue = list(uses_external)
    while queue:
        key = queue.pop()
        for caller_key in callers_of.get(key, set()):
            if caller_key not in uses_external:
                uses_external.add(caller_key)
                queue.append(caller_key)

    # Filter and build result
    func_info: _FuncInfo = {k: v for k, v in all_funcs.items() if k not in uses_external}

    # Violators: functions in non-_torch files that directly use external libs
    violators: list[KernelLocation] = sorted(
        [
            KernelLocation(fp, line, nm)
            for (fp, nm), (line, _) in all_funcs.items()
            if (fp, nm) in uses_external and _uses_external_lib(all_funcs[(fp, nm)][1])
        ],
        key=lambda x: (x.filepath, x.line),
    )

    # Build dependencies for topological sort
    name_to_keys: dict[str, list[_FuncKey]] = {}
    for key in func_info:
        name_to_keys.setdefault(key[1], []).append(key)

    filtered_names: set[str] = {k[1] for k in func_info}
    dependencies: dict[_FuncKey, set[_FuncKey]] = {}
    for key, (_line, node) in func_info.items():
        called_names = _get_called_functions(node, filtered_names)
        deps: set[_FuncKey] = set()
        for name in called_names:
            for dep_key in name_to_keys.get(name, []):
                if dep_key != key:
                    deps.add(dep_key)
        dependencies[key] = deps

    sorted_keys = _topological_sort(func_info, dependencies)

    return [KernelLocation(key[0], func_info[key][0], key[1]) for key in sorted_keys], violators
