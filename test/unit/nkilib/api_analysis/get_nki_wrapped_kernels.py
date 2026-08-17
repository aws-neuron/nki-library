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

"""Find wrapped public kernels via single-pass reachability analysis."""

import ast

from .get_nki_functions import CORE_DIR, ISA_PATTERN, KernelLocation
from .get_nki_kernels import get_nki_kernels_split


def _get_called_kernels(source: str, kernel_names: set[str]):
    """Return set of kernel names called directly in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in kernel_names:
            called.add(node.func.id)
    return called


def _analyze_function(source: str, name: str, kernel_names: set[str]):
    """Return set of kernel names that are wrapped by this function.

    Single-pass: propagate a set of "candidate sole callers" through the function.
    A candidate set represents: which single kernels could be the only kernel called
    on some execution path reaching this point (with no ISA on that path).

    - None in the set means "a path with no kernel calls yet" (still viable).
    - A kernel name means "a path where only that kernel was called".
    - If a statement calls multiple kernels or has ISA, those paths die.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            wrapped: set[str] = set()
            _walk_body(node.body, {None}, kernel_names, name, wrapped)
            return wrapped
    return set()


def _apply_stmt(candidates: set[str | None], src: str, kernel_names: set[str], self_name: str):
    """Given candidates reaching a statement, return candidates after the statement."""
    if not candidates:
        return candidates

    has_isa = bool(ISA_PATTERN.search(src))
    called = _get_called_kernels(src, kernel_names) - {self_name}

    new: set[str | None] = set()
    for c in candidates:
        if has_isa:
            # ISA on this path: only survives if no kernel involvement
            if c is None and not called:
                continue  # path dies (ISA with no kernel = not a wrapper path)
            # path with kernel + ISA = dies
            continue
        if not called:
            new.add(c)  # no change
        elif len(called) == 1:
            k = next(iter(called))
            if c is None:
                new.add(k)  # first kernel on this path
            elif c == k:
                new.add(k)  # same kernel again, still ok
            # else: second different kernel, path dies
        # else: multiple kernels called, path dies
    return new


def _walk_body(
    stmts: list[ast.stmt], candidates: set[str | None], kernel_names: set[str], self_name: str, wrapped: set[str]
):
    """Walk a list of statements, propagating candidates. Collect wrapped kernels at returns."""
    for stmt in stmts:
        if not candidates:
            return candidates
        candidates = _walk_stmt(stmt, candidates, kernel_names, self_name, wrapped)
    return candidates


def _walk_stmt(stmt: ast.stmt, candidates: set[str | None], kernel_names: set[str], self_name: str, wrapped: set[str]):
    """Process one statement, return updated candidates."""
    if not candidates:
        return candidates

    if isinstance(stmt, ast.Return):
        src = ast.unparse(stmt)
        final = _apply_stmt(candidates, src, kernel_names, self_name)
        wrapped.update(c for c in final if c is not None)
        return set()  # after return, no paths continue

    if isinstance(stmt, ast.If):
        # Test expression applies to both branches
        test_src = ast.unparse(stmt.test)
        after_test = _apply_stmt(candidates, test_src, kernel_names, self_name)

        body_out = _walk_body(stmt.body, set(after_test), kernel_names, self_name, wrapped)
        else_out = _walk_body(stmt.orelse, set(after_test), kernel_names, self_name, wrapped)
        return body_out | else_out

    if isinstance(stmt, (ast.For, ast.While)):
        # Conservative: run body twice to stabilize, merge with skip path
        body_out = _walk_body(stmt.body, set(candidates), kernel_names, self_name, wrapped)
        body_out2 = _walk_body(stmt.body, set(body_out), kernel_names, self_name, wrapped)
        return candidates | body_out2

    if isinstance(stmt, ast.With):
        return _walk_body(stmt.body, candidates, kernel_names, self_name, wrapped)

    # Regular statement
    src = ast.unparse(stmt)
    return _apply_stmt(candidates, src, kernel_names, self_name)


_cache: dict[str, dict[KernelLocation, KernelLocation]] = {}


def get_nki_wrapped_kernels():
    """Return dict of KernelLocation -> KernelLocation (wrapped -> wrapper)."""
    if "wrapped" not in _cache:
        _cache["wrapped"] = _compute_nki_wrapped_kernels()
    return _cache["wrapped"]


def _compute_nki_wrapped_kernels():
    """A public kernel B is "wrapped" by public kernel A if A has a code path with no direct
    ISA calls where the only public kernel called (excluding itself) is B, and that path
    reaches a return statement.
    """
    public, _, _ = get_nki_kernels_split()
    public_names: set[str] = {loc.name for loc in public}
    public_info: dict[str, KernelLocation] = {loc.name: loc for loc in public}

    # Build func bodies for public kernels
    func_bodies: dict[tuple[str, str], str] = {}
    for filepath in sorted(CORE_DIR.rglob("*.py")):
        if "__pycache__" in filepath.parts:
            continue
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)

        lines = source.splitlines()
        fpath = str(filepath)
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name in public_names:
                start = node.lineno - 1
                if i + 1 < len(tree.body):
                    next_node = tree.body[i + 1]
                    if hasattr(next_node, 'decorator_list') and next_node.decorator_list:
                        end = next_node.decorator_list[0].lineno - 1
                    else:
                        end = next_node.lineno - 1
                else:
                    end = len(lines)
                func_bodies[(fpath, node.name)] = '\n'.join(lines[start:end])

    wrapped: dict[KernelLocation, KernelLocation] = {}

    for wrapper_loc in public:
        body = func_bodies.get((wrapper_loc.filepath, wrapper_loc.name), "")
        for wrapped_name in _analyze_function(body, wrapper_loc.name, public_names):
            wrapped[public_info[wrapped_name]] = wrapper_loc

    return wrapped
