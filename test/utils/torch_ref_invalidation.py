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
"""Robust cache-invalidation key for torch-ref output caching.

The torch-ref golden cache must invalidate whenever the reference computation
changes — including edits to *helper* functions the reference calls transitively
(e.g. mx_matmul, quantize_to_mx), not just the top-level function body. A naive
``inspect.getsource(ref_fn)`` misses helper edits and would silently serve a
stale golden, masking real kernel bugs.

This module computes a ``dep_hash`` over the full source of every in-package
module the reference transitively imports, discovered by a static AST import
walk starting from the reference function's own module file. Any edit to any
reached module changes the hash. It deliberately over-invalidates (an unrelated
edit in a shared module busts the cache) — over-invalidation costs a recompute,
never a wrong golden.

Third-party dependencies whose math affects the golden (the neuron_dtypes C
extensions that pack/unpack MX x4, plus ml_dtypes/numpy/torch) are NOT reached
by the source walk — they are captured separately by ``dependency_version_hash``
(C-extension .so bytes + library version strings), which the cache encodes in the
S3 path, mirroring how the NEFF cache encodes ``neuronxcc.__version__``.

REFERENCE PURITY CONTRACT
-------------------------
The cache is only correct if a cached reference is a *pure function of*::

    (kwargs, in-package source under _PKG_PREFIXES, pinned third-party deps)

Equivalently: given the same kwargs, the reference must return the same golden
across runs, with every input that affects the result flowing through one of the
three hashed surfaces. The following are OUTSIDE the hashes and therefore MUST
NOT influence a cached reference's output — any of them silently serves a stale
golden (the worst failure for a validation cache, since a broken kernel passes):

  1. First-party source outside _PKG_PREFIXES. The AST walk only follows imports
     whose top-level package is in _PKG_PREFIXES = ("nkilib_src", "nkilib",
     "test"). A reference importing numeric helpers from any OTHER first-party
     top-level package would not invalidate on edits there. If a reference's
     transitive import graph ever grows such a dependency, add its prefix here.
  2. Dynamic imports of FIRST-PARTY modules (importlib.import_module(name) /
     __import__(name) with a non-literal or out-of-prefix name) are invisible to
     the static walk. (Function-local `from ... import x` literals ARE caught —
     ast.walk descends into function bodies. Third-party dynamic imports such as
     __import__('ml_dtypes') are fine: their numerics are pinned by
     dependency_version_hash.)
  3. External data files read at runtime (lookup tables, .bin/.npy/config). Their
     contents are in no hash; edits change nothing. References MUST NOT read
     external data files.
  4. Impurity: unseeded RNG, env vars, wall-clock, or any global mutable state
     that affects numerics. A reference must be deterministic in its kwargs.

When adding a new cached reference, re-confirm the four points above hold for its
full transitive call graph.
"""

import ast
import functools
import hashlib
import importlib
import logging
import os
import sys
from typing import Callable

log = logging.getLogger(__name__)

# Package prefixes whose modules are treated as "ours" and hashed. Everything
# else (third-party) is considered stable / version-pinned and skipped.
_PKG_PREFIXES = ("nkilib_src", "nkilib", "test")

# Third-party libraries whose version/build affects golden numerics but are NOT
# reached by the in-package source walk. Captured by the version hash
# (dependency_version_hash), which the cache encodes in the S3 path so a bump
# re-namespaces the cache. ml_dtypes underlies the MX x4 dtypes; numpy/torch/scipy
# do reference math; neuronxcc supplies MX quantization helpers; nki the custom
# fp8 dtypes.
_VERSION_LIBS = ("ml_dtypes", "numpy", "torch", "scipy", "neuronxcc", "nki")

# Third-party top-level packages a reference may import and still be cacheable,
# because their numerics are pinned. Derived from _VERSION_LIBS plus neuron_dtypes
# (pinned by its C-extension .so bytes, not a version string), so the purity
# allowlist can never drift out of sync with what the version hash actually pins.
_PINNED_TOP_LEVEL = frozenset(_VERSION_LIBS) | {"neuron_dtypes"}

# Callables whose use makes a reference impure w.r.t. the cache key — either a
# file read (contents in no hash) or a dynamic import (target unresolvable, so its
# source can't be hashed). A reference reaching any of these is declined for
# caching (fail-closed to recompute), never served a possibly-stale golden.
# Matched on the call's attribute/name tail, so aliased module objects (np.load,
# numpy.load, ...) are all caught. Point 4 of the purity contract (RNG / env /
# clock / global state) is NOT covered here — deferred to a separate check.
_IMPURE_FILE_READ_CALLS = frozenset({"load", "fromfile", "loadtxt", "genfromtxt", "memmap"})
_IMPURE_DYNAMIC_IMPORT_CALLS = frozenset({"import_module", "__import__"})


def _module_file(module_name: str) -> str | None:
    """Resolve an imported module name to its .py file path, using sys.modules
    when available (handles installed-package layout) and falling back to None."""
    mod = sys.modules.get(module_name)
    if mod is not None:
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py") and os.path.exists(f):
            return f
    return None


def _resolve_relative(base_module: str, node: ast.ImportFrom) -> str | None:
    """Resolve a ``from . import x`` / ``from ..y import z`` to an absolute module."""
    if node.level == 0:
        return node.module
    parts = base_module.split(".")
    base = parts[: -node.level] if node.level <= len(parts) else []
    return ".".join(base + ([node.module] if node.module else []))


def _in_pkg(module_name: str) -> bool:
    return any(module_name == p or module_name.startswith(p + ".") for p in _PKG_PREFIXES)


def _transitive_module_files(root_module: str) -> dict[str, str]:
    """Return {module_name: file_path} for root_module plus every in-package
    module it transitively imports (resolved via static AST + sys.modules)."""
    seen: set[str] = set()
    stack = [root_module]
    files: dict[str, str] = {}
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = _module_file(mod)
        if not path:
            continue
        files[mod] = path
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = _resolve_relative(mod, node)
                if target and _in_pkg(target):
                    stack.append(target)
                    # also enqueue submodule targets imported by name
                    for alias in node.names:
                        stack.append(f"{target}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _in_pkg(alias.name):
                        stack.append(alias.name)
    return files


def _top_level(module_name: str) -> str:
    return module_name.split(".", 1)[0]


# Prefixes holding the golden computation. The purity scan runs only on these; the
# remaining in-package prefix "test" is scaffolding whose framework imports/file-reads
# never compute the golden. Test-module source is still hashed (edits invalidate);
# only the scan is scoped.
_NUMERIC_SOURCE_PREFIXES = ("nkilib_src", "nkilib")


def _is_numeric_source(module_name: str) -> bool:
    return any(module_name == p or module_name.startswith(p + ".") for p in _NUMERIC_SOURCE_PREFIXES)


def _purity_violation(files: dict[str, str]) -> str | None:
    """Statically scan the transitive in-package sources for a cache-purity
    violation the key can't otherwise capture. Returns a human-readable reason if
    the reference is unsafe to cache, else None. Covers points 1-3 of the purity
    contract; point 4 (RNG / env / clock / global state) is out of scope here.

      1. An import of a top-level package that is neither in-package (hashed) nor a
         version-pinned dep (namespaced by the version hash) nor stdlib — its
         source/version is in no key, so an edit there could serve a stale golden.
      2. A dynamic import (importlib.import_module / __import__) — target
         unresolvable, so its source can't be hashed.
      3. A file read (open / np.load / np.fromfile / ...) — file contents are in no
         key, so a changed file serves a stale golden.

    Scanned only on numeric-source modules (see _NUMERIC_SOURCE_PREFIXES).

    Detection is static: aliased/reflective forms (getattr, pre-bound aliases) can
    evade it, matching the reflective-call gap the contract already documents.
    """
    for mod in sorted(files):
        if not _is_numeric_source(mod):
            continue
        try:
            with open(files[mod], encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError):
            # Unreadable/unparseable source is handled as uncacheable by
            # dependency_hash (returns None); nothing to scan here.
            continue
        for node in ast.walk(tree):
            # (1) out-of-allowlist imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = _top_level(alias.name)
                    if not (_in_pkg(alias.name) or top in _PINNED_TOP_LEVEL or top in sys.stdlib_module_names):
                        return f"{mod}: imports unpinned package {top!r} (not in-package, version-pinned, or stdlib)"
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:  # absolute import
                    top = _top_level(node.module)
                    if not (_in_pkg(node.module) or top in _PINNED_TOP_LEVEL or top in sys.stdlib_module_names):
                        return f"{mod}: imports unpinned package {top!r} (not in-package, version-pinned, or stdlib)"
            # (2) dynamic imports, (3) file reads — matched on the call's tail name
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    name = fn.attr
                elif isinstance(fn, ast.Name):
                    name = fn.id
                else:
                    continue
                if name in _IMPURE_DYNAMIC_IMPORT_CALLS:
                    return f"{mod}: uses dynamic import {name!r} (target unhashable)"
                if name == "open" or name in _IMPURE_FILE_READ_CALLS:
                    return f"{mod}: reads a file via {name!r} (contents in no cache key)"
    return None


@functools.lru_cache(maxsize=128)
def dependency_hash(root_module: str) -> str | None:
    """SHA-256 over the full source of every in-package module transitively
    imported from root_module. Returns a 16-hex-char digest, or None if the
    reference is unsafe to cache: any reached module's source can't be read (a
    placeholder there would freeze the hash and stop tracking edits), or a static
    purity scan finds an out-of-allowlist import, a dynamic import, or a file read
    (see _purity_violation). Cached per module (module set is stable within a
    process)."""
    files = _transitive_module_files(root_module)
    violation = _purity_violation(files)
    if violation is not None:
        log.debug("torch-ref cache: declining to cache — %s", violation)
        return None
    h = hashlib.sha256()
    for mod in sorted(files):
        try:
            with open(files[mod], "rb") as f:
                src = f.read()
        except OSError:
            # A reached module's source is unreadable: no content-tracking hash possible.
            return None
        h.update(mod.encode())
        h.update(b"\x00")
        h.update(hashlib.sha256(src).digest())
    return h.hexdigest()[:16]


def ref_dependency_hash(torch_ref_fn: Callable) -> str | None:
    """Dependency hash for a torch_ref function, over its module's full transitive
    in-package import graph. Returns None (uncacheable -> recompute) when the
    module can't be resolved to a .py file.

    Without the module the transitive import graph is invisible, so a body-only
    hash would miss edits to helper functions the reference calls and skip the
    purity scan (points 1-3) — either could serve a stale golden. Declining is
    fail-closed and preserves the "never a wrong hit" invariant; an unresolvable
    module is a rare edge case (a builtin, or a ref defined in __main__).
    """
    module_name = getattr(torch_ref_fn, "__module__", None)
    if module_name and _module_file(module_name):
        # dependency_hash also runs the purity scan (points 1-3) and returns None
        # if the reference is unsafe to cache.
        return dependency_hash(module_name)
    log.debug(
        "torch-ref cache: module for %r unresolvable; declining to cache (fail-closed)",
        getattr(torch_ref_fn, "__qualname__", torch_ref_fn),
    )
    return None


def _neuron_dtypes_so_paths() -> list[str]:
    """Locate the neuron_dtypes compiled extensions (the MX x4 pack/unpack code).

    neuron_dtypes has no __version__, so we fingerprint its C-extension bytes
    directly: a rebuild that changes packing behavior changes the .so bytes."""
    try:
        mod = importlib.import_module("neuron_dtypes")
    except Exception:
        return []
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        return []
    impl_dir = os.path.join(os.path.dirname(pkg_file), "_impl")
    try:
        # listdir straight away (no isdir pre-check) — avoids a TOCTOU window and
        # returns [] uniformly if the dir is missing or unreadable.
        return sorted(os.path.join(impl_dir, f) for f in os.listdir(impl_dir) if f.endswith(".so"))
    except OSError:
        return []


@functools.lru_cache(maxsize=1)
def dependency_version_hash() -> str:
    """Fingerprint the environment whose change would alter golden numerics:
    the Python interpreter minor version, the neuron_dtypes C-extension .so bytes,
    plus ml_dtypes/numpy/torch/scipy/neuronxcc/nki versions.

    Encoded by the cache in the S3 path so any dependency bump or C-extension
    rebuild produces a fresh namespace (a clean miss, never a stale golden).
    Process-cached. Returns a 12-hex-char digest."""
    h = hashlib.sha256()
    # Python interpreter minor version: hash()/set-dict ordering and RNG-adjacent
    # semantics can shift across interpreter versions, so a Python upgrade must
    # re-namespace the cache rather than risk serving a golden computed under
    # different semantics. Over-invalidates (most bumps don't change numerics) —
    # but over-invalidation only costs a recompute, never a wrong golden.
    h.update(f"python={sys.version_info[0]}.{sys.version_info[1]}\x00".encode())
    # C-extension bytes (neuron_dtypes has no version string)
    for so in _neuron_dtypes_so_paths():
        h.update(os.path.basename(so).encode())
        h.update(b"\x00")
        try:
            with open(so, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"<unreadable>")
    # library version strings
    for name in _VERSION_LIBS:
        try:
            ver = getattr(importlib.import_module(name), "__version__", "<none>")
        except Exception:
            ver = "<missing>"
        h.update(name.encode())
        h.update(b"=")
        h.update(str(ver).encode())
        h.update(b"\x00")
    return h.hexdigest()[:12]
