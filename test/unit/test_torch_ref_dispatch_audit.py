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
Enforce torch_ref dispatch readiness.

Discovers all @nki.jit kernel entries in src/, then verifies:
1. dispatch() can resolve and return the torch_ref
2. The torch_ref signature matches the kernel signature
3. The torch_ref file doesn't import from test/

Known failures are excluded via DISPATCH_KNOWN_FAILURES (can only shrink).

Guidelines for adding a new kernel:
- Add @nki.jit kernel in src/ → this test auto-discovers it
- Add matching <name>_torch_ref in <name>_torch.py with same signature
- If not yet dispatch-ready, add to DISPATCH_KNOWN_FAILURES with a comment explaining why
"""

import importlib
import inspect
import re
from inspect import signature
from pathlib import Path
from typing import Any, Callable, List, Protocol, Tuple, runtime_checkable

import pytest
from nkilib_src.nkilib.core.utils import kernel_torch_dispatch
from nkilib_src.nkilib.core.utils.kernel_torch_dispatch import dispatch

from test.utils.test_validation_utils import fail_with_report

REPO_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = REPO_ROOT / "src" / "nkilib_src" / "nkilib"
KERNEL_DIRS = [SRC_DIR / "core", SRC_DIR / "experimental"]


@runtime_checkable
class _WrappedRef(Protocol):
    """A callable that keeps a reference to the function it wraps."""

    __wrapped__: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _find_kernel_entries() -> List[Tuple[str, str, str]]:
    """Find all @nki.jit kernel entries in src/.

    Scans for @nki.jit decorated functions, skipping private (_) and torch files.
    Returns list of (base_name, kernel_mod_path, kernel_fn_name).
    """
    entries = []
    seen = set()

    for kernel_dir in KERNEL_DIRS:
        if not kernel_dir.exists():
            continue
        for py_file in sorted(kernel_dir.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            if "_torch" in py_file.name:
                continue

            source = py_file.read_text()
            kernel_mod_path = str(py_file.relative_to(REPO_ROOT / "src")).replace("/", ".").replace(".py", "")

            # Find @nki.jit decorated functions
            for match in re.finditer(r"^@nki\.jit\b.*\ndef (\w+)\(", source, re.MULTILINE):
                fn_name = match.group(1)
                if fn_name.startswith("_"):
                    continue

                base_name = fn_name.replace("_kernel", "") if fn_name.endswith("_kernel") else fn_name

                if base_name in seen:
                    continue
                seen.add(base_name)

                entries.append((base_name, kernel_mod_path, fn_name))

    return sorted(entries)


# ═══════════════════════════════════════════════════════════════════════════════
# Known failures — this list can ONLY shrink.
# When a kernel is fixed, remove it. Adding new entries requires justification.
# ═══════════════════════════════════════════════════════════════════════════════

DISPATCH_KNOWN_FAILURES = frozenset(
    {
        # GDN kernels are validated via explicit torch_ref= in their integration tests
        # (test_gdn_*.py), not convention-based auto-dispatch. They are not yet wired for
        # the <name>_torch_ref naming + signature-parity contract:
        #   gdn_cte:       ref is gdn_cte_torch_nki_ref (name mismatch)
        #   gdn_block_tkg: ref is gdn_block_decode_fused_torch_ref (name mismatch)
        #   gdn_tkg:       no gdn_tkg_torch.py module (ref lives inline in the integration test)
        #   qkv_cte:       has qkv_cte_torch_ref but parameter names differ from the kernel
        # TODO(qqdong): wire these to convention dispatch and remove from this set.
        "gdn_cte",
        "gdn_block_tkg",
        "gdn_tkg",
        "qkv_cte",
    }
)


def _get_dispatchable_kernels() -> List[str]:
    """All kernel names excluding known failures."""
    return [name for name, _, _ in _find_kernel_entries() if name not in DISPATCH_KNOWN_FAILURES]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTorchRefDispatch:
    """Discover all @nki.jit kernels and verify dispatch works end-to-end."""

    @pytest.mark.parametrize("kernel_name", _get_dispatchable_kernels())
    def test_dispatch_works(self, kernel_name, monkeypatch):
        """dispatch() must resolve a callable torch_ref with matching signature."""
        # _USE_TORCH_REF is captured at module import; patch the attribute directly.
        monkeypatch.setattr(kernel_torch_dispatch, "_USE_TORCH_REF", True)

        # Find kernel module and function name
        kernel_mod_path = None
        kernel_fn_name = None
        for name, mod_path, fn_name in _find_kernel_entries():
            if name == kernel_name:
                kernel_mod_path = mod_path
                kernel_fn_name = fn_name
                break

        assert kernel_mod_path is not None, f"Could not find kernel entry for '{kernel_name}'"
        assert kernel_fn_name is not None, f"Kernel entry for '{kernel_name}' has no function name"

        # Stub with correct __module__ and __name__ for convention-based dispatch
        def stub():
            pass

        stub.__module__ = kernel_mod_path
        stub.__name__ = kernel_fn_name

        # Dispatch should resolve and return a wrapped torch_ref
        result = dispatch(stub)

        assert result is not stub, (
            f"dispatch({kernel_name}) returned the stub — no torch_ref found. "
            f"Expected: {kernel_mod_path.rsplit('.', 1)[0]}_torch module with {kernel_name}_torch_ref"
        )
        assert callable(result), f"dispatch({kernel_name}) returned non-callable: {type(result)}"

        # Validate signature parity via __wrapped__
        assert isinstance(result, _WrappedRef), f"dispatch({kernel_name}) result does not expose the wrapped torch_ref"
        torch_ref_func = result.__wrapped__
        kernel_func = getattr(importlib.import_module(kernel_mod_path), kernel_fn_name)
        kernel_params = set(signature(kernel_func).parameters.keys())
        ref_params = set(signature(torch_ref_func).parameters.keys())

        assert kernel_params == ref_params, (
            f"Signature mismatch for {kernel_name}: "
            f"missing in torch_ref={kernel_params - ref_params}, "
            f"extra in torch_ref={ref_params - kernel_params}"
        )

        # Validate torch_ref source doesn't import from test/.
        # Unwrap any LncSubscriptable / functools.wraps layers so getfile() finds
        # the underlying function module.
        ref_source_file = Path(inspect.getfile(inspect.unwrap(torch_ref_func)))
        ref_source = ref_source_file.read_text()
        test_imports = re.findall(r"^\s*(?:from|import)\s+test[.\s]", ref_source, re.MULTILINE)
        assert not test_imports, (
            f"Torch ref for {kernel_name} ({ref_source_file}) imports from test/: "
            f"{[t.strip() for t in test_imports]}. Move the imported utility to src/."
        )

    def test_known_failures_not_stale(self, monkeypatch):
        """DISPATCH_KNOWN_FAILURES entries should be removed once fixed."""
        monkeypatch.setattr(kernel_torch_dispatch, "_USE_TORCH_REF", True)

        stale = []
        for name, mod_path, fn_name in _find_kernel_entries():
            if name not in DISPATCH_KNOWN_FAILURES:
                continue

            def stub():
                pass

            stub.__module__ = mod_path
            stub.__name__ = fn_name

            try:
                result = dispatch(stub)
                if result is stub:
                    continue
                # Check signature
                if not isinstance(result, _WrappedRef):
                    continue
                torch_ref_func = result.__wrapped__
                kernel_func = getattr(importlib.import_module(mod_path), fn_name)
                if set(signature(kernel_func).parameters.keys()) == set(signature(torch_ref_func).parameters.keys()):
                    # Also check no test/ imports
                    ref_file = Path(inspect.getfile(inspect.unwrap(torch_ref_func)))
                    source = ref_file.read_text()
                    if not re.search(r"^\s*(?:from|import)\s+test[.\s]", source, re.MULTILINE):
                        stale.append(f"  {name} (now passes — remove from DISPATCH_KNOWN_FAILURES)")
            except (ImportError, AttributeError):
                pass

        if stale:
            fail_with_report(
                "\n\nStale DISPATCH_KNOWN_FAILURES — these kernels are now dispatch-ready:\n"
                + "\n".join(stale)
                + "\n\nFix: remove them from DISPATCH_KNOWN_FAILURES.\n"
            )
