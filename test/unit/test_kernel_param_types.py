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

"""Kernel signature validation tests.

Verifies that all framework-ready kernel parameters use only allowed types:
primitives, enums, frozen dataclasses, Optional[T] for any T,
or tensor types (nl.ndarray).

The kernel list is dynamically derived from the framework-ready kernels
identified by the API analysis tooling (same source as generate_api_report.py).
"""

import dataclasses
import enum
import importlib
import inspect
import types
import typing
from typing import Tuple, Union

import pytest

from test.unit.nkilib.api_analysis.framework_ready_api_spec import HBM_SAFE_PROOF
from test.unit.nkilib.api_analysis.get_nki_functions import rel_path
from test.unit.nkilib.api_analysis.get_nki_integration_kernels import get_nki_framework_kernels
from test.utils.pytest_test_metadata import pytest_test_metadata


def _get_framework_ready_kernels():
    """Get framework-ready kernels as (module_path, name) tuples.

    Uses the same logic as generate_api_report.py to identify framework-ready kernels.
    """
    proven_names = {proof.name for proof in HBM_SAFE_PROOF}
    framework_ready = get_nki_framework_kernels(proven_names)

    result = []
    for loc in framework_ready:
        # Convert filepath to module path: nkilib_src/nkilib/core/mlp/mlp.py -> nkilib_src.nkilib.core.mlp.mlp
        module_path = rel_path(loc.filepath).replace("/", ".").replace(".py", "")
        result.append((module_path, loc.name))
    return result


FRAMEWORK_KERNELS = _get_framework_ready_kernels()

# Primitive types allowed as kernel parameters
_PRIMITIVE_TYPES = (int, float, bool, str, type(None), type)


def _is_optional(annotation) -> bool:
    """Check if a type annotation is Optional[T] (i.e. Union[T, None])."""
    origin = typing.get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = typing.get_args(annotation)
        return type(None) in args
    return False


def _is_allowed_type(annotation) -> bool:
    """Check if a type annotation is an allowed kernel parameter type.

    Allowed types:
    - Primitives: int, float, bool, str, NoneType, type
    - Enums (subclasses of enum.Enum)
    - Frozen dataclasses
    - nl.ndarray / nl.tensor (tensor types)
    - Optional[T] for any T
    - Tuple of allowed types
    - Union of allowed types
    - list[T] where T is allowed
    """

    # Handle Optional[X] — allow for any T
    if _is_optional(annotation):
        return True

    # Handle Union[X, Y] (non-Optional), Tuple[X, ...], list[X]
    origin = typing.get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return all(_is_allowed_type(arg) for arg in typing.get_args(annotation))
    if origin is tuple or origin is Tuple:
        return all(_is_allowed_type(arg) for arg in typing.get_args(annotation))
    if origin is list:
        args = typing.get_args(annotation)
        return all(_is_allowed_type(arg) for arg in args) if args else True

    # Primitives
    if annotation in _PRIMITIVE_TYPES:
        return True

    # Check if it's a class (not a string annotation or generic alias)
    if not isinstance(annotation, type):
        # Could be a forward reference or string annotation — allow it
        return True

    # Enum
    if issubclass(annotation, enum.Enum):
        return True

    # Frozen dataclass
    if dataclasses.is_dataclass(annotation) and annotation.__dataclass_params__.frozen:
        return True

    # Tensor types (nl.ndarray, nl.tensor, and similar)
    if "ndarray" in annotation.__name__.lower() or "tensor" in annotation.__name__.lower():
        return True

    return False


def _check_kernel_signature(module_path, name):
    """Validate all parameters of a kernel have allowed types.

    Returns list of (param_name, annotation) tuples for violations.
    """
    mod = importlib.import_module(module_path)
    fn = getattr(mod, name)
    # inspect.signature already follows __wrapped__ chains set by functools.wraps,
    # so we don't need to manually unwrap nki.jit or any other decorator.
    sig = inspect.signature(fn, follow_wrapped=True)
    violations = []

    for param_name, param in sig.parameters.items():
        # Unannotated params are allowed only if they have a default value
        if param.annotation is inspect.Parameter.empty:
            if param.default is inspect.Parameter.empty:
                violations.append((param_name, "<no annotation and no default>"))
            continue

        if not _is_allowed_type(param.annotation):
            violations.append((param_name, param.annotation))

    return violations


@pytest_test_metadata(name="Kernel Param Types")
class TestKernelParamTypes:
    """All kernel parameters must be primitives, enums, frozen dataclasses, or tensor types."""

    @pytest.mark.parametrize(
        "module_path,name",
        FRAMEWORK_KERNELS,
        ids=[f"{m}.{n}" for m, n in FRAMEWORK_KERNELS],
    )
    def test_kernel_param_types(self, module_path, name):
        violations = _check_kernel_signature(module_path, name)
        if violations:
            details = "\n".join(f"  - {p}: {a}" for p, a in violations)
            pytest.fail(
                f"Kernel {module_path}.{name} has parameters with disallowed types:\n{details}\n"
                f"Allowed: primitives (int/float/bool/str), enums, frozen dataclasses, "
                f"nl.ndarray, Optional[T] for any T, Tuple/Union wrappers of allowed types."
            )
