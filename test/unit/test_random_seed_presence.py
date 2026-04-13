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
Unit test to verify that all integration tests using numpy random generation
set a deterministic seed before generating random data.

Unseeded random calls produce non-deterministic test inputs, which can cause
intermittent (flaky) numerical accuracy failures on hardware. This test
catches that class of issue at build time.

Files can opt-out by including the comment @IGNORE_RANDOM_SEED anywhere in the file.
"""

import ast
from test.utils.test_validation_utils import (
    IntegrationFileCollector,
    ParsedFileInfo,
    ValidationErrorReporter,
    ValidationViolation,
)
from typing import List, Set

import pytest

IGNORE_MARKER = "@IGNORE_RANDOM_SEED"

# np.random functions that produce non-deterministic output when unseeded
RANDOM_GENERATORS: Set[str] = {
    "rand",
    "randn",
    "randint",
    "random",
    "random_sample",
    "uniform",
    "normal",
    "choice",
    "shuffle",
    "permutation",
    "dirichlet",
    "standard_normal",
    "exponential",
    "poisson",
    "binomial",
    "beta",
    "gamma",
}

SEED_FUNCTIONS: Set[str] = {"seed", "default_rng"}

FIX_INSTRUCTIONS = [
    "Add np.random.seed(<value>) before any np.random calls in each function listed above.",
    "",
    "Option 1: Seed at the start of the test method:",
    "",
    "   def test_my_kernel(self, ...):",
    "       np.random.seed(42)",
    "       input_np = np.random.randn(b, n).astype(np.float32)",
    "       ...",
    "",
    "Option 2: Use np.random.default_rng (always requires a seed):",
    "",
    "   def test_my_kernel(self, ...):",
    "       rng = np.random.default_rng(42)",
    "       input_np = rng.standard_normal((b, n)).astype(np.float32)",
    "       ...",
    "",
    "Option 3: Add @IGNORE_RANDOM_SEED comment to opt-out (use sparingly):",
    "",
    "   # @IGNORE_RANDOM_SEED - intentionally non-deterministic",
    "",
    "WHY: Unseeded random inputs cause flaky hardware accuracy failures.",
]


def _is_np_random_call(node: ast.Call, names: Set[str]) -> bool:
    """Check if node is np.random.<name>(...) for any name in names."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in names:
        return False
    val = func.value
    # np.random.<name>
    if (
        isinstance(val, ast.Attribute)
        and val.attr == "random"
        and isinstance(val.value, ast.Name)
        and val.value.id == "np"
    ):
        return True
    return False


def _body_has_seed_before_generator(body: List[ast.stmt]) -> bool:
    """
    Walk a function body in order. Return True if every random generator call
    is preceded by a seed call (or no generators exist).
    """
    seeded = False
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        if _is_np_random_call(node, SEED_FUNCTIONS):
            seeded = True
        if _is_np_random_call(node, RANDOM_GENERATORS) and not seeded:
            return False
    return True


def _check_file(file_info: ParsedFileInfo) -> List[str]:
    """Return list of 'ClassName.method_name' or 'function_name' that violate the rule."""
    violations: List[str] = []

    for cls in file_info.test_classes:
        for item in ast.iter_child_nodes(cls):
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                if not _body_has_seed_before_generator(item.body):
                    violations.append(f"{cls.name}.{item.name}")

    for func in file_info.test_functions:
        if not _body_has_seed_before_generator(func.body):
            violations.append(func.name)

    return violations


def test_random_seed_presence():
    """
    Verify that all integration tests using np.random generators set a seed first.

    Scans test/integration/nkilib/core and test/integration/nkilib/experimental
    for test functions that call np.random.<generator> without a preceding
    np.random.seed() or np.random.default_rng() in the same function body.
    """
    repo_root = IntegrationFileCollector.get_repo_root()
    dirs = [
        repo_root / "test" / "integration" / "nkilib" / "core",
        repo_root / "test" / "integration" / "nkilib" / "experimental",
        repo_root / "test" / "integration" / "nkilib" / "private",
    ]

    all_files: list[ParsedFileInfo] = []
    for d in dirs:
        if d.exists():
            collector = IntegrationFileCollector(d)
            all_files.extend(collector.collect(ignore_marker=IGNORE_MARKER))

    assert len(all_files) > 0, "No integration test files found"

    reporter = ValidationErrorReporter(
        error_title="Found test functions using np.random without setting a seed",
    )
    reporter.add_fix_instructions(FIX_INSTRUCTIONS)

    for file_info in all_files:
        bad_methods = _check_file(file_info)
        if bad_methods:
            reporter.add_violation(
                ValidationViolation(
                    file_path=file_info.file_path,
                    message=f"Unseeded random in: {', '.join(bad_methods)}",
                    method_names=set(bad_methods),
                )
            )

    if reporter.has_violations():
        pytest.fail(reporter.build())

    print(f"\n✓ All {len(all_files)} integration test files have seeded random calls")
