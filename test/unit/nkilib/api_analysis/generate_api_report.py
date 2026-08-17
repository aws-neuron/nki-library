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

"""Generate NKI API status report."""

from .framework_ready_api_spec import HBM_SAFE_PROOF
from .get_nki_functions import KernelLocation, rel_path
from .get_nki_integration_kernels import get_nki_framework_kernels
from .get_nki_kernels import get_nki_kernels_split
from .get_nki_utils import get_nki_utils_with_violators
from .get_nki_wrapped_kernels import get_nki_wrapped_kernels


def _fmt(loc: KernelLocation):
    return f"{loc.name} ({rel_path(loc.filepath)}::{loc.line})"


def _fmt_list(kernels: list[KernelLocation]):
    return [f"  {_fmt(loc)}" for loc in sorted(kernels, key=lambda k: k.name)]


def _collect_all_violators():
    """Collect violators from functions, utils, and kernels into a single list."""
    from .get_nki_functions import get_nki_functions_with_violators

    _, func_violators = get_nki_functions_with_violators()
    _, util_violators = get_nki_utils_with_violators()
    _, _, kernel_violators = get_nki_kernels_split()

    all_violators: list[tuple[KernelLocation, str]] = []
    for loc in func_violators:
        all_violators.append((loc, "function uses torch/np in non-_torch file"))
    for loc in util_violators:
        all_violators.append((loc, "public util not in _utils file or utils/ folder"))
    for filepath, line, name, reason in kernel_violators:
        all_violators.append((KernelLocation(filepath, line, name), reason))
    return all_violators


def _categorize_public_kernels(other_public: list[KernelLocation]):
    """Split other public kernels into subsections by reason."""
    wrapped_kernels = get_nki_wrapped_kernels()
    wrapper_for: dict[str, tuple[str, str]] = {}
    for wrapped_loc, wrapper_loc in wrapped_kernels.items():
        wrapper_for[wrapped_loc.name] = (wrapper_loc.name, rel_path(wrapper_loc.filepath))

    thin_wrapped: list[tuple[KernelLocation, str, str]] = []
    helper_subkernels: list[KernelLocation] = []
    uncategorized: list[KernelLocation] = []

    for loc in other_public:
        if loc.name in wrapper_for:
            wrapper_name, wrapper_path = wrapper_for[loc.name]
            thin_wrapped.append((loc, wrapper_name, wrapper_path))
        elif "/subkernels/" in loc.filepath:
            helper_subkernels.append(loc)
        else:
            uncategorized.append(loc)

    return thin_wrapped, helper_subkernels, uncategorized


def generate_report():
    public_kernels, _, _ = get_nki_kernels_split()
    all_violators = _collect_all_violators()

    proven_names: set[str] = {proof.name for proof in HBM_SAFE_PROOF}
    framework_ready = get_nki_framework_kernels(proven_names)
    framework_ready_set: set[KernelLocation] = set(framework_ready)
    other_public = [loc for loc in public_kernels if loc not in framework_ready_set]

    # Header
    print(f"Framework Ready: {len(framework_ready)}, Public: {len(other_public)}, Violators: {len(all_violators)}")

    # Violators (only if any)
    if all_violators:
        print("\n=== VIOLATORS ===\n")
        for loc, reason in sorted(all_violators, key=lambda v: v[0].name):
            print(f"  {_fmt(loc)} - {reason}")

    # Framework Ready
    print("\n=== FRAMEWORK READY ===\n")
    print("\n".join(_fmt_list(framework_ready)))

    # Public subsections
    print("\n=== PUBLIC ===\n")
    thin_wrapped, helper_subkernels, uncategorized = _categorize_public_kernels(other_public)

    subsections: list[str] = []
    if thin_wrapped:
        lines = ["  -- Thin Wrapped --"]
        for loc, wrapper_name, wrapper_path in sorted(thin_wrapped, key=lambda k: k[0].name):
            lines.append(f"  {_fmt(loc)} [{wrapper_name} @ {wrapper_path}]")
        subsections.append("\n".join(lines))
    if helper_subkernels:
        lines = ["  -- Helper Subkernels --"]
        lines.extend(_fmt_list(helper_subkernels))
        subsections.append("\n".join(lines))
    if uncategorized:
        lines = ["  -- Other --"]
        lines.extend(_fmt_list(uncategorized))
        subsections.append("\n".join(lines))
    print("\n\n".join(subsections))


if __name__ == "__main__":
    generate_report()
