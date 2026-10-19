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
nki.debug integration - all debugger-dependent code in one place.

This module is only imported when debugger mode is active.
"""

import logging
from typing import Optional


def run_debugger_inference(
    kernel_func,
    kernel_input: dict,
    dump_dir: str,
    core_id: int = 0,
    interactive: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-1,
    lnc: int = 1,
    platform_target: Optional[str] = None,
):
    """Run kernel through nki.debug against device dumps.

    nki.debug() returns a callable DebugKernel; we invoke it with the kernel inputs.
    This mirrors the nki.simulate pattern: create backend, then call with inputs.

    Args:
        kernel_func: The NKI kernel function to debug.
        kernel_input: Dict of kernel inputs (may contain .must_alias_input suffixes).
        dump_dir: Path to the debug_output directory containing device dumps.
        core_id: NeuronCore ID to debug.
        interactive: If True, open replay interface after test completion
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        lnc: lnc count (must match the compilation setting).
        platform_target: Hardware target (e.g. "trn2") to match the compilation
            setting.  When provided the env var NEURON_PLATFORM_TARGET_OVERRIDE
            is set so that the debugger's resolve_target() returns the same
            target that was used during compilation.
    """
    import os

    from nki.debugger import debug_kernel  # ty: ignore[unresolved-import]

    logging.info(
        f"Running nki.debug: dump_dir={dump_dir}, core_id={core_id}, "
        f"lnc={lnc}, platform_target={platform_target}, "
        f"interactive={interactive}, rtol={rtol}, atol={atol}"
    )

    # Strip ".must_alias_input" suffix from parameter names - this suffix is added
    # for the graph compiler but nki.debug expects original param names
    cleaned_input = {k.removesuffix(".must_alias_input"): v for k, v in kernel_input.items()}

    # Ensure the debugger uses the same hardware target as compilation.
    # Without this, resolve_target() falls back to _detect_target() which
    # may return a different target (e.g. trn3 on a dev machine without
    # Trainium hardware), causing the kernel to take different code paths
    # and desyncing the device-dump counter.
    if platform_target is not None:
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = platform_target

    debug_kernel(
        kernel_func,
        (),
        cleaned_input,
        dump_dir,
        core_id=core_id,
        _lnc=lnc,
        rtol=rtol,
        atol=atol,
        interactive=interactive,
    )
