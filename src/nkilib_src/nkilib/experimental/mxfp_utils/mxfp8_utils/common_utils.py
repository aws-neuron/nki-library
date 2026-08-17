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

"""Global Level variables for using SbufManager"""

import functools

from ....core.utils.allocator import SbufManager
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.logging import get_logger

# Maximum available SBUF size for TRN3 (256 KiB minus reserved regions)
MAX_AVAILABLE_SBUF_SIZE_TRN3 = 256 * 1024 - 16384 - 8 - 256

# NOTE: These global variables are intended to be set and unset at a top-level kernel level only.
# Access them through getter and setter functions only.

# Using global variables as context-manager, singleton classes are not supported yet in NKI.
_state = {"sbm": None}


def create_and_set_active_sbm(
    sb_lower_bound=0,
    sb_upper_bound=MAX_AVAILABLE_SBUF_SIZE_TRN3,
    logger=get_logger("SBM"),
    use_auto_alloc=True,
    default_stack_alloc=True,
):
    kernel_assert(_state["sbm"] is None, "SbufManager is already set. Only one kernel should be active at a time.")
    sbm = SbufManager(sb_lower_bound, sb_upper_bound, logger, use_auto_alloc, default_stack_alloc)
    _state["sbm"] = sbm


def get_active_sbm():
    return _state["sbm"]


def clear_active_sbm():
    _state["sbm"] = None


def with_active_sbm(func):
    """Cleanup-only RAII for the module-global active SbufManager.

    The parser frontend ignores decorators, so this wrapper executes only on the
    real-execution (tracer) path. SBM setup stays in the kernel body so the parser
    path still has its manager; this wrapper releases the SBM on exit -- even if the
    body raises -- but only if this call created it. Computing ownership at entry
    keeps nested sub-kernels (which inherit a parent's SBM) from clearing it.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        owns = get_active_sbm() is None
        try:
            return func(*args, **kwargs)
        finally:
            if owns:
                clear_active_sbm()

    return wrapper
