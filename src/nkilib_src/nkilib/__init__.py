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

"""NKI Library - Neuron Kernel Interface Library."""

import importlib
import logging
import os
import pkgutil
import sys
from importlib.util import find_spec as _find_spec

_logger = logging.getLogger(__name__)

_SETUP_IN_PROGRESS = False
NKILIB_FORCE_BUNDLED_LIBRARY_ENV_VAR_KEY = "NKILIB_FORCE_BUNDLED_LIBRARY"


def _is_force_bundled() -> bool:
    """Check if NKILIB_FORCE_BUNDLED_LIBRARY env var is set to true."""
    val = os.environ.get(NKILIB_FORCE_BUNDLED_LIBRARY_ENV_VAR_KEY, '').lower()
    result = val in ('1', 'true', 'yes', 'on')
    _logger.debug(f"{NKILIB_FORCE_BUNDLED_LIBRARY_ENV_VAR_KEY}=%r, forcing bundled: %s", val, result)
    return result


def _is_bundled_context() -> bool:
    """Return True if running as bundled nkilib in neuronx-cc."""
    return __name__ == 'nkilib'


def _is_standalone_available() -> bool:
    try:
        return _find_spec('nkilib_src.nkilib') is not None
    except:
        return False


def _try_setup_src_nkilib() -> bool:
    """Attempt to use standalone nkilib, return True if successful.
    Returns False if we were not able to discover nkilib standalone.

    Raises:
        ImportError: If standalone nkilib is found but fails to import.
            This ensures users are aware of configuration issues rather than
            silently falling back to bundled implementation.
    """
    global _SETUP_IN_PROGRESS

    # Check if user explicitly wants bundled library
    if _is_force_bundled():
        _logger.debug("Returning False - NKILIB_FORCE_BUNDLED_LIBRARY is set, forcing bundled nkilib")
        return False

    # Guard: prevent re-entry during circular imports
    if _SETUP_IN_PROGRESS:
        _logger.debug("Returning False - setup already in progress (bundled nkilib)")
        return False

    # Context check: only do override in bundled context
    if not _is_bundled_context():
        _logger.debug("Returning False - not in bundled context (global nkilib)")
        return False

    # Guard: already swapped
    current = sys.modules.get(__name__)
    if current is not None and getattr(current, '__name__', '') == 'nkilib_src.nkilib':
        lib_path = getattr(current, '__file__', 'unknown')
        _logger.debug("Returning True - already swapped to global nkilib at %s", lib_path)
        return True

    # Check if standalone nkilib module specifically is discoverable
    if not _is_standalone_available():
        _logger.debug("Returning False - standalone nkilib not found (using bundled nkilib)")
        return False

    _SETUP_IN_PROGRESS = True
    try:
        # Double-check availability before import (validates deps, syntax, etc.)
        import nkilib_src.nkilib as _impl

        # Replace this module
        sys.modules[__name__] = _impl

        # Alias all submodules
        for key in list(sys.modules.keys()):
            if key.startswith('nkilib_src.nkilib.'):
                alias = key.replace('nkilib_src.nkilib.', 'nkilib.')
                sys.modules[alias] = sys.modules[key]

        standalone_path = getattr(_impl, '__file__', 'unknown')
        _logger.debug("Returning True - successfully swapped to global nkilib at %s", standalone_path)
        return True
    except Exception as e:
        # Standalone was found but failed to import - fail with clear error
        # instead of silently falling back to bundled implementation
        raise ImportError(
            f"nkilib_src package was found but failed to import: {e}\n\n"
            "This typically happens when nkilib_src has missing dependencies "
            "or is incorrectly installed.\n\n"
            "To fix this issue, either:\n"
            "  1. Install the missing dependencies for nkilib_src\n"
            "  2. Uninstall nkilib_src if you want to use the bundled version: "
            "pip uninstall nki-library"
        ) from e
    finally:
        _SETUP_IN_PROGRESS = False


if not _try_setup_src_nkilib():
    # was not able to setup standalone nkilib, import the bundled one instead
    # dynamically discover all submodules of the bundled nkilib
    for _importer, _modname, _ispkg in pkgutil.iter_modules(__path__):
        _ = importlib.import_module(f'.{_modname}', __name__)
