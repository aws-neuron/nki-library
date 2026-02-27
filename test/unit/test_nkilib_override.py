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

"""Unit tests for nkilib module override mechanism."""

import sys


class TestContextDetection:
    """Test the context detection mechanism that determines bundled vs standalone."""

    def test_is_bundled_context_when_standalone(self):
        """Test that _is_bundled_context() correctly identifies standalone context."""
        import nkilib_src.nkilib as nkilib

        # In standalone context, __name__ should be 'nkilib_src.nkilib'
        assert nkilib.__name__ == 'nkilib_src.nkilib'

        # _is_bundled_context() should return False
        assert nkilib._is_bundled_context() is False

    def test_try_setup_src_nkilib_skips_when_standalone(self):
        """Test that _try_setup_src_nkilib() doesn't run in standalone context."""
        import nkilib_src.nkilib as nkilib

        # In standalone context, _try_setup_src_nkilib() should return False immediately
        # because we ARE the standalone implementation
        assert nkilib._try_setup_src_nkilib() is False


class TestGuardMechanisms:
    """Test guard mechanisms that prevent problematic behavior."""

    def test_circular_import_prevention(self):
        """Test that _SETUP_IN_PROGRESS guard prevents circular imports."""
        import nkilib_src.nkilib as nkilib

        # Save original value
        original_setup_in_progress = nkilib._SETUP_IN_PROGRESS

        try:
            # Simulate being in the middle of setup
            nkilib._SETUP_IN_PROGRESS = True

            # Calling _try_setup_src_nkilib() should immediately return False
            # without attempting any import or swap
            assert nkilib._try_setup_src_nkilib() is False

            # The flag should still be True (not modified by the call)
            assert nkilib._SETUP_IN_PROGRESS is True
        finally:
            # Restore original value
            nkilib._SETUP_IN_PROGRESS = original_setup_in_progress


class TestSubmoduleDiscovery:
    """Test dynamic submodule discovery mechanism."""

    def test_core_submodule_loaded(self):
        """Test that the core submodule is automatically discovered and loaded."""
        import nkilib_src.nkilib as nkilib

        # The 'core' submodule should be auto-loaded by pkgutil.iter_modules
        assert hasattr(nkilib, 'core')
        assert nkilib.core is not None
        assert nkilib.core.__name__ == 'nkilib_src.nkilib.core'

    def test_submodule_in_sys_modules(self):
        """Test that discovered submodules are registered in sys.modules."""
        import nkilib_src.nkilib as nkilib

        # Access core to ensure it's loaded
        _ = nkilib.core

        # Verify it's in sys.modules with the correct name
        assert 'nkilib_src.nkilib.core' in sys.modules
        assert sys.modules['nkilib_src.nkilib.core'] is nkilib.core


class TestImportBehavior:
    """Test import behavior and module identity."""

    def test_repeated_imports_return_same_module(self):
        """Test that repeated imports return the same module object."""
        import nkilib_src.nkilib as nkilib1
        import nkilib_src.nkilib as nkilib2

        # Should be the exact same object (Python module caching)
        assert nkilib1 is nkilib2

    def test_module_has_expected_internals(self):
        """Test that the module has all expected internal functions and flags."""
        import nkilib_src.nkilib as nkilib

        # Check for internal functions and variables
        assert hasattr(nkilib, '_is_bundled_context')
        assert callable(nkilib._is_bundled_context)

        assert hasattr(nkilib, '_try_setup_src_nkilib')
        assert callable(nkilib._try_setup_src_nkilib)

        assert hasattr(nkilib, '_SETUP_IN_PROGRESS')
        assert isinstance(nkilib._SETUP_IN_PROGRESS, bool)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_find_spec_handles_missing_module(self):
        """Test that find_spec properly handles non-existent modules."""
        from importlib.util import find_spec

        # find_spec should return None for non-existent modules
        # This validates the check used in _try_setup_src_nkilib()
        result = find_spec('definitely_does_not_exist_nkilib_xyz123')
        assert result is None

    def test_try_setup_src_nkilib_raises_on_broken_standalone(self):
        """Test that _try_setup_src_nkilib() raises ImportError when standalone is found but broken.

        This ensures users get a clear error message instead of silently falling back
        to the bundled implementation, which could lead to unexpected behavior.
        """
        # Read the source to verify try/except structure
        import inspect

        import nkilib_src.nkilib as nkilib

        source = inspect.getsource(nkilib._try_setup_src_nkilib)

        # Verify exception handling is present and re-raises ImportError
        assert 'try:' in source
        assert 'except Exception as e:' in source
        # Should raise ImportError, not return False (no silent fallback)
        assert 'raise ImportError' in source

        # Verify the error message provides useful guidance
        assert 'nkilib_src package was found but failed to import' in source
        assert 'pip uninstall nki-library' in source

        # Verify the finally block to reset the flag
        assert 'finally:' in source
        assert '_SETUP_IN_PROGRESS = False' in source
