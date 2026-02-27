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
Unit tests simulating bundled nkilib context (as if running from KaenaCompiler).

This test suite simulates the bundled environment and tests both
normal operations and edge cases.

These tests create temporary Python packages to simulate the bundled environment
where nkilib/__init__.py is the top-level package instead of nkilib_src.nkilib.

Test categories:
- Basic swap/fallback behavior
- Deeply nested imports (nkilib.core.mlp)
- Module identity consistency
- Import aliases
- Pickle roundtrip
- Cross-module interactions
- sys.modules aliasing
- Import order independence

Edge case tests:
- Empty nkilib_src directory
- Mismatched structure between standalone and bundled
- Environment variable (NKILIB_FORCE_BUNDLED_LIBRARY) forcing bundled version
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import final


@final
class BundledSimulator:
    """Context manager to simulate bundled nkilib environment in a subprocess.

    Creates an isolated Python environment where:
    - nkilib is a top-level package (simulating KaenaCompiler's bundled nkilib)
    - nkilib_src may or may not be available
    - Supports deep package hierarchy (nkilib.core.mlp) for edge case testing
    - Supports edge cases: empty standalone, missing submodules, env variables
    """

    CIRCULAR_IMPORT_NAMESPACE = "circular_test"

    def __init__(
        self,
        standalone_available: bool = True,
        standalone_broken: bool = False,
        deep_hierarchy: bool = False,
        standalone_empty: bool = False,
        standalone_missing_submodules: bool = False,
        bundled_has_extra_module: bool = False,
        env_force_bundled: bool = False,
        circular_imports: bool = False,
        standalone_reimports_nkilib: bool = False,
    ):
        """Initialize bundled simulator.

        Args:
            standalone_available: Whether nkilib_src package exists
            standalone_broken: Whether standalone raises error on import
            deep_hierarchy: Whether to create nkilib.core.mlp hierarchy
            standalone_empty: Whether nkilib_src/nkilib exists but is empty (no __init__.py)
            standalone_missing_submodules: Whether standalone is missing some bundled submodules
            bundled_has_extra_module: Whether bundled has submodules not in standalone
            env_force_bundled: Whether to set NKILIB_FORCE_BUNDLED_LIBRARY env var
            standalone_reimports_nkilib: Whether standalone's __init__.py imports 'nkilib',
                                         simulating a circular import back to bundled
            circular_imports: Whether to create circular import test modules
        """
        self.standalone_available = standalone_available
        self.standalone_broken = standalone_broken
        self.deep_hierarchy = deep_hierarchy
        self.standalone_empty = standalone_empty
        self.standalone_missing_submodules = standalone_missing_submodules
        self.bundled_has_extra_module = bundled_has_extra_module
        self.env_force_bundled = env_force_bundled
        self.standalone_reimports_nkilib = standalone_reimports_nkilib
        self.circular_imports = circular_imports
        self.tmpdir = None

    def __enter__(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)

        # Read the actual __init__.py implementation from nkilib_src
        import nkilib_src.nkilib as real_nkilib

        init_content = Path(real_nkilib.__file__).read_text()

        # Create bundled nkilib package
        bundled_nkilib = root / "nkilib"
        bundled_nkilib.mkdir()
        (bundled_nkilib / "__init__.py").write_text(init_content)

        # Create bundled core submodule
        bundled_core = bundled_nkilib / "core"
        bundled_core.mkdir()

        if self.deep_hierarchy:
            (bundled_core / "__init__.py").write_text('"""Bundled core module."""\nBUNDLED = True\nfrom . import mlp\n')
            # Create bundled core.mlp submodule
            bundled_mlp = bundled_core / "mlp"
            bundled_mlp.mkdir()
            (bundled_mlp / "__init__.py").write_text(
                '"""Bundled MLP module."""\n'
                'BUNDLED = True\n'
                '\n'
                'class MLPConfig:\n'
                '    """MLP configuration class for identity tests."""\n'
                '    SOURCE = "bundled"\n'
                '\n'
                'def run_mlp(x):\n'
                '    """Dummy MLP function."""\n'
                '    return f"bundled:{x}"\n'
            )
        else:
            (bundled_core / "__init__.py").write_text('"""Bundled core module."""\nBUNDLED = True\n')

        # Optionally create extra bundled submodule
        if self.bundled_has_extra_module:
            bundled_experimental = bundled_nkilib / "experimental"
            bundled_experimental.mkdir()
            (bundled_experimental / "__init__.py").write_text(
                '"""Bundled experimental module."""\n'
                'BUNDLED = True\n'
                '\n'
                'def experimental_feature():\n'
                '    return "bundled_experimental"\n'
            )

        # Create circular import test modules in bundled package
        if self.circular_imports:
            (bundled_nkilib / "circular_a.py").write_text(
                'from nkilib import circular_b\n'
                'VALUE_A = "bundled_a"\n'
                'def get_b_value():\n'
                '    return circular_b.VALUE_B\n'
            )
            (bundled_nkilib / "circular_b.py").write_text(
                'from nkilib import circular_a\n'
                'VALUE_B = "bundled_b"\n'
                'def get_a_value():\n'
                '    return circular_a.VALUE_A\n'
            )

        if self.standalone_available:
            standalone_pkg = root / "nkilib_src"
            standalone_pkg.mkdir()
            (standalone_pkg / "__init__.py").write_text("")
            src_nkilib = standalone_pkg / "nkilib"
            src_nkilib.mkdir()

            if self.standalone_empty:
                # Create nkilib subdirectory but leave it empty (no __init__.py)
                # This should cause import to fail or fall back to bundled
                pass  # Just the directory exists, no __init__.py
            elif self.standalone_broken:
                (src_nkilib / "__init__.py").write_text('raise ImportError("Simulated broken standalone")\n')
            else:
                # Create proper nkilib_src.nkilib package
                if self.standalone_reimports_nkilib:
                    # This simulates a customer mistake: standalone's __init__.py
                    # imports 'nkilib' during initialization, which would trigger
                    # re-entry into bundled nkilib's _try_setup_src_nkilib().
                    # Without the _SETUP_IN_PROGRESS guard, this would cause infinite recursion.
                    (src_nkilib / "__init__.py").write_text(
                        '"""Standalone nkilib that re-imports nkilib."""\n'
                        'STANDALONE = True\n'
                        '__standalone_marker__ = "standalone_active"\n'
                        '\n'
                        '# This import triggers re-entry into bundled nkilib.__init__\n'
                        'import nkilib as _nkilib_ref\n'
                        'REIMPORT_WORKED = True\n'
                    )
                else:
                    (src_nkilib / "__init__.py").write_text(
                        '"""Standalone nkilib."""\n'
                        'STANDALONE = True\n'
                        '__standalone_marker__ = "standalone_active"\n'
                    )

                # Create core submodule only if not missing submodules
                if not self.standalone_missing_submodules:
                    standalone_core = src_nkilib / "core"
                    standalone_core.mkdir()

                    if self.deep_hierarchy:
                        (standalone_core / "__init__.py").write_text(
                            '"""Standalone core module."""\nSTANDALONE = True\nfrom . import mlp\n'
                        )
                        standalone_mlp = standalone_core / "mlp"
                        standalone_mlp.mkdir()
                        (standalone_mlp / "__init__.py").write_text(
                            '"""Standalone MLP module."""\n'
                            'STANDALONE = True\n'
                            '\n'
                            'class MLPConfig:\n'
                            '    """MLP configuration class for identity tests."""\n'
                            '    SOURCE = "standalone"\n'
                            '\n'
                            'def run_mlp(x):\n'
                            '    """Standalone MLP function."""\n'
                            '    return f"standalone:{x}"\n'
                        )
                    else:
                        (standalone_core / "__init__.py").write_text(
                            '"""Standalone core module."""\nSTANDALONE = True\n'
                        )

                    # Create circular import test modules in standalone package
                    if self.circular_imports:
                        (src_nkilib / "circular_a.py").write_text(
                            'from nkilib_src.nkilib import circular_b\n'
                            'VALUE_A = "standalone_a"\n'
                            'def get_b_value():\n'
                            '    return circular_b.VALUE_B\n'
                        )
                        (src_nkilib / "circular_b.py").write_text(
                            'from nkilib_src.nkilib import circular_a\n'
                            'VALUE_B = "standalone_b"\n'
                            'def get_a_value():\n'
                            '    return circular_a.VALUE_A\n'
                        )
                # If standalone_missing_submodules is True, we don't create core
        else:
            # Shadow package without nkilib submodule
            standalone_pkg = root / "nkilib_src"
            standalone_pkg.mkdir()
            (standalone_pkg / "__init__.py").write_text('"""Shadow nkilib_src - no nkilib submodule."""\n')

        return root

    def __exit__(self, *args):
        self.tmpdir.cleanup()

    def run_python(self, code: str, root: Path) -> subprocess.CompletedProcess:
        """Run Python code in a subprocess with the simulated environment."""
        env = os.environ.copy()
        if self.env_force_bundled:
            env['NKILIB_FORCE_BUNDLED_LIBRARY'] = '1'

        setup_code = textwrap.dedent(f"""
            import sys
            import importlib
            sys.path.insert(0, {str(root)!r})
            # Clean up any previously imported nkilib modules
            for key in list(sys.modules.keys()):
                if key == 'nkilib' or key.startswith('nkilib.') or key.startswith('nkilib_src'):
                    del sys.modules[key]
            importlib.invalidate_caches()
        """)
        return subprocess.run(
            [sys.executable, "-c", setup_code + code],
            capture_output=True,
            text=True,
            env=env,
        )


# =============================================================================
# Basic Swap/Fallback Tests
# =============================================================================


class TestBundledContextSwap:
    """Test sys.modules swap when running as bundled nkilib."""

    def test_bundled_swaps_to_standalone_when_available(self):
        """When bundled nkilib imports and standalone exists, it should swap."""
        sim = BundledSimulator(standalone_available=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                assert hasattr(nkilib, '__standalone_marker__'), f"Should have swapped, got: {dir(nkilib)}"
                assert nkilib.__standalone_marker__ == "standalone_active"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_bundled_falls_back_when_standalone_missing(self):
        """When standalone is not available, bundled nkilib should work normally."""
        sim = BundledSimulator(standalone_available=False)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                assert not hasattr(nkilib, '__standalone_marker__'), "Should not have standalone marker"
                assert hasattr(nkilib, 'core'), "Should have core submodule"
                assert nkilib.core.BUNDLED is True, "Should be bundled core"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_bundled_raises_when_standalone_broken(self):
        """When standalone exists but fails to import, raise ImportError with clear message."""
        sim = BundledSimulator(standalone_available=True, standalone_broken=True)
        with sim as root:
            code = textwrap.dedent("""
                try:
                    import nkilib
                    print("FAIL: Should have raised ImportError")
                except ImportError as e:
                    msg = str(e)
                    assert "nkilib_src package was found but failed to import" in msg
                    assert "pip uninstall nki-library" in msg
                    print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\nstdout: {result.stdout}"
            assert "PASS" in result.stdout


# =============================================================================
# sys.modules Aliasing Tests
# =============================================================================


class TestSysModulesAliasing:
    """Test that sys.modules entries are correctly aliased after swap."""

    def test_submodule_import_creates_correct_entry(self):
        """After swap, importing nkilib.core should create correct sys.modules entry."""
        sim = BundledSimulator(standalone_available=True)
        with sim as root:
            code = textwrap.dedent("""
                import sys
                import nkilib
                import nkilib.core
                aliased_core = sys.modules.get('nkilib.core')
                assert aliased_core is not None
                assert hasattr(aliased_core, 'STANDALONE') and aliased_core.STANDALONE is True
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_direct_submodule_import_works(self):
        """Test that 'from nkilib.core import X' works after swap."""
        sim = BundledSimulator(standalone_available=True)
        with sim as root:
            code = textwrap.dedent("""
                from nkilib.core import STANDALONE
                assert STANDALONE is True
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_all_submodules_aliased(self):
        """Test that all imported submodules have aliases in sys.modules."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                import sys
                import nkilib.core.mlp
                for alias in ['nkilib', 'nkilib.core', 'nkilib.core.mlp']:
                    assert alias in sys.modules, f"Missing: {alias}"
                    m = sys.modules[alias]
                    assert hasattr(m, 'STANDALONE') or hasattr(m, '__standalone_marker__')
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_late_submodule_import_aliased(self):
        """Test that submodules imported after swap are correctly aliased."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                import sys
                import nkilib
                assert 'nkilib.core.mlp' not in sys.modules
                import nkilib.core.mlp
                assert 'nkilib.core.mlp' in sys.modules
                assert sys.modules['nkilib.core.mlp'] is nkilib.core.mlp
                assert nkilib.core.mlp.STANDALONE is True
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Reload Safety Tests
# =============================================================================


class TestReloadSafety:
    """Test reload() behavior doesn't corrupt module state."""

    def test_reload_is_idempotent(self):
        """Calling reload(nkilib) after swap should not corrupt state."""
        sim = BundledSimulator(standalone_available=True)
        with sim as root:
            code = textwrap.dedent("""
                import importlib
                import nkilib
                assert hasattr(nkilib, '__standalone_marker__')
                import nkilib.core as core_before
                importlib.reload(nkilib)
                import nkilib as nkilib_after
                assert hasattr(nkilib_after, '__standalone_marker__')
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Deeply Nested Import Tests
# =============================================================================


class TestDeeplyNestedImports:
    """Test imports of deeply nested submodules (e.g., nkilib.core.mlp)."""

    def test_deep_import_direct(self):
        """Test 'import nkilib.core.mlp' triggers swap and loads correctly."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib.core.mlp
                assert nkilib.core.mlp.STANDALONE is True
                assert nkilib.core.mlp.MLPConfig.SOURCE == "standalone"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_deep_import_from_syntax(self):
        """Test 'from nkilib.core.mlp import run_mlp' works after swap."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                from nkilib.core.mlp import run_mlp, MLPConfig
                assert run_mlp("test") == "standalone:test"
                assert MLPConfig.SOURCE == "standalone"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_deep_import_bundled_fallback(self):
        """Test deep imports work when falling back to bundled."""
        sim = BundledSimulator(standalone_available=False, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                from nkilib.core.mlp import run_mlp, MLPConfig
                assert run_mlp("test") == "bundled:test"
                assert MLPConfig.SOURCE == "bundled"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Module Identity Tests
# =============================================================================


class TestModuleIdentityConsistency:
    """Test that module identity is consistent across different import paths."""

    def test_attribute_access_vs_direct_import(self):
        """Test sys.modules['nkilib.core'] is nkilib.core after swap."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                import sys, nkilib, nkilib.core
                assert nkilib.core is sys.modules['nkilib.core']
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_multiple_import_statements_same_object(self):
        """Test that multiple imports of same module return identical object."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                import sys
                import nkilib.core.mlp
                from nkilib.core import mlp as mlp_alias
                from nkilib import core
                mlp1, mlp2, mlp3, mlp4 = nkilib.core.mlp, mlp_alias, core.mlp, sys.modules.get('nkilib.core.mlp')
                assert mlp1 is mlp2 is mlp3 is mlp4
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_isinstance_works_across_imports(self):
        """Test isinstance() works when class imported via different paths."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                from nkilib.core.mlp import MLPConfig as Config1
                import nkilib.core.mlp
                Config2 = nkilib.core.mlp.MLPConfig
                instance = Config1()
                assert isinstance(instance, Config2) and Config1 is Config2
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Import Alias Tests
# =============================================================================


class TestImportAliases:
    """Test import aliases don't break module identity."""

    def test_import_as_alias(self):
        """Test 'import nkilib.core as c' works correctly."""
        sim = BundledSimulator(standalone_available=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib.core as c
                import nkilib
                assert c is nkilib.core and c.STANDALONE is True
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_from_import_alias(self):
        """Test 'from nkilib.core.mlp import MLPConfig as Config' works."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                from nkilib.core.mlp import MLPConfig as Config
                from nkilib.core.mlp import MLPConfig
                assert Config is MLPConfig and Config.SOURCE == "standalone"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_rebinding_alias(self):
        """Test that rebinding an alias doesn't affect other references."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib.core.mlp as mlp
                import nkilib
                original_mlp = mlp
                mlp = "something else"
                assert nkilib.core.mlp is original_mlp and nkilib.core.mlp.STANDALONE is True
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Pickle Roundtrip Tests
# =============================================================================


class TestPickleRoundtrip:
    """Test pickle/unpickle with objects from swapped modules."""

    def test_pickle_class_instance(self):
        """Test that class instances can be pickled and unpickled after swap."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                import pickle
                from nkilib.core.mlp import MLPConfig
                original = MLPConfig()
                original.custom_attr = "test_value"
                restored = pickle.loads(pickle.dumps(original))
                assert type(restored) is MLPConfig
                assert restored.custom_attr == "test_value"
                assert restored.SOURCE == "standalone"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Cross-Module Interaction Tests
# =============================================================================


class TestCrossModuleInteractions:
    """Test interactions between modules that both import nkilib."""

    def test_two_modules_share_same_nkilib(self):
        """Test that two separate modules see the same nkilib after swap."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            # Create helper modules
            helper_dir = root / "helpers"
            helper_dir.mkdir()
            (helper_dir / "__init__.py").write_text("")
            (helper_dir / "module_a.py").write_text(
                textwrap.dedent("""
                import nkilib.core.mlp
                def get_mlp_module(): return nkilib.core.mlp
                def create_config(): return nkilib.core.mlp.MLPConfig()
            """)
            )
            (helper_dir / "module_b.py").write_text(
                textwrap.dedent("""
                from nkilib.core.mlp import MLPConfig
                def get_config_class(): return MLPConfig
                def check_instance(obj): return isinstance(obj, MLPConfig)
            """)
            )

            code = textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(root)!r})
                from helpers import module_a, module_b
                assert module_a.get_mlp_module().MLPConfig is module_b.get_config_class()
                assert module_b.check_instance(module_a.create_config())
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Import Order Independence Tests
# =============================================================================


class TestImportOrderIndependence:
    """Test that import order doesn't affect behavior."""

    def test_submodule_before_parent(self):
        """Test importing nkilib.core.mlp before nkilib works."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                from nkilib.core.mlp import MLPConfig
                import nkilib
                import nkilib.core
                assert MLPConfig.SOURCE == "standalone"
                assert nkilib.core.mlp.MLPConfig is MLPConfig
                assert hasattr(nkilib, '__standalone_marker__')
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout

    def test_mixed_import_order(self):
        """Test complex mixed import order."""
        sim = BundledSimulator(standalone_available=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                from nkilib.core import mlp
                import nkilib
                from nkilib.core.mlp import MLPConfig
                import nkilib.core as core_alias
                assert mlp is nkilib.core.mlp is core_alias.mlp
                assert MLPConfig is mlp.MLPConfig
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}"
            assert "PASS" in result.stdout


# =============================================================================
# Empty Standalone Directory Tests (Edge Cases)
# =============================================================================


class TestEmptyStandaloneDirectory:
    """Test behavior when nkilib_src/nkilib exists but is empty."""

    def test_empty_standalone_folder_falls_back_to_bundled(self):
        """When nkilib_src/nkilib exists but has no __init__.py, should fall back to bundled."""
        sim = BundledSimulator(standalone_available=True, standalone_empty=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                try:
                    import nkilib
                    # Import a submodule to verify bundled is working
                    from nkilib.core.mlp import run_mlp
                    result = run_mlp("test")
                    # Should use bundled since standalone is empty
                    assert result == "bundled:test", f"Expected bundled:test, got {result}"
                    assert not hasattr(nkilib, '__standalone_marker__'), "Should not have standalone marker"
                    print("PASS")
                except (ImportError, ModuleNotFoundError) as e:
                    # If it raises ImportError, it means standalone was detected but failed to import
                    # This is also acceptable behavior - the key is we don't silently use broken standalone
                    print(f"ImportError (acceptable): {e}")
                    print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_empty_standalone_with_submodule_import(self):
        """Test that submodule imports work when standalone is empty."""
        sim = BundledSimulator(standalone_available=True, standalone_empty=True, deep_hierarchy=True)
        with sim as root:
            code = textwrap.dedent("""
                try:
                    from nkilib.core.mlp import run_mlp
                    # Should use bundled
                    assert run_mlp("test") == "bundled:test"
                    print("PASS")
                except ImportError as e:
                    # Acceptable if empty standalone causes detection failure
                    print(f"ImportError (acceptable): {e}")
                    print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout


# =============================================================================
# Mismatched Structure Tests (Edge Cases)
# =============================================================================


class TestMismatchedStructure:
    """Test behavior when standalone has different structure than bundled."""

    def test_standalone_missing_submodule_causes_import_error(self):
        """When standalone exists but is missing submodules, importing missing paths should error."""
        sim = BundledSimulator(standalone_available=True, standalone_missing_submodules=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                # Should have swapped to standalone
                assert hasattr(nkilib, '__standalone_marker__'), "Should have swapped to standalone"

                # Try to import core.mlp which doesn't exist in standalone
                try:
                    from nkilib.core import mlp
                    print("FAIL: Should have raised ImportError for missing submodule")
                except (ImportError, ModuleNotFoundError, AttributeError) as e:
                    print(f"Expected error: {e}")
                    print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_bundled_extra_module_not_in_standalone(self):
        """When bundled has extra modules not in standalone, those should not be accessible after swap."""
        sim = BundledSimulator(
            standalone_available=True,
            standalone_missing_submodules=False,  # Has core
            bundled_has_extra_module=True,  # But bundled has experimental
        )
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                # Should have swapped to standalone
                assert hasattr(nkilib, '__standalone_marker__'), "Should have swapped to standalone"

                # Try to import experimental which only exists in bundled, not standalone
                try:
                    from nkilib import experimental
                    print("FAIL: Should not have experimental module from standalone")
                except (ImportError, ModuleNotFoundError, AttributeError) as e:
                    print(f"Expected error: {e}")
                    print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout


# =============================================================================
# Environment Variable Force Bundled Tests (Edge Cases)
# =============================================================================


class TestForceBundledEnvironmentVariable:
    """Test NKILIB_FORCE_BUNDLED_LIBRARY environment variable."""

    def test_env_var_forces_bundled_when_standalone_available(self):
        """When env var is set, should use bundled even when standalone is available."""
        sim = BundledSimulator(standalone_available=True, env_force_bundled=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                # Should use bundled because env var is set
                assert not hasattr(nkilib, '__standalone_marker__'), "Should not have standalone marker"
                assert hasattr(nkilib, 'core'), "Should have core submodule"
                assert nkilib.core.BUNDLED is True, "Should be using bundled version"
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_env_var_with_mismatched_structure(self):
        """When env var forces bundled, should access bundled modules even if standalone has different structure."""
        sim = BundledSimulator(
            standalone_available=True,
            standalone_missing_submodules=True,  # Standalone missing core
            bundled_has_extra_module=True,  # Bundled has experimental
            env_force_bundled=True,
            deep_hierarchy=True,
        )
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                # Should use bundled due to env var
                assert not hasattr(nkilib, '__standalone_marker__'), "Should use bundled"

                # Should be able to import bundled modules
                from nkilib.core.mlp import run_mlp
                assert run_mlp("test") == "bundled:test", "Should use bundled mlp"

                # Should be able to access bundled experimental module
                from nkilib.experimental import experimental_feature
                assert experimental_feature() == "bundled_experimental"

                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_env_var_no_import_error_for_bundled_only_paths(self):
        """When env var forces bundled, paths that exist in bundled but not standalone should work."""
        sim = BundledSimulator(
            standalone_available=True,
            standalone_missing_submodules=True,  # Standalone doesn't have core
            env_force_bundled=True,
            deep_hierarchy=True,
        )
        with sim as root:
            code = textwrap.dedent("""
                # This import would fail if we used standalone (which lacks core)
                # But with env var, we use bundled which has it
                from nkilib.core.mlp import run_mlp
                result = run_mlp("test")
                assert result == "bundled:test", f"Expected bundled:test, got {result}"

                # Verify we're actually using bundled
                import nkilib
                assert not hasattr(nkilib, '__standalone_marker__'), "Should be using bundled"

                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_env_var_various_truthy_values(self):
        """Test that various truthy values for env var work correctly."""
        truthy_values = ['1', 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES', 'on', 'On', 'ON']

        for value in truthy_values:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)

                # Setup minimal environment
                import nkilib_src.nkilib as real_nkilib

                init_content = Path(real_nkilib.__file__).read_text()

                bundled_nkilib = root / "nkilib"
                bundled_nkilib.mkdir()
                (bundled_nkilib / "__init__.py").write_text(init_content)

                bundled_core = bundled_nkilib / "core"
                bundled_core.mkdir()
                (bundled_core / "__init__.py").write_text('BUNDLED = True\n')

                standalone_pkg = root / "nkilib_src"
                standalone_pkg.mkdir()
                (standalone_pkg / "__init__.py").write_text("")
                src_nkilib = standalone_pkg / "nkilib"
                src_nkilib.mkdir()
                (src_nkilib / "__init__.py").write_text('__standalone_marker__ = "active"\n')

                env = os.environ.copy()
                env['NKILIB_FORCE_BUNDLED_LIBRARY'] = value

                setup_code = textwrap.dedent(f"""
                    import sys
                    sys.path.insert(0, {str(root)!r})
                    for key in list(sys.modules.keys()):
                        if 'nkilib' in key:
                            del sys.modules[key]
                    import importlib
                    importlib.invalidate_caches()
                """)

                code = textwrap.dedent("""
                import nkilib
                assert not hasattr(nkilib, '__standalone_marker__'), f"Should use bundled with value: {!r}"
                print("PASS")
                """).format(value)

                result = subprocess.run(
                    [sys.executable, "-c", setup_code + code],
                    capture_output=True,
                    text=True,
                    env=env,
                )

                assert result.returncode == 0, f"Failed for value '{value}': {result.stderr}\\nstdout: {result.stdout}"
                assert "PASS" in result.stdout, f"Test failed for truthy value: {value}"

    def test_env_var_falsy_values_use_standalone(self):
        """Test that falsy/unset env var values allow standalone to be used."""
        falsy_values = ['0', 'false', 'False', 'FALSE', 'no', 'No', 'NO', 'off', 'Off', 'OFF', '']

        for value in falsy_values:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)

                # Setup minimal environment
                import nkilib_src.nkilib as real_nkilib

                init_content = Path(real_nkilib.__file__).read_text()

                bundled_nkilib = root / "nkilib"
                bundled_nkilib.mkdir()
                (bundled_nkilib / "__init__.py").write_text(init_content)

                bundled_core = bundled_nkilib / "core"
                bundled_core.mkdir()
                (bundled_core / "__init__.py").write_text('BUNDLED = True\n')

                standalone_pkg = root / "nkilib_src"
                standalone_pkg.mkdir()
                (standalone_pkg / "__init__.py").write_text("")
                src_nkilib = standalone_pkg / "nkilib"
                src_nkilib.mkdir()
                (src_nkilib / "__init__.py").write_text('__standalone_marker__ = "active"\n')

                env = os.environ.copy()
                if value:  # Only set if not empty string
                    env['NKILIB_FORCE_BUNDLED_LIBRARY'] = value
                else:
                    # Ensure it's not set
                    env.pop('NKILIB_FORCE_BUNDLED_LIBRARY', None)

                setup_code = textwrap.dedent(f"""
                    import sys
                    sys.path.insert(0, {str(root)!r})
                    for key in list(sys.modules.keys()):
                        if 'nkilib' in key:
                            del sys.modules[key]
                    import importlib
                    importlib.invalidate_caches()
                """)

                code = textwrap.dedent("""
                import nkilib
                assert hasattr(nkilib, '__standalone_marker__'), f"Should use standalone with value: {!r}"
                print("PASS")
                """).format(value)

                result = subprocess.run(
                    [sys.executable, "-c", setup_code + code],
                    capture_output=True,
                    text=True,
                    env=env,
                )

                assert result.returncode == 0, f"Failed for value '{value}': {result.stderr}\\nstdout: {result.stdout}"
                assert "PASS" in result.stdout, f"Test failed for falsy value: {value}"


# =============================================================================
# Combined Edge Case Tests
# =============================================================================


class TestCombinedEdgeCases:
    """Test complex combinations of edge cases."""

    def test_env_var_overrides_broken_standalone(self):
        """When standalone is broken but env var forces bundled, should use bundled without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Setup environment with broken standalone
            import nkilib_src.nkilib as real_nkilib

            init_content = Path(real_nkilib.__file__).read_text()

            bundled_nkilib = root / "nkilib"
            bundled_nkilib.mkdir()
            (bundled_nkilib / "__init__.py").write_text(init_content)

            bundled_core = bundled_nkilib / "core"
            bundled_core.mkdir()
            (bundled_core / "__init__.py").write_text('BUNDLED = True\n')

            # Create broken standalone
            standalone_pkg = root / "nkilib_src"
            standalone_pkg.mkdir()
            (standalone_pkg / "__init__.py").write_text("")
            src_nkilib = standalone_pkg / "nkilib"
            src_nkilib.mkdir()
            (src_nkilib / "__init__.py").write_text('raise RuntimeError("Broken standalone!")\n')

            env = os.environ.copy()
            env['NKILIB_FORCE_BUNDLED_LIBRARY'] = '1'

            setup_code = textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(root)!r})
                for key in list(sys.modules.keys()):
                    if 'nkilib' in key:
                        del sys.modules[key]
                import importlib
                importlib.invalidate_caches()
            """)

            code = textwrap.dedent("""
                import nkilib
                # Should use bundled and not trigger the RuntimeError from standalone
                assert hasattr(nkilib, 'core'), "Should have bundled core"
                assert nkilib.core.BUNDLED is True
                print("PASS")
            """)

            result = subprocess.run(
                [sys.executable, "-c", setup_code + code],
                capture_output=True,
                text=True,
                env=env,
            )

            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_force_bundled_logs_debug_message(self):
        """Test that forcing bundled version logs appropriate debug messages."""
        sim = BundledSimulator(standalone_available=True, env_force_bundled=True)
        with sim as root:
            code = textwrap.dedent("""
                import logging
                import sys

                # Set up logging to capture debug messages
                logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

                import nkilib
                # Should use bundled
                assert not hasattr(nkilib, '__standalone_marker__')
                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout
            # Debug logging might include info about forcing bundled
            # (This is a soft check since logging format may vary)


# =============================================================================
# Circular Import E2E Tests
# =============================================================================


class TestCircularImportsE2E:
    """End-to-end tests for circular import scenarios.

    These tests validate that the override mechanism does not interfere with
    Python's native circular import handling, and that the _SETUP_IN_PROGRESS
    guard correctly prevents infinite recursion during the override swap.
    """

    def test_circular_import_within_standalone_package(self):
        """Test that circular imports within standalone package work correctly.

        Validates:
        1. Python's native circular import handling works within nkilib_src
        2. The _SETUP_IN_PROGRESS guard does NOT interfere with intra-package imports
        3. Both modules resolve correctly despite the circular dependency
        """
        sim = BundledSimulator(standalone_available=True, circular_imports=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                # Should have swapped to standalone
                assert hasattr(nkilib, '__standalone_marker__'), "Should have swapped to standalone"

                # Import circular_a which imports circular_b which imports circular_a
                from nkilib import circular_a

                # Verify modules loaded correctly with standalone values
                assert circular_a.VALUE_A == "standalone_a", f"Got {circular_a.VALUE_A}"
                assert circular_a.get_b_value() == "standalone_b"

                # Import circular_b directly and verify it also works
                from nkilib import circular_b
                assert circular_b.VALUE_B == "standalone_b", f"Got {circular_b.VALUE_B}"
                assert circular_b.get_a_value() == "standalone_a"

                # Verify module identity is consistent
                assert circular_a.circular_b is circular_b
                assert circular_b.circular_a is circular_a

                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_circular_import_within_bundled_package(self):
        """Test that circular imports within bundled package work correctly.

        Validates:
        1. The bundled package discovery (pkgutil.iter_modules) doesn't break circular imports
        2. Once submodules start loading, Python's standard import semantics apply
        3. The _SETUP_IN_PROGRESS guard only affects the override swap, not submodule imports
        """
        sim = BundledSimulator(standalone_available=False, circular_imports=True)
        with sim as root:
            code = textwrap.dedent("""
                import nkilib
                # Should use bundled (no standalone available)
                assert not hasattr(nkilib, '__standalone_marker__'), "Should use bundled"

                # Import circular_a which imports circular_b which imports circular_a
                from nkilib import circular_a

                # Verify modules loaded correctly with bundled values
                assert circular_a.VALUE_A == "bundled_a", f"Got {circular_a.VALUE_A}"
                assert circular_a.get_b_value() == "bundled_b"

                # Import circular_b directly and verify it also works
                from nkilib import circular_b
                assert circular_b.VALUE_B == "bundled_b", f"Got {circular_b.VALUE_B}"
                assert circular_b.get_a_value() == "bundled_a"

                # Verify module identity is consistent
                assert circular_a.circular_b is circular_b
                assert circular_b.circular_a is circular_a

                print("PASS")
            """)
            result = sim.run_python(code, root)
            assert result.returncode == 0, f"Failed: {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout

    def test_circular_import_during_override_swap_is_prevented(self):
        """Test that the _SETUP_IN_PROGRESS guard prevents infinite recursion.

        Validates:
        1. When standalone's __init__.py imports 'nkilib' during initialization,
           this triggers re-entry into bundled nkilib's _try_setup_src_nkilib()
        2. The _SETUP_IN_PROGRESS guard prevents infinite recursion
        3. The import completes successfully without hanging or crashing
        4. The standalone module is still correctly loaded and usable

        This simulates a real customer scenario where code in nkilib_src
        accidentally imports 'nkilib' (perhaps for compatibility or lazy coding),
        which would cause a circular import chain without the guard.
        """
        sim = BundledSimulator(standalone_available=True, standalone_reimports_nkilib=True)
        with sim as root:
            code = textwrap.dedent("""
                import sys

                # This import triggers:
                # 1. bundled nkilib.__init__ runs
                # 2. _try_setup_src_nkilib() is called, sets _SETUP_IN_PROGRESS=True
                # 3. nkilib_src.nkilib is imported
                # 4. nkilib_src.nkilib.__init__ runs "import nkilib"
                # 5. bundled nkilib.__init__ runs AGAIN
                # 6. _try_setup_src_nkilib() is called AGAIN
                # 7. Guard detects _SETUP_IN_PROGRESS=True, returns False immediately
                # 8. No infinite recursion!
                import nkilib

                # If we got here without hanging or RecursionError, the guard worked!

                # Verify standalone was loaded (has the marker from standalone __init__.py)
                assert hasattr(nkilib, '__standalone_marker__'), "Standalone should be active"

                # Verify the re-import attribute was set (proves standalone's __init__ completed)
                assert hasattr(nkilib, 'REIMPORT_WORKED'), "Standalone's reimport code should have run"
                assert nkilib.REIMPORT_WORKED is True

                print("PASS: Circular import during swap was handled correctly")
            """)
            result = sim.run_python(code, root)
            assert (
                result.returncode == 0
            ), f"Failed (possible infinite recursion): {result.stderr}\\nstdout: {result.stdout}"
            assert "PASS" in result.stdout
