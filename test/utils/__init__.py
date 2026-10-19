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
"""NKI Library Testing — shared test utilities and pytest fixtures."""

import sys

_PREFIX = "test.utils."
_ALIAS_PREFIX = "nkilib_testing."


def _alias_for(fullname: str) -> str:
    return fullname.replace(_PREFIX, _ALIAS_PREFIX, 1)


# The redirector unifies ``test.utils.*`` (source tree) and ``nkilib_testing.*``
# (installed wheel, loaded by the pytest11 entry point) onto a single module
# object.  Without it Python creates separate module objects per path, breaking
# isinstance checks, mock.patch targets, class registries, and cross-module Enum
# identity (e.g. ``TraceMode`` comparisons).
#
# The finder API differs by interpreter version, so we branch on sys.version_info:
#   * Python >= 3.12 removed the legacy ``find_module``/``load_module`` finder
#     API, so we MUST implement the modern ``find_spec``/``exec_module`` protocol.
#   * Python < 3.12 (3.10/3.11) still honors the legacy API, which we keep to
#     avoid depending on import-machinery internals on the older runtime.
if sys.version_info >= (3, 12):
    from importlib.abc import Loader, MetaPathFinder
    from importlib.util import spec_from_loader

    class _NkilibTestingRedirector(MetaPathFinder, Loader):
        """Modern find_spec/exec_module redirector (Python >= 3.12)."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith(_PREFIX) and _alias_for(fullname) in sys.modules:
                # is_package mirrors the already-loaded alias module so submodule
                # imports beneath a redirected package keep resolving.
                is_package = hasattr(sys.modules[_alias_for(fullname)], "__path__")
                return spec_from_loader(fullname, self, is_package=is_package)
            return None

        def create_module(self, spec):
            # Reuse the existing alias module object so both names share identity.
            return sys.modules[_alias_for(spec.name)]

        def exec_module(self, module):
            # Module was already executed under its nkilib_testing name; no-op.
            sys.modules[module.__name__] = module

else:

    class _NkilibTestingRedirector:
        """Legacy find_module/load_module redirector (Python < 3.12)."""

        def find_module(self, fullname, path=None):
            if fullname.startswith(_PREFIX) and _alias_for(fullname) in sys.modules:
                return self
            return None

        def load_module(self, fullname):
            if fullname not in sys.modules:
                sys.modules[fullname] = sys.modules[_alias_for(fullname)]
            return sys.modules[fullname]


# Only install when running from the source tree (loaded as test.utils),
# not from the installed wheel (loaded as nkilib_testing).
#
# The nkilib_testing pytest plugin is auto-loaded from the installed wheel via
# its pytest11 entry point, creating nkilib_testing.* module objects.  Source-tree
# code imports the same modules as test.utils.* via relative imports.
if __name__ == "test.utils":
    sys.meta_path.insert(0, _NkilibTestingRedirector())
