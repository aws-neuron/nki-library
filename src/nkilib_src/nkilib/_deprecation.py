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

"""Deprecation marking for NKI Lib public APIs.

Two symbol-level decorators, one per rollout path, so intent is explicit at the call
site and nothing is easy to omit by mistake:

* ``@deprecated(since=..., use_instead=...)`` — a **shipped** public API. ``since``
  (the release it shipped in) is **required**.
* ``@deprecated_unshipped(use_instead=...)`` — the **tight-loop** case: a consumed but
  not-yet-shipped API. Transient, so there is no release to record.

Both:

1. Emit a ``DeprecationWarning`` **when the decorator runs** (i.e. at import /
   definition time). It deliberately does **not** wrap the decorated object in a
   call-time wrapper: an @nki.jit kernel must stay the underlying kernel so the NKI
   compiler frontends compile it correctly. The tracer frontend tolerates a wrapper
   but the parser frontend statically compiles the object it is given, so a wrapper
   would break parser-based compilation. Returning the object unchanged keeps both
   frontends working; decorator ordering relative to ``@nki.jit`` therefore does not
   matter.
2. Stamp a structured ``__nki_deprecated__`` record so tooling can enumerate
   deprecations at runtime; a static AST scan distinguishes the two paths by decorator
   name without importing kernels.
3. Drive the documentation deprecation note (via the same record).

*Whether and when* a deprecated symbol is removed is decided by the removal /
enforcement policy, not by these decorators: the marker only records that a symbol is
deprecated and, for a shipped API, the release it was deprecated in. This keeps the
removal policy in one place and changeable without editing call sites. This module is
an internal (underscore-prefixed) utility and is not itself a public API.
"""

from __future__ import annotations

import re
import warnings
from typing import Callable, Optional, TypeVar

_T = TypeVar("_T")

#: A release identifier is ``MAJOR.MINOR`` or ``MAJOR.MINOR.PATCH`` (e.g. ``"2.33"``, or
#: ``"2.33.2"`` for a patch release). Release branches are the ground truth for what has
#: shipped; there are no git tags.
_RELEASE_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?$")


def _validate_release(value: str, field: str) -> None:
    if not isinstance(value, str) or not _RELEASE_RE.match(value):
        raise ValueError(f"{field} must be a release identifier like '2.33' or '2.33.2'; got {value!r}")


def _build_message(qualname: str, since: Optional[str], use_instead: Optional[str]) -> str:
    if since:
        msg = f"{qualname} is deprecated (since release {since}); it will be removed in a future release."
    else:
        msg = f"{qualname} is deprecated; it will be removed once downstream consumers integrate the replacement."
    if use_instead:
        msg += f" Use {use_instead} instead."
    return msg


def _make_decorator(since: Optional[str], use_instead: Optional[str]) -> Callable[[_T], _T]:
    record = {"since": since, "use_instead": use_instead}

    def _decorate(obj: _T) -> _T:
        qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
        message = _build_message(qualname, since, use_instead)
        warnings.warn(message, category=DeprecationWarning, stacklevel=2)
        # Stamp metadata for tooling/introspection. Return the object unchanged so
        # both NKI frontends still compile the underlying kernel; guard the attribute
        # writes in case an object disallows them.
        try:
            obj.__nki_deprecated__ = record
            obj.__deprecated__ = message  # PEP 702 interop attribute
        except (AttributeError, TypeError):
            pass
        return obj

    return _decorate


def deprecated(*, since: str, use_instead: Optional[str] = None) -> Callable[[_T], _T]:
    """Mark a **shipped** public function or class as deprecated.

    Args:
        since: Release the deprecation ships in (e.g. ``"2.33"``). **Required** — a
            shipped deprecation must record the release it was deprecated in.
        use_instead: Fully qualified replacement symbol, surfaced in the warning and
            documentation. Optional.

    Emits the ``DeprecationWarning`` at import / definition time and returns the object
    unchanged. Whether and when the symbol is removed is decided by the removal /
    enforcement policy, not here. For a consumed-but-not-yet-shipped API, use
    :func:`deprecated_unshipped` instead.

    Raises:
        ValueError: if ``since`` is not a release identifier.
    """
    _validate_release(since, "since")
    return _make_decorator(since, use_instead)


def deprecated_unshipped(*, use_instead: Optional[str] = None) -> Callable[[_T], _T]:
    """Mark a consumed-but-not-yet-shipped public function or class as deprecated.

    This is the tight-loop case: the API is on ``mainline`` but has not shipped in a
    release, so there is no release to record and removal happens once downstream
    consumers integrate the replacement (not on a release schedule).

    Emits the ``DeprecationWarning`` at import / definition time and returns the object
    unchanged.

    Args:
        use_instead: Fully qualified replacement symbol, surfaced in the warning and
            documentation. Optional.
    """
    return _make_decorator(None, use_instead)
