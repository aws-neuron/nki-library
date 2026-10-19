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

"""Host-side tests for the pure-Python TreeLogger.

TreeLogger buffers hierarchical log entries and renders them as an indented
tree. It is plain Python (no device tensors), so these tests construct it with
a real Logger and call its methods directly; they run under the simulator with
no device kernel or marker.
"""

import pytest
from nkilib_src.nkilib.core.utils.logging import Logger, LogLevel
from nkilib_src.nkilib.core.utils.tree_logger import LogEntry, TreeLogger


def _debug_logger():
    """Logger with DEBUG enabled so TreeLogger._enabled is True."""
    return Logger("test", LogLevel.DEBUG)


def _disabled_logger():
    """Logger at INFO level so DEBUG (and thus TreeLogger) is disabled."""
    return Logger("test", LogLevel.INFO)


@pytest.mark.fast
def test_log_disabled_early_return():
    """Disabled logger: log() takes the early return and buffers nothing (L46 -> L47)."""
    tl = TreeLogger("t", _disabled_logger())
    assert tl._enabled is False
    tl.log("ignored", 0)
    assert tl.stack_logs == []


@pytest.mark.fast
def test_log_enabled_appends_entry():
    """Enabled logger: log() appends a LogEntry (L46 false arc -> L48)."""
    tl = TreeLogger("t", _debug_logger())
    assert tl._enabled is True
    tl.log("hello", 1, is_scope_boundary=True)
    assert len(tl.stack_logs) == 1
    assert tl.stack_logs[0].msg == "hello"
    assert tl.stack_logs[0].depth == 1
    assert tl.stack_logs[0].is_scope_boundary is True


@pytest.mark.fast
def test_flush_empty_returns_without_printing(capsys):
    """flush() with no buffered logs takes the early return (L73 -> L74)."""
    tl = TreeLogger("t", _debug_logger())
    tl.flush()
    assert capsys.readouterr().out == ""


@pytest.mark.fast
def test_flush_renders_tree_and_clears(capsys):
    """flush() with a crafted depth sequence exercises _print_tree fully.

    Depth sequence [0, 1, 1, 2, 0] covers, in the first pass:
      - the ``depth_j == depth_i`` sibling match (L92 -> L93)
      - the ``depth_j < depth_i`` break (L90 -> L91)
      - the inner-loop fall-through continue (L92 -> L88)
    and, in the second pass:
      - a depth-0 entry (L100 false arc -> L107 directly)
      - a depth-1 entry with an empty ``range(depth-1)`` (L101 skips body)
      - a depth-2 entry that enters ``range(depth-1)`` (L101 -> L102)
      - both the last-child ``is_last`` True (L103 -> L104) and False (L103 -> L106) arcs.
    """
    tl = TreeLogger("root", _debug_logger())
    for depth in (0, 1, 1, 2, 0):
        tl.log(f"n{depth}", depth)
    tl.flush()
    out = capsys.readouterr().out
    assert "[root] Allocations:" in out
    assert "├── " in out  # non-last sibling
    assert "└── " in out  # last sibling
    assert "│   " in out  # deeper indentation
    # Buffer is cleared after flushing.
    assert tl.stack_logs == []


@pytest.mark.fast
def test_has_depth_after_found_and_not_found():
    """_has_depth_after: hit (L52 -> L53), skip-and-continue (L52 -> L51), miss (L51 -> L54)."""
    tl = TreeLogger("t", _debug_logger())
    logs = [LogEntry("a", 0, True), LogEntry("b", 1, True)]
    # start at 0, target depth 1: index 0 (depth 0) is skipped, index 1 matches.
    assert tl._has_depth_after(logs, 0, 1) is True
    # target depth never present -> loop exhausts and returns False.
    assert tl._has_depth_after(logs, 0, 5) is False


@pytest.mark.fast
def test_tree_prefix_depth_zero():
    """_tree_prefix at depth 0 returns empty string (L57 -> L58)."""
    tl = TreeLogger("t", _debug_logger())
    assert tl._tree_prefix(0, True, []) == ""


@pytest.mark.fast
def test_tree_prefix_parent_continues_and_last():
    """_tree_prefix with a continuing parent and a last child.

    depth=3 -> range(2): i=0 uses parent_continues[0]=True (L61 -> L62, vertical bar),
    i=1 uses parent_continues[1]=False (L61 -> L64, blank); is_last=True (L65 -> L66).
    """
    tl = TreeLogger("t", _debug_logger())
    prefix = tl._tree_prefix(3, True, [True, False])
    assert prefix == "│   " + "    " + "└── "


@pytest.mark.fast
def test_tree_prefix_short_parent_continues_and_not_last():
    """_tree_prefix where parent_continues is too short and the node is not last.

    depth=3 with empty parent_continues: both loop iterations take the blank branch
    via the ``i < len(parent_continues)`` guard being False (L61 -> L64); is_last=False
    (L65 -> L68).
    """
    tl = TreeLogger("t", _debug_logger())
    prefix = tl._tree_prefix(3, False, [])
    assert prefix == "    " + "    " + "├── "
