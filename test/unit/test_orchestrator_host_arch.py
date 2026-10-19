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

import contextlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, create_autospec

import pytest

from test.utils.common_dataclasses import KernelArgs, Platforms, TraceMode
from test.utils.exceptions import InferenceException
from test.utils.host_management import SshHost, SshHostGroup
from test.utils.test_orchestrator import FilesystemArgs, Orchestrator


class _StopAfterArchCheck(Exception):
    pass


def _node(alias: str) -> SshHost:
    """A 128-core node: 16 devices of 8 cores each."""
    member = create_autospec(SshHost, instance=True)
    member.get_host_id.return_value = alias
    member.get_neuron_device_info.return_value = [SimpleNamespace(nc_count=8) for _ in range(16)]
    return member


def test_run_inference_validates_each_node_of_a_multi_node_target(tmp_path):
    """The per-node architecture check runs on every member, so a two-node target whose combined
    core count is not a single-node size still passes and proceeds to the queue join."""
    group = SshHostGroup(members=(_node("m0"), _node("m1")))
    group.soft_join_queue = MagicMock(side_effect=_StopAfterArchCheck())

    def one_attempt(**_kwargs):
        yield contextlib.nullcontext(group)

    orchestrator = object.__new__(Orchestrator)
    orchestrator.trace_mode = TraceMode.CompileAndInfer
    orchestrator.ucode_lib_path = None
    orchestrator.collector = MagicMock()
    orchestrator.fs_config = cast(
        FilesystemArgs,
        SimpleNamespace(
            artifacts_output_directory_path=str(tmp_path),
            host_manager=SimpleNamespace(get_host_assignment_with_retry=one_attempt),
        ),
    )
    kernel_under_test = SimpleNamespace(
        compiler_input=SimpleNamespace(platform_target=Platforms.TRN2, logical_nc_config=1),
        inference_args=SimpleNamespace(collective_ranks=256),
    )

    with pytest.raises(InferenceException) as exc:
        orchestrator._run_inference(cast(KernelArgs, kernel_under_test), {})

    assert isinstance(exc.value.__cause__, _StopAfterArchCheck)
    for member in group.hosts:
        cast(MagicMock, member.get_neuron_device_info).assert_called_once_with()


def test_run_inference_tags_arch_check_failure_with_the_failing_node(tmp_path):
    """An arch-check failure on the second node is attributed to that node, not the group's primary."""
    bad = _node("m1")
    cast(MagicMock, bad.get_neuron_device_info).side_effect = OSError("neuron-ls failed")
    group = SshHostGroup(members=(_node("m0"), bad))
    group.soft_join_queue = MagicMock(side_effect=_StopAfterArchCheck())

    def one_attempt(**_kwargs):
        yield contextlib.nullcontext(group)

    orchestrator = object.__new__(Orchestrator)
    orchestrator.trace_mode = TraceMode.CompileAndInfer
    orchestrator.ucode_lib_path = None
    orchestrator.collector = MagicMock()
    orchestrator.fs_config = cast(
        FilesystemArgs,
        SimpleNamespace(
            artifacts_output_directory_path=str(tmp_path),
            host_manager=SimpleNamespace(get_host_assignment_with_retry=one_attempt),
        ),
    )
    kernel_under_test = SimpleNamespace(
        compiler_input=SimpleNamespace(platform_target=Platforms.TRN2, logical_nc_config=1),
        inference_args=SimpleNamespace(collective_ranks=256),
    )

    with pytest.raises(InferenceException) as exc:
        orchestrator._run_inference(cast(KernelArgs, kernel_under_test), {})

    cause = exc.value.__cause__
    assert isinstance(cause, OSError)
    assert cause.failed_host_id == "m1"  # ty: ignore[unresolved-attribute]
    group.soft_join_queue.assert_not_called()
