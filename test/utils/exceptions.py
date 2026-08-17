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
import subprocess
from enum import Enum
from typing import final

import fabric2


@final
class TestStatus(str, Enum):
    """Test execution status."""

    SUCCESS = "SUCCESS"
    COMPILATION_FAILURE = "COMPILATION_FAILURE"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    EXPECTED_FAILURE = "EXPECTED_FAILURE"
    TEST_EXECUTION_FAILURE = "TEST_EXECUTION_FAILURE"


class RemoteExecutionException(Exception):
    def __init__(self, message: str, result: fabric2.Result, *args: object) -> None:
        super().__init__(
            f"{message}\n===STDOUT===\n{result.stdout}\n===STDERR===\n{result.stderr}\n",
            *args,
        )


class RemoteFileTransferException(Exception):
    def __init__(self, message: str, e: Exception, *args: object) -> None:
        super().__init__(f"{message}\n", e, *args)


class LocalExecutionException(Exception):
    def __init__(self, message: str, result: subprocess.CompletedProcess, *args: object) -> None:
        super().__init__(
            f"{message}\n===STDOUT===\n{result.stdout}\n===STDERR===\n{result.stderr}\n",
            *args,
        )


class TimeoutException(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)


class QueuePatienceRotation(TimeoutException):
    """Transient signal that this host's core-allocation queue ETA exceeds the
    caller's patience right now, so the caller should release its slot and try a
    different host. This is not a host failure; the host stays eligible for
    re-selection. Subclasses the timeout error so existing catch sites that expect
    a timeout continue to work until a caller branches on this type explicitly."""


class NoNeuronDevicesException(Exception):
    def __init__(self, host_alias: str):
        super().__init__(f"No Neuron devices found on {host_alias}")


class CompilationException(Exception):
    """Raised when kernel compilation fails."""

    status = TestStatus.COMPILATION_FAILURE


class InferenceException(Exception):
    """Raised when kernel execution fails on hardware."""

    status = TestStatus.INFERENCE_FAILURE


class FleetEmptyError(InferenceException):
    """Raised when a host claim is attempted while the fleet is poisoned — no hosts
    are available and the fleet stayed empty long enough, so an individual claim must fail
    fast. Unretriable.

    Subclasses ``InferenceException`` so the orchestrator surfaces it as a clean
    inference failure."""


class ValidationException(Exception):
    """Raised when output validation fails."""

    status = TestStatus.VALIDATION_FAILURE


class HostsBusyError(Exception):
    """Raised when no host could be reserved for an exclusive session."""


class UnimplementedException(Exception):
    def __init__(self, message: str = "Unimplemented", *args: object) -> None:
        super().__init__(message, *args)
