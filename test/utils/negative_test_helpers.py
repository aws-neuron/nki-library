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
"""Utilities for negative (expected-failure) test cases.

Provides a context manager and helpers for tests that are expected to raise
kernel validation exceptions.  The ContextVar allows Orchestrator.execute()
to automatically detect negative tests without requiring callers to pass an
explicit flag.
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Optional

# Context variable to track if we're in a negative test case
# This allows Orchestrator.execute() to automatically detect negative tests
_is_negative_test_context: contextvars.ContextVar[bool] = contextvars.ContextVar('is_negative_test', default=False)


def is_in_negative_test_context() -> bool:
    """Check if we're currently inside an assert_negative_test_case context.

    This allows Orchestrator.execute() to automatically detect negative tests
    without requiring tests to explicitly pass is_negative_test to KernelArgs.
    """
    return _is_negative_test_context.get()


def call_with_invalid_argument(fn: Callable[..., object], *args: Any, **kwargs: Any) -> object:
    """Call ``fn`` with arguments that deliberately violate its declared parameter types.

    Use this only inside a ``pytest.raises`` block that asserts the callee rejects the
    bad value, so that the intent is explicit at the call site rather than looking like
    an accidental mistake.
    """
    return fn(*args, **kwargs)


def assert_kernel_validation_exception(expected_validation_error: Optional[str], exception: Exception):
    assert exception is not None, "Expected to receive neuron assertion exception, but none was given"
    actual_exception_message = exception.__str__()

    assert "[NCC_INKI016] Kernel validation exception:" in actual_exception_message
    if expected_validation_error is not None:
        assert actual_exception_message.__contains__(expected_validation_error)


@contextmanager
def assert_negative_test_case(
    is_negative_test_case: bool,
    expected_validation_error: Optional[str] = None,
):
    token = _is_negative_test_context.set(is_negative_test_case)
    try:
        yield
    except Exception as e:
        if not is_negative_test_case:
            raise e
        assert_kernel_validation_exception(expected_validation_error, e)
    else:
        assert not is_negative_test_case, "Test case was expected to fail, but it hasn't!"
    finally:
        _is_negative_test_context.reset(token)
