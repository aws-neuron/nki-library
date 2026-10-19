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
"""Unit tests for negative_test_helpers module."""

import pytest

from test.utils.negative_test_helpers import (
    assert_kernel_validation_exception,
    assert_negative_test_case,
    call_with_invalid_argument,
    is_in_negative_test_context,
)


class TestIsInNegativeTestContext:
    def test_default_is_false(self):
        assert is_in_negative_test_context() is False

    def test_true_inside_negative_context(self):
        with assert_negative_test_case(is_negative_test_case=True):
            assert is_in_negative_test_context() is True
            # Raise expected exception to satisfy the context manager
            raise Exception("[NCC_INKI016] Kernel validation exception: dummy")

    def test_false_inside_positive_context(self):
        with assert_negative_test_case(is_negative_test_case=False):
            assert is_in_negative_test_context() is False

    def test_reset_after_context_exit(self):
        with assert_negative_test_case(is_negative_test_case=True):
            raise Exception("[NCC_INKI016] Kernel validation exception: dummy")
        assert is_in_negative_test_context() is False


class TestAssertNegativeTestCase:
    def test_negative_passes_on_expected_exception(self):
        """Negative test passes when kernel validation exception is raised."""
        with assert_negative_test_case(is_negative_test_case=True):
            raise Exception("[NCC_INKI016] Kernel validation exception: bad shape")

    def test_negative_fails_when_no_exception(self):
        """Negative test fails assertion when no exception is raised."""
        with pytest.raises(AssertionError, match="expected to fail"):
            with assert_negative_test_case(is_negative_test_case=True):
                pass  # no exception raised

    def test_negative_validates_error_message(self):
        """Negative test checks expected_validation_error substring."""
        with assert_negative_test_case(
            is_negative_test_case=True,
            expected_validation_error="bad shape",
        ):
            raise Exception("[NCC_INKI016] Kernel validation exception: bad shape")

    def test_negative_fails_on_wrong_error_message(self):
        """Negative test fails when exception message doesn't match."""
        with pytest.raises(AssertionError):
            with assert_negative_test_case(
                is_negative_test_case=True,
                expected_validation_error="bad shape",
            ):
                raise Exception("[NCC_INKI016] Kernel validation exception: wrong dims")

    def test_positive_reraises_exception(self):
        """Positive test re-raises unexpected exceptions."""
        with pytest.raises(ValueError, match="unexpected"):
            with assert_negative_test_case(is_negative_test_case=False):
                raise ValueError("unexpected")

    def test_positive_passes_without_exception(self):
        """Positive test passes when no exception is raised."""
        with assert_negative_test_case(is_negative_test_case=False):
            pass


class TestAssertKernelValidationException:
    def test_passes_with_valid_exception(self):
        exc = Exception("[NCC_INKI016] Kernel validation exception: bad shape")
        assert_kernel_validation_exception(None, exc)

    def test_passes_with_matching_error_string(self):
        exc = Exception("[NCC_INKI016] Kernel validation exception: bad shape")
        assert_kernel_validation_exception("bad shape", exc)

    def test_fails_on_none_exception(self):
        with pytest.raises(AssertionError, match="Expected to receive"):
            call_with_invalid_argument(assert_kernel_validation_exception, None, None)

    def test_fails_on_missing_marker(self):
        exc = Exception("some other error")
        with pytest.raises(AssertionError):
            assert_kernel_validation_exception(None, exc)

    def test_fails_on_wrong_error_string(self):
        exc = Exception("[NCC_INKI016] Kernel validation exception: bad shape")
        with pytest.raises(AssertionError):
            assert_kernel_validation_exception("wrong dims", exc)
