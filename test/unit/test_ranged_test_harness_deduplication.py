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
Unit tests for ranged_test_harness deduplication logic.

Tests the cross-strategy deduplication behavior where tests with identical
tensor dimensions are deduplicated regardless of generator strategy.
"""

from test.utils.ranged_test_harness import RangeTestCase, TensorRangeConfig


class TestRangeTestCaseDeduplication:
    """Test deduplication behavior of RangeTestCase."""

    def test_dedup_key_uses_only_dimensions(self):
        """Test that dedup_key() generates key based only on tensor dimensions."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)
        # Create test case with specific dimensions
        test_case = RangeTestCase(
            test_type="random",
            tensors={
                "input": {"dim0": 128, "dim1": 256},
                "output": {"dim0": 128, "dim1": 512},
            },
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        dedup_key = test_case.dedup_key()

        # Key should contain tensor names and dimensions
        assert "input" in dedup_key
        assert "output" in dedup_key
        assert "dim0-128" in dedup_key
        assert "dim1-256" in dedup_key
        assert "dim1-512" in dedup_key

        # Key should NOT contain test_type
        assert "random" not in dedup_key

    def test_dedup_key_sorts_dimension_items(self):
        """Test that dedup_key() sorts dimension items for consistency."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)

        # Create two test cases with same dimensions in different order
        test_case1 = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 128, "dim1": 256}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        test_case2 = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim1": 256, "dim0": 128}},  # Different dict order
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        # Dedup keys should be identical due to sorting
        assert test_case1.dedup_key() == test_case2.dedup_key()

    def test_cross_strategy_deduplication_same_dedup_key(self):
        """Test that tests with identical dimensions but different strategies have same dedup_key."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)

        # Create test cases with identical dimensions but different test_type
        random_test = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 128, "dim1": 256}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        monotonic_test = RangeTestCase(
            test_type="monotonic",
            tensors={"input": {"dim0": 128, "dim1": 256}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        manual_test = RangeTestCase(
            test_type="manual",
            tensors={"input": {"dim0": 128, "dim1": 256}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        # All should have the same dedup_key (for cross-strategy deduplication)
        assert random_test.dedup_key() == monotonic_test.dedup_key()
        assert monotonic_test.dedup_key() == manual_test.dedup_key()

    def test_cross_strategy_deduplication_different_unique_id(self):
        """Test that tests with identical dimensions but different strategies have different unique_id."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)

        # Create test cases with identical dimensions but different test_type
        random_test = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 128, "dim1": 256}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        monotonic_test = RangeTestCase(
            test_type="monotonic",
            tensors={"input": {"dim0": 128, "dim1": 256}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        # unique_id should include test_type prefix for display/debugging
        random_id = random_test.unique_id()
        monotonic_id = monotonic_test.unique_id()

        assert random_id != monotonic_id
        assert "random" in random_id
        assert "monotonic" in monotonic_id

    def test_different_dimensions_different_dedup_key(self):
        """Test that tests with different dimensions have different dedup_key."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)

        test_case1 = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 128, "dim1": 256}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        test_case2 = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 256, "dim1": 512}},  # Different dimensions
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        # Should have different dedup_keys
        assert test_case1.dedup_key() != test_case2.dedup_key()

    def test_different_tensor_names_different_dedup_key(self):
        """Test that tests with different tensor names have different dedup_key."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)

        test_case1 = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 128}},
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        test_case2 = RangeTestCase(
            test_type="random",
            tensors={"output": {"dim0": 128}},  # Different tensor name
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        # Should have different dedup_keys
        assert test_case1.dedup_key() != test_case2.dedup_key()

    def test_multiple_tensors_deduplication(self):
        """Test deduplication with multiple tensors."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)

        test_case1 = RangeTestCase(
            test_type="random",
            tensors={
                "input": {"dim0": 128, "dim1": 256},
                "output": {"dim0": 128, "dim1": 512},
            },
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        test_case2 = RangeTestCase(
            test_type="monotonic",  # Different strategy
            tensors={
                "input": {"dim0": 128, "dim1": 256},
                "output": {"dim0": 128, "dim1": 512},
            },
            additional_params={},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        # Same dimensions, different strategies → same dedup_key
        assert test_case1.dedup_key() == test_case2.dedup_key()

        # But different unique_id for display
        assert test_case1.unique_id() != test_case2.unique_id()

    def test_additional_params_not_in_dedup_key(self):
        """Test that additional_params don't affect dedup_key."""
        test_config = TensorRangeConfig(tensor_configs={}, monotonic_step_size=1)

        test_case1 = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 128}},
            additional_params={"param1": "value1"},
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        test_case2 = RangeTestCase(
            test_type="random",
            tensors={"input": {"dim0": 128}},
            additional_params={"param2": "value2"},  # Different additional params
            test_config_ref=test_config,
            is_negative_test_case=False,
        )

        # Should have the same dedup_key (additional_params not included)
        assert test_case1.dedup_key() == test_case2.dedup_key()
