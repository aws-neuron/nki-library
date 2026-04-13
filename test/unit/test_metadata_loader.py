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
"""Unit tests for metadata_loader module."""

from enum import Enum
from test.utils.metadata_loader import (
    _coerce_json_value,
    compute_config_version_id,
    match_model_config_id,
)


class SampleEnum(Enum):
    SOFTMAX = 0
    SIGMOID = 1


class AnotherEnum(Enum):
    SiLU = 0
    Swish = 3


def _make_entry(test_settings, model_settings=None):
    """Helper to build a metadata entry."""
    entry = {"test_settings": test_settings}
    if model_settings is not None:
        entry["model_settings"] = model_settings
    return entry


class TestCoerceJsonValue:
    def test_exact_match_returns_json_value(self):
        assert _coerce_json_value(5, 5) == 5
        assert _coerce_json_value("hello", "hello") == "hello"
        assert _coerce_json_value(True, True) is True

    def test_enum_coercion_success(self):
        result = _coerce_json_value("SampleEnum.SOFTMAX", SampleEnum.SOFTMAX)
        assert result == SampleEnum.SOFTMAX
        assert isinstance(result, SampleEnum)

    def test_enum_coercion_different_member(self):
        result = _coerce_json_value("SampleEnum.SIGMOID", SampleEnum.SOFTMAX)
        assert result == SampleEnum.SIGMOID

    def test_enum_coercion_invalid_member_returns_json_value(self):
        result = _coerce_json_value("SampleEnum.INVALID", SampleEnum.SOFTMAX)
        assert result == "SampleEnum.INVALID"

    def test_enum_coercion_no_dot_returns_json_value(self):
        result = _coerce_json_value("SOFTMAX", SampleEnum.SOFTMAX)
        assert result == "SOFTMAX"

    def test_enum_coercion_mixed_case_name(self):
        result = _coerce_json_value("AnotherEnum.Swish", AnotherEnum.Swish)
        assert result == AnotherEnum.Swish

    def test_nl_prefix_stripped(self):
        result = _coerce_json_value("nl.bfloat16", "bfloat16")
        assert result == "bfloat16"

    def test_nl_prefix_different_dtype(self):
        result = _coerce_json_value("nl.float8_e4m3fn_x4", "float8_e4m3fn_x4")
        assert result == "float8_e4m3fn_x4"

    def test_nl_prefix_no_match(self):
        result = _coerce_json_value("nl.float16", "bfloat16")
        assert result == "float16"

    def test_none_both(self):
        assert _coerce_json_value(None, None) is None

    def test_none_json_val(self):
        assert _coerce_json_value(None, 5) is None

    def test_none_test_val(self):
        assert _coerce_json_value(5, None) == 5

    def test_different_types_no_coercion(self):
        assert _coerce_json_value(5, "5") == 5
        assert _coerce_json_value("hello", 123) == "hello"


class TestComputeConfigVersionId:
    def test_deterministic(self):
        config = {"kernel_key": "moe", "test_settings": {"a": 1}}
        assert compute_config_version_id(config) == compute_config_version_id(config)

    def test_different_inputs_different_hashes(self):
        config_a = {"a": 1}
        config_b = {"a": 2}
        assert compute_config_version_id(config_a) != compute_config_version_id(config_b)

    def test_key_order_irrelevant(self):
        config_a = {"a": 1, "b": 2}
        config_b = {"b": 2, "a": 1}
        assert compute_config_version_id(config_a) == compute_config_version_id(config_b)

    def test_returns_hex_string(self):
        result = compute_config_version_id({"x": 1})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestMatchModelConfigId:
    def test_exact_match_returns_hash(self):
        entry = _make_entry({"a": 1, "b": 2})
        result = match_model_config_id({"a": 1, "b": 2}, [entry])
        assert result == compute_config_version_id(entry)

    def test_no_match_returns_none(self):
        entry = _make_entry({"a": 1})
        assert match_model_config_id({"a": 999}, [entry]) is None

    def test_empty_metadata_list_returns_none(self):
        assert match_model_config_id({"a": 1}, []) is None

    def test_partial_match_returns_none(self):
        entry = _make_entry({"a": 1, "b": 2})
        assert match_model_config_id({"a": 1, "b": 999}, [entry]) is None

    def test_subset_key_match_succeeds(self):
        entry = _make_entry({"a": 1, "b": 2, "c": 3})
        result = match_model_config_id({"a": 1, "b": 2}, [entry])
        assert result == compute_config_version_id(entry)

    def test_extra_key_in_test_key_no_match(self):
        entry = _make_entry({"a": 1})
        assert match_model_config_id({"a": 1, "b": 2}, [entry]) is None

    def test_second_entry_matches(self):
        entry1 = _make_entry({"a": 1})
        entry2 = _make_entry({"a": 2})
        result = match_model_config_id({"a": 2}, [entry1, entry2])
        assert result == compute_config_version_id(entry2)

    def test_enum_coercion_in_matching(self):
        entry = _make_entry({"rf": "SampleEnum.SOFTMAX", "x": 1})
        result = match_model_config_id({"rf": SampleEnum.SOFTMAX, "x": 1}, [entry])
        assert result == compute_config_version_id(entry)

    def test_dtype_coercion_in_matching(self):
        entry = _make_entry({"wd": "nl.bfloat16", "id": "nl.float16"})
        result = match_model_config_id({"wd": "bfloat16", "id": "float16"}, [entry])
        assert result == compute_config_version_id(entry)

    def test_mixed_coercion_in_matching(self):
        entry = _make_entry(
            {
                "rf": "SampleEnum.SIGMOID",
                "wd": "nl.bfloat16",
                "ba": 8,
                "se": False,
            }
        )
        test_key = {
            "rf": SampleEnum.SIGMOID,
            "wd": "bfloat16",
            "ba": 8,
            "se": False,
        }
        result = match_model_config_id(test_key, [entry])
        assert result == compute_config_version_id(entry)

    def test_missing_test_settings_no_match(self):
        entry = {"model_settings": {"model_name": "test"}}
        assert match_model_config_id({"a": 1}, [entry]) is None

    def test_returns_first_match(self):
        entry1 = _make_entry({"a": 1})
        entry2 = _make_entry({"a": 1})
        entry2["extra"] = "different"
        result = match_model_config_id({"a": 1}, [entry1, entry2])
        assert result == compute_config_version_id(entry1)
