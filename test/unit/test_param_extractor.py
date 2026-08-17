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
Unit tests for param_extractor module.
"""

from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock

import pytest

from ..utils.param_extractor import (
    _stable_param_value,
    _unwrap_kernel_func,
    compute_params_hash,
    derive_test_method_id,
    extract_pytest_params,
    normalize_param_names,
    normalize_param_value,
)


class SampleEnum(Enum):
    """Sample enum for testing."""

    VALUE_A = 0
    VALUE_B = 1
    VALUE_C = 2


class TestNormalizeParamValue:
    """Tests for normalize_param_value function."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert normalize_param_value(None) is None

    def test_enum_returns_name(self):
        """Enum values are converted to their name string."""
        assert normalize_param_value(SampleEnum.VALUE_A) == "VALUE_A"
        assert normalize_param_value(SampleEnum.VALUE_B) == "VALUE_B"

    def test_empty_string_returns_none(self):
        """Empty or whitespace-only strings return None."""
        assert normalize_param_value("") is None
        assert normalize_param_value("   ") is None
        assert normalize_param_value("\t\n") is None

    def test_non_empty_string_passes_through(self):
        """Non-empty strings pass through unchanged."""
        assert normalize_param_value("hello") == "hello"
        assert normalize_param_value("  hello  ") == "  hello  "

    def test_primitives_pass_through(self):
        """Primitive types pass through unchanged."""
        assert normalize_param_value(42) == 42
        assert normalize_param_value(3.14) == 3.14
        assert normalize_param_value(True) is True
        assert normalize_param_value(False) is False

    def test_object_with_dict_converted_to_string(self):
        """Objects with __dict__ are converted to string representation."""

        class CustomObj:
            def __init__(self):
                self.value = 123

            def __str__(self):
                return "CustomObj(123)"

        obj = CustomObj()
        assert normalize_param_value(obj) == "CustomObj(123)"

    def test_homogeneous_list_kept_native(self):
        """Homogeneous lists are kept as native arrays for correct OpenSearch mapping."""
        assert normalize_param_value([1, 2, 3]) == [1, 2, 3]

    def test_mixed_type_tuple_stringified(self):
        """Mixed-type tuples are JSON-stringified to avoid dynamic mapping conflicts."""
        assert normalize_param_value((8000, 512, 4, 4, False)) == "[8000, 512, 4, 4, false]"

    def test_homogeneous_tuple_kept_native(self):
        """Homogeneous tuples are kept as native arrays."""
        assert normalize_param_value((1, 2)) == [1, 2]

    def test_empty_list_kept_native(self):
        """Empty lists are kept as native arrays."""
        assert normalize_param_value([]) == []


class TestUnwrapKernelFunc:
    """Tests for _unwrap_kernel_func function."""

    def test_plain_function_returns_unchanged(self):
        """Plain function returns unchanged."""

        def my_func():
            pass

        assert _unwrap_kernel_func(my_func) is my_func

    def test_unwraps_func_attribute(self):
        """Objects with 'func' attribute are unwrapped."""

        def original():
            return None

        mock_kernel = Mock()
        mock_kernel.func = original

        result = _unwrap_kernel_func(mock_kernel)
        assert result is original

    def test_unwraps_wrapped_attribute(self):
        """Decorated functions with __wrapped__ are unwrapped."""

        def original():
            return None

        def decorated():
            return None

        decorated.__wrapped__ = original

        assert _unwrap_kernel_func(decorated) is original

    def test_recursive_unwrap(self):
        """Nested wrappers are unwrapped recursively."""

        def original():
            return None

        # Create nested wrapper: GenericKernel(decorated(original))
        def decorated():
            return None

        decorated.__wrapped__ = original
        mock_kernel = Mock()
        mock_kernel.func = decorated

        result = _unwrap_kernel_func(mock_kernel)
        assert result is original


class TestExtractPytestParams:
    """Tests for extract_pytest_params function."""

    def test_empty_params_returns_empty(self):
        """Empty params dict returns empty dict."""
        assert extract_pytest_params({}) == {}

    def test_none_values_filtered(self):
        """None values are filtered out."""
        params = {"a": 1, "b": None, "c": 3}
        result = extract_pytest_params(params)
        assert result == {"a": 1, "c": 3}

    def test_enum_converted_to_name(self):
        """Enum values are converted to their name."""
        params = {"mode": SampleEnum.VALUE_A}
        result = extract_pytest_params(params)
        assert result == {"mode": "VALUE_A"}

    def test_mixed_params(self):
        """Mixed parameter types are handled correctly."""
        params = {
            "count": 42,
            "name": "test",
            "mode": SampleEnum.VALUE_B,
            "empty": "",
            "flag": True,
        }
        result = extract_pytest_params(params)

        assert result["count"] == 42
        assert result["name"] == "test"
        assert result["mode"] == "VALUE_B"
        assert result["flag"] is True
        assert "empty" not in result  # Empty string filtered


class TestNormalizeParamNames:
    """Tests for normalize_param_names function."""

    def test_empty_dict(self):
        assert normalize_param_names({}) == {}

    def test_no_mapping_passes_through(self):
        params = {"batch": 32, "seqlen": 1024, "custom_param": 7}
        assert normalize_param_names(params) == params

    def test_batch_size_normalized(self):
        assert normalize_param_names({"batch_size": 32}) == {"batch": 32}

    def test_seq_len_normalized(self):
        assert normalize_param_names({"seq_len": 1024}) == {"seqlen": 1024}

    def test_sequence_length_normalized(self):
        assert normalize_param_names({"sequence_length": 512}) == {"seqlen": 512}

    def test_hidden_size_normalized(self):
        assert normalize_param_names({"hidden_size": 4096}) == {"hidden": 4096}

    def test_hidden_dim_normalized(self):
        assert normalize_param_names({"hidden_dim": 8192}) == {"hidden": 8192}

    def test_head_names_normalized_to_n_prefix(self):
        """num_* head names normalize to n_* for OpenSearch consistency."""
        assert normalize_param_names({"num_q_heads": 8}) == {"n_q_heads": 8}
        assert normalize_param_names({"num_kv_heads": 4}) == {"n_kv_heads": 4}
        assert normalize_param_names({"num_heads": 64}) == {"n_heads": 64}

    def test_n_prefix_head_names_unchanged(self):
        """n_* head names are already canonical — pass through."""
        assert normalize_param_names({"n_q_heads": 16}) == {"n_q_heads": 16}
        assert normalize_param_names({"n_kv_heads": 8}) == {"n_kv_heads": 8}
        assert normalize_param_names({"n_heads": 4}) == {"n_heads": 4}

    def test_mixed_params(self):
        params = {"batch_size": 16, "seq_len": 2048, "hidden": 4096, "custom": "abc"}
        expected = {"batch": 16, "seqlen": 2048, "hidden": 4096, "custom": "abc"}
        assert normalize_param_names(params) == expected

    def test_kernel_specific_params_not_normalized(self):
        """S_tkg, S_ctx, BxS should NOT be normalized — they have distinct semantics."""
        params = {"S_tkg": 1, "S_ctx": 9216, "BxS": 64}
        assert normalize_param_names(params) == params


# ---------------------------------------------------------------------------
# Tests for _stable_param_value
# ---------------------------------------------------------------------------


class TestStableParamValue:
    """Tests for _stable_param_value function."""

    def test_none_returns_none(self):
        assert _stable_param_value(None) is None

    def test_bool_returns_bool(self):
        assert _stable_param_value(True) is True
        assert _stable_param_value(False) is False

    def test_int_returns_int(self):
        assert _stable_param_value(42) == 42

    def test_float_returns_float(self):
        assert _stable_param_value(3.14) == 3.14

    def test_str_returns_str(self):
        assert _stable_param_value("hello") == "hello"

    def test_enum_returns_value(self):
        """Enums use .value for stability against member renames."""
        assert _stable_param_value(SampleEnum.VALUE_A) == 0
        assert _stable_param_value(SampleEnum.VALUE_B) == 1

    def test_type_returns_name(self):
        class bfloat16:
            pass

        assert _stable_param_value(bfloat16) == "bfloat16"

    def test_list_recursively_normalizes(self):
        result = _stable_param_value([SampleEnum.VALUE_A, 1, "x"])
        assert result == [0, 1, "x"]

    def test_tuple_recursively_normalizes(self):
        result = _stable_param_value((SampleEnum.VALUE_B, 2))
        assert result == [1, 2]

    def test_dict_sorted_and_recursively_normalizes(self):
        result = _stable_param_value({"b": SampleEnum.VALUE_A, "a": 1})
        assert result == {"a": 1, "b": 0}

    def test_set_sorted_and_normalized(self):
        result = _stable_param_value({3, 1, 2})
        assert result == [1, 2, 3]

    def test_frozenset_sorted_and_normalized(self):
        result = _stable_param_value(frozenset([3, 1, 2]))
        assert result == [1, 2, 3]

    def test_numpy_scalar(self):
        """Numpy scalars are converted via .item()."""

        class FakeInt64:
            def item(self):
                return 128

        assert _stable_param_value(FakeInt64()) == 128

    def test_dataclass_decomposed(self):
        """Objects with __dict__ are decomposed into sorted attribute dicts."""

        @dataclass
        class Config:
            batch: int = 4
            seqlen: int = 2048

        result = _stable_param_value(Config(batch=8, seqlen=1024))
        assert result == {"batch": 8, "seqlen": 1024}

    def test_dataclass_with_enum_field(self):
        @dataclass
        class Config:
            norm: SampleEnum = SampleEnum.VALUE_A

        result = _stable_param_value(Config(norm=SampleEnum.VALUE_B))
        assert result == {"norm": 1}

    def test_dataclass_with_set_field(self):
        @dataclass
        class Config:
            dims: set = None

            def __post_init__(self):
                if self.dims is None:
                    self.dims = {3, 1, 2}

        result = _stable_param_value(Config())
        assert result == {"dims": [1, 2, 3]}

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported parameter type"):
            _stable_param_value(b"bytes_value")


# ---------------------------------------------------------------------------
# Tests for compute_params_hash
# ---------------------------------------------------------------------------


class TestComputeParamsHash:
    """Tests for compute_params_hash function."""

    def test_empty_params(self):
        h = compute_params_hash({})
        assert len(h) == 64  # full SHA-256 hex

    def test_deterministic(self):
        params = {"batch": 4, "seqlen": 2048}
        assert compute_params_hash(params) == compute_params_hash(params)

    def test_order_independent(self):
        """Key order doesn't affect the hash."""
        h1 = compute_params_hash({"a": 1, "b": 2})
        h2 = compute_params_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_none_preserved(self):
        """Params with None differ from params without the key."""
        h1 = compute_params_hash({"a": 1, "b": None})
        h2 = compute_params_hash({"a": 1})
        assert h1 != h2

    def test_enum_uses_value(self):
        """Enum params use .value, not .name."""
        h1 = compute_params_hash({"x": SampleEnum.VALUE_A})
        h2 = compute_params_hash({"x": 0})
        assert h1 == h2

    def test_different_params_different_hash(self):
        h1 = compute_params_hash({"batch": 1})
        h2 = compute_params_hash({"batch": 2})
        assert h1 != h2


# ---------------------------------------------------------------------------
# Tests for derive_test_method_id
# ---------------------------------------------------------------------------


class TestDeriveTestMethodId:
    """Tests for derive_test_method_id function."""

    def test_with_class(self):
        node = Mock()
        node.module.__name__ = "test.integration.nkilib.attention.test_attn"
        node.cls.__name__ = "TestAttention"
        node.originalname = "test_forward"
        assert derive_test_method_id(node) == "test.integration.nkilib.attention.test_attn::TestAttention::test_forward"

    def test_without_class(self):
        node = Mock()
        node.module.__name__ = "test.integration.nkilib.test_simple"
        node.cls = None
        node.originalname = "test_basic"
        assert derive_test_method_id(node) == "test.integration.nkilib.test_simple::test_basic"

    def test_without_module(self):
        node = Mock()
        node.module = None
        node.cls.__name__ = "TestFoo"
        node.originalname = "test_bar"
        assert derive_test_method_id(node) == "::TestFoo::test_bar"
