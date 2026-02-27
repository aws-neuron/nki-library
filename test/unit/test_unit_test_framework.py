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

"""Unit tests for UnitTestFramework utilities."""

import inspect
from test.utils.unit_test_framework import (
    UnitTestFramework,
    check_unused_parameters,
    torch_ref_wrapper,
    validate_torch_ref_signature,
)
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestValidateTorchRefSignature:
    """Tests for validate_torch_ref_signature."""

    def test_matching_signatures_pass(self):
        """Matching signatures should not raise."""

        def kernel(a, b, c=None):
            pass

        def torch_ref(a, b, c=None):
            pass

        validate_torch_ref_signature(kernel, torch_ref)  # Should not raise

    def test_missing_param_in_ref_raises(self):
        """Missing parameter in torch_ref should raise ValueError."""

        def kernel(a, b, c):
            pass

        def torch_ref(a, b):
            pass

        with pytest.raises(ValueError, match="Missing in torch_ref.*'c'"):
            validate_torch_ref_signature(kernel, torch_ref)

    def test_extra_param_in_ref_raises(self):
        """Extra parameter in torch_ref should raise ValueError."""

        def kernel(a, b):
            pass

        def torch_ref(a, b, extra):
            pass

        with pytest.raises(ValueError, match="Extra in torch_ref.*'extra'"):
            validate_torch_ref_signature(kernel, torch_ref)


class TestCheckUnusedParameters:
    """Tests for check_unused_parameters."""

    def test_all_params_used_passes(self):
        """Function with all parameters used should not raise."""

        def func(a, b):
            return a + b

        check_unused_parameters(func)  # Should not raise

    def test_unused_param_raises(self):
        """Function with unused parameter should raise ValueError."""

        def func(a, b, unused):
            return a + b

        with pytest.raises(ValueError, match="unused.*may be unused"):
            check_unused_parameters(func)

    def test_multiple_unused_params_raises(self):
        """Function with multiple unused parameters should list all."""

        def func(a, unused1, unused2):
            return a

        with pytest.raises(ValueError, match="unused1.*unused2|unused2.*unused1"):
            check_unused_parameters(func)


class TestTorchRefWrapper:
    """Tests for torch_ref_wrapper."""

    def test_numpy_to_torch_conversion(self):
        """Wrapper should convert numpy arrays to torch tensors."""

        @torch_ref_wrapper
        def ref_func(x):
            assert isinstance(x, torch.Tensor)
            return x * 2

        result = ref_func(x=np.array([1.0, 2.0], dtype=np.float32))
        assert "out" in result
        assert isinstance(result["out"], np.ndarray)
        np.testing.assert_array_equal(result["out"], [2.0, 4.0])

    def test_float16_upcast_to_float32(self):
        """Wrapper should upcast float16 to float32 for CPU compatibility."""

        @torch_ref_wrapper
        def ref_func(x):
            assert x.dtype == torch.float32  # Should be upcasted
            return x

        result = ref_func(x=np.array([1.0], dtype=np.float16))
        assert result["out"].dtype == np.float32

    def test_uint32_to_int32_conversion(self):
        """Wrapper should convert uint32 to int32 (torch doesn't support uint32)."""

        @torch_ref_wrapper
        def ref_func(x):
            assert x.dtype == torch.int32
            return x

        result = ref_func(x=np.array([1, 2], dtype=np.uint32))
        assert result["out"].dtype == np.int32

    def test_dict_output_conversion(self):
        """Wrapper should handle dict outputs with multiple tensors."""

        @torch_ref_wrapper
        def ref_func(x):
            return {"a": x, "b": x * 2}

        result = ref_func(x=np.array([1.0], dtype=np.float32))
        assert "a" in result and "b" in result
        np.testing.assert_array_equal(result["a"], [1.0])
        np.testing.assert_array_equal(result["b"], [2.0])

    def test_none_values_skipped(self):
        """Wrapper should pass None values through to torch_ref."""

        @torch_ref_wrapper
        def ref_func(x, bias=None):
            assert bias is None  # None should be passed through
            return x

        result = ref_func(x=np.array([1.0], dtype=np.float32), bias=None)
        np.testing.assert_array_equal(result["out"], [1.0])

    def test_bfloat16_to_float32_conversion(self):
        """Wrapper should convert bfloat16 to float32 (torch.from_numpy doesn't support bfloat16)."""
        from neuron_dtypes import bfloat16

        @torch_ref_wrapper
        def ref_func(x):
            assert x.dtype == torch.float32  # Should be converted from bfloat16
            return x

        bf16_array = np.array([1.0, 2.0]).astype(bfloat16)
        result = ref_func(x=bf16_array)
        assert result["out"].dtype == np.float32

    def test_non_tensor_return_passthrough(self):
        """Wrapper should pass through non-tensor/non-dict returns."""

        @torch_ref_wrapper
        def ref_func(x):
            return {"value": 42, "name": "test"}  # Non-tensor values in dict

        result = ref_func(x=np.array([1.0], dtype=np.float32))
        assert result["value"] == 42
        assert result["name"] == "test"

    def test_scalar_passthrough(self):
        """Wrapper should pass through scalar non-array values."""

        @torch_ref_wrapper
        def ref_func(x, scale=2.0):
            assert scale == 2.0  # Scalar should pass through
            return x * scale

        result = ref_func(x=np.array([1.0], dtype=np.float32), scale=2.0)
        np.testing.assert_array_equal(result["out"], [2.0])

    def test_non_dict_non_tensor_return(self):
        """Wrapper should pass through non-dict/non-tensor returns as-is."""

        @torch_ref_wrapper
        def ref_func(x):
            return [1, 2, 3]  # Return a list (not dict or tensor)

        result = ref_func(x=np.array([1.0], dtype=np.float32))
        assert result == [1, 2, 3]


class TestCheckUnusedParametersEdgeCases:
    """Edge case tests for check_unused_parameters."""

    def test_builtin_function_skipped(self):
        """Built-in functions without source should be skipped."""
        # Built-in functions can't have source retrieved
        check_unused_parameters(len)  # Should not raise


class TestUnitTestFramework:
    """Tests for UnitTestFramework class."""

    def test_init_validates_signature(self):
        """Framework should validate signatures on init."""

        def kernel(a, b):
            pass

        def torch_ref(a, c):  # Mismatched param name
            pass

        mock_manager = MagicMock()
        with pytest.raises(ValueError, match="Missing in torch_ref"):
            UnitTestFramework(
                test_manager=mock_manager,
                kernel_entry=kernel,
                torch_ref=torch_ref,
                kernel_input_generator=lambda x: {},
                output_tensor_descriptor=lambda x: {},
            )

    def test_init_with_check_unused_params(self):
        """Framework should check unused params when enabled."""

        def kernel(a, unused):
            return a

        def torch_ref(a, unused):
            return a

        mock_manager = MagicMock()
        with pytest.raises(ValueError, match="unused.*may be unused"):
            UnitTestFramework(
                test_manager=mock_manager,
                kernel_entry=kernel,
                torch_ref=torch_ref,
                kernel_input_generator=lambda x: {},
                output_tensor_descriptor=lambda x: {},
                check_unused_params=True,
            )

    def test_run_test_validates_extra_keys(self):
        """run_test should raise on extra keys in kernel_input."""

        def kernel(a, b):
            pass

        def torch_ref(a, b):
            pass

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": 1, "b": 2, "extra_typo": 3},
            output_tensor_descriptor=lambda x: {},
        )

        with pytest.raises(ValueError, match="extra_typo.*don't match"):
            framework.run_test(test_config=None, compiler_args=MagicMock())

    def test_run_test_validates_missing_required(self):
        """run_test should raise on missing required parameters."""

        def kernel(a, b):  # Both required
            pass

        def torch_ref(a, b):
            pass

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": 1},  # Missing 'b'
            output_tensor_descriptor=lambda x: {},
        )

        with pytest.raises(ValueError, match="missing required.*b"):
            framework.run_test(test_config=None, compiler_args=MagicMock())

    def test_run_test_allows_optional_missing(self):
        """run_test should allow missing optional parameters."""

        def kernel(a, b=None):  # b is optional
            pass

        def torch_ref(a, b=None):
            return {"out": a}

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": np.array([1.0])},  # b missing but optional
            output_tensor_descriptor=lambda x: {"out": np.array([1.0])},
        )

        # Should not raise - b is optional
        framework.run_test(test_config=None, compiler_args=MagicMock())
        mock_manager.execute.assert_called_once()

    def test_run_test_with_inference_args(self):
        """run_test should pass inference_args to kernel_args when provided."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a}

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": np.array([1.0])},
            output_tensor_descriptor=lambda x: {"out": np.array([1.0])},
        )

        mock_inference_args = MagicMock()
        framework.run_test(test_config=None, compiler_args=MagicMock(), inference_args=mock_inference_args)
        mock_manager.execute.assert_called_once()
        # Verify inference_args was set on kernel_args
        call_args = mock_manager.execute.call_args[0][0]
        assert call_args.inference_args == mock_inference_args

    def test_run_test_lazy_golden_generator(self):
        """run_test should create lazy golden generator that returns cached ref_result."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a * 2}

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": np.array([1.0, 2.0])},
            output_tensor_descriptor=lambda x: {"out": np.array([0.0, 0.0])},
        )

        framework.run_test(test_config=None, compiler_args=MagicMock())
        # Get the lazy_golden from kernel_args and call its generator
        kernel_args = mock_manager.execute.call_args[0][0]
        lazy_golden = kernel_args.validation_args.golden_output
        result = lazy_golden.lazy_golden_generator()
        np.testing.assert_array_equal(result["out"], [2.0, 4.0])

    def test_run_test_handles_must_alias_input(self):
        """run_test should handle .must_alias_input suffix correctly."""

        def kernel(a, output):
            pass

        def torch_ref(a, output):
            return {"out": a}

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {
                "a": np.array([1.0]),
                "output.must_alias_input": np.array([0.0]),
            },
            output_tensor_descriptor=lambda x: {"out": np.array([1.0])},
        )

        framework.run_test(test_config=None, compiler_args=MagicMock())
        mock_manager.execute.assert_called_once()

    def test_run_test_filters_extra_key_for_ref(self):
        """run_test should filter keys present in kernel but not in torch_ref."""

        def kernel(a, extra_kernel_param):
            pass

        def torch_ref(a, extra_kernel_param):  # Must match for signature validation
            return {"out": a}

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {
                "a": np.array([1.0]),
                "extra_kernel_param": 42,
            },
            output_tensor_descriptor=lambda x: {"out": np.array([1.0])},
        )

        framework.run_test(test_config=None, compiler_args=MagicMock())
        mock_manager.execute.assert_called_once()

    def test_run_test_defers_torch_ref_to_lazy_golden(self):
        """run_test should not call torch_ref eagerly; it should be deferred to lazy golden."""

        def kernel(a):
            pass

        mock_torch_ref = MagicMock(return_value={"out": np.array([2.0])})
        # Signature must match kernel for __init__ validation
        mock_torch_ref.__signature__ = inspect.signature(kernel)

        mock_manager = MagicMock()

        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=mock_torch_ref,
            kernel_input_generator=lambda x: {"a": np.array([1.0])},
            output_tensor_descriptor=lambda x: {"out": np.array([1.0])},
        )

        framework.run_test(test_config=None, compiler_args=MagicMock())
        # torch_ref should NOT have been called during run_test
        mock_torch_ref.assert_not_called()
        # But lazy_golden_generator should be set (not None)
        kernel_args = mock_manager.execute.call_args[0][0]
        lazy_golden = kernel_args.validation_args.golden_output
        assert lazy_golden.lazy_golden_generator is not None
        # Calling .golden triggers torch_ref
        result = lazy_golden.golden
        mock_torch_ref.assert_called_once()
        np.testing.assert_array_equal(result["out"], [2.0])

    def test_run_test_lazy_golden_validates_output_keys(self):
        """Lazy golden should validate output keys when accessed."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a, "extra": a * 2}

        mock_manager = MagicMock()

        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": np.array([1.0])},
            output_tensor_descriptor=lambda x: {"out": np.array([1.0])},
        )

        framework.run_test(test_config=None, compiler_args=MagicMock())
        kernel_args = mock_manager.execute.call_args[0][0]
        lazy_golden = kernel_args.validation_args.golden_output
        # Output key mismatch should raise when .golden is accessed
        with pytest.raises(ValueError, match="Output tensor mismatch"):
            lazy_golden.golden

    def test_run_test_lazy_golden_validates_extra_output_keys(self):
        """Lazy golden should catch extra keys in output_tensor_descriptor."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a}

        mock_manager = MagicMock()

        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": np.array([1.0])},
            output_tensor_descriptor=lambda x: {"out": np.array([1.0]), "unused": np.array([0.0])},
        )

        framework.run_test(test_config=None, compiler_args=MagicMock())
        kernel_args = mock_manager.execute.call_args[0][0]
        with pytest.raises(ValueError, match="Output tensor mismatch"):
            kernel_args.validation_args.golden_output.golden

    def test_run_test_calls_collector_with_metadata(self):
        """run_test should call collector.match_and_add_metadata_dimensions when metadata is provided."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {}

        mock_manager = MagicMock()
        mock_collector = MagicMock()
        metadata_key = {"ln": 2, "ae": True}
        metadata_list = [{"test_settings": {"ln": 2, "ae": True}, "model_settings": {"model": "test"}}]

        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": 1},
            output_tensor_descriptor=lambda x: {},
            collector=mock_collector,
        )

        with patch("test.utils.unit_test_framework.load_model_configs", return_value=metadata_list) as mock_load:
            framework.run_test(
                test_config=None,
                compiler_args=MagicMock(),
                metadata={"config_name": "test_moe_block", "key": metadata_key},
            )
            mock_load.assert_called_once_with("test_moe_block")
        mock_collector.match_and_add_metadata_dimensions.assert_called_once_with(metadata_key, metadata_list)

    def test_run_test_skips_collector_when_no_metadata(self):
        """run_test should not call collector when metadata is None."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {}

        mock_manager = MagicMock()
        mock_collector = MagicMock()

        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda x: {"a": 1},
            output_tensor_descriptor=lambda x: {},
            collector=mock_collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=MagicMock(),
        )
        mock_collector.match_and_add_metadata_dimensions.assert_not_called()
