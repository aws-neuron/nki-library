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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from neuron_dtypes import bfloat16 as np_bfloat16
from neuron_dtypes import float8_e4m3fn as np_float8_e4m3fn
from neuron_dtypes import static_cast as np_static_cast

from test.utils.common_dataclasses import (
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ValidationArgs,
)
from test.utils.unit_test_collective_framework import (
    validate_cross_rank_consistency,
)
from test.utils.unit_test_framework import (
    UnitTestFramework,
    check_unused_parameters,
    filter_kernel_input,
    filter_ref_input,
    torch_ref_wrapper,
    validate_input_keys,
    validate_torch_ref_signature,
)


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

        @torch_ref_wrapper
        def ref_func(x):
            assert x.dtype == torch.float32  # Should be converted from bfloat16
            return x

        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
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
            return a, output

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
            _ = lazy_golden.golden

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
            _ = kernel_args.validation_args.golden_output.golden

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

    def test_custom_comparator_receives_torch_ref_golden(self):
        """custom_comparator should receive golden from torch_ref, not compute it independently."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a * 3}

        received = {}

        def comparator(golden_dict, output_tensors):
            received.update(golden_dict)
            return {
                "out": CustomValidatorWithOutputTensorData(
                    validator=type("V", (CustomValidator,), {"validate": lambda self, x: True}),
                    output_ndarray=output_tensors["out"],
                )
            }

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: {"a": np.array([2.0])},
            output_tensor_descriptor=lambda _: {"out": np.array([0.0])},
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), custom_comparator=comparator)

        # Trigger lazy golden to invoke comparator
        kernel_args = mock_manager.execute.call_args[0][0]
        _ = kernel_args.validation_args.golden_output.golden
        np.testing.assert_array_equal(received["out"], [6.0])

    def test_custom_comparator_result_used_as_validation(self):
        """Framework should use comparator's returned CustomValidatorWithOutputTensorData."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a}

        validator_cls = type("V", (CustomValidator,), {"validate": lambda self, x: True})

        def comparator(golden_dict, output_tensors):
            return {
                "out": CustomValidatorWithOutputTensorData(
                    validator=validator_cls,
                    output_ndarray=np.zeros((2,)),
                )
            }

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: {"a": np.array([1.0, 2.0])},
            output_tensor_descriptor=lambda _: {"out": np.zeros((2,))},
        )
        framework.run_test(test_config=None, compiler_args=MagicMock(), custom_comparator=comparator)

        kernel_args = mock_manager.execute.call_args[0][0]
        result = kernel_args.validation_args.golden_output.golden
        assert isinstance(result["out"], CustomValidatorWithOutputTensorData)
        assert result["out"].validator is validator_cls

    def test_custom_comparator_and_custom_validation_args_exclusive(self):
        """Passing both custom_comparator and custom_validation_args should raise."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a}

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: {"a": np.array([1.0])},
            output_tensor_descriptor=lambda _: {"out": np.array([0.0])},
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            framework.run_test(
                test_config=None,
                compiler_args=MagicMock(),
                custom_validation_args=MagicMock(spec=ValidationArgs),
                custom_comparator=lambda g, o: {},
            )

    # ── trace_only mode ──

    def test_trace_only_init_without_torch_ref(self):
        """trace_only=True should allow torch_ref=None and output_tensor_descriptor=None."""

        def kernel(a):
            pass

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            kernel_input_generator=lambda x: {"a": 1},
            trace_only=True,
        )
        assert framework.trace_only is True
        assert framework.reference is None

    def test_trace_only_false_without_torch_ref_raises(self):
        """trace_only=False (default) with torch_ref=None should raise."""

        def kernel(a):
            pass

        mock_manager = MagicMock()
        with pytest.raises(ValueError, match="torch_ref is required"):
            UnitTestFramework(
                test_manager=mock_manager,
                kernel_entry=kernel,
                kernel_input_generator=lambda x: {"a": 1},
            )

    def test_trace_only_false_without_output_tensor_descriptor_raises(self):
        """trace_only=False with output_tensor_descriptor=None should raise."""

        def kernel(a):
            pass

        def torch_ref(a):
            return {"out": a}

        mock_manager = MagicMock()
        with pytest.raises(ValueError, match="output_tensor_descriptor is required"):
            UnitTestFramework(
                test_manager=mock_manager,
                kernel_entry=kernel,
                torch_ref=torch_ref,
                kernel_input_generator=lambda x: {"a": 1},
            )

    def test_trace_only_run_test_calls_execute(self):
        """trace_only run_test should call test_manager.execute with no golden generator."""

        def kernel(a):
            pass

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            kernel_input_generator=lambda x: {"a": np.array([1.0])},
            trace_only=True,
        )
        framework.run_test(test_config=None, compiler_args=MagicMock())
        mock_manager.execute.assert_called_once()
        kernel_args = mock_manager.execute.call_args[0][0]
        assert kernel_args.validation_args.golden_output.lazy_golden_generator is None

    def test_trace_only_run_test_validates_input_keys(self):
        """trace_only run_test should still validate kernel_input keys."""

        def kernel(a):
            pass

        mock_manager = MagicMock()
        framework = UnitTestFramework(
            test_manager=mock_manager,
            kernel_entry=kernel,
            kernel_input_generator=lambda x: {"a": 1, "extra_typo": 2},
            trace_only=True,
        )
        with pytest.raises(ValueError, match="extra_typo.*don't match"):
            framework.run_test(test_config=None, compiler_args=MagicMock())


class TestTorchRefWrapperPreserveLowerPrecision:
    """Tests for torch_ref_wrapper with preserve_lower_precision=True."""

    def test_bfloat16_input_preserved_as_torch_bfloat16(self):
        """With preserve_lower_precision, bfloat16 inputs should arrive as torch.bfloat16."""

        @torch_ref_wrapper
        def ref_default(x):
            assert x.dtype == torch.float32, "Default should promote to float32"
            return x

        wrapped_preserve = torch_ref_wrapper(
            lambda x: (setattr(wrapped_preserve, '_seen_dtype', x.dtype), x)[1],
            preserve_lower_precision=True,
        )

        # Need proper signature for the lambda
        def ref_preserve(x):
            assert x.dtype == torch.bfloat16, f"Expected bfloat16, got {x.dtype}"
            return x

        wrapped = torch_ref_wrapper(ref_preserve, preserve_lower_precision=True)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        wrapped(x=bf16_array)  # Should not raise

    def test_bfloat16_output_cast_back(self):
        """With preserve_lower_precision, output should be cast back to bfloat16 numpy."""

        def ref_func(x):
            return x * 2  # bfloat16 in, bfloat16 out

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array)
        assert str(result["out"].dtype) == "bfloat16"

    def test_bfloat16_default_returns_float32(self):
        """Without preserve_lower_precision, bfloat16 output should be float32."""

        def ref_func(x):
            return x * 2

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=False)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array)
        assert result["out"].dtype == np.float32

    def test_bfloat16_intermediate_truncation_matches_numpy(self):
        """preserve_lower_precision should produce bit-identical results to numpy bfloat16 arithmetic."""

        np.random.seed(123)
        a = np_static_cast(np.random.randn(64), np_bfloat16)
        b = np_static_cast(np.random.randn(64), np_bfloat16)

        # Numpy bfloat16 arithmetic (truncates intermediates)
        expected = np_static_cast(a * b + a, np_bfloat16)

        # Torch ref with preserve_lower_precision
        def ref_func(x, y):
            return {"out": x * y + x}

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        result = wrapped(x=a, y=b)

        assert np.array_equal(expected.view(np.uint16), result["out"].view(np.uint16)), (
            "Should be bit-identical to numpy bfloat16 arithmetic"
        )

    def test_float32_input_unaffected(self):
        """preserve_lower_precision should not affect float32 inputs/outputs."""

        def ref_func(x):
            return x * 2

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        f32_array = np.array([1.5, 2.5], dtype=np.float32)
        result = wrapped(x=f32_array)
        assert result["out"].dtype == np.float32
        np.testing.assert_array_equal(result["out"], [3.0, 5.0])

    def test_dict_output_all_keys_cast_back(self):
        """With preserve_lower_precision, all dict output tensors should be cast back."""

        def ref_func(x):
            return {"a": x, "b": x * 2}

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        bf16_array = np.array([1.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array)
        assert str(result["a"].dtype) == "bfloat16"
        assert str(result["b"].dtype) == "bfloat16"

    def test_non_tensor_values_in_dict_unaffected(self):
        """Non-tensor values in dict output should pass through unchanged."""

        def ref_func(x):
            return {"out": x, "count": 42}

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        bf16_array = np.array([1.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array)
        assert str(result["out"].dtype) == "bfloat16"
        assert result["count"] == 42

    def test_scalar_passthrough_with_bfloat16(self):
        """Scalar kwargs should pass through even with bfloat16 inputs."""

        def ref_func(x, backward=False):
            assert backward is True
            return x

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        bf16_array = np.array([1.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array, backward=True)
        assert str(result["out"].dtype) == "bfloat16"

    def test_single_tensor_return_with_preserve(self):
        """preserve_lower_precision should work with single tensor return (not dict)."""

        def ref_func(x):
            return x * 2  # Returns a single tensor, not a dict

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        bf16_array = np.array([1.0, 3.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array)
        assert str(result["out"].dtype) == "bfloat16"


class TestTorchRefWrapperDtypeConverters:
    """Tests for torch_ref_wrapper with input_dtype_converter and output_dtype_converter."""

    def test_input_dtype_converter_called_for_bfloat16(self):
        """input_dtype_converter should be called with numpy array for bfloat16 inputs."""
        seen = {}

        def converter(value):
            seen['dtype_str'] = str(value.dtype)
            seen['value_type'] = type(value)
            return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)

        def ref_func(x):
            assert x.dtype == torch.bfloat16
            return x.float()

        wrapped = torch_ref_wrapper(ref_func, input_dtype_converter=converter)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        wrapped(x=bf16_array)
        assert seen['dtype_str'] == 'bfloat16'
        assert seen['value_type'] == np.ndarray

    def test_input_dtype_converter_none_falls_through(self):
        """Returning None from input_dtype_converter should use default behavior."""

        def converter(value):
            return None  # fall through to default

        def ref_func(x):
            assert x.dtype == torch.float32, "Should be float32 (default behavior)"
            return x

        wrapped = torch_ref_wrapper(ref_func, input_dtype_converter=converter)
        bf16_array = np.array([1.0]).astype(np_bfloat16)
        wrapped(x=bf16_array)  # Should not raise

    def test_input_dtype_converter_called_for_int32(self):
        """input_dtype_converter is called for all numpy inputs; returning None falls back to default."""
        called = {'count': 0}

        def converter(value):
            called['count'] += 1
            return None

        def ref_func(x):
            return x.float()

        wrapped = torch_ref_wrapper(ref_func, input_dtype_converter=converter)
        int_array = np.array([1, 2, 3], dtype=np.int32)
        wrapped(x=int_array)
        assert called['count'] == 1

    def test_output_dtype_converter_called_for_custom_dtype(self):
        """output_dtype_converter should be called and its result used when non-None."""

        def converter(tensor):
            if tensor.dtype == torch.bfloat16:
                return tensor.float().numpy().astype(np.float64)
            return None

        def ref_func(x):
            return x.to(torch.bfloat16)

        wrapped = torch_ref_wrapper(ref_func, output_dtype_converter=converter)
        f32_array = np.array([1.0, 2.0], dtype=np.float32)
        result = wrapped(x=f32_array)
        assert result["out"].dtype == np.float64

    def test_output_dtype_converter_none_falls_through(self):
        """Returning None from output_dtype_converter should use default behavior."""

        def converter(tensor):
            return None

        def ref_func(x):
            return x

        wrapped = torch_ref_wrapper(ref_func, output_dtype_converter=converter)
        f32_array = np.array([1.0], dtype=np.float32)
        result = wrapped(x=f32_array)
        assert result["out"].dtype == np.float32

    def test_output_dtype_converter_with_dict_output(self):
        """output_dtype_converter should apply to each tensor in dict output."""
        call_count = {'n': 0}

        def converter(tensor):
            call_count['n'] += 1
            return None  # use default for all

        def ref_func(x):
            return {"a": x, "b": x * 2}

        wrapped = torch_ref_wrapper(ref_func, output_dtype_converter=converter)
        f32_array = np.array([1.0], dtype=np.float32)
        wrapped(x=f32_array)
        assert call_count['n'] == 2

    def test_both_converters_together(self):
        """input_dtype_converter and output_dtype_converter should work together."""

        def in_converter(value):
            if 'bfloat16' in str(value.dtype):
                return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
            return None

        def out_converter(tensor):
            if tensor.dtype == torch.bfloat16:
                return np_static_cast(tensor.float().numpy(), np_bfloat16)
            return None

        def ref_func(x):
            assert x.dtype == torch.bfloat16
            return x * 2

        wrapped = torch_ref_wrapper(ref_func, input_dtype_converter=in_converter, output_dtype_converter=out_converter)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array)
        assert str(result["out"].dtype) == "bfloat16"


class TestTorchRefWrapperBackwardCompat:
    """Verify preserve_lower_precision behavior is unchanged when no custom converters are provided."""

    def test_bfloat16_still_upcasts_to_f32_without_preserve(self):
        """Without preserve_lower_precision, bf16 input should arrive as fp32."""

        def ref_func(x):
            assert x.dtype == torch.float32
            return x

        wrapped = torch_ref_wrapper(ref_func)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        wrapped(x=bf16_array)

    def test_bfloat16_preserved_with_preserve_flag(self):
        """With preserve_lower_precision, bf16 input should arrive as torch.bfloat16."""

        def ref_func(x):
            assert x.dtype == torch.bfloat16
            return x

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        wrapped(x=bf16_array)

    def test_bfloat16_output_castback_with_preserve_flag(self):
        """With preserve_lower_precision, output should be cast back to bf16 numpy."""

        def ref_func(x):
            return x * 2

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        bf16_array = np.array([1.0, 2.0]).astype(np_bfloat16)
        result = wrapped(x=bf16_array)
        assert str(result["out"].dtype) == "bfloat16"

    def test_fp8_still_upcasts_to_f32_without_converter(self):
        """Without custom converter, fp8 input should arrive as fp32."""

        def ref_func(x):
            assert x.dtype == torch.float32
            return x

        wrapped = torch_ref_wrapper(ref_func)
        fp8_array = np.array([1.0, 2.0]).astype(np_float8_e4m3fn)
        wrapped(x=fp8_array)

    def test_fp8_still_upcasts_to_f32_with_preserve_flag(self):
        """With preserve_lower_precision but no converter, fp8 input should still be fp32."""

        def ref_func(x):
            assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
            return x

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        fp8_array = np.array([1.0, 2.0]).astype(np_float8_e4m3fn)
        wrapped(x=fp8_array)

    def test_fp8_output_castback_with_preserve_flag(self):
        """With preserve_lower_precision, fp8 output should be cast back to fp8 numpy."""

        def ref_func(x):
            return x * 2

        wrapped = torch_ref_wrapper(ref_func, preserve_lower_precision=True)
        fp8_array = np.array([1.0, 0.5]).astype(np_float8_e4m3fn)
        result = wrapped(x=fp8_array)
        assert str(result["out"].dtype) == "float8_e4m3fn"


class TestValidateInputKeys:
    """Tests for validate_input_keys shared helper."""

    def test_valid_keys_pass(self):
        def kernel(a, b, c=None):
            pass

        validate_input_keys({"a": 1, "b": 2}, kernel)  # Should not raise

    def test_extra_key_raises(self):
        def kernel(a):
            pass

        with pytest.raises(ValueError, match="extra.*don't match"):
            validate_input_keys({"a": 1, "extra": 2}, kernel)

    def test_missing_required_raises(self):
        def kernel(a, b):
            pass

        with pytest.raises(ValueError, match="missing required.*b"):
            validate_input_keys({"a": 1}, kernel)

    def test_optional_missing_ok(self):
        def kernel(a, b=None):
            pass

        validate_input_keys({"a": 1}, kernel)  # Should not raise

    def test_must_alias_input_accepted(self):
        def kernel(a, output):
            return a, output

        validate_input_keys({"a": 1, "output.must_alias_input": 2}, kernel)

    def test_must_alias_input_satisfies_required(self):
        def kernel(a, output):
            return a, output

        # "output" is required but provided as "output.must_alias_input"
        validate_input_keys({"a": 1, "output.must_alias_input": 2}, kernel)

    def test_must_alias_input_unknown_base_raises(self):
        def kernel(a):
            pass

        with pytest.raises(ValueError, match="unknown.must_alias_input.*don't match"):
            validate_input_keys({"a": 1, "unknown.must_alias_input": 2}, kernel)


class TestFilterKernelInput:
    """Tests for filter_kernel_input shared helper."""

    def test_filters_to_kernel_params(self):
        def kernel(a, b):
            pass

        result = filter_kernel_input({"a": 1, "b": 2, "extra": 3}, kernel)
        # extra is not in kernel params but validate_input_keys would catch it;
        # filter_kernel_input just silently drops it
        assert result == {"a": 1, "b": 2}

    def test_preserves_must_alias_input(self):
        def kernel(a, output):
            return a, output

        result = filter_kernel_input({"a": 1, "output.must_alias_input": 2}, kernel)
        assert result == {"a": 1, "output.must_alias_input": 2}

    def test_drops_unknown_must_alias(self):
        def kernel(a):
            pass

        result = filter_kernel_input({"a": 1, "unknown.must_alias_input": 2}, kernel)
        assert result == {"a": 1}


class TestFilterRefInput:
    """Tests for filter_ref_input shared helper."""

    def test_filters_to_ref_params(self):
        def torch_ref(a, b):
            pass

        result = filter_ref_input({"a": 1, "b": 2, "extra": 3}, torch_ref)
        assert result == {"a": 1, "b": 2}

    def test_must_alias_input_stripped_and_copied(self):
        def torch_ref(a, output):
            pass

        arr = np.array([1.0, 2.0])
        result = filter_ref_input({"a": 1, "output.must_alias_input": arr}, torch_ref)
        assert "output" in result
        # Should be a copy, not the same object
        assert result["output"] is not arr
        np.testing.assert_array_equal(result["output"], arr)

    def test_non_copyable_must_alias_passed_through(self):
        def torch_ref(a, scale):
            pass

        result = filter_ref_input({"a": 1, "scale.must_alias_input": 2.0}, torch_ref)
        assert result == {"a": 1, "scale": 2.0}


class TestValidateCrossRankConsistency:
    """Tests for validate_cross_rank_consistency shared helper."""

    def test_consistent_shapes_pass(self):
        rank0 = {"a": np.zeros((2, 3)), "b": np.ones((4,))}

        def gen(rank_id):
            return {"a": np.zeros((2, 3)), "b": np.ones((4,))}

        validate_cross_rank_consistency(gen, rank0, 2)  # Should not raise

    def test_shape_mismatch_raises(self):
        rank0 = {"a": np.zeros((2, 3))}

        def gen(rank_id):
            return {"a": np.zeros((2, 4))}

        with pytest.raises(ValueError, match="'a'.*shape mismatch"):
            validate_cross_rank_consistency(gen, rank0, 2)

    def test_dtype_mismatch_raises(self):
        rank0 = {"a": np.zeros((2,), dtype=np.float32)}

        def gen(rank_id):
            return {"a": np.zeros((2,), dtype=np.float16)}

        with pytest.raises(ValueError, match="'a'.*dtype mismatch"):
            validate_cross_rank_consistency(gen, rank0, 2)

    def test_non_array_values_ignored(self):
        rank0 = {"a": np.zeros((2,)), "scale": 1.0, "flag": True}

        def gen(rank_id):
            return {"a": np.zeros((2,)), "scale": 99.0, "flag": False}

        validate_cross_rank_consistency(gen, rank0, 2)  # Should not raise

    def test_mismatch_on_later_rank_raises(self):
        rank0 = {"a": np.zeros((2, 3))}

        def gen(rank_id):
            if rank_id == 3:
                return {"a": np.zeros((2, 4))}
            return {"a": np.zeros((2, 3))}

        with pytest.raises(ValueError, match="rank 0.*vs rank 3"):
            validate_cross_rank_consistency(gen, rank0, 4)
