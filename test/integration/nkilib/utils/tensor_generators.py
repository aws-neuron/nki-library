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
import inspect
from dataclasses import dataclass
from test.integration.nkilib.utils.dtype_helper import dt, static_cast
from test.utils.mx_utils import dequantize_mx_golden, get_mx_fp_max, get_mx_max_exp, quantize_mx_golden

import nki.language as nl
import numpy as np
from typing_extensions import override


def update_func_str(**arg_vals):
    """Decorator that creates a custom string representation for a function.

    This decorator wraps a function to provide a descriptive string representation
    that includes the outer function's name and any provided argument values.
    The wrapped function preserves the original function's behavior and metadata
    while overriding its __str__ method.

    Args:
        **arg_vals: Keyword arguments to include in the string representation.
                   These are formatted as key_value pairs in the final string.

    Returns:
        A decorator function that wraps the target function with a custom
        string representation.

    Example:
        >>> @update_func_str(mean=0.5, std=1.0)
        ... def my_func():
        ...     pass
        >>> str(my_func)
        'func_outer_function_name_mean_0.5_std_1.0'

    Note:
        The string format is: "func_{outer_func_name}_{key1}_{val1}_{key2}_{val2}..."
        where outer_func_name is obtained from the call stack.
    """

    def decorator(func):
        stack = inspect.stack()
        outer_func_name = stack[1].function

        class FunctionWithCustomStr:
            def __call__(self, *args, **kwargs):
                return func(*args, **kwargs)

            @override
            def __str__(self):
                return f"func_{outer_func_name}_{'_'.join(f'{k}_{v}' for k, v in arg_vals.items())}"

            # Preserve function metadata
            @property
            def __name__(self):
                return func.__name__

            @property
            def __doc__(  # pyright: ignore[reportImplicitOverride, reportIncompatibleVariableOverride]
                self,
            ):
                return func.__doc__

        return FunctionWithCustomStr()

    return decorator


@dataclass
class TensorTemplate:
    """
    Lightweight template for tensor metadata without allocating actual array memory.
    """

    name: str
    shape: tuple
    dtype: type


def guess_tensor_dtype(dtype):
    """Convert and normalize data types to their appropriate NumPy equivalents.

    This function maps various data types (including NKI-specific types) to their
    corresponding NumPy data types for tensor generation. It handles boolean,
    floating-point, integer, and special floating-point formats.

    Args:
        dtype: The input data type to convert. Can be a NumPy dtype, Python built-in
               type (bool), or NKI language type (nl.tfloat32, nl.bfloat16, nl.float8_e4m3).

    Returns:
        A NumPy dtype that can be used for tensor creation:
        - Boolean and floating types are returned as-is
        - Integer types are normalized to np.int32
        - NKI floating types (tfloat32, bfloat16, float8_e4m3) are mapped to np.float32

    Raises:
        AssertionError: If the dtype is not one of the supported types.

    Example:
        >>> guess_tensor_dtype(np.int64)
        numpy.int32
        >>> guess_tensor_dtype(nl.bfloat16)
        numpy.float32
    """
    if dtype == np.bool_ or dtype is bool or np.issubdtype(dtype, np.floating):
        return dtype
    elif np.issubdtype(dtype, np.integer):
        return np.int32
    elif dtype in (nl.tfloat32, nl.bfloat16, nl.float8_e4m3):
        return np.float32
    else:
        assert False, f"unsupported dtype {dtype}"


def gaussian_tensor_generator(mean: float = 0.0, std: float = 1.0, modifier_fn=None, lnc=None, seed: int = 0):
    """Create a tensor generator function that produces Gaussian-distributed tensors.

    This factory function returns a tensor generator that creates tensors filled with
    values drawn from a Gaussian (normal) distribution. The generator uses a
    configurable random seed for reproducibility.

    Args:
        mean (float, optional): The mean (center) of the Gaussian distribution.
                               Defaults to 0.0.
        std (float, optional): The standard deviation (spread) of the Gaussian
                              distribution. Defaults to 1.0.
        modifier_fn (callable, optional): A function to apply post-processing to the
                                         generated tensor. Should accept
                                         (tensor_template, tensor, lnc) as arguments.
                                         Defaults to None.
        lnc (optional): Additional context passed to the modifier function.
                       Defaults to None.
        seed (int, optional): Random seed for reproducible tensor generation.
                             Defaults to 0.

    Returns:
        callable: A tensor generator function that accepts a tensor_template and
                 returns a NumPy array with the same shape and dtype as the template,
                 filled with Gaussian-distributed values.

    Example:
        >>> generator = gaussian_tensor_generator(mean=0.5, std=2.0, seed=42)
        >>> template = np.zeros((3, 3), dtype=np.float32)
        >>> tensor = generator(template)
        >>> tensor.shape
        (3, 3)
        >>> tensor.dtype
        dtype('float32')

    Note:
        - Uses the specified random seed for reproducible tensor generation
        - The returned generator preserves the dtype of the input template
        - If a modifier_fn is provided, it is applied after generating the base tensor
    """
    rng = np.random.default_rng(seed)

    @update_func_str()
    def tensor_generator(shape, dtype, name):
        """Generate tensor with specified shape, dtype, and name.

        Args:
            shape: Tuple specifying tensor dimensions
            dtype: Data type for the tensor
            name: Name for the tensor
        """
        guessed_dtype = dtype  # guess_tensor_dtype(tensor_template.dtype)
        tensor = (rng.normal(size=shape) * std + mean).astype(guessed_dtype)
        if modifier_fn is not None:
            tensor = modifier_fn(TensorTemplate(name=name, shape=shape, dtype=dtype), tensor, lnc)
        return dt.static_cast(tensor, dtype=dtype)

    return tensor_generator


def np_random_sample_fp8(seed=0):
    """Create a random sampler for FP8 tensors with range [-240, 240]."""
    np.random.seed(seed)

    @update_func_str(seed=seed)
    def generator(shape, dtype, name=None):
        rand_arr = np.random.random_sample(shape) * 480.0 - 240.0
        return dt.static_cast(rand_arr, dtype)

    return generator


def np_random_sample(seed=0):
    """Create a random sampler generator function.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Generator function with signature (shape, dtype, name=None) -> np.ndarray
    """
    np.random.seed(0)

    @update_func_str(seed=seed)
    def generator(shape, dtype, name=None):
        if dtype == np.bool_ or dtype == bool:
            return np.random.randint(low=0, high=2, size=shape, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            return np.random.randint(
                low=max(np.iinfo(dtype).min, -100),
                high=min(np.iinfo(dtype).max, 100),
                size=shape,
                dtype=np.int32,
            )
        rand_arr = np.random.random_sample(shape)
        if np.issubdtype(dtype, np.floating):
            return rand_arr.astype(dtype)
        elif dtype in (nl.tfloat32, nl.bfloat16, nl.float8_e4m3, nl.float8_e5m2):
            rand_arr_casted = dt.static_cast(dt.static_cast(rand_arr, dtype), np.float32)
            return dt.static_cast(rand_arr_casted, dtype)
        else:
            assert False, f"unsupported dtype {dtype} for random number generation"

    return generator


def np_random_sample_static_quantize_inp(seed=0):
    """Create a random sampler generator function that generate a FP8 weight, a input scale, and a weight scale.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Generator function with signature (shape, dtype, name=None) -> np.ndarray
    """
    np.random.seed(0)

    @update_func_str(seed=seed)
    def generator(shape, dtype, name=None):
        rand_arr = np.random.random_sample(shape)
        w_scale = np.random.random_sample()
        in_scale = np.random.random_sample()
        if dtype in (nl.float8_e4m3, nl.float8_e5m2):
            # For FP8, generate numbers from -range to range
            max_range_map = {
                nl.float8_e4m3: 240.0,
                nl.float8_e5m2: 57344.0,
            }
            scale_map = {nl.float8_e4m3: 1 / 256.0, nl.float8_e5m2: 1 / 65536.0}
            max_range = max_range_map[dtype]
            rand_arr_casted = dt.static_cast(dt.static_cast(rand_arr * max_range - max_range, dtype), np.float32)
            w_scale = dt.static_cast(w_scale * scale_map[dtype], np.float32)
            in_scale = dt.static_cast(in_scale * scale_map[dtype], np.float32)
            return dt.static_cast(rand_arr_casted, dtype), w_scale, in_scale
        else:
            assert False, f"unsupported dtype {dtype} for random number generation"

    return generator


def duplicate_row_rmsnorm_inp_generator(all_ones: bool = False):
    """Generate inputs for RMSNorm kernel with shape [B, S, H] and each row being the same."""
    rng = np.random.default_rng(0)

    @update_func_str(all_ones=all_ones)
    def tensor_generator(shape, dtype, name):
        if name == "hidden":
            B, S, H = shape

            if all_ones:
                return np.ones(shape=(B, S, H), dtype=dtype)

            # Generate [B, 1, H] rmsnorm output, then broadcast to [B, S, H]
            inp1 = rng.normal(size=(B, 1, H)).astype(dtype)
            inp1 = np.broadcast_to(inp1, shape=(B, S, H)) / 100.0

            return inp1

        elif name == "gamma":
            if all_ones:
                return np.ones(shape=shape, dtype=dtype)

        return np_random_sample()(shape, dtype, name)

    return tensor_generator


def sparse_nonzero_tensor_generator(num_nonzero: int, value_range: tuple, seed: int):
    """Create a tensor generator that produces sparse tensors with specific nonzero patterns.

    This generator creates tensors where most elements are zero, with a controlled number
    of nonzero elements at random positions. Nonzero values are uniformly distributed
    within the specified range.

    Args:
        num_nonzero (int): Number of nonzero elements to generate.
        value_range (tuple): (min, max) range for nonzero values.
        seed (int): Random seed for reproducible tensor generation.

    Returns:
        callable: A tensor generator function that accepts shape, dtype, and name,
                 and returns a sparse NumPy array with the specified nonzero pattern.
    """
    rng = np.random.default_rng(seed)

    @update_func_str(num_nonzero=num_nonzero, value_range=value_range, seed=seed)
    def tensor_generator(shape, dtype):
        # Create zero tensor
        tensor = np.zeros(shape, dtype=dtype)

        # Determine number of nonzero elements
        total_elements = np.prod(shape)
        n_nonzero = num_nonzero if num_nonzero is not None else rng.integers(0, total_elements + 1)
        n_nonzero = min(n_nonzero, total_elements)

        if n_nonzero > 0:
            # Generate random positions for nonzero elements
            flat_tensor = tensor.flatten()
            nonzero_positions = rng.choice(total_elements, size=n_nonzero, replace=False)

            # Generate random nonzero values within the specified range
            min_val, max_val = value_range
            nonzero_values = rng.uniform(min_val, max_val, size=n_nonzero).astype(dtype)

            # Set nonzero values
            flat_tensor[nonzero_positions] = nonzero_values
            tensor = flat_tensor.reshape(shape)

        return dt.static_cast(tensor, dtype=dtype)

    return tensor_generator


def generate_stabilized_mx_data(mx_dtype, shape, val_range=1.0):
    """
    Generate stabilized floating-point data and its equivalent MX quantized representation.

    This function returns standard floating-point numbers along with their equivalent
    MX quantized data and scale tensors that are stabilized in the sense that the
    floating-point data and MX data can convert to each other exactly without losing precision.

    Args:
        mx_dtype: MX quantization dtype (float8_e5m2_x4, float8_e4m3fn_x4, float4_e2m1fn_x4)
        shape: 2D shape for the output tensor, each 8x4 block is a scaling group; e.g.,
               fp_data[8*row : 8*(row+1), 4*col : 4*(col+1)] is a scaling group
        val_range: fp_data output will be in (-val_range, val_range), (default: 1.0)

    Returns:
        tuple: (fp_data, quantized_mx_data, quantized_mx_scale)
               - fp_data: floating-point data
               - quantized_mx_data: MX quantized data that can be de-quantized to fp_data
               - quantized_mx_scale: MX scale tensor
    """
    _q_height, _q_width = 8, 4
    assert shape[0] % _q_height == 0, f'shape[0] must be a multiple of {_q_height}, but got {shape[0]}'
    assert shape[1] % _q_width == 0, f'shape[1] must be a multiple of {_q_width}, but got {shape[1]}'

    if val_range == 0:
        zeros = np.zeros(shape)
        return zeros, *quantize_mx_golden(zeros, mx_dtype)

    # Generate initial random data within the representable range of mx_dtype
    max_val = get_mx_fp_max(mx_dtype)
    max_exp = get_mx_max_exp(mx_dtype)

    # Generate initial random mxfp data within the mxfp dtype's range.
    rand_data = (np.random.random(shape) * 2 - 1) * max_val

    # For each scaling block, randomly select one element to have max exponent.
    # This prevents change in mx_scale after quantize(dequantize(rand_mx_data, rand_mx_scale)), causing precision loss.
    for i in range(0, shape[0], _q_height):
        for j in range(0, shape[1], _q_width):
            # Random position within the tile
            tile_i = np.random.randint(0, _q_height - 1)
            tile_j = np.random.randint(0, _q_width - 1)

            # Set this element to have maximum exponent
            # Value = ±1.xxx × 2^max_exp (where 1.xxx is the mantissa)
            sign = np.random.choice([-1, 1])
            # Within the range of [1.0, 1.5) (could be upto 1.75 for mxfp8).
            mantissa = 1.0 + np.random.random() * 0.5
            rand_data[i + tile_i, j + tile_j] = sign * mantissa * (2**max_exp)

    rand_quantized_data = static_cast(rand_data.astype(np.float32), mx_dtype)

    # Calculate mx_scale bounds based on val_range
    # max_val already takes max_exp into account
    float32_exp_bias = 127
    mx_scale_upper_bound = min(255, int(np.log2(val_range / max_val) + float32_exp_bias))
    mx_scale_lower_bound = max(0, mx_scale_upper_bound - 10)

    rand_quantized_scale = np.random.randint(
        mx_scale_lower_bound,
        mx_scale_upper_bound + 1,
        size=(shape[0] // _q_height, shape[1] // _q_width),
        dtype=np.uint8,
    )

    dequantized_fp_data = dequantize_mx_golden(rand_quantized_data, rand_quantized_scale)

    return dequantized_fp_data, rand_quantized_data, rand_quantized_scale
