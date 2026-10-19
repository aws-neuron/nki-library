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
"""Numpy<->Torch dtype conversion wrapper for torch reference functions.

Handles custom dtypes (bfloat16, float8, MX packed) that torch.from_numpy
cannot handle natively, enabling torch refs to be called with numpy inputs
from the kernel test framework or downstream E2E model debugging.
"""

import functools
from typing import Callable, Optional

import neuron_dtypes as ndt
import numpy as np
import torch


def torch_ref_wrapper(
    torch_ref_func: Callable,
    preserve_lower_precision: bool = False,
    input_dtype_converter: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    output_dtype_converter: Optional[Callable[[torch.Tensor], np.ndarray]] = None,
) -> Callable:
    """Wrap a torch reference function to handle numpy<->torch conversion.

    Converts numpy arrays to torch tensors (float16->float32 for CPU compatibility),
    calls the torch reference, and converts results back to numpy.

    Args:
        torch_ref_func: Torch reference function that takes torch tensors as kwargs
        preserve_lower_precision: If True, cast output tensors back to the original
            input dtype (e.g., bfloat16) using neuron_dtypes.static_cast.
        input_dtype_converter: Optional callback to customize numpy->torch conversion.
            Return a torch tensor to override default, or None to fall back.
        output_dtype_converter: Optional callback to customize torch->numpy conversion.
            Return a numpy array to override default, or None to fall back.

    Returns:
        Wrapped function that takes numpy arrays and returns numpy arrays.
    """

    # Options that change the golden but aren't in the ref's kwargs; the torch-ref
    # cache folds these into its key so wrappers differing only in an option don't
    # collide. locals() before any other binding is exactly the parameters.
    _params = dict(locals())
    cache_options = {name: value for name, value in _params.items() if name != "torch_ref_func"}

    @functools.wraps(torch_ref_func)
    def wrapped(**kwargs):
        torch_kwargs = {}
        original_dtype = None
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                dtype_str = str(value.dtype)
                # Try custom converter first
                if input_dtype_converter is not None:
                    converted = input_dtype_converter(value)
                    if converted is not None:
                        torch_kwargs[key] = converted
                        continue
                # MX packed x4 types: pass as numpy
                if 'x4' in dtype_str:
                    torch_kwargs[key] = value
                    continue
                # Track original dtype for output cast-back
                if original_dtype is None and (
                    'bfloat16' in dtype_str or 'float8' in dtype_str or 'float16' in dtype_str
                ):
                    original_dtype = dtype_str
                # Make value safe for torch.from_numpy
                if value.dtype == np.uint32:
                    value = value.astype(np.int32)
                elif 'bfloat16' in dtype_str or 'float8' in dtype_str:
                    value = value.astype(np.float32)
                tensor = torch.from_numpy(value)
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()
                elif preserve_lower_precision and 'bfloat16' in dtype_str:
                    tensor = tensor.to(torch.bfloat16)
                torch_kwargs[key] = tensor
            else:
                torch_kwargs[key] = value

        result = torch_ref_func(**torch_kwargs)

        def _tensor_to_numpy(t):
            if isinstance(t, torch.Tensor):
                if output_dtype_converter is not None:
                    converted = output_dtype_converter(t)
                    if converted is not None:
                        return converted
                if t.dtype == torch.bfloat16:
                    t = t.float()
                np_val = t.numpy()
                if preserve_lower_precision and original_dtype is not None:
                    np_val = ndt.static_cast(np_val, original_dtype)
                return np_val
            return t

        if isinstance(result, torch.Tensor):
            return {"out": _tensor_to_numpy(result)}
        elif isinstance(result, dict):
            return {k: _tensor_to_numpy(v) for k, v in result.items()}
        else:
            return result

    wrapped._torch_ref_cache_options = cache_options
    return wrapped
