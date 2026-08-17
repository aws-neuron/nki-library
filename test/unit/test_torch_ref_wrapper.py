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
"""Unit tests for torch_ref_wrapper."""

import numpy as np
from nkilib_src.nkilib.core.utils.torch_ref_wrapper import torch_ref_wrapper


def _ref(input_a, input_b=None, num_m_tiles=None):
    return input_a if input_b is None else input_a + input_b


class TestTorchRefWrapper:
    def test_exposes_all_construction_options_except_the_reference(self):
        # Every parameter except the reference itself is captured generically.
        w = torch_ref_wrapper(_ref)
        opts = w._torch_ref_cache_options
        assert "torch_ref_func" not in opts
        assert set(opts) == {
            "preserve_lower_precision",
            "input_dtype_converter",
            "output_dtype_converter",
        }
        assert opts["preserve_lower_precision"] is False

    def test_preserve_lower_precision_changes_the_captured_options(self):
        # Two wrappers over the same ref differing only in this flag must not share
        # captured options (they produce different goldens).
        w_default = torch_ref_wrapper(_ref)
        w_preserve = torch_ref_wrapper(_ref, preserve_lower_precision=True)
        assert w_default._torch_ref_cache_options != w_preserve._torch_ref_cache_options
        assert w_preserve._torch_ref_cache_options["preserve_lower_precision"] is True

    def test_wrapper_still_computes_correctly(self):
        # The cache attribute doesn't disturb the wrapper's compute behavior.
        w = torch_ref_wrapper(_ref)
        out = w(input_a=np.ones((2, 2), dtype=np.float32), input_b=np.ones((2, 2), dtype=np.float32))
        assert np.array_equal(out["out"], np.full((2, 2), 2.0, dtype=np.float32))
