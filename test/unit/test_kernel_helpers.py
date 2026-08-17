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
import nki.language as nl
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType
from nkilib_src.nkilib.core.utils.kernel_helpers import get_nl_act_fn_from_type


class TestGetNlActFnFromType:
    @pytest.mark.parametrize(
        "act_fn,expected",
        [
            (ActFnType.SiLU, nl.silu),
            (ActFnType.GELU, nl.gelu),
            (ActFnType.GELU_Tanh_Approx, nl.gelu_apprx_tanh),
            (ActFnType.Swish, nl.gelu_apprx_sigmoid),
        ],
    )
    def test_returns_correct_activation_function(self, act_fn, expected):
        assert get_nl_act_fn_from_type(act_fn) == expected

    def test_raises_on_invalid_type(self):
        with pytest.raises(AssertionError):
            get_nl_act_fn_from_type("invalid")
