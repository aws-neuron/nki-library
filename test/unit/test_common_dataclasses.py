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
from test.utils.common_dataclasses import Platforms

import pytest


class TestPlatformsIsTrn3:
    @pytest.mark.parametrize(
        "platform,expected",
        [
            (Platforms.TRN1, False),
            (Platforms.TRN2, False),
            (Platforms.TRN3, True),
            (Platforms.TRN3_A0, True),
        ],
    )
    def test_is_trn3(self, platform, expected):
        assert platform.is_trn3() == expected


class TestPlatformsGetCompileTarget:
    @pytest.mark.parametrize(
        "platform,expected",
        [
            (Platforms.TRN1, "trn1"),
            (Platforms.TRN2, "trn2"),
            (Platforms.TRN3, "trn3"),
            (Platforms.TRN3_A0, "trn3"),
        ],
    )
    def test_get_compile_target(self, platform, expected):
        assert platform.get_compile_target() == expected
