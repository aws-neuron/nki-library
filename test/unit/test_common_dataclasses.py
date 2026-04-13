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


from test.utils.common_dataclasses import (
    ModelTestType,
    _iter_model_configs,
    is_model_test_type,
    prepare_model_parametrize,
)


class TestModelTestType:
    def test_test_id_prefix(self):
        assert ModelTestType.BROAD.test_id_prefix == "MODEL_WIP_BROAD"
        assert ModelTestType.GENERALITY.test_id_prefix == "MODEL_WIP_GENERALITY"
        assert ModelTestType.OPTIMAL.test_id_prefix == "MODEL_WIP_OPTIMAL"


class TestIsModelTestType:
    @pytest.mark.parametrize(
        "test_type,expected",
        [
            ("MODEL_WIP", True),
            ("MODEL_WIP_BROAD", True),
            ("MODEL_WIP_GENERALITY", True),
            ("MODEL_WIP_OPTIMAL", True),
            ("manual", False),
            ("random", False),
        ],
    )
    def test_is_model_test_type(self, test_type, expected):
        assert is_model_test_type(test_type) == expected


class TestIterModelConfigs:
    def test_dict_format(self):
        configs = {
            ModelTestType.BROAD: [[1, 2], [3, 4]],
            ModelTestType.OPTIMAL: [[5, 6]],
        }
        result = list(_iter_model_configs(configs))
        assert result == [
            (ModelTestType.BROAD, [1, 2]),
            (ModelTestType.BROAD, [3, 4]),
            (ModelTestType.OPTIMAL, [5, 6]),
        ]

    def test_flat_list_format(self):
        configs = [[1, 2], [3, 4]]
        result = list(_iter_model_configs(configs))
        assert result == [
            (ModelTestType.BROAD, [1, 2]),
            (ModelTestType.BROAD, [3, 4]),
        ]


class TestPrepareModelParametrize:
    def test_dict_format(self):
        configs = {
            ModelTestType.BROAD: [[1, 2]],
            ModelTestType.GENERALITY: [[3, 4]],
        }
        params, ids = prepare_model_parametrize(configs)
        assert params == [[1, 2], [3, 4]]
        assert ids == ["MODEL_WIP_BROAD_1-2", "MODEL_WIP_GENERALITY_3-4"]

    def test_flat_list_format(self):
        configs = [[1, 2], [3, 4]]
        params, ids = prepare_model_parametrize(configs)
        assert params == [[1, 2], [3, 4]]
        assert ids == ["MODEL_WIP_BROAD_1-2", "MODEL_WIP_BROAD_3-4"]

    def test_custom_id_formatter(self):
        configs = {ModelTestType.OPTIMAL: [[10, 20]]}
        params, ids = prepare_model_parametrize(configs, id_formatter=lambda p: f"x{p[0]}")
        assert ids == ["MODEL_WIP_OPTIMAL_x10"]

    def test_empty_dict(self):
        params, ids = prepare_model_parametrize({})
        assert params == []
        assert ids == []


from test.utils.common_dataclasses import unpack_model_config


class TestUnpackModelConfig:
    def test_tuple_entry(self):
        mt, params = unpack_model_config((ModelTestType.GENERALITY, [1, 2, 3]))
        assert mt == ModelTestType.GENERALITY
        assert params == [1, 2, 3]

    def test_raw_list_defaults_to_broad(self):
        mt, params = unpack_model_config([1, 2, 3])
        assert mt == ModelTestType.BROAD
        assert params == [1, 2, 3]
