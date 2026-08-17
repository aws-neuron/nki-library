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
from unittest.mock import patch

import pytest

from test.utils.common_dataclasses import (
    PYTEST_XDIST_WORKER_ENV,
    InstanceSize,
    LazyGoldenGenerator,
    ModelTestType,
    Platforms,
    ValidationArgs,
    _iter_model_configs,
    is_model_test_type,
    is_xdist_worker,
    prepare_model_parametrize,
    resolve_probe_worker_count,
    unpack_model_config,
)


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
            (Platforms.TRN3_A0, "trn3pre"),
            (Platforms.TRN3_PDS, "trn3"),
            (Platforms.TRN3_PDS_A0, "trn3pre"),
        ],
    )
    def test_get_compile_target(self, platform, expected):
        assert platform.get_compile_target() == expected


class TestPlatformsFromStrSafe:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("trn2", Platforms.TRN2),
            ("trn3_pds_a0", Platforms.TRN3_PDS_A0),
        ],
    )
    def test_known_value_parses(self, value, expected):
        assert Platforms.from_str_safe(value) == expected

    @pytest.mark.parametrize("value", ["bogus", "TRN2", "", "trn2.48xlarge"])
    def test_unknown_value_returns_none(self, value):
        # Unrecognized strings degrade to None rather than raising.
        assert Platforms.from_str_safe(value) is None


class TestModelTestType:
    def test_test_id_prefix(self):
        assert ModelTestType.BROAD.test_id_prefix == "BROAD"
        assert ModelTestType.GENERALITY.test_id_prefix == "GENERALITY"
        assert ModelTestType.OPTIMAL.test_id_prefix == "OPTIMAL"


class TestIsModelTestType:
    @pytest.mark.parametrize(
        "test_type,expected",
        [
            ("BROAD", True),
            ("BROAD_some_test", True),
            ("GENERALITY_test", True),
            ("OPTIMAL_config", True),
            ("TIER0_ln-2", True),
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
            (ModelTestType.BROAD, [1, 2], None),
            (ModelTestType.BROAD, [3, 4], None),
            (ModelTestType.OPTIMAL, [5, 6], None),
        ]

    def test_flat_list_format(self):
        configs = [[1, 2], [3, 4]]
        result = list(_iter_model_configs(configs))
        assert result == [
            (ModelTestType.BROAD, [1, 2], None),
            (ModelTestType.BROAD, [3, 4], None),
        ]

    def test_platform_restricted_entry(self):
        platforms = {Platforms.TRN3, Platforms.TRN3_A0}
        configs = {
            ModelTestType.TIER0: [
                ([1, 2], platforms),
                [3, 4],
            ],
        }
        result = list(_iter_model_configs(configs))
        assert result == [
            (ModelTestType.TIER0, [1, 2], platforms),
            (ModelTestType.TIER0, [3, 4], None),
        ]


class TestPrepareModelParametrize:
    def test_dict_format(self):
        configs = {
            ModelTestType.BROAD: [[1, 2]],
            ModelTestType.GENERALITY: [[3, 4]],
        }
        params, ids = prepare_model_parametrize(configs)
        assert params == [[1, 2], [3, 4]]
        assert ids == ["BROAD_1-2", "GENERALITY_3-4"]

    def test_flat_list_format(self):
        configs = [[1, 2], [3, 4]]
        params, ids = prepare_model_parametrize(configs)
        assert params == [[1, 2], [3, 4]]
        assert ids == ["BROAD_1-2", "BROAD_3-4"]

    def test_custom_id_formatter(self):
        configs = {ModelTestType.OPTIMAL: [[10, 20]]}
        params, ids = prepare_model_parametrize(configs, id_formatter=lambda p: f"x{p[0]}")
        assert ids == ["OPTIMAL_x10"]

    def test_empty_dict(self):
        params, ids = prepare_model_parametrize({})
        assert params == []
        assert ids == []


class TestUnpackModelConfig:
    def test_tuple_entry(self):
        mt, params = unpack_model_config((ModelTestType.GENERALITY, [1, 2, 3]))
        assert mt == ModelTestType.GENERALITY
        assert params == [1, 2, 3]

    def test_raw_list_defaults_to_broad(self):
        mt, params = unpack_model_config([1, 2, 3])
        assert mt == ModelTestType.BROAD
        assert params == [1, 2, 3]


class TestValidationArgsEqualNanInf:
    def test_default_equal_nan_inf_is_false(self):
        golden = LazyGoldenGenerator(lazy_golden_generator=lambda: {}, output_ndarray={})
        args = ValidationArgs(golden_output=golden)
        assert args.equal_nan_inf is False

    def test_equal_nan_inf_can_be_set_true(self):
        golden = LazyGoldenGenerator(lazy_golden_generator=lambda: {}, output_ndarray={})
        args = ValidationArgs(golden_output=golden, equal_nan_inf=True)
        assert args.equal_nan_inf is True


class TestResolveProbeWorkerCount:
    """Sizing for the host-probe thread pool, shared by the static capacity probe and the
    fleet reachability probe: bound by the run's parallelism, floored at 1, never exceeding
    the host count."""

    def test_bounded_by_parallelism_cap(self):
        # More hosts than the --maxprocesses cap: the cap wins so probing never outruns the run.
        assert resolve_probe_worker_count(10, 4) == 4

    def test_bounded_by_host_count(self):
        # Fewer hosts than the cap: no point spawning idle workers.
        assert resolve_probe_worker_count(3, 64) == 3

    @pytest.mark.parametrize("cap", [None, 0, -5])
    def test_non_positive_cap_falls_back_to_cpu_count(self, cap):
        # An unset/invalid cap falls back to CPU count (then still clamped to host count).
        with patch("test.utils.common_dataclasses.os.cpu_count", return_value=8):
            assert resolve_probe_worker_count(20, cap) == 8

    def test_never_below_one(self):
        # Defensive: callers short-circuit an empty host list, but the pool size stays >= 1.
        assert resolve_probe_worker_count(0, 8) == 1


class TestIsXdistWorker:
    def test_true_when_worker_env_set(self, monkeypatch):
        # xdist sets the var (to the worker id) in each worker process.
        monkeypatch.setenv(PYTEST_XDIST_WORKER_ENV, "gw0")
        assert is_xdist_worker() is True

    def test_false_when_worker_env_unset(self, monkeypatch):
        # The controller/master process (and a non-xdist run) has no such var.
        monkeypatch.delenv(PYTEST_XDIST_WORKER_ENV, raising=False)
        assert is_xdist_worker() is False

    def test_true_even_for_empty_worker_id(self, monkeypatch):
        # Presence is what marks a worker, not a truthy value — key by membership so an
        # (unusual) empty id still reads as a worker rather than silently as the controller.
        monkeypatch.setenv(PYTEST_XDIST_WORKER_ENV, "")
        assert is_xdist_worker() is True


class TestInstanceSize:
    """InstanceSize parses catalog size strings and owns each size's nominal physical-core count
    (used to core-weight the shared-fleet capacity gate)."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("48xlarge", InstanceSize.X_48XLARGE),
            ("3xlarge", InstanceSize.X_3XLARGE),
        ],
    )
    def test_from_str_parses_known_sizes(self, value, expected):
        assert InstanceSize.from_str(value) == expected

    @pytest.mark.parametrize("value", ["99xlarge", "48XLARGE", "", None])
    def test_from_str_maps_unrecognized_to_unknown(self, value):
        # An unrecognized/absent size degrades to UNKNOWN (with a warning) rather than raising,
        # so a new size is flagged rather than silently mis-weighted.
        assert InstanceSize.from_str(value) is InstanceSize.UNKNOWN

    @pytest.mark.parametrize(
        "size,cores",
        [
            (InstanceSize.X_48XLARGE, 128),
            (InstanceSize.X_3XLARGE, 8),
            (InstanceSize.UNKNOWN, 1),  # a host's existence implies at least one core
        ],
    )
    def test_get_core_count(self, size, cores):
        assert size.get_core_count() == cores
