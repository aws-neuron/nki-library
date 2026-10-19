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
import json
import subprocess
from unittest.mock import patch

import pytest

from test.utils.common_dataclasses import (
    PYTEST_XDIST_WORKER_ENV,
    InstanceSize,
    LazyGoldenGenerator,
    ModelTestType,
    NeuronDeviceInfo,
    NeuronLsReport,
    Platforms,
    ValidationArgs,
    _iter_model_configs,
    is_model_test_type,
    is_xdist_worker,
    looks_like_json,
    neuron_ls_json_command,
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


def _device(
    neuron_device: int = 0,
    neuroncore_ids: list[int] | None = None,
    lnc: int = 2,
    instance_type: str | None = None,
) -> dict:
    """One neuron-ls device entry (from_dict requires every key below). ``instance_type`` is
    omitted by default, as ``--output-format json`` does; legacy ``--json-output`` sets it."""
    device = {
        "neuron_device": neuron_device,
        "bdf": "0000:aa:00.0",
        "cpu_affinity": "48-95",
        "numa_node": "1",
        "connected_to": [1, 2],
        "nc_count": 4,
        "logical_neuroncore_config": lnc,
        "memory_size": 154618822656,
        "neuroncore_ids": [0, 1, 2, 3] if neuroncore_ids is None else neuroncore_ids,
        "neuron_processes": [],
    }
    if instance_type is not None:
        device["instance_type"] = instance_type
    return device


_ULTRASERVER_INFO = {
    "ultraserver_id": "4569742665279272783",
    "ultraserver_mode": "unset",
    "ultraserver_type": "Switch",
    "ultraserver_size": 4,
    "ultraserver_instance_index": 2,
    "ultraserver_supported_modes": "1: 4-node, 3: 2-node-vertical, 4: single-node",
}


class TestNeuronLsJsonCommand:
    """The remote/local neuron-ls invocation must survive a fleet where only some hosts have a
    neuron-ls new enough to support instance-level output.

    These tests execute the generated command against stub binaries rather than asserting on its
    text, because the failure that matters is behavioural: an old neuron-ls reports an unknown flag
    on STDOUT, so a command that streams the first attempt prepends that complaint to the
    fallback's JSON and the parse fails.
    """

    @staticmethod
    def _stub(tmp_path, body: str):
        """Write an executable stub neuron-ls that logs each invocation's arguments."""
        log = tmp_path / "calls.log"
        stub = tmp_path / "neuron-ls"
        stub.write_text(f'#!/bin/sh\necho "$@" >> {log}\n{body}\n')
        stub.chmod(0o755)
        return stub, log

    @staticmethod
    def _run(cmd: str) -> str:
        result = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True, timeout=30)
        return result.stdout

    def test_legacy_neuron_ls_yields_only_the_fallback_json(self, tmp_path):
        # Reproduces the real old binary: "unknown flag" on STDOUT with a non-zero exit.
        legacy_json = json.dumps([_device(neuroncore_ids=[0, 1], lnc=1)])
        stub, log = self._stub(
            tmp_path,
            'case "$*" in\n'
            "  *--output-format*) echo \"unknown flag \\`output-format'\"; exit 1 ;;\n"
            f"  *--json-output*) echo '{legacy_json}' ;;\n"
            "esac",
        )
        stdout = self._run(neuron_ls_json_command(str(stub)))

        # The complaint must not reach the parser, and the payload must be the legacy device list.
        assert "unknown flag" not in stdout
        assert NeuronLsReport.from_stdout(stdout).total_physical_cores == 2
        assert "--output-format json" in log.read_text()  # the new flag was still attempted first

    def test_modern_neuron_ls_skips_the_legacy_flag(self, tmp_path):
        new_json = json.dumps({"devices": [_device(neuroncore_ids=[0], lnc=1)]})
        stub, log = self._stub(
            tmp_path,
            'case "$*" in\n'
            f"  *--output-format*) echo '{new_json}' ;;\n"
            '  *--json-output*) echo "legacy should not run" ;;\n'
            "esac",
        )
        stdout = self._run(neuron_ls_json_command(str(stub)))

        assert NeuronLsReport.from_stdout(stdout).total_physical_cores == 1
        assert "--json-output" not in log.read_text()  # one round trip, no wasted invocation

    def test_falls_back_when_the_unknown_flag_exits_zero(self, tmp_path):
        # Acceptance keys off the payload shape, so a usage message on a zero exit also falls back.
        legacy_json = json.dumps([_device(neuroncore_ids=[0, 1, 2, 3], lnc=1)])
        stub, _ = self._stub(
            tmp_path,
            'case "$*" in\n'
            '  *--output-format*) echo "usage: neuron-ls [options]"; exit 0 ;;\n'
            f"  *--json-output*) echo '{legacy_json}' ;;\n"
            "esac",
        )
        stdout = self._run(neuron_ls_json_command(str(stub)))

        assert "usage:" not in stdout
        assert NeuronLsReport.from_stdout(stdout).total_physical_cores == 4

    def test_forces_lnc_for_both_attempts(self):
        # LNC is forced so the probe counts physical cores the way the worker will see them;
        # the fallback is a separate process and needs the same environment.
        env = f"NEURON_LOGICAL_NC_CONFIG={NeuronDeviceInfo.logical_neuroncore_config}"
        assert neuron_ls_json_command("/bin/neuron-ls").count(env) == 2


class TestLooksLikeJson:
    """Shape check that decides whether a neuron-ls attempt produced a payload or a complaint."""

    @pytest.mark.parametrize("stdout", ['{"devices": []}', "[]", "  \n [1]", '\n{"a": 1}'])
    def test_accepts_json_payloads(self, stdout):
        assert looks_like_json(stdout)

    @pytest.mark.parametrize("stdout", ["", "   ", "unknown flag `output-format'", "usage: neuron-ls", "null"])
    def test_rejects_non_payloads(self, stdout):
        # "null" is valid JSON but never a device list, so treating it as a miss is intended.
        assert not looks_like_json(stdout)


class TestNeuronLsReportFromStdout:
    """One parser for both neuron-ls JSON formats: the newer object (with instance-level info,
    including the multi-node deployment) and the legacy bare device list."""

    def test_parses_instance_format_with_multi_node_deployment(self):
        stdout = json.dumps(
            {
                "instance_info": {
                    "instance_type": "trn3-dev1.48xlarge",
                    "instance_id": "i-05aa9dd8c8eba4ae4",
                    "logical_neuroncore_config": 2,
                    "ultraserver_info": _ULTRASERVER_INFO,
                },
                "devices": [_device(0), _device(1)],
            }
        )
        report = NeuronLsReport.from_stdout(stdout)
        assert len(report.devices) == 2
        assert [d.instance_type for d in report.devices] == ["trn3-dev1.48xlarge"] * 2
        assert report.logical_host_id == "4569742665279272783"
        assert (report.logical_host_rank, report.logical_host_size) == (2, 4)
        assert report.total_physical_cores == 16  # 2 devices * 4 neuroncore_ids * lnc 2

    def test_single_node_host_reports_no_deployment(self):
        # A host outside a multi-node deployment omits ultraserver_info; its devices still parse.
        stdout = json.dumps({"instance_info": {"instance_type": "trn2.48xlarge"}, "devices": [_device(0)]})
        report = NeuronLsReport.from_stdout(stdout)
        assert len(report.devices) == 1
        assert report.devices[0].instance_type == "trn2.48xlarge"
        assert report.logical_host_id is None

    def test_device_level_instance_type_wins_over_instance_info(self):
        stdout = json.dumps(
            {
                "instance_info": {"instance_type": "trn2.48xlarge"},
                "devices": [_device(0, instance_type="trn3-dev1.48xlarge")],
            }
        )
        assert NeuronLsReport.from_stdout(stdout).devices[0].instance_type == "trn3-dev1.48xlarge"

    def test_instance_format_without_instance_type_leaves_it_empty(self):
        stdout = json.dumps({"instance_info": {}, "devices": [_device(0)]})
        assert NeuronLsReport.from_stdout(stdout).devices[0].instance_type == ""

    @pytest.mark.parametrize("ultraserver_info", [None, {}, {"ultraserver_id": ""}, {"ultraserver_size": 4}])
    def test_deployment_membership_requires_an_id(self, ultraserver_info):
        # The id alone decides membership: without it there is no deployment to join, whatever
        # else the host reports.
        stdout = json.dumps(
            {"instance_info": {"ultraserver_info": ultraserver_info}, "devices": [_device(0)]},
        )
        report = NeuronLsReport.from_stdout(stdout)
        assert (report.logical_host_id, report.logical_host_rank, report.logical_host_size) == (None, None, None)

    def test_deployment_is_reported_whatever_the_mode_is(self):
        # An unset/undecided mode still names a deployment — the id is what groups hosts.
        stdout = json.dumps(
            {
                "instance_info": {"ultraserver_info": {**_ULTRASERVER_INFO, "ultraserver_mode": "unset"}},
                "devices": [_device(0)],
            }
        )
        assert NeuronLsReport.from_stdout(stdout).logical_host_id == "4569742665279272783"

    def test_parses_legacy_device_list_without_a_deployment(self):
        # Legacy --json-output carries no instance-level info, so the deployment is unknown (None)
        # rather than wrongly reported as single-node.
        report = NeuronLsReport.from_stdout(json.dumps([_device(0, instance_type="trn2.48xlarge")]))
        assert [d.neuron_device for d in report.devices] == [0]
        assert report.devices[0].instance_type == "trn2.48xlarge"
        assert report.logical_host_id is None
        assert report.total_physical_cores == 8

    @pytest.mark.parametrize("stdout", ["[]", "{}", "null"])
    def test_empty_output_yields_no_devices(self, stdout):
        report = NeuronLsReport.from_stdout(stdout)
        assert report.devices == []
        assert report.total_physical_cores == 0

    def test_instance_format_without_devices_yields_no_devices(self):
        report = NeuronLsReport.from_stdout(json.dumps({"instance_info": {"ultraserver_info": _ULTRASERVER_INFO}}))
        assert report.devices == []
        assert report.logical_host_id == "4569742665279272783"

    def test_malformed_output_raises(self):
        # The caller decides what an unreadable host means; the parser does not guess.
        with pytest.raises(json.JSONDecodeError):
            NeuronLsReport.from_stdout("neuron-ls: unrecognized option")
        with pytest.raises(ValueError, match="unexpected neuron-ls JSON type"):
            NeuronLsReport.from_stdout("42")
