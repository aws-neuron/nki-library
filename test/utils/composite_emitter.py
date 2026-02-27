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
"""Composite emitter that fans out to multiple emitters."""

import logging
from typing import final

from typing_extensions import override

from .metrics_collector import IMetricsCollector
from .metrics_emitter import IMetricsEmitter, OutputMode


@final
class CompositeEmitter(IMetricsEmitter):
    """Emitter that fans out to multiple child emitters."""

    def __init__(self, collector: IMetricsCollector, emitters: list[IMetricsEmitter]):
        """
        Initialize composite emitter.

        Args:
            collector: Shared MetricsCollector instance
            emitters: List of emitters to fan out to
        """
        self._collector = collector
        self._emitters = emitters
        self._logger = logging.getLogger(__name__)

    @override
    def get_metrics_enabled(self) -> bool:
        """Returns True if any child emitter has metrics enabled."""
        return any(emitter.get_metrics_enabled() for emitter in self._emitters)

    @override
    def get_output_mode(self) -> OutputMode | None:
        # Return first emitter's mode if any
        for emitter in self._emitters:
            mode = emitter.get_output_mode()
            if mode is not None:
                return mode
        return None

    @override
    def set_output_dir(self, output_dir: str) -> None:
        for emitter in self._emitters:
            emitter.set_output_dir(output_dir)

    @override
    def emit(self) -> None:
        """Call emit() on all child emitters, log errors but don't fail."""
        for emitter in self._emitters:
            try:
                emitter.emit()
            except Exception as e:
                self._logger.error(f"Emitter {type(emitter).__name__} failed: {e}")

    @override
    def get_collector(self) -> IMetricsCollector:
        return self._collector

    def emit_run_complete(self, kernel_name: str, tests_passed: int, tests_total: int) -> None:
        """Fan out run_complete to emitters that support it."""
        for emitter in self._emitters:
            if hasattr(emitter, "emit_run_complete"):
                try:
                    emitter.emit_run_complete(kernel_name, tests_passed, tests_total)
                except Exception as e:
                    self._logger.error(f"emit_run_complete failed for {type(emitter).__name__}: {e}")
