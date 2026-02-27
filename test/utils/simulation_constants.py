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
"""
Simulation mode constants - lightweight module with no heavy imports.

This module can be safely imported before deciding whether to enable simulation mode.
"""

# Environment variable to enable simulation mode.
# pytest-xdist workers don't inherit sys.argv, so we use an env var to propagate
# the simulation mode flag to worker processes.
# Values: "1" = force simulation mode, any other value (including "0") = defer to CLI flag.
SIMULATION_MODE_ENV_VAR = "NKILIB_SIMULATION_MODE"

# Environment variable to run all tests in simulation mode, including large tensor tests.
# Values: "1" = run all tests regardless of size, any other value = auto-skip large tests.
SIMULATION_RUN_ALL_ENV_VAR = "NKILIB_SIMULATION_RUN_ALL"
