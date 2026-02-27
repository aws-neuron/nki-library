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

import platform

"""
Helper for stepping through code in VS Code.

To use, add a launch.json profile in VS code for debugging via Remote Attach.  Use a profile that looks like this:
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ],
            "justMyCode": false
        }
    ]
}

NOTE: This does also work for other debugging tools that support the Debugging Adapter Protocol (DAP)

In your code, insert the following where appropriate to pause and wait for the debugger to attach:
  from test.utils.debug_utils import wait_for_debugger
  wait_for_debugger()

You can insert breakpoints where you want by adding:
  breakpoint()

To launch the debugger, go to the Run and Debug panel and hit the green triangle.
"""


def wait_for_debugger(port=5678, output_msg=True):
    import debugpy

    debugpy.listen(("localhost", port))
    if output_msg:
        print(f"debugpy listening on {platform.node()}:{port}", flush=True)
    debugpy.wait_for_client()
