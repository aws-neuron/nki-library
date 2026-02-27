#!/usr/bin/env python3
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

"""Standalone tool to display zoom-in differences between actual and golden outputs."""

import argparse
import os
import sys

import numpy as np


def infer_dtype(file_path, shape):
    """Infer dtype from file size and shape."""
    file_size = os.path.getsize(file_path)
    num_elements = np.prod(shape)

    if file_size % num_elements != 0:
        raise ValueError(
            f"File size {file_size} is not divisible by num_elements {num_elements}. Shape {shape} is incorrect."
        )

    dtype_size = file_size / num_elements
    dtype_map = {1: 'int8', 2: 'float16', 4: 'float32', 8: 'float64'}

    if dtype_size not in dtype_map:
        raise ValueError(
            f"Cannot infer dtype: file_size={file_size}, num_elements={num_elements}, bytes_per_element={dtype_size}"
        )

    return dtype_map[dtype_size]


def print_2d_zoom_values(actual, golden, x_start, x_num, y_start, y_num):
    """Print expected:actual values in a 2D map format."""
    # Flatten to 2D if needed
    if actual.ndim > 2:
        actual = actual.reshape(-1, actual.shape[-1])
    if golden.ndim > 2:
        golden = golden.reshape(-1, golden.shape[-1])
    if actual.ndim == 1:
        actual = actual.reshape(1, -1)
    if golden.ndim == 1:
        golden = golden.reshape(1, -1)

    h, w = actual.shape
    x_end = min(x_start + x_num, w)
    y_end = min(y_start + y_num, h)
    zoom_w = x_end - x_start

    zone_size = 8 if zoom_w % 8 == 0 else 4

    print(f"\nZOOM-IN VALUES (rows {y_start}:{y_end}, cols {x_start}:{x_end})")

    # Column header
    col_header = "     "
    for j in range(x_start, x_end):
        if j % zone_size == 0:
            col_header += f"{j:<15}"
        else:
            col_header += " " * 15
    print(col_header)

    # Top border
    border = "    ┌"
    for j in range(zoom_w):
        border += "─" * 15
        if j < zoom_w - 1 and (j + 1) % zone_size == 0:
            border += "┬"
    border += "┐"
    print(border)

    # Data rows
    for i in range(y_start, y_end):
        row = f"{i:3d} │"
        for j in range(x_start, x_end):
            exp_val = golden[i, j]
            act_val = actual[i, j]

            exp_str = "nan" if np.isnan(exp_val) else ("inf" if np.isinf(exp_val) else f"{exp_val:.4g}")
            act_str = "nan" if np.isnan(act_val) else ("inf" if np.isinf(act_val) else f"{act_val:.4g}")

            row += f"{exp_str}:{act_str}".ljust(15)
            if j < x_end - 1 and (j - x_start + 1) % zone_size == 0:
                row += "│"
        row += "│"
        print(row)

    # Bottom border
    border = "    └"
    for j in range(zoom_w):
        border += "─" * 15
        if j < zoom_w - 1 and (j + 1) % zone_size == 0:
            border += "┴"
    border += "┘"
    print(border)


def main():
    parser = argparse.ArgumentParser(
        description="Display zoom-in view of differences between golden and actual tensor outputs"
    )
    parser.add_argument("x_start", type=int, help="Starting column index")
    parser.add_argument("x_num", type=int, help="Number of columns")
    parser.add_argument("y_start", type=int, help="Starting row index")
    parser.add_argument("y_num", type=int, help="Number of rows")
    parser.add_argument("--actual", default="out", help="Path to actual output file (default: out)")
    parser.add_argument(
        "--golden",
        default="golden-out.bin",
        help="Path to golden/expected output file (default: golden-out.bin)",
    )
    parser.add_argument("--dtype", default=None, help="Data type for golden (auto-inferred if not specified)")
    parser.add_argument("--actual-dtype", default=None, help="Data type for actual (auto-inferred if not specified)")
    parser.add_argument("--shape", required=True, help="Reshape tensor to (rows, cols), e.g., '8,1536'")

    args = parser.parse_args()

    # Parse shape
    shape = tuple(int(x) for x in args.shape.split(','))

    # Infer dtypes if not provided
    golden_dtype = args.dtype or infer_dtype(args.golden, shape)
    actual_dtype = args.actual_dtype or infer_dtype(args.actual, shape)

    print(f"Using dtypes: golden={golden_dtype}, actual={actual_dtype}")

    # Load tensors
    try:
        actual = np.fromfile(args.actual, dtype=getattr(np, actual_dtype))
        golden = np.fromfile(args.golden, dtype=getattr(np, golden_dtype))
    except Exception as e:
        print(f"Error loading files: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert actual to same dtype as golden for comparison
    if actual.dtype != golden.dtype:
        actual = actual.astype(golden.dtype)

    if actual.shape != golden.shape:
        print(f"Shape mismatch: actual {actual.shape} vs golden {golden.shape}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded tensors with shape: {actual.shape}")

    # Reshape
    try:
        actual = actual.reshape(shape)
        golden = golden.reshape(shape)
        print(f"Reshaped to: {actual.shape}")
    except Exception as e:
        print(f"Error reshaping: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate coordinates
    if actual.ndim == 1:
        h, w = 1, actual.shape[0]
    elif actual.ndim == 2:
        h, w = actual.shape
    else:
        h, w = actual.reshape(-1, actual.shape[-1]).shape

    if args.x_start < 0 or args.x_start >= w:
        print(f"Error: x_start={args.x_start} is out of bounds [0, {w})", file=sys.stderr)
        sys.exit(1)
    if args.y_start < 0 or args.y_start >= h:
        print(f"Error: y_start={args.y_start} is out of bounds [0, {h})", file=sys.stderr)
        sys.exit(1)
    if args.x_start + args.x_num > w:
        print(f"Error: x_start + x_num = {args.x_start + args.x_num} exceeds width {w}", file=sys.stderr)
        sys.exit(1)
    if args.y_start + args.y_num > h:
        print(f"Error: y_start + y_num = {args.y_start + args.y_num} exceeds height {h}", file=sys.stderr)
        sys.exit(1)

    # Display zoom-in view
    print_2d_zoom_values(
        actual,
        golden,
        args.x_start,
        args.x_num,
        args.y_start,
        args.y_num,
    )


if __name__ == "__main__":
    main()
