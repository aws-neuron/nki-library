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
from typing import Optional, TextIO

import numpy as np
from neuron_dtypes import is_float_type


def maxAllClose(
    a,
    b,
    rtol=1e-05,
    atol=1e-08,
    equal_nan_inf=False,
    verbose=0,
    mode="max",
    logfile: Optional[TextIO] = None,
    min_pass_rate: float = 1.0,
) -> bool:
    """
    Compare two arrays for approximate equality with configurable tolerance modes.

    This function provides a custom comparison utility with lower sensitivity by considering
    the overall value range. Unlike numpy.allclose, it offers detailed mismatch reporting and
    uses the overall value range for relative difference calculations, making it suitable for
    validating numerical computation results where global tolerance is more appropriate than
    per-element tolerance.

    Reference: Spec Issue NCC-181

    Args:
        a (np.ndarray): Test array to compare (typically the computed result).
        b (np.ndarray): Reference/gold array to compare against (ground truth).
        rtol (float, optional): Relative tolerance multiplier. Elements are considered equal if
            abs(a - b) <= atol + rtol * brange. Defaults to 1e-05.
        atol (float, optional): Absolute tolerance threshold. Provides a minimum tolerance
            independent of magnitude. Defaults to 1e-08.
        equal_nan_inf (bool, optional): If True, matching NaN values and matching infinity
            values (same sign) are treated as equal. Defaults to False.
        verbose (int, optional): Verbosity level controlling output detail. Defaults to 0.
            - 0: Silent mode, returns boolean only
            - 1: Prints summary with largest absolute and relative differences
            - 3: Prints detailed information for every mismatched element
        mode (str, optional): Comparison mode determining how relative differences are
            calculated. Defaults to "max".
            - "py": Python-style allclose with per-element range. Computes
              relDiff = absDiff / abs(b[i]) for each element. Reports the largest
              relative difference, matching np.testing.assert_allclose behavior.
            - "max": Uses global maximum absolute value in b for all comparisons.
              Computes relErr = (absDiff - atol) / max(abs(b)) as percentage.
              Provides looser, more uniform match criteria across all elements.
              Reports relErr for the element with largest absDiff.
        logfile (str, optional): File path for logging verbose output. When specified,
            all verbose output is appended to this file in addition to stdout.
            Defaults to None (no file logging).
        min_pass_rate (float, optional): Minimum fraction of elements that must pass
            tolerance check (0 to 1). Defaults to 1.0 (all elements must pass).
            When < 1.0, comparison passes if at least this fraction of elements
            are within tolerance.

    Returns:
        bool: True if all elements are approximately equal within specified tolerances.
            False if any element exceeds tolerance or if array shapes don't match.

    Raises:
        No exceptions are raised. Shape mismatches return False with optional error message.

    Notes:
        - Shape validation: Arrays must have identical shapes or comparison fails immediately.
        - Tolerance formula: abs(a - b) <= atol + rtol * brange
        - In "max" mode: brange = max(abs(b)), excluding NaN values
        - In "py" mode: brange = abs(b) computed per-element
        - Special value handling for floating point types:
          * inf - inf = 0.0 (matching infinities considered equal)
          * -inf - (-inf) = 0.0 (matching negative infinities considered equal)
          * NaN comparisons handled via equal_nan_inf parameter
        - Invalid operations (like inf - inf) are suppressed to prevent numpy warnings.

    Examples:
        Basic usage with default settings:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.0, 2.0, 3.0001])
        >>> maxAllClose(a, b, rtol=1e-3)
        True

        Verbose output with summary:
        >>> maxAllClose(a, b, rtol=1e-3, verbose=1)
        INFO: allclose comparison summary: largest abs diff = 0.0001, ...
        True

        Per-element detailed output:
        >>> maxAllClose(a, b, rtol=1e-5, verbose=3, mode="py")
        INFO: allclose comparison failed at (2,): abs(3 - 3.0001) = 0.0001 ...
        False

        Handling special values:
        >>> a = np.array([1.0, np.nan, np.inf])
        >>> b = np.array([1.0, np.nan, np.inf])
        >>> maxAllClose(a, b, equal_nan_inf=True)
        True

        Logging to file:
        >>> maxAllClose(a, b, rtol=1e-5, verbose=1, logfile="comparison.log")
        False
    """
    assert 0 <= rtol <= 1, f"Relative tolerance has to be between 0 and 1 but got {rtol=}"

    # b is gold/refernce, a is new
    if not a.shape == b.shape:
        if verbose > 0:
            print(f"ERROR: allclose mismatching shapes {a.shape} != {b.shape}")
        return False

    brange = abs(b)
    if mode == "max":
        brange = np.amax(brange[np.isfinite(brange)])

    # We ignore invalid errors here because inf-inf will cause numpy to produce nan
    # and throw an invalid error
    with np.errstate(invalid="ignore"):
        abs_diff = abs(a.astype(float) - b.astype(float))
    # Calling np.sign() on boolean types will crash, so we do this only for floats
    if is_float_type(a.dtype):
        # Replace matching inf or matching -inf with 0.0
        abs_diff = np.where((np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))), 0.0, abs_diff)
    close = abs_diff <= atol + rtol * brange
    if equal_nan_inf:
        close |= (a == b) | (np.isnan(a) & np.isnan(b))

    finiteness_mismatch_tensor = np.isfinite(a) != np.isfinite(b)
    is_finiteness_matching = not finiteness_mismatch_tensor.any()

    def printWithLog(s: str):
        print(s)
        if logfile is not None:
            print(s, file=logfile)

    sf = ""

    if not is_finiteness_matching:
        if verbose >= 1:
            sf += f"ERROR: There are indices with finite-infinite mismatches\n"

            if verbose >= 3:
                indices = zip(*np.where(finiteness_mismatch_tensor))
                for index in indices:
                    sf += f"{index}: {a[index]} vs {b[index]}\n"

            printWithLog(sf)

    if verbose >= 3:
        indices_array = np.argwhere(~close)
        s = (
            "INFO: allclose comparison failed at {indices}: "
            "abs({a:.4g} - {b:.4g}) = {absDiff:.4g} is greater than "
            "({atol:.4g} + {rtol:.4g} * {brange:.4g}) = {val:.4g}, "
        )
        sPy = "relative difference {relDiff:4g} "
        sMax = "relative error {relErr:4g} % "
        for indices in indices_array:
            indices = tuple(indices)
            brangep = brange[indices] if mode == "py" else brange
            val = atol + rtol * brangep
            absDiff = abs_diff[indices]
            if mode == "py":
                ss = s + sPy
                sf = ss.format(
                    a=a[indices],
                    b=b[indices],
                    atol=atol,
                    rtol=rtol,
                    brange=brangep,
                    indices=indices,
                    val=val,
                    absDiff=absDiff,
                    relDiff=absDiff / brangep,
                )
            else:
                ss = s + sMax
                relErr = (absDiff - atol) / brange
                sf = ss.format(
                    a=a[indices],
                    b=b[indices],
                    atol=atol,
                    rtol=rtol,
                    brange=brangep,
                    indices=indices,
                    val=val,
                    absDiff=absDiff,
                    relErr=relErr * 100,
                )
            printWithLog(sf)

    if verbose >= 1:
        largest_abs_diff = np.amax(abs_diff)
        s = "INFO: allclose comparison summary: largest abs diff = {:g}, "
        sPy = "largest relative difference = {:g} \n"
        sMax = "rel diff = {:g} % (check against current rel tolerance of {:g} %)\n"
        if mode == "py":
            with np.errstate(invalid="ignore", divide="ignore"):
                largest_relative_diff = np.amax(abs_diff / brange)
            sf = (s + sPy).format(largest_abs_diff, largest_relative_diff)
        else:
            maxAbsDiffElementRelDiff = (largest_abs_diff - atol) / brange if largest_abs_diff >= atol else 0.0
            sf = (s + sMax).format(largest_abs_diff, maxAbsDiffElementRelDiff * 100.0, rtol * 100.0)
        printWithLog(sf)

    all_close = np.all(close) and is_finiteness_matching
    if all_close or min_pass_rate >= 1.0:
        return all_close

    # Check if pass rate meets minimum threshold
    pass_rate = np.mean(close)
    passed = pass_rate >= min_pass_rate and is_finiteness_matching
    if verbose >= 1:
        printWithLog(
            f"INFO: pass rate = {pass_rate:.2%} (required: {min_pass_rate:.2%}) - {'PASSED' if passed else 'FAILED'}\n"
        )
    return passed


def get_largest_abs_diff(a, b, atol=1e-8):
    """
    Returns the relative error of the largest absolute difference as a decimal value.

    Formula: (largest_absolute_difference - atol) / max(abs(b))
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} != {b.shape}"

    with np.errstate(invalid="ignore"):
        abs_diff = abs(a.astype(float) - b.astype(float))

    if is_float_type(a.dtype):
        abs_diff = np.where((np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))), 0.0, abs_diff)

    largest_abs_diff = float(np.amax(abs_diff))

    # Handle case where all values are NaN
    b_non_nan = b[~np.isnan(b)]
    if b_non_nan.size == 0:
        return 0.0

    brange = float(np.amax(np.abs(b_non_nan)))

    if largest_abs_diff < atol or brange == 0:
        return 0.0

    return (largest_abs_diff - atol) / brange
