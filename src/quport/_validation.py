# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from numbers import Integral
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np


def is_boolean(value: object) -> bool:
    """True for a Python or NumPy boolean.

    ``type(value) is bool`` misses ``numpy.bool_``, which ``float()`` then
    accepts as 0.0 or 1.0, so a validator that promises to reject booleans has
    to check for both before converting.
    """
    return isinstance(value, bool | np.bool_)


def validate_nonnegative_integral(value: object, *, label: str) -> int:
    """Return a non-negative integer, rejecting booleans and non-integral values."""
    if type(value) is bool or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a non-negative integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def validate_nonnegative_finite_float(value: object, *, label: str) -> float:
    """Return a finite non-negative float, rejecting booleans."""
    if is_boolean(value):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        out = float(cast(SupportsFloat | SupportsIndex | str, value))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    if out < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return out


def validate_finite_result(value: float, *, label: str) -> float:
    """Return a computed result only if it remains finite."""
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value
