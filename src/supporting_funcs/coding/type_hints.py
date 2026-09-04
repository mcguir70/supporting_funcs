"""Type aliases for common array types used in physics calculations."""

from typing import Any, TypeAlias
from numpy import float64, ndarray, dtype, bool as np_bool
from numpy.typing import NDArray
from matplotlib.axes import Axes


OneDArray: TypeAlias = ndarray[tuple[int], dtype[float64]] # pylint: disable=C0103
OneDBoolArray: TypeAlias = ndarray[tuple[int], dtype[np_bool]] # pylint: disable=C0103
OneDFloatArray: TypeAlias = ndarray[tuple[int], dtype[float64]] # pylint: disable=C0103
TwoDArray: TypeAlias = ndarray[tuple[int, int], dtype[float64]] # pylint: disable=C0103
NPFloatArray: TypeAlias = ndarray[tuple[int] | tuple[int, int], dtype[float64]]

AxesArr: TypeAlias = ndarray[tuple[int] | tuple[int, int], dtype[Any]]
# Matplotlib just returns an ndarray of Axes objects, so we can't specify the dtype as Axes.

FloatArray: TypeAlias = NDArray[float64]
BoolArray: TypeAlias = NDArray[np_bool]

FloatOrArray: TypeAlias = float | FloatArray
BoolOrArray: TypeAlias = bool | BoolArray

AxesOrArray: TypeAlias = Axes | AxesArr
