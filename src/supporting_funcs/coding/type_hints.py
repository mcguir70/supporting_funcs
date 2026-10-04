"""Type aliases for common array types used in physics calculations."""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import bool as np_bool
from numpy import dtype, float64, int64, ndarray, uint64
from numpy.typing import NDArray

type OneDArray = ndarray[tuple[int], dtype[float64]]
type OneDBoolArray = ndarray[tuple[int], dtype[np_bool]]
type OneDIntArray = ndarray[tuple[int], dtype[int64]]
type OneDUIntArray = ndarray[tuple[int], dtype[uint64]]
type OneDFloatArray = ndarray[tuple[int], dtype[float64]]
type TwoDArray = ndarray[tuple[int, int], dtype[float64]]
type NPFloatArray = ndarray[tuple[int] | tuple[int, int], dtype[float64]]

type AxesArr = ndarray[tuple[int] | tuple[int, int], dtype[Any]]
type MatPlotLibFigAxes = tuple[Figure, Axes | AxesArr]
# Matplotlib just returns an ndarray of Axes objects, so we can't specify the dtype as Axes.

type IntArray = NDArray[int64]
type UIntArray = NDArray[uint64]
type FloatArray = NDArray[float64]
type BoolArray = NDArray[np_bool]

type FloatOrArray = float | FloatArray
type BoolOrArray = bool | BoolArray

type AxesOrArray = Axes | AxesArr
