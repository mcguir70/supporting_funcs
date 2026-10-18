"""Geometric calculations for common shapes.

Functions
---------
circle_circumference
    Calculate the circumference of a circle given its radius.
sphere_area
    Calculate the surface area of a sphere given its radius.
cylinder_radial_area
    Calculate the radial surface area of a cylinder given its radius and length.
cylinder_total_area
    Calculate the total surface area of a cylinder given its radius and length.
sphere_volume
    Calculate the volume of a sphere given its radius.
cylinder_volume
    Calculate the volume of a cylinder given its radius and length.
"""
from typing import overload

import numpy as np

from supporting_funcs.coding.type_hints import FloatArray, FloatOrArray


# region Length
@overload
def circle_circumference(radius: float) -> float: ...
@overload
def circle_circumference(radius: FloatArray) -> FloatArray: ...
def circle_circumference(radius: FloatOrArray) -> FloatOrArray:
    """Calculate the circumference of a circle given its radius.

    Parameters
    ----------
    radius: [length]
        The radius of the circle.

    Returns
    -------
    circumference: [length]
        The circumference of the circle.
    """
    return 2 * np.pi * radius
# endregion Length
# region Area
@overload
def sphere_area(radius: float) -> float: ...
@overload
def sphere_area(radius: FloatArray) -> FloatArray: ...
def sphere_area(radius: FloatOrArray) -> FloatOrArray:
    """Calculate the surface area of a sphere given its radius.

    Parameters
    ----------
    radius: [length]
        The radius of the sphere.

    Returns
    -------
    area: [length**2]
        The surface area of the sphere.
    """
    return 4 * np.pi * radius**2

@overload
def cylinder_radial_area(radius: float, length: float) -> float: ...
@overload
def cylinder_radial_area(radius: FloatArray, length: FloatOrArray) -> FloatArray: ...
@overload
def cylinder_radial_area(radius: FloatOrArray, length: FloatArray) -> FloatArray: ...
def cylinder_radial_area(radius: FloatOrArray, length: FloatOrArray) -> FloatOrArray:
    """Calculate the radial (lateral) surface area of a cylinder given its radius and length.

    Parameters
    ----------
    radius: [length]
        The radius of the cylinder.
    length: [length]
        The length of the cylinder.

    Returns
    -------
    area: [length**2]
        The radial (lateral) surface area of the cylinder.
    """
    return 2 * np.pi * radius * length

@overload
def cylinder_total_area(radius: float, length: float) -> float: ...
@overload
def cylinder_total_area(radius: FloatOrArray, length: FloatArray) -> FloatArray: ...
@overload
def cylinder_total_area(radius: FloatArray, length: FloatOrArray) -> FloatArray: ...
def cylinder_total_area(radius: FloatOrArray, length: FloatOrArray) -> FloatOrArray:
    """Calculate the total surface area of a cylinder given its radius and length.

    Parameters
    ----------
    radius: [length]
        The radius of the cylinder.
    length: [length]
        The length of the cylinder.

    Returns
    -------
    area: [length**2]
        The total surface area of the cylinder.
    """
    return 2 * np.pi * radius * (length + radius)
# endregion Area
# region Volume
@overload
def sphere_volume(radius: float) -> float: ...
@overload
def sphere_volume(radius: FloatArray) -> FloatArray: ...
def sphere_volume(radius: FloatOrArray) -> FloatOrArray:
    """Calculate the volume of a sphere given its radius.

    Parameters
    ----------
    radius: [length]
        The radius of the sphere.

    Returns
    -------
    volume: [length**3]
        The volume of the sphere.
    """
    return (4 / 3) * np.pi * radius**3

@overload
def cylinder_volume(radius: float, length: float) -> float: ...
@overload
def cylinder_volume(radius: FloatArray, length: FloatOrArray) -> FloatArray: ...
@overload
def cylinder_volume(radius: FloatOrArray, length: FloatArray) -> FloatArray: ...
def cylinder_volume(radius: FloatOrArray, length: FloatOrArray) -> FloatOrArray:
    """Calculate the volume of a cylinder given its radius and length.

    Parameters
    ----------
    radius: [length]
        The radius of the cylinder.
    length: [length]
        The length of the cylinder.

    Returns
    -------
    volume: [length**3]
        The volume of the cylinder.
    """
    return np.pi * radius**2 * length
# endregion Volume
