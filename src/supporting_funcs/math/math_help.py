"""A collection of math helper functions.

Functions
---------
solve_cubic_positive_real : Solve a cubic equation for positive real roots.
"""
import numpy as np


def solve_cubic_positive_real(
    a: float, b: float, c: float, d: float, imag_tol: float = 1e-9
) -> np.ndarray:
    """
    Real, positive roots of a*x**3 + b*x**2 + c*x + d = 0.

    Parameters
    ----------
    a, b, c, d : float
        Real cubic coefficients (descending order).
    imag_tol : float
        Max |imag part| to treat a root as real (absolute, not relative).

    Returns
    -------
    roots : ndarray of float
        Real, positive roots, ascending order. Empty if none.
    """
    roots = np.roots([a, b, c, d])
    real_mask = np.abs(roots.imag) < imag_tol
    real_roots = roots[real_mask].real
    positive_roots = real_roots[real_roots > 0.0]
    return np.sort(positive_roots)
