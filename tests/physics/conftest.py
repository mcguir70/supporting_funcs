"""Shared fixtures for the supporting_funcs test suite.

Note: this file deliberately imports ``supporting_funcs`` as an installed
package.  If that import fails, the src-layout packaging is wrong -- do not
"fix" it with ``sys.path.insert``.
"""

import numpy as np
import pytest

# Relative tolerance for comparison against published table values.
# NACA TR-1135 tables are given to 5 significant figures, so 1e-5 is the
# tightest meaningful bound.
RTOL_TABLE = 1e-5

# Relative tolerance for iterative solvers (root finders).  Looser than
# RTOL_TABLE because the default solver tolerance in the module is 1e-6.
RTOL_SOLVER = 1e-5

# Ratio of specific heats for dry air at standard conditions [1].
GAMMA_AIR = 1.4

# Representative ratio of specific heats for rocket combustion products [1].
GAMMA_COMBUSTION = 1.2


@pytest.fixture(scope="session")
def gamma_air() -> float:
    """Ratio of specific heats for air, [1]."""
    return GAMMA_AIR


@pytest.fixture(scope="session")
def gamma_combustion() -> float:
    """Ratio of specific heats for typical combustion products, [1]."""
    return GAMMA_COMBUSTION


@pytest.fixture(scope="session")
def gamma_sweep() -> np.ndarray:
    """Range of physically plausible specific heat ratios, [1]."""
    return np.array([1.1, 1.2, 1.3, 1.4, 1.667], dtype=np.float64)


@pytest.fixture(scope="session")
def mach_supersonic() -> np.ndarray:
    """Supersonic Mach numbers avoiding the M -> 1 singularity, [1]."""
    return np.array([1.05, 1.5, 2.0, 3.0, 5.0, 8.0], dtype=np.float64)

@pytest.fixture(scope="session")
def mach_subsonic() -> np.ndarray:
    """Subsonic Mach numbers avoiding the M -> 0 singularity, [1]."""
    return np.array([0.05, 0.2, 0.5, 0.8, 0.95], dtype=np.float64)

@pytest.fixture(scope="session")
def mach_sweep() -> np.ndarray:
    """Range of physically plausible Mach numbers, [1]."""
    return np.array([0.05, 0.2, 0.5, 0.8, 0.95, 1.05, 1.5, 2.0, 3.0, 5.0, 8.0], dtype=np.float64)
