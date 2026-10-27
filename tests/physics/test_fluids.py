"""Fluid dynamics tests for the supporting_funcs.physics.fluids module."""
# TODO: verify d_dt_blowdown maintains a constant entropy process when heat_in is zero

# TODO: verify d_dt_ullage_liquid:
# rho_ullage * m_ullage + rho_prop * m_prop = V_tot
import numpy as np
import pytest
from tests.physics.conftest import RTOL_SOLVER

from supporting_funcs.physics import fluids
from supporting_funcs.physics.units import Q_


@pytest.fixture(scope="session")
def dp_qty():
    """Pressure drop values for testing, [Pa]."""
    return Q_(np.array([1e3, 5e3, 1e4, 5e4, 1e5], dtype=np.float64), 'Pa')

@pytest.fixture(scope="session")
def rho_qty():
    """Density values for testing, [kg/m^3]."""
    return Q_(np.array([0.5, 1.5, 2, 700, 1200], dtype=np.float64), 'kg/m**3')

@pytest.fixture(scope="session")
def cda_qty():
    """Flow area values for testing, [m^2]."""
    return Q_(np.array([1e-12, 5e-4, 1e-2, 14., 1200], dtype=np.float64), 'm**2')

@pytest.fixture(scope="session")
def mdot_qty():
    """Mass flow rate values for testing, [kg/s]."""
    return Q_(np.array([3.16227766e-11,
                        6.12372436e-2,
                        2.0,
                        1.17132404e5,
                        1.85903201e7], dtype=np.float64), 'kg/s')

class TestSteadyState:
    """Test suite for the steady state functions in supporting_funcs.physics.fluids."""

    def test_cda_from_dp(self, dp_qty: Q_, rho_qty: Q_, mdot_qty: Q_, cda_qty: Q_):
        """Check that the cda_from_dp function returns the correct cda value for a given pressure drop, density, and mass flow rate.""" # noqa: E501
        actual_cda = fluids.cda_from_dp(mdot_qty.m_as('kg/s'),
                                        rho_qty.m_as('kg/m**3'),
                                        dp_qty.m_as('Pa'),
                                        )
        cda_dbl =cda_qty.m_as('m**2')

        np.testing.assert_allclose(actual_cda, cda_dbl, rtol=RTOL_SOLVER)

    def test_dp_from_cda(self, dp_qty: Q_, rho_qty: Q_, mdot_qty: Q_, cda_qty: Q_):
        """Check that the dp_from_cda function returns the correct pressure drop for a given cda value, density, and mass flow rate.""" # noqa: E501
        actual_dp = fluids.dp_from_cda(mdot_qty.m_as('kg/s'),
                                        rho_qty.m_as('kg/m**3'),
                                        cda_qty.m_as('m**2'),
                                        )
        dp_dbl = dp_qty.m_as('Pa')

        np.testing.assert_allclose(actual_dp, dp_dbl, rtol=RTOL_SOLVER)

    def test_round_trip_cda_from_dp(self, dp_qty: Q_, rho_qty: Q_, mdot_qty: Q_):
        """Check that the cda_from_dp and dp_from_cda functions are inverses of each other."""
        cda_qty = fluids.cda_from_dp(dp=dp_qty.m_as('Pa'),
                                     rho=rho_qty.m_as('kg/m**3'),
                                     mdot=mdot_qty.m_as('kg/s'))
        round_trip_dp = fluids.dp_from_cda(cda=cda_qty,
                                            rho=rho_qty.m_as('kg/m**3'),
                                            mdot=mdot_qty.m_as('kg/s'))

        np.testing.assert_allclose(round_trip_dp, dp_qty.m_as('Pa'), rtol=RTOL_SOLVER)
