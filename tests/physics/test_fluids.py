"""Fluid dynamics tests for the supporting_funcs.physics.fluids module."""
# TODO: verify d_dt_blowdown maintains a constant entropy process when heat_in is zero

# TODO: verify d_dt_ullage_liquid:
# rho_ullage * m_ullage + rho_prop * m_prop = V_tot
import numpy as np
import pytest
from CoolProp import constants as CPc
from CoolProp.CoolProp import AbstractState
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

@pytest.fixture(scope="session")
def coolprop_gases():
    """List of fluids available in CoolProp for testing."""
    return ['Nitrogen', 'Oxygen', 'CarbonDioxide', 'Methane', 'Helium']

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

    def test_size_copv_adiabatic(self, coolprop_gases, rng_seed):
        """Check that the size_copv_adiabatic function obeys the boundary conditions."""
        rng_obj = np.random.default_rng(seed=rng_seed)
        starting_temp = 293.15  # K
        tank_fill_pressure = 101325  # Pa
        ullage_perc_vol = 0.1  # 10% ullage
        copv_start_pressure = rng_obj.uniform(tank_fill_pressure, 100e6)
        p_tank = rng_obj.uniform(1e5*0.9, copv_start_pressure*0.9)
        liquid_volume_start = 10 # m^3

        for fluid in coolprop_gases:
            copv_obj = fluids.size_copv_adiabatic(p1_copv = copv_start_pressure,
                                    p2_copv = p_tank * 1.2,
                                    temp_start = starting_temp,
                                    p_tank_start = tank_fill_pressure,
                                    p_tank = p_tank,
                                    liquid_volume = liquid_volume_start,
                                    ullage_volume_start = ullage_perc_vol * liquid_volume_start,
                                    fluid = fluid)

            state = AbstractState('HEOS', fluid)
            state.update(CPc.PT_INPUTS, copv_start_pressure, starting_temp)
            s_copv_start = state.smass()
            rho_copv_start = state.rhomass()
            u_copv_start = state.umass()

            state.update(CPc.PT_INPUTS, tank_fill_pressure, starting_temp)
            rho_ullage_start = state.rhomass()
            u_ullage_start = state.umass()
            s_ullage_start = state.smass()

            mass_ullage_start = rho_ullage_start * ullage_perc_vol * liquid_volume_start
            mass_copv_start = rho_copv_start * copv_obj.copv_volume

            state.update(CPc.PT_INPUTS, p_tank, copv_obj.temp_ullage_end)
            rho_ullage_end = state.rhomass()
            u_ullage_end = state.umass()
            s_ullage_end = state.smass()
            mass_ullage_end = rho_ullage_end * (1 + ullage_perc_vol) * liquid_volume_start

            state.update(CPc.PT_INPUTS, p_tank * 1.2, copv_obj.temp_copv_end)
            rho_copv_end = state.rhomass()
            u_copv_end = state.umass()
            s_copv_end = state.smass()
            mass_copv_end = rho_copv_end * copv_obj.copv_volume
            boundary_work = p_tank * liquid_volume_start

            U_err_ox = ((mass_ullage_start * u_ullage_start + mass_copv_start * u_copv_start  # noqa: N806
                        - (mass_ullage_end * u_ullage_end + mass_copv_end * u_copv_end + boundary_work)) # noqa: E501
                    / (mass_ullage_start * u_ullage_start + mass_copv_start * u_copv_start))
            m_err_ox = ((mass_ullage_start + mass_copv_start - (mass_ullage_end + mass_copv_end))  # noqa: E501
                        / (mass_ullage_start + mass_copv_start))

            s_copv_err_ox = (s_copv_start - s_copv_end) / s_copv_start
            S_ox_final = mass_copv_end * s_copv_end + mass_ullage_end * s_ullage_end  # noqa: N806
            S_ox_initial = mass_copv_start * s_copv_start + mass_ullage_start * s_ullage_start  # noqa: N806
            S_diff = (S_ox_final - S_ox_initial) / S_ox_initial  # noqa: N806

            assert 0 == pytest.approx(m_err_ox, rel=RTOL_SOLVER, abs=1e-7), (
                "Mass not conserved during ullage calculation for oxidizer.")
            assert 0 == pytest.approx(U_err_ox, rel=RTOL_SOLVER, abs=1e-7), (
                "Energy not conserved during ullage calculation for oxidizer.")
            assert 0 == pytest.approx(s_copv_err_ox, rel=RTOL_SOLVER, abs=1e-7), (
                "Entropy not conserved during ullage calculation for oxidizer.")
            assert S_diff > 0, "Entropy decreased during ullage calculation for oxidizer."
