"""Tests for supporting_funcs.physics.compressibleFlow.

Reference values are taken from NACA TR-1135, "Equations, Tables, and Charts
for Compressible Flow" (Ames Research Staff, 1953), for gamma = 1.4.

Four kinds of test appear here, in rough order of value:

1. Table tests   -- pin the function to a published number.
2. Invariants    -- roundtrips, limits, monotonicity.  These catch algebra
                    errors at Mach numbers no table covers.
3. Type contract -- float in / float out, ndarray in / ndarray out,
                    broadcasting.  The module hand-rolls this per function,
                    so it needs checking per function.
4. Known bugs    -- xfail(strict=True), so the test suite fails loudly when
                    a bug is fixed and the marker is left behind.

All angles are radians.  All pressures are Pa.  Ratios are dimensionless [1].
"""

import warnings

import numpy as np
import pytest
import scipy.optimize as scipy_optimize

from supporting_funcs.physics import compressible_flow as cf

from tests.physics.conftest import RTOL_SOLVER, RTOL_TABLE


# ---------------------------------------------------------------------------
# Reference data (NACA TR-1135, gamma = 1.4)
# ---------------------------------------------------------------------------

# (mach, p/p0, T/T0, rho/rho0, A/A*)
ISENTROPIC_TABLE = [
    (0.5, 0.843019, 0.952381, 0.885170, 1.33984),
    (1.0, 0.528282, 0.833333, 0.633938, 1.00000),
    (2.0, 0.127805, 0.555556, 0.230048, 1.68750),
    (3.0, 0.0272237, 0.357143, 0.0762263, 4.23457),
    (5.0, 0.00189004, 0.166667, 0.0113402, 25.0000),
]

# (mach_1, mach_2, p2/p1, p02/p01)
NORMAL_SHOCK_TABLE = [
    (1.5, 0.701089, 2.45833, 0.929787),
    (2.0, 0.577350, 4.50000, 0.720874),
    (3.0, 0.475191, 10.3333, 0.328344),
    (5.0, 0.415227, 29.0000, 0.0617163),
]

# (mach, prandtl_meyer_angle_deg)
PRANDTL_MEYER_TABLE = [
    (1.0, 0.0),
    (1.5, 11.9052),
    (2.0, 26.3798),
    (3.0, 49.7573),
    (5.0, 76.9202),
]

# Weak-solution oblique shock, gamma = 1.4 (NACA TR-1135 chart 2).
# (mach_1, shock_angle_deg, deflection_angle_deg)
OBLIQUE_WEAK_TABLE = [
    (2.0, 39.31393, 10.0),
    (3.0, 27.38269, 10.0),
    (2.0, 53.42294, 20.0),
    (3.0, 37.76363, 20.0),
]

# Shock angle at maximum deflection, gamma = 1.4.
# Note theta_max is NOT monotonic in M1 -- it dips near M1 = 2 and rises
# again, so M1 = 1.5 and M1 = 5.0 give almost the same angle.  Easy to
# transcribe wrong; that is why the reference deflection is carried too.
# (mach_1, theta_max_deg, delta_max_deg)
OBLIQUE_MAX_TABLE = [
    (1.5, 66.58888, 12.11267),
    (2.0, 64.66898, 22.97353),
    (3.0, 65.24085, 34.07344),
    (5.0, 66.58424, 41.11766),
    (10.0, 67.45435, 44.42902),
]

class MiscTests:
    """Miscellaneous tests that don't fit anywhere else."""
    def try_help(self, func):
        """Check that the function has a docstring and type hints."""
        assert func.__doc__ is not None
        assert func.__annotations__ is not None
        help(func)  # smoke test -- no exception

# ---------------------------------------------------------------------------
# Isentropic relations
# ---------------------------------------------------------------------------
class TestIsentropicRelations:
    """Closed-form isentropic ratios -- no solver involved."""

    @pytest.mark.parametrize(
        "mach, p_rat, t_rat, rho_rat, a_rat", ISENTROPIC_TABLE
    )
    def test_matches_naca_table(
        self, mach, p_rat, t_rat, rho_rat, a_rat, gamma_air
    ):
        """Test that the isentropic functions match the NACA table values."""
        assert cf.P_ratio_from_mach(mach, gamma_air) == pytest.approx(
            p_rat, rel=RTOL_TABLE
        )
        assert cf.T_ratio_from_mach(mach, gamma_air) == pytest.approx(
            t_rat, rel=RTOL_TABLE
        )
        assert cf.rho_ratio_from_mach(mach, gamma_air) == pytest.approx(
            rho_rat, rel=RTOL_TABLE
        )
        assert cf.area_ratio_from_mach(mach, gamma_air) == pytest.approx(
            a_rat, rel=RTOL_TABLE
        )
    def test_non_air_gamma(self):
        """Test that the isentropic functions work for non-air gamma values."""
        gammas = [1.1, 1.2, 1.3, 1.667]
        p_rats =  [ 0.13458798, 0.13281030, 0.13046081, 0.12023303]
        rho_rats =[ 0.16150558, 0.18593443, 0.20873729, 0.28062390]
        t_rats =  [ 0.83333333, 0.71428571, 0.62500000, 0.42844901]
        a_rats =  [ 2.03182101, 1.88371168, 1.77318840, 1.53109954]
        mach = 2.0
        for gamma_idx, gamma in enumerate(gammas):
            p_rat = cf.P_ratio_from_mach(mach, gamma)
            t_rat = cf.T_ratio_from_mach(mach, gamma)
            rho_rat = cf.rho_ratio_from_mach(mach, gamma)
            a_rat = cf.area_ratio_from_mach(mach, gamma)
            assert p_rat == pytest.approx(p_rats[gamma_idx], rel=RTOL_TABLE)
            assert t_rat == pytest.approx(t_rats[gamma_idx], rel=RTOL_TABLE)
            assert rho_rat == pytest.approx(rho_rats[gamma_idx], rel=RTOL_TABLE)
            assert a_rat == pytest.approx(a_rats[gamma_idx], rel=RTOL_TABLE)

    def test_stagnation_limit(self, gamma_sweep):
        """At M = 0 every ratio is unity, for any gamma."""
        assert cf.P_ratio_from_mach(0.0, gamma_sweep) == pytest.approx(1.0)
        assert cf.T_ratio_from_mach(0.0, gamma_sweep) == pytest.approx(1.0)
        assert cf.rho_ratio_from_mach(0.0, gamma_sweep) == pytest.approx(1.0)

    def test_throat_area_ratio_is_unity(self, gamma_sweep):
        """A/A* == 1 at M = 1 by definition, independent of gamma."""
        assert cf.area_ratio_from_mach(1.0, gamma_sweep) == pytest.approx(1.0)

    def test_enthalpy_ratio_equals_temperature_ratio(
        self, mach_sweep, gamma_air
    ):
        """Calorically perfect gas: h/h0 == T/T0."""
        np.testing.assert_allclose(
            cf.h_ratio_from_mach(mach_sweep, gamma_air),
            cf.T_ratio_from_mach(mach_sweep, gamma_air),
        )

    def test_ideal_gas_consistency(self, mach_sweep, gamma_air):
        """p/p0 == (rho/rho0) * (T/T0) follows from p = rho * R * T."""
        p_rat = cf.P_ratio_from_mach(mach_sweep, gamma_air)
        rho_rat = cf.rho_ratio_from_mach(mach_sweep, gamma_air)
        t_rat = cf.T_ratio_from_mach(mach_sweep, gamma_air)
        np.testing.assert_allclose(p_rat, rho_rat * t_rat, rtol=1e-12)

    def test_ratios_decrease_monotonically(self, gamma_air):
        """All ratios are strictly decreasing in Mach number."""
        mach = np.linspace(0.0, 8.0, 200)
        for func in (cf.P_ratio_from_mach, cf.T_ratio_from_mach, cf.rho_ratio_from_mach):
            values = func(mach, gamma_air)
            assert np.all(np.diff(values) < 0.0), func.__name__

    def test_area_ratio_has_minimum_at_throat(self, gamma_air):
        """A/A* is convex with its minimum at M = 1."""
        mach = np.concatenate(
            [np.linspace(0.05, 0.99, 50), np.linspace(1.01, 6.0, 50)]
        )
        assert np.all(cf.area_ratio_from_mach(mach, gamma_air) > 1.0)

    def test_psi_matches_definition(self, mach_sweep, gamma_air):
        """Psi is defined as 1 + (gamma - 1)/2 * M^2."""
        expected = 1.0 + 0.5 * (gamma_air - 1.0) * mach_sweep**2
        np.testing.assert_allclose(
            cf.Psi(gamma=gamma_air, mach=mach_sweep), expected
        )

class TestMachFromStagnation:
    """ Test the mach_from_stagnation function."""
    def test_guard_against_invalid_gamma(self, gamma_sweep):
        """Test that the function returns NaN for invalid gamma values."""
        T0 = 300.0 #pylint: disable=C0103
        P0 = 101325.0 #pylint: disable=C0103
        R = 287.05 #pylint: disable=C0103
        area = 11.0
        mdot = 1.0
        invalid_gammas = np.asarray([1.0, 0.9, 0.5])
        mixed_gammas = np.concatenate((invalid_gammas, gamma_sweep))
        for gamma in invalid_gammas:
            result = cf.mach_from_stagnation(T0, P0, mdot, invalid_gammas[0], R, area)
            assert np.isnan(result), f"Expected NaN for gamma={gamma}, got {result}"
        if np.any(~np.isnan(cf.mach_from_stagnation(T0, P0, mdot, invalid_gammas, R, area))):
            raise ValueError("Expected NaN for all invalid gammas, got "
                             + f"{cf.mach_from_stagnation(T0, P0, mdot, invalid_gammas, R, area)}")
        for gamma in gamma_sweep:
            result = cf.mach_from_stagnation(T0, P0, mdot, gamma, R, area)
            assert not np.isnan(result), f"Expected valid result for gamma={gamma}, got NaN"
        if np.any(np.isnan(cf.mach_from_stagnation(T0, P0, mdot, gamma_sweep, R, area))):
            raise ValueError("Expected valid results for all valid gammas, got "
                             + f"{cf.mach_from_stagnation(T0, P0, mdot, gamma_sweep, R, area)}")
        result = cf.mach_from_stagnation(T0, P0, mdot, mixed_gammas, R, area)
        if np.any(~np.isnan(result[:len(invalid_gammas)-1])):
            raise ValueError("Expected NaN for invalid gammas, got "
                             + f"{result[:len(invalid_gammas)-1]}")
        if np.any(np.isnan(result[len(invalid_gammas):])):
            raise ValueError("Expected valid results for valid gammas, got "
                             + f"{result[len(invalid_gammas):]}")

    def test_calculation(self, gamma_sweep, mach_sweep):
        """Value Test:
        Test that the function returns the correct Mach number"""
        T0 = 300.0 #pylint: disable=C0103
        P0 = 101325.0 #pylint: disable=C0103
        R = 287.05 #pylint: disable=C0103
        A = 11.0 #pylint: disable=C0103

        rho_0 = P0 / (R * T0)
        for gamma in gamma_sweep:
            rhos = cf.rho_ratio_from_mach(mach_sweep, gamma) * rho_0
            Ts = T0 * cf.T_ratio_from_mach(mach_sweep, gamma) #pylint: disable=C0103
            mdots = rhos * np.sqrt(gamma * R * Ts) * mach_sweep * A
            mach_test = cf.mach_from_stagnation(T0, P0, mdots, gamma, R, A, supersonic=mach_sweep>1)
            rho_test = cf.rho_ratio_from_mach(mach_test, gamma) * rho_0
            Ts_test = T0 * cf.T_ratio_from_mach(mach_test, gamma) #pylint: disable=C0103
            c_test = np.sqrt(gamma * R * Ts_test)
            mdot_test = rho_test * c_test * mach_test * A

            np.testing.assert_allclose(
                mdot_test, mdots, rtol=1e-10, err_msg=f"gamma = {gamma}"
            )

    def test_mach_from_stagnation_rejects_non_boolean_supersonic(self, gamma_air):
        """Test that the function rejects non-boolean values for the supersonic parameter."""
        mach_test = 2.312
        T0 = 300.0 #pylint: disable=C0103
        P0 = 101325.0 #pylint: disable=C0103
        R = 287.05 #pylint: disable=C0103
        area = 11.0
        mdot = area * cf.m_flux_from_stagnation(T0, P0, gamma_air, R, mach_test)

        with pytest.raises(TypeError, match="Invalid supersonic type"):
            cf.mach_from_stagnation(
                T0, P0, mdot, gamma_air, R, area, supersonic="yes" #type: ignore
            ) # type: ignore

class TestMachFromPressureRatio:
    """Analytic inversion of the isentropic pressure relation."""

    def test_roundtrip(self, mach_sweep, gamma_air):
        """Test that mach_from_P_ratio inverts P_ratio_from_mach for supersonic Mach numbers."""
        p_rat = cf.P_ratio_from_mach(mach_sweep, gamma_air)
        np.testing.assert_allclose(
            cf.mach_from_P_ratio(p_rat, gamma_air), mach_sweep, rtol=1e-10
        )

    def test_unity_ratio_gives_zero_mach(self, gamma_air):
        """Test that mach_from_P_ratio returns 0 for a pressure ratio of 1."""
        assert cf.mach_from_P_ratio(1.0, gamma_air) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Area-ratio inversion (iterative)
# ---------------------------------------------------------------------------
class TestMachFromAreaRatio:
    """Newton / bisection inversion of A/A*.  This is the risky one."""

    @pytest.mark.parametrize("mach_expected", [1.5, 2.0, 3.0, 4.5])
    def test_supersonic_roundtrip_scalar(self, mach_expected, gamma_air):
        """Test the value of mach_from_area_ratio for a given area ratio matches 
        the expected Mach number from the isentropic relations.
        """
        a_rat = cf.area_ratio_from_mach(mach_expected, gamma_air)
        assert cf.mach_from_area_ratio(a_rat, gamma_air, supersonic=True) == pytest.approx(
            mach_expected, rel=RTOL_SOLVER
        )

    def test_roundtrip_vectorised(self, gamma_air, mach_sweep):
        """Test that mach_from_area_ratio inverts area_ratio_from_mach arrays too"""
        a_rat = cf.area_ratio_from_mach(mach_sweep, gamma_air)
        np.testing.assert_allclose(
            cf.mach_from_area_ratio(a_rat,
                                    gamma_air,
                                    supersonic = mach_sweep > 1),
            mach_sweep,
            rtol=RTOL_SOLVER
        )

    def test_broadcasts_gamma_against_area(self, gamma_sweep):
        """Check that the function can handle broadcasting of gamma against area ratio."""
        a_rat = np.full(gamma_sweep.shape, 4.0)
        result = cf.mach_from_area_ratio(a_rat,
                                         gamma_sweep,
                                         supersonic = np.full(gamma_sweep.shape, True))
        assert result.shape == gamma_sweep.shape
        np.testing.assert_allclose(
            cf.area_ratio_from_mach(result, gamma_sweep), a_rat, rtol=RTOL_SOLVER
        )

    def test_bracketed_subsonic_root(self, gamma_air):
        """With explicit bounds the subsonic branch resolves correctly.

        A/A* = 1.6875 has roots at M = 0.37224 and M = 2.0 for gamma = 1.4.
        """
        result = cf.mach_from_area_ratio(
            np.array([1.6875, 1.6875, 2.03506526]),
            gamma_air,
            1000,
            # 1e-10,
            mach_lims=np.array([[0.05, 1.], [1, 5.0], [0., 1]]),
        )
        np.testing.assert_allclose(
                    result,
                    np.array([0.372244, 2, 0.3]),
                    rtol=1e-5,
                )

    def test_rejects_bracket_not_spanning_root(self, gamma_air):
        """Implementation Test:
        Test that when the provided bracket does not span a root,
        the function raises a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid bracket"):
            cf.mach_from_area_ratio(
                1.6875, gamma_air, 1000, 1e-6, mach_lims=np.array([2.5, 5.0])
            )
        with pytest.raises(ValueError, match="Invalid bracket"):
            cf.mach_from_area_ratio(
                1.6875, gamma_air, 1000, 1e-6, mach_lims=np.array([0.001, 0.1])
            )

    def test_rejects_malformed_bracket_shape(self, gamma_air):
        """Implementation Test:
        Test that when the provided bracket has an invalid shape,
        the function raises a ValueError.
        """
        with pytest.raises(ValueError, match="mach_lims must be a tuple of"):
            cf.mach_from_area_ratio(
                1.6875, gamma_air, 1000, 1e-6, mach_lims=np.array([1.0, 2.0, 3.0])
            )

# ---------------------------------------------------------------------------
# Normal shocks
# ---------------------------------------------------------------------------
class TestNormalShock:
    """Test the normal shock relations"""

    @pytest.mark.parametrize(
        "mach_1, mach_2, p_rat, pt_rat", NORMAL_SHOCK_TABLE
    )
    def test_matches_naca_table(
        self, mach_1, mach_2, p_rat, pt_rat, gamma_air
    ):
        """Test that the normal shock functions match the NACA table values."""
        assert cf.normal_shock_mach_2(mach_1, gamma_air) == pytest.approx(
            mach_2, rel=RTOL_TABLE
        )
        assert cf.normal_shock_P_ratio(mach_1, gamma_air) == pytest.approx(
            p_rat, rel=RTOL_TABLE
        )
        assert cf.normal_shock_P_tot_ratio(mach_1, gamma_air) == pytest.approx(
            pt_rat, rel=RTOL_TABLE
        )

    def test_sonic_shock_is_identity(self, gamma_sweep):
        """A shock at M1 = 1 has vanishing strength for any gamma."""
        assert cf.normal_shock_mach_2(1.0, gamma_sweep) == pytest.approx(1.0)
        assert cf.normal_shock_P_ratio(1.0, gamma_sweep) == pytest.approx(1.0)
        assert cf.normal_shock_P_tot_ratio(1.0, gamma_sweep) == pytest.approx(1.0)

    def test_downstream_is_always_subsonic(self, gamma_sweep):
        """The downstream Mach number is always less than 1 valid Mach number."""
        mach_1 = np.linspace(1.01, 20.0, 100)[:, None]
        mach_2 = cf.normal_shock_mach_2(mach_1, gamma_sweep[None, :])
        assert np.all(mach_2 < 1.0)

    def test_total_pressure_loss_is_monotonic(self, gamma_air):
        """Second law: p02/p01 <= 1 and strictly decreasing in M1."""
        mach_1 = np.linspace(1.0, 10.0, 200)
        pt_rat = cf.normal_shock_P_tot_ratio(mach_1, gamma_air)
        assert np.all(pt_rat <= 1.0 + 1e-12)
        assert np.all(np.diff(pt_rat) < 0.0)

    def test_limiting_downstream_mach(self, gamma_air):
        """As M1 -> inf, M2 -> sqrt((gamma - 1) / (2 * gamma))."""
        expected = np.sqrt((gamma_air - 1.0) / (2.0 * gamma_air))
        assert cf.normal_shock_mach_2(1e6, gamma_air) == pytest.approx(
            expected, rel=1e-9
        )

    def test_normal_shock_matches_components(self, mach_supersonic, gamma_air):
        """The normal_shock function returns the same values as the individual components."""
        mach_2, pt_rat = cf.normal_shock(mach_supersonic, gamma_air)
        np.testing.assert_allclose(
            mach_2, cf.normal_shock_mach_2(mach_supersonic, gamma_air)
        )
        np.testing.assert_allclose(
            pt_rat, cf.normal_shock_P_tot_ratio(mach_supersonic, gamma_air)
        )


# ---------------------------------------------------------------------------
# Prandtl-Meyer / expansion fans
# ---------------------------------------------------------------------------
class TestPrandtlMeyer:
    """Test the Prandtl-Meyer expansion fan relations."""
    @pytest.mark.parametrize("mach, nu_deg", PRANDTL_MEYER_TABLE)
    def test_matches_naca_table(self, mach, nu_deg, gamma_air):
        """Value Test: Check that the Prandtl-Meyer function matches the NACA table values."""
        result_rad = cf.prandtl_meyer(mach, gamma_air)
        assert np.degrees(result_rad) == pytest.approx(nu_deg, abs=1e-4)

    def test_sonic_angle_is_zero(self, gamma_sweep):
        """Value Test: nu = 0 at M = 1 for any gamma."""
        assert cf.prandtl_meyer(1.0, gamma_sweep) == pytest.approx(0.0)

    def test_monotonic_in_mach(self, gamma_air):
        """Value Test: nu is strictly increasing in M for M > 1."""
        mach = np.linspace(1.0, 30.0, 300)
        nu_rad = cf.prandtl_meyer(mach, gamma_air)
        assert np.all(np.diff(nu_rad) > 0.0)

    def test_bounded_by_maximum(self, gamma_air):
        """Value Test: nu < nu_max for any M > 1."""
        mach = np.array([1.5, 5.0, 50.0, 500.0])
        nu_max_rad = cf.max_prandtl_meyer(gamma_air)
        assert np.all(cf.prandtl_meyer(mach, gamma_air) < nu_max_rad)

    def test_maximum_matches_closed_form(self, gamma_sweep):
        """Value Test: nu_max matches the closed-form expression."""
        expected_rad = (
            0.5
            * np.pi
            * (np.sqrt((gamma_sweep + 1.0) / (gamma_sweep - 1.0)) - 1.0)
        )
        np.testing.assert_allclose(
            cf.max_prandtl_meyer(gamma_sweep), expected_rad
        )

    def test_asymptotic_approach_to_maximum(self, gamma_air):
        """As M -> inf, nu -> nu_max."""
        nu_max_rad = cf.max_prandtl_meyer(gamma_air)
        assert cf.prandtl_meyer(1e5, gamma_air) == pytest.approx(
            nu_max_rad, rel=1e-4
        )

    def test_max_turning_angle_is_complement(self, gamma_air):
        """Value Test: The maximum turning angle for an expansion fan is nu_max - nu(M1)."""
        mach_1 = 2.0
        expected_rad = cf.max_prandtl_meyer(gamma_air) - cf.prandtl_meyer(
            mach_1, gamma_air
        )
        assert cf.expansion_fan_max_turning_angle(
            mach_1, gamma_air
        ) == pytest.approx(expected_rad)

    def test_subsonic_input_is_not_a_number(self, gamma_air):
        """Value Test: nu is undefined below M = 1; NaN is acceptable, a wrong float
        is not."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cf.prandtl_meyer(0.5, gamma_air)
        assert np.isnan(result)


class TestExpansionFan:
    """Test the expansion fan relations."""

    def test_zero_deflection_is_identity(self, gamma_air):
        """Value Test: A zero deflection angle returns the same Mach number."""
        result = cf.expansion_fan_mach_2(2.0, 0.0, gamma_air)
        assert result == pytest.approx(2.0, rel=1e-4)

    @pytest.mark.parametrize("delta_deg", [5.0, 10.0, 20.0])
    def test_inverts_prandtl_meyer(self, delta_deg, gamma_air):
        """Value Test: nu(M2) - nu(M1) must equal the turning angle."""
        mach_1 = 2.0
        delta_rad = np.radians(delta_deg)
        mach_2 = cf.expansion_fan_mach_2(mach_1, delta_rad, gamma_air)
        nu_1 = cf.prandtl_meyer(mach_1, gamma_air)
        nu_2 = cf.prandtl_meyer(mach_2, gamma_air)
        assert nu_2 - nu_1 == pytest.approx(delta_rad, abs=1e-5)

    def test_expansion_accelerates_flow(self, gamma_air):
        """Value Test: An expansion fan always accelerates the flow."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mach_2 = cf.expansion_fan_mach_2(2.0, np.radians(15.0), gamma_air)
        assert mach_2 > 2.0


# ---------------------------------------------------------------------------
# Oblique shocks
# ---------------------------------------------------------------------------
class TestObliqueShockAngles:
    """maxObliqueShockAngle is correct; the theta -> delta map is not."""

    @pytest.mark.parametrize(
        "mach_1, theta_max_deg, delta_max_deg", OBLIQUE_MAX_TABLE
    )
    def test_max_shock_angle_matches_reference(
        self, mach_1, theta_max_deg, _, gamma_air
    ):
        """Value Test:
        Test that the maximum shock angle for an oblique shock matches the reference values."""
        result_rad = cf.oblique_shock_max_angle(mach_1, gamma_air)
        assert np.degrees(result_rad) == pytest.approx(
            theta_max_deg, abs=1e-3
        )

    def test_max_shock_angle_bounded_by_normal_shock(self, gamma_air):
        """Value Test: The maximum shock angle is always less than 90 degrees."""
        mach_1 = np.linspace(1.01, 20.0, 100)
        theta_rad = cf.oblique_shock_max_angle(mach_1, gamma_air)
        assert np.all(theta_rad > 0.0)
        assert np.all(theta_rad < 0.5 * np.pi)

    def test_sonic_limit_is_mach_wave(self, gamma_air):
        """Value Test:
        At M1 = 1 the shock degenerates to a normal Mach wave."""
        assert cf.oblique_shock_max_angle(1.0, gamma_air) == pytest.approx(
            0.5 * np.pi, abs=1e-6
        )

    @pytest.mark.parametrize(
        "mach_1, theta_deg, delta_deg", OBLIQUE_WEAK_TABLE
    )
    def test_deflection_from_shock_angle(
        self, mach_1, theta_deg, delta_deg, gamma_air
    ):
        """Value Test:
        Test that the deflection angle for an oblique shock matches the reference values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_rad = cf.oblique_shock_delta_from_theta(
                mach_1, np.radians(theta_deg), gamma_air
            )
        assert np.degrees(result_rad) == pytest.approx(delta_deg, abs=1e-3)

    @pytest.mark.parametrize(
        "mach_1, theta_max_deg, delta_max_deg", OBLIQUE_MAX_TABLE
    )
    def test_max_deflection_matches_reference(
        self, mach_1, _, delta_max_deg, gamma_air
    ):
        """Value Test: Test that the maximum deflection angle for an 
        oblique shock matches the reference values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_rad = cf.oblique_shock_max_deflection(mach_1, gamma_air)
        assert np.degrees(result_rad) == pytest.approx(
            delta_max_deg, abs=1e-3
        )

    def test_subsonic_normal_component_returns_nan(self, gamma_air):
        """Implementation Test: theta below the Mach angle gives Mn1 < 1: physically invalid."""
        mach_angle_rad = np.arcsin(1.0 / 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cf.oblique_shock_delta_from_theta(
                2.0, 0.5 * mach_angle_rad, gamma_air
            )
        assert np.isnan(result)

    @pytest.mark.parametrize(
        "mach_1, theta_deg, delta_deg", OBLIQUE_WEAK_TABLE
    )
    def test_weak_solution_recovers_shock_angle(
        self, mach_1, theta_deg, delta_deg, gamma_air
    ):
        """Value Test: The weak solution should recover the expected shock angle."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, theta_rad = cf.oblique_shock(
                mach_1, np.radians(delta_deg), gamma_air
            )
        assert np.degrees(theta_rad) == pytest.approx(theta_deg, abs=1e-2)

    def test_strong_solution_exceeds_weak(self, gamma_air):
        """Both roots must be finite -- the NaN check is load-bearing.

        Without it a root finder fed a NaN residual returns its initial
        bracket midpoint, which satisfies the ordering by accident.
        """
        delta_rad = np.radians(10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, theta_weak = cf.oblique_shock(
                2.0, delta_rad, gamma_air, is_strong_shock=False
            )
            _, _, theta_strong = cf.oblique_shock(
                2.0, delta_rad, gamma_air, is_strong_shock=True
            )
        assert np.isfinite(theta_weak)
        assert np.isfinite(theta_strong)
        assert np.degrees(theta_weak) == pytest.approx(39.31393, abs=1e-2)
        assert np.degrees(theta_strong) == pytest.approx(83.70008, abs=1e-2)

    # @pytest.mark.xfail(strict=True, reason="See above.")
    def test_normal_shock_limit(self, gamma_air: float):
        """delta -> 0 with a strong shock recovers the normal shock."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mach_2, pt_rat, theta_rad = cf.oblique_shock(
                2.0, 1e-8, gamma_air, is_strong_shock=True
            )
        assert theta_rad == pytest.approx(0.5 * np.pi, abs=1e-4)
        assert mach_2 == pytest.approx(
            cf.normal_shock_mach_2(2.0, gamma_air), rel=1e-4
        )
        assert pt_rat == pytest.approx(
            cf.normal_shock_P_tot_ratio(2.0, gamma_air), rel=1e-4
        )


# ---------------------------------------------------------------------------
# Nozzle performance coefficients
# ---------------------------------------------------------------------------
class TestThrustCoefficients:
    """Test the thrust coefficient functions."""
    def test_cf(self, gamma_sweep):
        "Value Test: Check that the thrust coefficient matches the definition."
        T0 = 3000.0 #pylint: disable=C0103
        cp = 1011.0
        pressure_chamber_pa = 2.0e6
        pressure_back_pa = 101325.0

        for gamma in gamma_sweep:
            R = cp * (gamma - 1.0) / gamma #pylint: disable=C0103
            for area_ratio in [1.0, 2.0, 4.0, 10.0]:
                mach_exit = cf.mach_from_area_ratio(area_ratio, gamma, supersonic=True)
                vel_exit = np.sqrt(
                    2.0 * cp * T0
                    * (1. - cf.T_ratio_from_mach(mach_exit, gamma)))
                pressure_exit_pa = pressure_chamber_pa * cf.P_ratio_from_mach(mach_exit, gamma)
                mass_flux = cf.m_flux_from_stagnation(T0, pressure_chamber_pa, gamma, R, 1.0)

                c_f = (mass_flux
                       * vel_exit
                       + (pressure_exit_pa - pressure_back_pa) * area_ratio
                    ) / (pressure_chamber_pa)

                cf_value = cf.get_cf(
                    gamma,
                    area_ratio,
                    pressure_chamber_pa,
                    pressure_back_pa,
                )
                assert cf_value == pytest.approx(c_f, rel=1e-8)

    def test_cfv_exceeds_cf_at_positive_back_pressure(
        self, gamma_combustion: float
    ):
        """Value Test:
        The vacuum thrust coefficient exceeds the actual thrust coefficient
        when the nozzle is firing into an atmosphere."""
        area_ratio = 4.0
        pressure_chamber_pa = 2.0e6
        pressure_back_pa = 101325.0
        cfv = cf.get_cfv(gamma_combustion, area_ratio)
        cf_value = cf.get_cf(
            gamma_combustion,
            area_ratio,
            pressure_chamber_pa,
            pressure_back_pa,
        )
        assert cf_value < cfv

    def test_cf_equals_cfv_in_vacuum(self, gamma_combustion: float):
        """Value Test:
        the vacuum thrust coefficient equals the actual thrust coefficient
        when the nozzle is firing into a vacuum."""
        area_ratio = 4.0
        cfv = cf.get_cfv(gamma_combustion, area_ratio)
        cf_value = cf.get_cf(gamma_combustion, area_ratio, 2.0e6, 0.0)
        assert cf_value == pytest.approx(cfv)

    def test_cf_from_cfv_matches_get_cf(self, gamma_combustion: float):
        """Implementation Test: The get_cf_from_cfv function returns the same value as get_cf."""
        area_ratio = 4.0
        pressure_chamber_pa = 2.0e6
        pressure_back_pa = 101325.0
        cfv = cf.get_cfv(gamma_combustion, area_ratio)
        assert cf.get_cf_from_cfv(
            cfv, area_ratio, pressure_chamber_pa, pressure_back_pa
        ) == pytest.approx(
            cf.get_cf(
                gamma_combustion,
                area_ratio,
                pressure_chamber_pa,
                pressure_back_pa,
            )
        )

    def test_cfv_increases_with_area_ratio(self, gamma_combustion):
        """Value Test: The vacuum thrust coefficient increases with the area ratio."""
        area_ratio = np.array([2.0, 5.0, 10.0, 25.0, 50.0])
        cfv = cf.get_cfv(gamma_combustion, area_ratio)
        assert np.all(np.diff(cfv) > 0.0)

    def test_cfv_bounded_by_theoretical_maximum(self, gamma_combustion):
        """cfv < sqrt(2*g**2/(g-1) * (2/(g+1))**((g+1)/(g-1))) as Pe -> 0."""
        gamma = gamma_combustion
        cfv_max = np.sqrt(
            (2.0 * gamma**2 / (gamma - 1.0))
            * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        )
        cfv = cf.get_cfv(gamma, np.array([2.0, 10.0, 50.0]))
        assert np.all(cfv < cfv_max)

    def test_explicit_pressure_ratio_overrides_area_ratio(
        self, gamma_combustion
    ):
        """Implementation Test:
        The get_cfv function returns the same value when the pressure ratio is provided."""
        area_ratio = 4.0
        pressure_ratio_computed = cf.P_ratio_from_mach(
            cf.mach_from_area_ratio(area_ratio, gamma_combustion, supersonic=True), gamma_combustion
        )
        assert cf.get_cfv(
            gamma_combustion, area_ratio, pressure_ratio_computed
        ) == pytest.approx(cf.get_cfv(gamma_combustion, area_ratio))

    def test_cf_max_at_is_at_ambient(
            self, gamma_combustion
            ):
        """Value Test:
        At Pe = Pa, the nozzle is perfectly expanded and Cf is maximised."""

        def _objective(eps, pressure_chamber_pa, pressure_back_pa):
            return -cf.get_cf(gamma_combustion, eps, pressure_chamber_pa, pressure_back_pa)
        pressure_chamber_pa = 2.0e6
        pressure_back_pa = 101325.0

        res = scipy_optimize.minimize(
            _objective,
            (2.0e6 / 101325.0) ** 0.5,
            args=(pressure_chamber_pa, pressure_back_pa),
            method="Nelder-Mead",
        )

        area_ratio_optimal = res.x[0]
        pressure_exit_optimal = pressure_chamber_pa * cf.P_ratio_from_mach(
                            cf.mach_from_area_ratio(area_ratio_optimal,
                                                    gamma_combustion,
                                                    supersonic=True),
                            gamma_combustion
                            )
        assert pressure_exit_optimal == pytest.approx(pressure_back_pa, rel=1e-6)

# ---------------------------------------------------------------------------
# Mass flux
# ---------------------------------------------------------------------------
class TestMassFlux:
    """Test the mass flux function"""
    def test_choked_flux_matches_closed_form(self, gamma_combustion):
        """At M = 1, mdot/A = P0 * sqrt(g/(R*T0)) *
        (2/(g+1))**((g+1)/(2*(g-1)))."""
        gamma = gamma_combustion
        temperature_stagnation_k = 3000.0
        pressure_stagnation_pa = 2.0e6
        gas_constant_j_per_kg_k = 320.0

        expected_kg_per_m2_s = (
            pressure_stagnation_pa
            * np.sqrt(
                gamma
                / (gas_constant_j_per_kg_k * temperature_stagnation_k)
            )
            * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
        assert cf.m_flux_from_stagnation(
            temperature_stagnation_k,
            pressure_stagnation_pa,
            gamma,
            gas_constant_j_per_kg_k,
            1.0,
        ) == pytest.approx(expected_kg_per_m2_s, rel=1e-12)

    def test_flux_is_zero_at_rest(self, gamma_combustion):
        """At M = 0, mdot/A = 0."""
        assert cf.m_flux_from_stagnation(
            3000.0, 2.0e6, gamma_combustion, 320.0, 0.0
        ) == pytest.approx(0.0)

    def test_flux_peaks_at_sonic_condition(self, gamma_combustion):
        """The D-function is maximised at M = 1 -- this is choking."""
        mach = np.linspace(0.01, 4.0, 400)
        flux = cf.m_flux_from_stagnation(
            3000.0, 2.0e6, gamma_combustion, 320.0, mach
        )
        assert mach[np.argmax(flux)] == pytest.approx(1.0, abs=0.02)

    def test_flux_scales_linearly_with_stagnation_pressure(
        self, gamma_combustion
    ):
        """Value Test:
        The mass flux is proportional to the stagnation pressure."""
        base = cf.m_flux_from_stagnation(
            3000.0, 1.0e6, gamma_combustion, 320.0, 1.0
        )
        doubled = cf.m_flux_from_stagnation(
            3000.0, 2.0e6, gamma_combustion, 320.0, 1.0
        )
        assert doubled == pytest.approx(2.0 * base, rel=1e-12)

    def test_flux_scales_with_inverse_sqrt_stagnation_temperature(
        self, gamma_combustion
    ):
        """Value Test:
        The mass flux scales with the inverse square root of the stagnation temperature."""
        base = cf.m_flux_from_stagnation(
            1000.0, 1.0e6, gamma_combustion, 320.0, 1.0
        )
        hotter = cf.m_flux_from_stagnation(
            4000.0, 1.0e6, gamma_combustion, 320.0, 1.0
        )
        assert hotter == pytest.approx(0.5 * base, rel=1e-12)


class TestRecoveryTemperature:
    """Test the recovery temperature function."""

    def test_zero_mach_recovers_freestream(self):
        """At M = 0, Tr = T0."""
        temperature_freestream_k = 300.0
        assert cf.T_recovery(
            temperature_freestream_k, 0.71, 1.4, 0.0
        ) == pytest.approx(temperature_freestream_k)

    def test_unity_prandtl_gives_stagnation_temperature(self):
        """r = Pr**(1/3) = 1 when Pr = 1, so Tr collapses to T0."""
        temperature_freestream_k = 300.0
        gamma = 1.4
        mach = 3.0
        expected_k = temperature_freestream_k / cf.T_ratio_from_mach(mach, gamma)
        assert cf.T_recovery(
            temperature_freestream_k, 1.0, gamma, mach
        ) == pytest.approx(expected_k)

    def test_bounded_below_stagnation_for_air(self):
        """Pr < 1 for air, so Tr < T0."""
        temperature_freestream_k = 300.0
        gamma = 1.4
        mach = 4.0
        temperature_recovery_k = cf.T_recovery(
            temperature_freestream_k, 0.71, gamma, mach
        )
        temperature_stagnation_k = temperature_freestream_k / cf.T_ratio_from_mach(
            mach, gamma
        )
        assert (
            temperature_freestream_k
            < temperature_recovery_k
            < temperature_stagnation_k
        )


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------
SCALAR_IN_SCALAR_OUT = [
    ("P_ratio_from_mach", (2.0, 1.4), {}),
    ("T_ratio_from_mach", (2.0, 1.4), {}),
    ("h_ratio_from_mach", (2.0, 1.4), {}),
    ("rho_ratio_from_mach", (2.0, 1.4), {}),
    ("area_ratio_from_mach", (2.0, 1.4), {}),
    ("mach_from_P_ratio", (0.1278, 1.4), {}),
    ("normal_shock_P_ratio", (2.0, 1.4), {}),
    ("normal_shock_mach_2", (2.0, 1.4), {}),
    ("normal_shock_P_tot_ratio", (2.0, 1.4), {}),
    ("prandtl_meyer", (2.0, 1.4), {}),
    ("max_prandtl_meyer", (1.4,), {}),
    ("oblique_shock_max_angle", (2.0, 1.4), {}),
    ("mach_from_area_ratio", (1.6875, 1.4), {"supersonic": True}),
]

ARRAY_IN_ARRAY_OUT = [
    (name, args, kwargs)
    for name, args, kwargs in SCALAR_IN_SCALAR_OUT
]

class TestTypeContract:
    """The module hand-rolls float/ndarray dispatch per function.

    These are cheap and catch the copy-paste errors that dispatch style
    invites.
    """

    @pytest.mark.parametrize("name, args, kwargs", SCALAR_IN_SCALAR_OUT)
    def test_scalar_input_returns_python_float(self, name, args, kwargs):
        """Implementation Test: Check that the function returns a Python float for scalar inputs."""
        result = getattr(cf, name)(*args, **kwargs)
        assert isinstance(result, float), f"{name} returned {type(result)}"
        assert not isinstance(result, np.ndarray)

    @pytest.mark.parametrize("name, args, kwargs", ARRAY_IN_ARRAY_OUT)
    def test_array_input_returns_ndarray(self, name, args, kwargs):
        """Implementation Test: Check that the function returns a NumPy array for array inputs."""
        array_args = tuple(np.atleast_1d(np.asarray(a)) for a in args)
        result = getattr(cf, name)(*array_args, **kwargs)
        assert isinstance(result, np.ndarray), f"{name}: {type(result)}"

    def test_mach_from_area_ratio_preserves_array_shape_when_multi_element(self):
        """Implementation Test: Multi-element input does keep its shape -- only size 1 breaks."""
        result = cf.mach_from_area_ratio(np.array([1.6875, 4.0]),
                                         1.4,
                                         supersonic=np.array([True, True]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_mach_from_area_ratio_preserves_array_shape_when_size_one(self):
        """Implementation Test: Size-1 input does keep its shape."""
        result = cf.mach_from_area_ratio(np.array([1.6875]), 1.4, supersonic=np.array([True]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    @pytest.mark.parametrize(
        "name",
        [
            "P_ratio_from_mach",
            "T_ratio_from_mach",
            "h_ratio_from_mach",
            "rho_ratio_from_mach",
            "area_ratio_from_mach",
            "normal_shock_mach_2",
            "normal_shock_P_tot_ratio",
        ],
    )
    def test_broadcasting_shape(self, name):
        """Implementation Test: 
        Check that the function can handle broadcasting of Mach number against gamma."""
        mach = np.linspace(1.5, 4.0, 7)[:, None]
        gamma = np.array([1.2, 1.3, 1.4])[None, :]
        result = getattr(cf, name)(mach, gamma)
        assert result.shape == (7, 3)

    @pytest.mark.parametrize(
        "name", 
        ["P_ratio_from_mach", "T_ratio_from_mach", "h_ratio_from_mach", "normal_shock_mach_2"]
    )
    def test_scalar_and_array_agree(self, name, gamma_air):
        """Implementation Test: 
        Check that the function returns the same values for scalar and array inputs."""
        func = getattr(cf, name)
        mach = np.array([1.5, 2.0, 3.0])
        np.testing.assert_allclose(
            func(mach, gamma_air),
            [func(float(m), gamma_air) for m in mach],
            rtol=1e-14,
        )

    def test_normal_shock_returns_two_element_tuple(self, gamma_air):
        """Implementation Test:
        Check that the normal_shock function returns a two-element tuple."""
        result = cf.normal_shock(2.0, gamma_air)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_oblique_shock_returns_three_element_tuple(self, gamma_air):
        """Implementation Test:
        Check that the oblique_shock function returns a three-element tuple."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cf.oblique_shock(2.0, float(np.radians(10.0)), gamma_air)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
        assert all(np.isfinite(v) for v in result)
