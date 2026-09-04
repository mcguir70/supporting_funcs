"""# compressible_flow
Equations and functions for compressible flow calculations. 
Includes relationships for:
- isentropic flow
- normal shock
- oblique shock
- expansion fan"""
from typing import Optional
from warnings import warn

from numpy import (
    ndarray,
    pi as PI,
    arctan as np_atan,
    tan as np_tan,
    sin as np_sin,
    cos as np_cos,
    arcsin as np_asin,
)
import numpy as np
from scipy.optimize import elementwise

from supporting_funcs.coding.type_hints import BoolArray

#region Isentropic Flow Relations
def Psi(gamma: float | ndarray, mach: float | ndarray) -> float | ndarray: #pylint: disable=C0103
    """Calculate the Compressibility function Psi for a given specific heat ratio and Mach number.

    Parameters
    ----------
        gamma: [1]
            Specific heat ratio (Cp/Cv)
        mach: [1]
            Mach number
    Returns
    -------
        Psi: [1] 
            Compressibility function value
    """
    return 1 + (gamma - 1) / 2 * mach**2

def P_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray: #pylint: disable=C0103
    """
    Calculate the isentropic pressure ratio from Mach number and specific heat ratio.
    
    Parameters
    ----------
        mach: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)
    
    Returns
    -------
        p/p0: [1]
            Isentropic pressure ratio
    """
    return Psi(gamma=gamma, mach=mach) ** (-gamma / (gamma - 1))

def mach_from_P_ratio(p_rat: float | ndarray, gamma: float | ndarray) -> float | ndarray: #pylint: disable=C0103
    """
    Calculate the Mach number from pressure ratio and specific heat ratio.

    Parameters
    ----------
    p_rat: float | ndarray
        [1] (p/p0) Isentropic pressure ratio 

    gamma: float | ndarray
        [1] (Cp/Cv) Specific heat ratio 
    
    Returns
    -------
    mach: [1]
        Mach number
    """
    return ((2 / (gamma - 1)) * (p_rat ** ((1 - gamma) / (gamma)) - 1))**0.5

def T_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray: #pylint: disable=C0103
    """
    Calculate the temperature ratio from Mach number and specific heat ratio.
    
    Parameters
    ----------
        mach: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)
    
    Returns
    -------
        T/T0: [1]
            Temperature ratio
    """
    return Psi(gamma=gamma, mach=mach) ** (-1)

def h_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the enthalpy ratio from Mach number and specific heat ratio.

    Parameters
    ----------
        mach: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)
    Returns
    -------
        h/h0: [1]
            Enthalpy ratio
    """
    return T_ratio_from_mach(mach, gamma)

def rho_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the isentropic density ratio from Mach number and specific heat ratio.
    
    Parameters
    ----------
        mach: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)
    
    Returns
    -------
        rho/rho0: [1]
            Density ratio
    """
    return T_ratio_from_mach(mach, gamma) ** (1 / (gamma - 1))

def area_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the area ratio from Mach number and specific heat ratio.
    
    Parameters
    ----------
        mach: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)
    
    Returns
    -------
        A/A*: [1]
            Area ratio
    """
    return (1/mach) * (
        (2/(gamma + 1)) * (1 + 0.5*(gamma - 1)*mach**2)) ** ((gamma + 1)/(2 * (gamma - 1)))

def mach_from_stagnation(T0: float | ndarray, #pylint: disable=C0103
                         P0: float | ndarray, #pylint: disable=C0103
                         mdot: float | ndarray,
                         gamma: float | ndarray,
                         R: float | ndarray, #pylint: disable=C0103
                         area: float | ndarray,
                         supersonic: bool | ndarray = False) -> float | ndarray:
    """
    Calculate the Mach number from stagnation conditions, 
    mass flow rate, specific heat ratio, and gas constant.
    
    Parameters
    ----------
        T0: [K] 
            Stagnation temperature
        P0: [Pa] 
            Stagnation pressure
        mdot: [kg/s] 
            Mass flow rate
        gamma: [1]
            Specific heat ratio (Cp/Cv)
        R: [J/(kg*K)]
            Specific Gas constant
        area: [m^2]
            Flow area
        supersonic: [bool]
            Whether to calculate the super- or sub-sonic Mach number
    Returns
    -------
        mach: [1]
            Mach number
    """
    shape = ()
    args = (T0, P0, mdot, gamma, R, area)
    for arg in args:
        if isinstance(arg, ndarray):
            if shape == (): # pylint: disable=C1803
                shape = arg.shape
            elif shape != arg.shape:
                raise ValueError(
                    "T0, P0, mdot, gamma, R, and A must have the same shape or be scalars.")
        elif isinstance(arg, (float, int)):
            continue
        else:
            raise TypeError(f"Argument {arg} must be a float, int, or numpy ndarray.")
    T0 = np.broadcast_to(T0, shape)
    P0 = np.broadcast_to(P0, shape)
    mdot = np.broadcast_to(mdot, shape)
    gamma = np.broadcast_to(gamma, shape)
    R = np.broadcast_to(R, shape)
    area = np.broadcast_to(area, shape)

    if isinstance(supersonic, bool | np.bool_ | ndarray):
        supersonic = np.broadcast_to(supersonic, shape=shape)
    else:
        raise TypeError(f"Invalid supersonic type: {type(supersonic)}, expected bool or bool array")

    def _rhs(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray:
        return (mach
                * P_ratio_from_mach(mach=mach, gamma=gamma)
                * T_ratio_from_mach(mach=mach, gamma=gamma) ** -0.5)

    def _lhs(T0: ndarray, #pylint: disable=C0103
             P0: ndarray, #pylint: disable=C0103
             mdot: ndarray,
             gamma: ndarray,
             R: ndarray, #pylint: disable=C0103
             area: ndarray
             ) -> ndarray:
        return mdot / (P0 * area) * (R * T0 / gamma) ** 0.5

    def _error_func(mach: ndarray, gamma: ndarray, lhs_val: ndarray) -> ndarray:
        return lhs_val - _rhs(mach=mach, gamma=gamma)

    lims = np.empty(shape=[*shape, 2])

    lims[ supersonic, :] = [1.00001, 10.0]
    lims[~supersonic, :] = [0., 0.99999]

    lhs_flat = _lhs(T0=T0, P0=P0, mdot=mdot, gamma=gamma, R=R, area=area)
    gamma_flat = gamma
    valid_mask = np.ones_like(lhs_flat, dtype=bool)
    valid_mask[gamma_flat <= 1] = False
    if np.all(~valid_mask):
        if shape == ():
            return float(np.nan)
        return np.full(shape, np.nan, dtype=float)
    max_rhs = _rhs(mach=1, gamma=gamma_flat[valid_mask])
    valid_mask[valid_mask] = lhs_flat[valid_mask] < max_rhs

    mach = np.full(shape, float('nan'), dtype=float)
    res = elementwise.find_root(_error_func,
                                    (lims[valid_mask, 0], lims[valid_mask, 1]),
                                    args=(gamma_flat[valid_mask],
                                            lhs_flat[valid_mask]),
                                    tolerances={'xatol': 1e-10, 'frtol': 1e-10},)

    if shape == (): #pylint: disable=C1803
        if not (len(res.x.shape) == 1 and res.x.shape[0] == 1):
            raise ValueError("Expected a single root for scalar input.")
        if res.success.item():
            return float(res.x.item())
        else:
            return float(np.nan)

    valid_idx = np.flatnonzero(valid_mask)
    mach[valid_idx[res.success]] = res.x[res.success]

    return np.asarray(mach).reshape(shape)

def m_flux_from_stagnation(T0: float | ndarray, #pylint: disable=C0103
                           P0: float | ndarray, #pylint: disable=C0103
                           gamma: float | ndarray,
                           R: float | ndarray, #pylint: disable=C0103
                           mach: float | ndarray
                           ) -> float | ndarray:
    """
    Calculate the mass flux (mdot/A) from stagnation conditions, 
    specific heat ratio, gas constant, and Mach number.
    
    Parameters
    ----------
        T0: [K]
            Stagnation temperature
        P0: [Pa]
            Stagnation pressure
        gamma: [1]
            Specific heat ratio (Cp/Cv)
        R: [J/(kg*K)]
            specific Gas constant
        mach: [1]
            Mach number

    Returns
    -------
        mdot/A: [kg/(m^2*s)] 
            Mass flux (mdot/A)
    """
    mass_flux = P0 * (gamma / (R * T0)) ** 0.5 * _D_func(mach=mach, gamma=gamma)
    return mass_flux

def _D_func(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray: #pylint: disable=C0103
    """
    Helper function for mass flow rate calculation, 
    representing the Mach number dependent part of the mass flow rate equation.
    
    Parameters
    ----------
        mach: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        val: [1]
            Value of the D function for mass flow rate calculation
    """
    return mach / (1 + 0.5 * (gamma - 1) * mach**2) ** ((gamma + 1) / (2 * (gamma - 1)))

def mach_from_area_ratio(area_rat: float | ndarray,
                         gamma: float | ndarray,
                         max_iter: int = 1000,
                         tol: float = 1e-6,
                         *,
                         mach_lims: Optional[ndarray | tuple[float, float]] = None,
                         supersonic: Optional[bool | BoolArray] = None,
                         ) -> float | ndarray:
    """
    Calculate the Mach number from area ratio and specific heat ratio.
    Supports scalar or numpy array inputs for A and gamma.

    Parameters
    ----------
        area_rat: [1]
            Area ratio (A/A*)
        gamma: [1]
            Specific heat ratio (Cp/Cv)
        mach_lims: [1]
            Limits for the Mach ratio to constrain the search space (min, max)
        max_iter (int): [-]
            Maximum number of iterations for convergence
        tol (float): [1]
            Tolerance for convergence

    Returns
    -------
        Mach: [1]
            Calculated Mach number(s)
    """
    if isinstance(area_rat, ndarray) or isinstance(gamma, ndarray):
        return_float = False
    else:
        return_float = True

    # Vectorized numpy version
    area_rat = np.asarray(area_rat, dtype=float)
    if np.any(area_rat < 1):
        raise ValueError(f"Area ratio must be greater than 1., got {area_rat}.")
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma <= 1):
        raise ValueError(f"Specific heat ratio gamma must be greater than 1., got {gamma}.")

    shape = np.broadcast_shapes(area_rat.shape, gamma.shape, (1,))

    area_rat = np.broadcast_to(area_rat, shape)
    gamma = np.broadcast_to(gamma, shape)

    if supersonic is None and mach_lims is None:
        raise ValueError("Either supersonic or mach_lims must be specified.")

    if mach_lims is not None:
        mach_lims = np.asarray(mach_lims, dtype=float).copy()
        if mach_lims.shape[-1] != 2:
            raise ValueError(
                "mach_lims must be a tuple of (min, max) or an array of shape (..., 2).")
        mach_lims[mach_lims == 0] = np.finfo(float).eps #pylint: disable=E1101
        try:
            mach_lims = np.broadcast_to(mach_lims, shape + (2,))
        except ValueError as e:
            raise ValueError("Failed to broadcast mach_lims to the shape of area_rat.") from e

        if np.any(mach_lims[..., 0] >= mach_lims[..., 1]):
            raise ValueError("mach_lims must be such that min < max.")
        if np.any(mach_lims <= 0):
            raise ValueError("mach_lims must be positive.")
        # raise NotImplementedError("Here") # TODO
        if np.any(((mach_lims[..., 0] < 1) == (mach_lims[..., 1] > 1))):
            raise ValueError(
                "mach_lims must be such that " \
                "the first and second columns are (both or neither) supersonic.")
        if supersonic is not None:
            supersonic = np.broadcast_to(np.asarray(supersonic, dtype=bool), shape)
            if np.any((supersonic & (mach_lims[..., 1] <= 1))
                      | (~supersonic & (mach_lims[..., 0] >= 1))):
                raise ValueError(
                    "Conflicting specification: cannot specify both supersonic and mach_lims. " \
                    "Use mach_lims to specify the Mach number range, "
                    "or set supersonic to True or False to select the super or sub-sonic branch.")
    else: # mach_lims is None, supersonic is not None
        supersonic = np.broadcast_to(np.asarray(supersonic, dtype=bool), shape)
        mach_lims = np.ones(shape + (2,), dtype=float)
        mach_lims[~supersonic, 0] = 1e-6
        mach_lims[ supersonic, 1] = 10.

    def _error_func(mach: float | ndarray,
                    gamma: float | ndarray,
                    area: float | ndarray
                    ) -> float | ndarray:
        return area_ratio_from_mach(mach, gamma) - area

    res = elementwise.find_root(_error_func,
                                (mach_lims[..., 0], mach_lims[..., 1]),
                                args = (gamma, area_rat),
                                tolerances={'xrtol': tol, 'frtol': tol},
                                maxiter=max_iter)

    if not np.all(res.success):
        invalid_idx = np.flatnonzero(res.status == -1)
        failed_idx = np.flatnonzero(res.status == -2)
        infinite_idx = np.flatnonzero(res.status == -3)
        raise ValueError(
            "Failed to converge to a solution for some elements: "
            + (f'\n  Invalid bracket indices: {invalid_idx}' if invalid_idx.size > 0 else '')
            + (f'\n  Failed after {max_iter} iterations indices: {failed_idx}'
               if failed_idx.size > 0 else '')
            + (f'\n  Infinite values indices: {infinite_idx}' if infinite_idx.size > 0 else ''))

    if return_float:
        return res.x.item()
    return res.x

def T_recovery(temp_inf: float, #pylint: disable=C0103
               prandtl: float,
               gamma: float,
               mach: float,
               flow_type: str = 'turbulent') -> float:
    """
    Calculate the recovery temperature for a given 
    free stream temperature, Prandtl number, specific heat ratio, Mach number, and flow type.

    Parameters
    ----------
        temp_inf: [K]
            Free stream temperature
        prandtl: [1]
            Prandtl number
        gamma: [1]
            Specific heat ratio (Cp/Cv)
        mach: [1]
            Mach number
        flow_type: [-]
            Type of flow ('turbulent' or 'laminar')

    Returns
    -------
        T_recovery: [K]
            Recovery temperature
    """
    if flow_type != 'turbulent':
        raise NotImplementedError(
            f"Only 'turbulent' flow recovery temperature is implemented. (got {flow_type})")
    r_recovery = prandtl**(1/3) # Recovery factor for Turbulent flow
    return temp_inf * (1 + r_recovery * (gamma - 1) / 2 * mach**2)

#region Expansion Fan Relations
def prandtl_meyer(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the Prandtl-Meyer function for a given Mach number and specific heat ratio.
    
    Parameters
    ----------
        mach: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        nu: [rad]
            Prandtl-Meyer angle
    """
    return_float = True
    if not isinstance(mach, ndarray):
        mach = np.asarray(mach, dtype=float)
    else:
        return_float = False
    if not isinstance(gamma, ndarray):
        gamma = np.asarray(gamma, dtype=float)
    else:
        return_float = False
    result = (((gamma + 1) / (gamma - 1)) ** 0.5
              * np_atan(((gamma - 1) * (mach ** 2 - 1) / (gamma + 1)) ** 0.5)
              - np_atan((mach**2 - 1) ** 0.5))
    if return_float:
        return float(np.asarray(result).item())
    return result

def max_prandtl_meyer(gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the maximum Prandtl-Meyer angle for a given specific heat ratio.
    
    Parameters
    ----------
        gamma: [1]
            Specific heat ratio (Cp/Cv)
    
    Returns
    -------
        nu_{max}: [rad]
            Maximum Prandtl-Meyer angle in radians
    """
    return_float = True
    if not isinstance(gamma, ndarray):
        gamma = np.asarray(gamma, dtype=float)
    else:
        return_float = False
    result = PI/2 * (((gamma + 1) / (gamma - 1)) ** 0.5 - 1)
    if return_float:
        return float(np.asarray(result).item())
    return result

def expansion_fan_mach_2(mach_1: float | ndarray,
                         delta: float | ndarray,
                         gamma: float | ndarray
                         ) -> float | ndarray:
    """
    Calculate the downstream Mach number after an expansion fan for a given 
    upstream Mach number, turning angle, and specific heat ratio.
    
    Parameters
    ----------
        mach_1: [1]
            Upstream Mach number
        delta: [rad]
            Turning angle
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        mach_2: [1]
            Downstream Mach number
    """
    # Inverse Prandtl-Meyer to find mach_2
    def _error_func(mach_2: ndarray, nu_target: ndarray, gamma: ndarray) -> ndarray:
        """Equation to solve for mach_2 given target Prandtl-Meyer angle."""
        return prandtl_meyer(mach_2, gamma) - nu_target

    return_float = True
    if not isinstance(mach_1, ndarray):
        mach_1 = np.asarray(mach_1, dtype=float)
    else:
        return_float = False
    if not isinstance(delta, ndarray):
        delta = np.asarray(delta, dtype=float)
    else:
        return_float = False
    if not isinstance(gamma, ndarray):
        gamma = np.asarray(gamma, dtype=float)
    else:
        return_float = False

    nu1 = prandtl_meyer(mach_1, gamma)
    nu2 = nu1 + delta

    nu2 = prandtl_meyer(mach_1, gamma) + delta
    res = elementwise.find_root(_error_func,
                                (np.full_like(mach_1, 1.0001), np.full_like(mach_1, 50.0)),
                                args=(nu2, gamma),
                                tolerances={'xatol': 1e-6, 'xrtol': 1e-6, 'frtol': 1e-6},
                                )
    if not np.all(res.success):
        invalid_idx = np.flatnonzero(res.status == -1)
        failed_idx = np.flatnonzero(res.status == -2)
        infinite_idx = np.flatnonzero(res.status == -3)
        raise ValueError("Failed to converge to a solution for some elements:"
                    + (f'\n  Invalid bracket indices: {invalid_idx}'
                       if invalid_idx.size > 0 else '')
                    + (f'\n  Failed after 100 iterations indices: {failed_idx}'
                        if failed_idx.size > 0 else '')
                    + (f'\n  Infinite values indices: {infinite_idx}'
                       if infinite_idx.size > 0 else ''))

    if return_float:
        return res.x.item()
    return res.x
def expansion_fan_max_turning_angle(mach_1: float | ndarray,
                                    gamma: float | ndarray
                                    ) -> float | ndarray:
    """
    Calculate the maximum turning angle for a given Mach number and specific heat ratio.
    
    Parameters
    ----------
        mach_1: [1]
            Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)
    
    Returns
    -------
        delta_max: [rad]
            Maximum turning angle
    """
    # Type conversions handled in called functions
    return max_prandtl_meyer(gamma) - prandtl_meyer(mach_1, gamma)
#endregion Expansion Fan Relations
#endregion Isentropic Flow Relations
#region Normal Shock Equations
#  https://www.grc.nasa.gov/www/k-12/airplane/normal.html

def normal_shock_P_ratio(mach_1: float | ndarray, gamma: float | ndarray) -> float | ndarray: #pylint: disable=C0103
    """
    Calculate the pressure ratio across a normal shock.

    Parameters
    ----------
        mach_1: [1]
            Upstream Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        p2/p1: [1]
            Pressure ratio (p2/p1) across the shock
    """
    return (2 * gamma * mach_1**2 - (gamma - 1)) / (gamma + 1)

def normal_shock_mach_2(mach_1: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the downstream Mach number across a normal shock.

    Parameters
    ----------
        mach_1: [1]
            Upstream Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        mach_2: [1]
            Downstream Mach number across the shock
    """
    return (((gamma-1) * mach_1**2 +2) / (2 * gamma * mach_1**2 - (gamma - 1))) ** 0.5

def normal_shock_P_tot_ratio( #pylint: disable=C0103
        mach_1: float | ndarray,
        gamma: float | ndarray
        ) -> float | ndarray:
    """
    Calculate the total pressure ratio across a normal shock.

    Parameters
    ----------
        mach_1: [1] 
            Upstream Mach number
        gamma: [1] 
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        p02/p01: [1]
        Total pressure ratio across the shock
    """
    p2_p1 = normal_shock_P_ratio(mach_1, gamma)
    mach_2 = normal_shock_mach_2(mach_1, gamma)
    p2_p02 = P_ratio_from_mach(mach_2, gamma)
    p1_p01 = P_ratio_from_mach(mach_1, gamma)
    p02_p01 = p2_p1 * (p1_p01 / p2_p02)
    return p02_p01

def normal_shock(mach_1: float | ndarray,
                 gamma: float | ndarray
                 ) -> tuple[float | ndarray, float | ndarray]:
    """
    Calculate the downstream Mach number, pressure ratio, and total pressure ratio 
    for a normal shock.

    Parameters
    ----------
        mach_1: [1] 
            Upstream Mach number
        gamma: [1] 
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        mach_2: [1] 
            Downstream Mach number
        p02/p01: [1] 
            Total pressure ratio across the shock
    """
    mach_2 = normal_shock_mach_2(mach_1, gamma)
    p02_p01 = normal_shock_P_tot_ratio(mach_1, gamma)
    return mach_2, p02_p01
#endregion Normal Shock Equations
#region Oblique Shock Equations
def oblique_shock_delta_from_theta(mach_1: float | ndarray,
                                   theta: float | ndarray,
                                   gamma: float | ndarray
                                   ) -> float | ndarray:
    """
    Calculate the deflection angle for an oblique shock given the shock angle.

    Parameters
    ----------
        mach_1: [1] 
            Upstream Mach number
        theta: [rad]
            Shock angle
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        delta: [rad]
            Flow deflection angle
    """
    return_float = not any(isinstance(x, ndarray) for x in [mach_1, theta, gamma])

    mach_1 = np.asarray(mach_1, dtype=float)
    theta = np.asarray(theta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    shape = np.broadcast_shapes(mach_1.shape, theta.shape, gamma.shape, (1,))
    mach_1 = np.broadcast_to(mach_1, shape)
    theta = np.broadcast_to(theta, shape)
    gamma = np.broadcast_to(gamma, shape)

    mach_normal_1 = mach_1 * np_sin(theta)
    mach_normal_1_is_sup = mach_normal_1 >= 1.0 # Only valid for supersonic normal Mach number

    delta = np.full_like(mach_normal_1, float('nan'))
    if np.any(~mach_normal_1_is_sup):
        warn("obliqueShockDeltaFromTheta is only valid "
        "for supersonic upstream normal Mach numbers (Mn1 > 1). "
        "Results may be invalid for some inputs.")

    delta[mach_normal_1_is_sup] = np_atan(2 * (1 / np_tan(theta[mach_normal_1_is_sup]))
                                          * (mach_normal_1[mach_normal_1_is_sup]**2 - 1)
                                          / (mach_1[mach_normal_1_is_sup]**2
                                             * (gamma[mach_normal_1_is_sup]
                                                + np_cos(2 * theta[mach_normal_1_is_sup]))
                                             + 2))

    if return_float:
        return float(np.asarray(delta).item())
    return delta

def oblique_shock(mach_1: float | ndarray,
                  delta: float | ndarray,
                  gamma: float | ndarray,
                  is_strong_shock: bool | ndarray = False,
                  x0: None | tuple[float | ndarray, float | ndarray] | ndarray = None
                  ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
    """
    Calculate the downstream Mach number, pressure ratio, and shock angle for an oblique shock.
    https://kyleniemeyer.github.io/gas-dynamics-notes/compressible-flows/oblique-shocks.html

    Parameters
    ----------
        mach_1: [1] 
            Upstream Mach number
        delta: [rad]
            Deflection angle
        gamma: [1]
           Specific heat ratio (Cp/Cv)
        x0: [rad]
            Bracket for root finding, tuple of (theta_min, theta_max) or array of shape (..., 2)
        is_strong_shock: [bool]
            If True, find the strong shock solution. If False, find the weak shock solution.

    Returns
    -------
        mach_2: [1] 
            Downstream Mach number
        p02_p01: [1] 
            Total pressure ratio
        theta: [rad]
            Shock angle
    """
    def _error_func(theta: ndarray, delta: ndarray, mach_1: ndarray, gamma: ndarray) -> ndarray:
        """Equation to solve for shock angle given deflection angle, 
            A positive delta means the shock angle guess is too small."""
        lhs = delta
        rhs = oblique_shock_delta_from_theta(mach_1=mach_1, theta=theta, gamma=gamma)
        return lhs - rhs

    return_float = not any(isinstance(x, ndarray) for x in [mach_1, delta, gamma, is_strong_shock])

    mach_1 = np.asarray(mach_1, dtype=float)
    delta = np.asarray(delta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    is_strong_shock = np.asarray(is_strong_shock, dtype=bool)

    shape = np.broadcast_shapes(mach_1.shape, delta.shape, gamma.shape, is_strong_shock.shape, (1,))
    mach_1 = np.broadcast_to(mach_1, shape)
    delta = np.broadcast_to(delta, shape)
    gamma = np.broadcast_to(gamma, shape)
    is_strong_shock = np.broadcast_to(is_strong_shock, shape)

    if x0 is not None:
        x0 = np.broadcast_to(np.asarray(x0, dtype=float), shape + (2,))
    if x0 is None:
        x0 = np.empty(shape=shape + (2,), dtype=float)
        theta_max = oblique_shock_max_angle(mach_1, gamma)
        assert isinstance(theta_max, ndarray), "thetaMax should be an ndarray after broadcasting."

        x0[is_strong_shock, 0] = theta_max[is_strong_shock]
        x0[is_strong_shock, 1] = PI / 2
        x0[~is_strong_shock, 0] = np_asin(1 / mach_1[~is_strong_shock])
        x0[~is_strong_shock, 1] = theta_max[~is_strong_shock]

    res = elementwise.find_root(_error_func,
                                (np.asarray(x0[..., 0], dtype=float),
                                 np.asarray(x0[..., 1], dtype=float)),
                                 args=(delta, mach_1, gamma),
                                 tolerances={'xatol': 1e-6, 'xrtol': 1e-6, 'frtol': 1e-6},
                                 maxiter=100)
    if not np.all(res.success):
        invalid_idx = np.flatnonzero(res.status == -1)
        failed_idx = np.flatnonzero(res.status == -2)
        infinite_idx = np.flatnonzero(res.status == -3)
        raise ValueError("Failed to converge to a solution for some elements:"
                         + (f'\n  Invalid bracket indices: {invalid_idx}'
                            if invalid_idx.size > 0 else '')
                         + (f'\n  Failed after 100 iterations indices: {failed_idx}'
                            if failed_idx.size > 0 else '')
                         + (f'\n  Infinite values indices: {infinite_idx}'
                            if infinite_idx.size > 0 else ''))
    theta_sol = res.x

    mach_1_normal = mach_1 * np_sin(theta_sol)
    p02_p01 = normal_shock_P_tot_ratio(mach_1_normal, gamma)
    mach_2_normal = normal_shock_mach_2(mach_1_normal, gamma)
    mach_2 = mach_2_normal / np_sin(theta_sol - delta)

    return_val = [mach_2, p02_p01, theta_sol]

    if return_float:
        for i, val in enumerate(return_val):
            if isinstance(val, ndarray):
                return_val[i] = float(np.asarray(val).item())

    return tuple(return_val)

def oblique_shock_max_angle(mach_1: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the oblique shock angle that results in the maximum deflection angle
    https://kyleniemeyer.github.io/gas-dynamics-notes/compressible-flows/oblique-shocks.html#equation-eq-delta-max
    
    Parameters
    ----------
        mach_1: [1]
            Upstream Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        max_theta: [rad]
            Maximum oblique shock angle (rad)
    """
    return_float = not any(isinstance(x, ndarray) for x in [mach_1, gamma])
    mach_1 = np.asarray(mach_1, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    # Implicit broadcasting works here
    sin_2_theta_max = ((1 / (gamma * mach_1**2))
                       * ((gamma + 1) * mach_1**2 / 4
                          - 1
                          + ((gamma + 1)
                             * ((gamma + 1)* mach_1**4 / 16 + (gamma - 1) * mach_1**2 / 2 + 1)
                            ) ** 0.5))
    theta_max = np_asin(sin_2_theta_max ** 0.5)
    if return_float:
        return np.asarray(theta_max).item()
    return theta_max

def oblique_shock_max_deflection(mach_1: float | ndarray,
                                 gamma: float | ndarray
                                 ) -> float | ndarray:
    """
    Calculate the maximum oblique shock deflection angle.

    Parameters
    ----------
        mach_1: [1]
            Upstream Mach number
        gamma: [1]
            Specific heat ratio (Cp/Cv)

    Returns
    -------
        delta_max: [rad]
            Maximum oblique shock deflection angle (rad)
    """
    theta_max = oblique_shock_max_angle(mach_1, gamma)
    delta_max = oblique_shock_delta_from_theta(mach_1, theta_max, gamma)
    return delta_max

#endregion Oblique Shock Equations
#region Rocket Nozzle Flow
def get_cfv(gamma: float | ndarray, #pylint: disable=C0103
            Ae_At: float | ndarray, #pylint: disable=C0103
            Pe_Pc: Optional[float | ndarray] = None #pylint: disable=C0103
            ) -> float | ndarray:
    """ Calculate the vacuum specific impulse coefficient (cfv) for a rocket nozzle.
    Parameters
    ----------
        gamma: [1]
            Specific heat ratio (Cp/Cv)
        Ae_At: [1]
            Area ratio (Ae/At)
        Pe_Pc: [1]
           Exit to chamber pressure ratio (Pe/Pc). If not provided, will be calculated from Ae_At

    Returns
    -------
        cfv: [1]
            Vacuum nozzle specific impulse coefficient
    """
    if Pe_Pc is None:
        Pe_Pc = P_ratio_from_mach(mach_from_area_ratio(Ae_At, gamma, supersonic=True), gamma)
    return ((2 * gamma**2 / (gamma - 1)) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
           * (1 - Pe_Pc ** ((gamma - 1) / gamma))) ** 0.5 + Pe_Pc * Ae_At

def get_cf_from_cfv(cfv: float | ndarray,
                    Ae_At: float | ndarray, #pylint: disable=C0103
                    Pc: float | ndarray, #pylint: disable=C0103
                    Pb: float | ndarray #pylint: disable=C0103
                    ) -> float | ndarray:
    """Calculate the specific impulse coefficient (cf) 
    from the vacuum specific impulse coefficient (cfv)
    
    Parameters
    ----------
        cfv: [1]
            Vacuum nozzle specific impulse coefficient
        Ae_At: [1] 
            Area ratio (Ae/At)
        Pc: [Pa]
            Chamber pressure
        Pb: [Pa]
            Back pressure
    
    Returns
    -------
        cf: [1]
           Nozzle specific impulse coefficient
    """
    cf = cfv - (Pb / Pc) * Ae_At
    return cf

def get_cf(gamma: float | ndarray,
           Ae_At: float | ndarray, #pylint: disable=C0103
           Pc: float | ndarray, #pylint: disable=C0103
           Pb: float | ndarray, #pylint: disable=C0103
           Pe_Pc: Optional[float | ndarray] = None #pylint: disable=C0103
           ) -> float | ndarray:
    """Calculate the specific impulse coefficient (cf) for a given area ratio and pressure ratio

    Parameters
    ----------
        gamma: [1]
            Specific heat ratio (Cp/Cv)
        Ae_At: [1]
            Area ratio (Ae/At)
        Pc: [Pa]
            Chamber pressure (Pa)
        Pb: [Pa]
            Back pressure (Pa)
        Pe_Pc: [1]
            Exit to chamber pressure ratio (Pe/Pc). If not provided, will be calculated from Ae_At
    
    Returns
    -------
        cf: [1]
            Specific impulse coefficient
    """
    if Pe_Pc is None:
        Pe_Pc = P_ratio_from_mach(mach_from_area_ratio(Ae_At, gamma, supersonic=True), gamma)
    cfv = get_cfv(gamma, Ae_At, Pe_Pc)
    return get_cf_from_cfv(cfv, Ae_At, Pc, Pb)
#endregion Rocket Nozzle Flow
