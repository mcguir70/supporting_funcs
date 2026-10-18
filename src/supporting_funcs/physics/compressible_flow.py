"""
Equations and functions for compressible flow calculations.

Functions
---------
Psi:
    - Calculate the compressibility function Psi for a given specific heat ratio and Mach number.
press_ratio_from_mach:
    - Calculate the isentropic pressure ratio from Mach number and specific heat ratio.
mach_from_press_ratio:
    - Calculate the Mach number from pressure ratio and specific heat ratio.
temp_ratio_from_mach:
    - Calculate the temperature ratio from Mach number and specific heat ratio.
h_ratio_from_mach:
    - Calculate the enthalpy ratio from Mach number and specific heat ratio.
rho_ratio_from_mach:
    - Calculate the isentropic density ratio from Mach number and specific heat ratio.
area_ratio_from_mach:
    - Calculate the area ratio from Mach number and specific heat ratio.
mach_from_stagnation:
    - Calculate the Mach number from stagnation conditions.
m_flux_from_stagnation:
    - Calculate the mass flux (mdot/A) from stagnation conditions.
mach_from_area_ratio:
    - Calculate the Mach number from area ratio and specific heat ratio.
temp_recovery:
    - Calculate the recovery temperature.

prandtl_meyer:
    - Calculate the Prandtl-Meyer function for a given Mach number and specific heat ratio.
max_prandtl_meyer:
    - Calculate the maximum Prandtl-Meyer angle for a given specific heat ratio.
expansion_fan_mach_2:
    - Calculate the downstream Mach number after an expansion fan.
expansion_fan_max_turning_angle:
    - Calculate the maximum turning angle for a given Mach number and specific heat ratio.

normal_shock_press_ratio:
    - Calculate the pressure ratio across a normal shock.
normal_shock_mach_2:
    - Calculate the downstream Mach number across a normal shock.
normal_shock_press_tot_ratio:
    - Calculate the total pressure ratio across a normal shock.
normal_shock:
    - Calculate the downstream normal shock properties.

oblique_shock_delta_from_theta:
    - Calculate the deflection angle (delta) from the shock angle (theta) and Mach number.
oblique_shock:
    - Calculate the downstream Mach number and pressure ratio across an oblique shock.
oblique_shock_max_angle:
    - Calculate the maximum shock angle for a given Mach number and specific heat ratio.
oblique_shock_max_deflection:
    - Calculate the maximum deflection angle for a given Mach number and specific heat ratio.

get_cfv:
    - Calculate the vacuum thrust coefficient.
get_cf_from_cfv:
    - Calculate the actual thrust coefficient (cf) from the vacuum thrust coefficient (cfv).
get_cf:
    - Calculate the actual thrust coefficient (cf).
"""
from typing import Literal, Never, overload
from warnings import warn

import numpy as np
from numpy import arcsin as np_asin
from numpy import arctan as np_atan
from numpy import cos as np_cos
from numpy import logical_not as np_not
from numpy import ndarray
from numpy import pi as PI
from numpy import sin as np_sin
from numpy import tan as np_tan
from scipy.optimize import elementwise

from supporting_funcs.coding.type_hints import (
    BoolArray,
    BoolOrArray,
    FloatArray,
    FloatOrArray,
    OneDFloatArray,
)

type MachLims = FloatArray | tuple[float, float]


#region Isentropic Flow Relations
@overload
def Psi(gamma: float, mach: float) -> float: ...
@overload
def Psi(gamma: FloatArray, mach: FloatOrArray) -> FloatArray: ...
@overload
def Psi(gamma: FloatOrArray, mach: FloatArray) -> FloatArray: ...
def Psi(gamma: float | ndarray, mach: float | ndarray) -> float | ndarray: #noqa: N802
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

@overload
def press_ratio_from_mach(mach: float, gamma: float) -> float: ...
@overload
def press_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def press_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
def press_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray:
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

@overload
def mach_from_press_ratio(p_rat: float, gamma: float) -> float: ...
@overload
def mach_from_press_ratio(p_rat: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def mach_from_press_ratio(p_rat: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
def mach_from_press_ratio(p_rat: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the Mach number from pressure ratio and specific heat ratio.

    Parameters
    ----------
    p_rat: [1]
        (p/p0) Isentropic pressure ratio

    gamma: [1]
        (Cp/Cv) Specific heat ratio

    Returns
    -------
    mach: [1]
        Mach number
    """
    return ((2 / (gamma - 1)) * (p_rat ** ((1 - gamma) / (gamma)) - 1))**0.5

@overload
def temp_ratio_from_mach(mach: float, gamma: float) -> float: ...
@overload
def temp_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def temp_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
def temp_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray:
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

@overload
def h_ratio_from_mach(mach: float, gamma: float) -> float: ...
@overload
def h_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def h_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
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
    return temp_ratio_from_mach(mach, gamma)

@overload
def rho_ratio_from_mach(mach: float, gamma: float) -> float: ...
@overload
def rho_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def rho_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
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
    return temp_ratio_from_mach(mach, gamma) ** (1 / (gamma - 1))

@overload
def area_ratio_from_mach(mach: float, gamma: float) -> float: ...
@overload
def area_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def area_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
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

@overload
def mach_from_stagnation(T0: float, #noqa N803
                         P0: float, #noqa N803
                         mdot: float,
                         gamma: float,
                         R: float, #noqa N803
                         area: float,
                         supersonic: bool = ...) -> float: ...
@overload
def mach_from_stagnation(T0: FloatArray, #noqa N803
                         P0: FloatOrArray, #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #noqa N803
                         P0: FloatArray, #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #noqa N803
                         P0: FloatOrArray, #noqa N803
                         mdot: FloatArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #noqa N803
                         P0: FloatOrArray, #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatArray,
                         R: FloatOrArray, #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #noqa N803
                         P0: FloatOrArray, #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatArray, #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #noqa N803
                         P0: FloatOrArray, #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #noqa N803
                         area: FloatArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
def mach_from_stagnation(T0: FloatOrArray, #noqa: N803
                         P0: FloatOrArray, #noqa: N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #noqa: N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = False) -> FloatOrArray:
    """
    Calculate the Mach number from stagnation conditions.

    Returns a NaN if
    - the supersonic Mach > 10 or the subsonic Mach < 0.001. (Gamma stops being constant at M > 10)
    - gamma <= 1
    - Nonphysical stagnation conditions (e.g. negative pressure, temperature, or mass flow rate)

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
    args = (T0, P0, mdot, gamma, R, area, supersonic)
    return_float = all(np.ndim(arg) == 0 for arg in args)
    arg_arrays = [np.asarray(arg) for arg in args]
    shape = np.broadcast_shapes(*[arg.shape for arg in arg_arrays])
    (T0, P0, mdot, gamma, R, area, supersonic #noqa: N806
     ) = (np.broadcast_to(arg_array, shape) for arg_array in arg_arrays)

    def _rhs(mach: FloatOrArray, gamma: FloatOrArray) -> FloatOrArray:
        return (mach
                * press_ratio_from_mach(mach=mach, gamma=gamma)
                * temp_ratio_from_mach(mach=mach, gamma=gamma) ** -0.5)

    def _lhs(temp_stag: FloatArray,
             press_stag: FloatArray,
             mdot: FloatArray,
             gamma: FloatArray,
             R: FloatArray, #noqa: N803
             area: FloatArray
             ) -> FloatArray:
        return mdot / (press_stag * area) * (R * temp_stag / gamma) ** 0.5

    def _error_func(mach: FloatArray, gamma: FloatArray, lhs_val: FloatArray) -> FloatArray:
        return lhs_val - _rhs(mach=mach, gamma=gamma)

    lims = np.empty(shape=[*shape, 2])

    lims[ supersonic, :] = [1.00001, 10.0]
    lims[np_not(supersonic), :] = [0., 0.99999]

    lhs = _lhs(temp_stag=T0, press_stag=P0, mdot=mdot, gamma=gamma, R=R, area=area)
    valid_mask = np.ones_like(lhs, dtype=bool)
    valid_mask[gamma <= 1] = False
    max_rhs = _rhs(mach=1, gamma=gamma[valid_mask])
    valid_mask[valid_mask] = lhs[valid_mask] < max_rhs

    mach = np.empty(shape=shape, dtype=float)
    res = elementwise.find_root(_error_func,
                                    (lims[valid_mask, 0], lims[valid_mask, 1]),
                                    args=(gamma[valid_mask],
                                            lhs[valid_mask]),
                                    tolerances={'xatol': 1e-10, 'frtol': 1e-10},)

    mach[valid_mask] = np.where(res.success, res.x, np.nan)
    mach[~valid_mask] = np.nan
    if return_float:
        return float(mach.item())
    return mach

@overload
def m_flux_from_stagnation(T0: float, #noqa N803
                           P0: float, #noqa N803
                           gamma: float,
                           R: float, #noqa N803
                           mach: float
                           ) -> float: ...
@overload
def m_flux_from_stagnation(T0: FloatArray, #noqa N803
                           P0: FloatOrArray, #noqa N803
                           gamma: FloatOrArray,
                           R: FloatOrArray, #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #noqa N803
                           P0: FloatArray, #noqa N803
                           gamma: FloatOrArray,
                           R: FloatOrArray, #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #noqa N803
                           P0: FloatOrArray, #noqa N803
                           gamma: FloatArray,
                           R: FloatOrArray, #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #noqa N803
                           P0: FloatOrArray, #noqa N803
                           gamma: FloatOrArray,
                           R: FloatArray, #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #noqa N803
                           P0: FloatOrArray, #noqa N803
                           gamma: FloatOrArray,
                           R: FloatOrArray, #noqa N803
                           mach: FloatArray
                           ) -> FloatArray: ...
def m_flux_from_stagnation(T0: float | ndarray, #noqa: N803
                           P0: float | ndarray, #noqa: N803
                           gamma: float | ndarray,
                           R: float | ndarray, #noqa: N803
                           mach: float | ndarray
                           ) -> float | ndarray:
    """
    Calculate the mass flux (mdot/A) from stagnation conditions.

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

@overload
def _D_func(mach: float, gamma: float) -> float: ...
@overload
def _D_func(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def _D_func(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
def _D_func(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray: #noqa: N802
    """
    Helper function for mass flow rate calculation.

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
    """  # noqa: D401
    return mach / (1 + 0.5 * (gamma - 1) * mach**2) ** ((gamma + 1) / (2 * (gamma - 1)))

@overload
def mach_from_area_ratio( # Supersonic required
    area_rat: float,
    gamma: float,
    max_iter: int = ...,
    tol: float = ...,
) -> Never: ...
@overload
def mach_from_area_ratio( # Supersonic required
    area_rat: float,
    gamma: float,
    max_iter: int = ...,
    tol: float = ...,
    *,
    supersonic: bool,
    mach_lims: MachLims | None = ...,
) -> float: ...
@overload
def mach_from_area_ratio( # Mach lims required
    area_rat: float,
    gamma: float,
    max_iter: int = ...,
    tol: float = ...,
    *,
    mach_lims: tuple[float, float] | OneDFloatArray,
    supersonic: bool | None = ...,
) -> float: ...
@overload
def mach_from_area_ratio( # Supersonic required
    area_rat: FloatArray,
    gamma: FloatArray | float,
    max_iter: int = ...,
    tol: float = ...,
    *,
    supersonic: BoolOrArray,
    mach_lims: MachLims | None = ...,
) -> FloatArray: ...
@overload
def mach_from_area_ratio( # Mach lims required
    area_rat: FloatArray,
    gamma: FloatArray | float,
    max_iter: int = ...,
    tol: float = ...,
    *,
    mach_lims: MachLims,
    supersonic: BoolOrArray | None = ...,
) -> FloatArray: ...
@overload
def mach_from_area_ratio( # Supersonic required
    area_rat: float,
    gamma: FloatArray,
    max_iter: int = ...,
    tol: float = ...,
    *,
    supersonic: BoolOrArray,
    mach_lims: MachLims | None = ...,
) -> FloatArray: ...
@overload
def mach_from_area_ratio( # Mach lims required
    area_rat: float,
    gamma: FloatArray,
    max_iter: int = ...,
    tol: float = ...,
    *,
    mach_lims: MachLims,
    supersonic: BoolOrArray | None = ...,
) -> FloatArray: ...
def mach_from_area_ratio(area_rat: float | ndarray,
                         gamma: float | ndarray,
                         max_iter: int = 1000,
                         tol: float = 1e-6,
                         *,
                         mach_lims: ndarray | tuple[float, float] | None = None,
                         supersonic: bool | BoolArray | None = None,
                         ) -> float | ndarray:
    """
    Calculate the Mach number from area ratio and specific heat ratio.

    Parameters
    ----------
    area_rat: [1]
        Area ratio (A/A*)
    gamma: [1]
        Specific heat ratio (Cp/Cv)
    max_iter (int): [-]
        Maximum number of iterations for convergence
    tol (float): [1]
        Tolerance for convergence

    mach_lims: [1]
        Limits for the Mach ratio to constrain the search space (min, max)
    supersonic: [bool]
        Whether to calculate the super- or sub-sonic Mach number.
        If None, mach_lims must be specified to constrain the search space.
        Sets bounds to (1, 10) for supersonic and (0, 1) for subsonic.

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
        if np.any((mach_lims[..., 0] < 1) == (mach_lims[..., 1] > 1)):
            raise ValueError(
                "mach_lims must be such that " \
                "the first and second columns are (both or neither) supersonic.")
        if supersonic is not None:
            supersonic = np.broadcast_to(np.asarray(supersonic, dtype=bool), shape)
            if np.any((supersonic & (mach_lims[..., 1] <= 1))
                      | (np_not(supersonic) & (mach_lims[..., 0] >= 1))):
                raise ValueError(
                    "Conflicting specification: cannot specify both supersonic and mach_lims. " \
                    "Use mach_lims to specify the Mach number range, "
                    "or set supersonic to True or False to select the super or sub-sonic branch.")
    else: # mach_lims is None, supersonic is not None
        supersonic = np.broadcast_to(np.asarray(supersonic, dtype=bool), shape)
        mach_lims = np.ones(shape + (2,), dtype=float)
        mach_lims[np_not(supersonic), 0] = 1e-6
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

@overload
def get_recovery_scalar(prandtl: float, flow_type: Literal['turbulent']) -> float: ...
@overload
def get_recovery_scalar(prandtl: FloatArray, flow_type: Literal['turbulent']) -> FloatArray: ...
def get_recovery_scalar(prandtl: FloatOrArray, flow_type: Literal['turbulent']) -> FloatOrArray:
    """
    Calculate the recovery factor for a given Prandtl number and specific heat ratio.

    Parameters
    ----------
    prandtl: [1]
        Prandtl number
    flow_type: [-]
        Type of flow ('turbulent')

    Returns
    -------
    r_recovery: [1]
        Recovery factor
    """
    match flow_type:
        case 'turbulent':
            return prandtl**(1/3)
    raise ValueError(f"Unsupported flow type: {flow_type}")

@overload
def temp_recovery(temp_inf: float,
               prandtl: float,
               gamma: float,
               mach: float,
               flow_type: Literal['turbulent'] = ...
               ) -> float: ...
@overload
def temp_recovery(temp_inf: FloatArray,
               prandtl: FloatOrArray,
               gamma: FloatOrArray,
               mach: FloatOrArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...
@overload
def temp_recovery(temp_inf: FloatOrArray,
               prandtl: FloatArray,
               gamma: FloatOrArray,
               mach: FloatOrArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...
@overload
def temp_recovery(temp_inf: FloatOrArray,
               prandtl: FloatOrArray,
               gamma: FloatArray,
               mach: FloatOrArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...
@overload
def temp_recovery(temp_inf: FloatOrArray,
               prandtl: FloatOrArray,
               gamma: FloatOrArray,
               mach: FloatArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...
def temp_recovery(temp_inf: FloatOrArray, #noqa: N802
               prandtl: FloatOrArray,
               gamma: FloatOrArray,
               mach: FloatOrArray,
               flow_type: Literal['turbulent'] = 'turbulent') -> FloatOrArray:
    """
    Calculate the recovery temperature.

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
    r_recovery = get_recovery_scalar(prandtl=prandtl, flow_type=flow_type)
    return temp_inf * (1 + r_recovery * (gamma - 1) / 2 * mach**2)

#region Expansion Fan Relations
@overload
def prandtl_meyer(mach: float, gamma: float) -> float: ...
@overload
def prandtl_meyer(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def prandtl_meyer(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
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

@overload
def max_prandtl_meyer(gamma: float) -> float: ...
@overload
def max_prandtl_meyer(gamma: FloatArray) -> FloatArray: ...
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

@overload
def expansion_fan_mach_2(mach_1: float, delta: float, gamma: float) -> float: ...
@overload
def expansion_fan_mach_2(mach_1: FloatArray,
                         delta: FloatOrArray,
                         gamma: FloatOrArray
                         ) -> FloatArray: ...
@overload
def expansion_fan_mach_2(mach_1: FloatOrArray,
                         delta: FloatArray,
                         gamma: FloatOrArray
                         ) -> FloatArray: ...
@overload
def expansion_fan_mach_2(mach_1: FloatOrArray,
                         delta: FloatOrArray,
                         gamma: FloatArray
                         ) -> FloatArray: ...
def expansion_fan_mach_2(mach_1: float | ndarray,
                         delta: float | ndarray,
                         gamma: float | ndarray
                         ) -> float | ndarray:
    """
    Calculate the downstream Mach number after an expansion fan.

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

@overload
def expansion_fan_max_turning_angle(mach_1: float, gamma: float) -> float: ...
@overload
def expansion_fan_max_turning_angle(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def expansion_fan_max_turning_angle(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
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
@overload
def normal_shock_press_ratio(mach_1: float, gamma: float) -> float: ...
@overload
def normal_shock_press_ratio(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def normal_shock_press_ratio(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
def normal_shock_press_ratio(mach_1: float | ndarray,
                         gamma: float | ndarray) -> float | ndarray:
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

@overload
def normal_shock_mach_2(mach_1: float, gamma: float) -> float: ...
@overload
def normal_shock_mach_2(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def normal_shock_mach_2(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
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

@overload
def normal_shock_press_tot_ratio(mach_1: float, gamma: float) -> float: ...
@overload
def normal_shock_press_tot_ratio(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def normal_shock_press_tot_ratio(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
def normal_shock_press_tot_ratio(
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
    p2_p1 = normal_shock_press_ratio(mach_1, gamma)
    mach_2 = normal_shock_mach_2(mach_1, gamma)
    p2_p02 = press_ratio_from_mach(mach_2, gamma)
    p1_p01 = press_ratio_from_mach(mach_1, gamma)
    p02_p01 = p2_p1 * (p1_p01 / p2_p02)
    return p02_p01

@overload
def normal_shock(mach_1: float, gamma: float) -> tuple[float, float]: ...
@overload
def normal_shock(mach_1: FloatArray, gamma: FloatOrArray) -> tuple[FloatArray, FloatArray]: ...
@overload
def normal_shock(mach_1: FloatOrArray, gamma: FloatArray) -> tuple[FloatArray, FloatArray]: ...
def normal_shock(mach_1: float | ndarray,
                 gamma: float | ndarray
                 ) -> tuple[float | ndarray, float | ndarray]:
    """
    Calculate the downstream normal shock properties.

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
    p02_p01 = normal_shock_press_tot_ratio(mach_1, gamma)
    return mach_2, p02_p01
#endregion Normal Shock Equations
#region Oblique Shock Equations
@overload
def oblique_shock_delta_from_theta(mach_1: float,
                                   theta: float,
                                   gamma: float
                                   ) -> float: ...
@overload
def oblique_shock_delta_from_theta(mach_1: FloatArray,
                                   theta: FloatOrArray,
                                   gamma: FloatOrArray
                                   ) -> FloatArray: ...
@overload
def oblique_shock_delta_from_theta(mach_1: FloatOrArray,
                                   theta: FloatArray,
                                   gamma: FloatOrArray
                                   ) -> FloatArray: ...
@overload
def oblique_shock_delta_from_theta(mach_1: FloatOrArray,
                                   theta: FloatOrArray,
                                   gamma: FloatArray
                                   ) -> FloatArray: ...
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
    if np.any(np_not(mach_normal_1_is_sup)):
        warn("obliqueShockDeltaFromTheta is only valid "
        "for supersonic upstream normal Mach numbers (Mn1 > 1). "
        "Results may be invalid for some inputs.",
        stacklevel=2)

    delta[mach_normal_1_is_sup] = np_atan(2 * (1 / np_tan(theta[mach_normal_1_is_sup]))
                                          * (mach_normal_1[mach_normal_1_is_sup]**2 - 1)
                                          / (mach_1[mach_normal_1_is_sup]**2
                                             * (gamma[mach_normal_1_is_sup]
                                                + np_cos(2 * theta[mach_normal_1_is_sup]))
                                             + 2))

    if return_float:
        return float(np.asarray(delta).item())
    return delta

@overload
def oblique_shock(mach_1: float,
                  delta: float,
                  gamma: float,
                  is_strong_shock: bool = ...,
                  x0: tuple[float, float] | OneDFloatArray | None = ...
                  ) -> tuple[float, float, float]: ...
@overload
def oblique_shock(mach_1: FloatArray,
                  delta: FloatOrArray,
                  gamma: FloatOrArray,
                  is_strong_shock: BoolOrArray = ...,
                  x0: tuple[FloatOrArray, FloatOrArray] | FloatArray | None = ...
                  ) -> tuple[FloatArray, FloatArray, FloatArray]: ...
@overload
def oblique_shock(mach_1: FloatOrArray,
                  delta: FloatArray,
                  gamma: FloatOrArray,
                  is_strong_shock: BoolOrArray = ...,
                  x0: tuple[FloatOrArray, FloatOrArray] | FloatArray | None = ...
                  ) -> tuple[FloatArray, FloatArray, FloatArray]: ...
@overload
def oblique_shock(mach_1: FloatOrArray,
                  delta: FloatOrArray,
                  gamma: FloatArray,
                  is_strong_shock: BoolOrArray = ...,
                  x0: tuple[FloatOrArray, FloatOrArray] | FloatArray | None = ...
                  ) -> tuple[FloatArray, FloatArray, FloatArray]: ...
@overload
def oblique_shock(mach_1: FloatOrArray,
                  delta: FloatOrArray,
                  gamma: FloatOrArray,
                  is_strong_shock: BoolArray,
                  x0: tuple[FloatOrArray, FloatOrArray] | FloatArray | None = ...
                  ) -> tuple[FloatArray, FloatArray, FloatArray]: ...
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
        """Equation to solve for shock angle given deflection angle.

        A positive delta means the shock angle guess is too small.
        """
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
        x0[np_not(is_strong_shock), 0] = np_asin(1 / mach_1[np_not(is_strong_shock)])
        x0[np_not(is_strong_shock), 1] = theta_max[np_not(is_strong_shock)]

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
    p02_p01 = normal_shock_press_tot_ratio(mach_1_normal, gamma)
    mach_2_normal = normal_shock_mach_2(mach_1_normal, gamma)
    mach_2 = mach_2_normal / np_sin(theta_sol - delta)

    return_val = [mach_2, p02_p01, theta_sol]

    if return_float:
        for i, val in enumerate(return_val):
            if isinstance(val, ndarray):
                return_val[i] = float(np.asarray(val).item())

    return tuple(return_val)

@overload
def oblique_shock_max_angle(mach_1: float, gamma: float) -> float: ...
@overload
def oblique_shock_max_angle(mach_1: FloatArray, gamma: FloatArray | float) -> FloatArray: ...
@overload
def oblique_shock_max_angle(mach_1: FloatArray | float, gamma: FloatArray) -> FloatArray: ...
def oblique_shock_max_angle(mach_1: float | ndarray, gamma: float | ndarray) -> float | ndarray:
    """
    Calculate the oblique shock angle that results in the maximum deflection angle.

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

@overload
def oblique_shock_max_deflection(mach_1: float, gamma: float) -> float: ...
@overload
def oblique_shock_max_deflection(mach_1: FloatArray, gamma: FloatArray | float) -> FloatArray: ...
@overload
def oblique_shock_max_deflection(mach_1: FloatArray | float, gamma: FloatArray) -> FloatArray: ...
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
@overload
def get_cfv(gamma: float,
            area_ratio: float,
            Pe_Pc: float | None = ... #noqa N803
            ) -> float: ...
@overload
def get_cfv(gamma: FloatArray,
            area_ratio: FloatOrArray,
            Pe_Pc: FloatOrArray | None = ... #noqa N803
            ) -> FloatArray: ...
@overload
def get_cfv(gamma: FloatOrArray,
            area_ratio: FloatArray,
            Pe_Pc: FloatOrArray | None = ... #noqa N803
            ) -> FloatArray: ...
@overload
def get_cfv(gamma: FloatOrArray,
            area_ratio: FloatOrArray,
            Pe_Pc: FloatArray #noqa N803
            ) -> FloatArray: ...
def get_cfv(gamma: float | ndarray,
            area_ratio: float | ndarray,
            Pe_Pc: float | ndarray | None = None #noqa: N803
            ) -> float | ndarray:
    """
    Calculate the vacuum thrust coefficient (cfv) for a rocket nozzle.

    Parameters
    ----------
    gamma: [1]
        Specific heat ratio (Cp/Cv)
    area_ratio: [1]
        Ae/At Area ratio
    Pe_Pc: [1]
        Exit to chamber pressure ratio (Pe/Pc). If not provided, will be calculated from Ae_At

    Returns
    -------
    cfv: [1]
        Vacuum nozzle thrust coefficient
    """
    if Pe_Pc is None:
        Pe_Pc = press_ratio_from_mach(mach_from_area_ratio(area_ratio, gamma, supersonic=True), #noqa: N806
                                  gamma)
    return ((2 * gamma**2 / (gamma - 1)) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
           * (1 - Pe_Pc ** ((gamma - 1) / gamma))) ** 0.5 + Pe_Pc * area_ratio

@overload
def get_cf_from_cfv(cfv: float,
                    area_ratio: float,
                    Pc: float, #noqa N803
                    Pb: float #noqa N803
                    ) -> float: ...
@overload
def get_cf_from_cfv(cfv: FloatArray,
                    area_ratio: FloatOrArray,
                    Pc: FloatOrArray, #noqa N803
                    Pb: FloatOrArray #noqa N803
                    ) -> FloatArray: ...
@overload
def get_cf_from_cfv(cfv: FloatOrArray,
                    area_ratio: FloatArray,
                    Pc: FloatOrArray, #noqa N803
                    Pb: FloatOrArray #noqa N803
                    ) -> FloatArray: ...
@overload
def get_cf_from_cfv(cfv: FloatOrArray,
                    area_ratio: FloatOrArray,
                    Pc: FloatArray, #noqa N803
                    Pb: FloatOrArray #noqa N803
                    ) -> FloatArray: ...
@overload
def get_cf_from_cfv(cfv: FloatOrArray,
                    area_ratio: FloatOrArray,
                    Pc: FloatOrArray, #noqa N803
                    Pb: FloatArray #noqa N803
                    ) -> FloatArray: ...
def get_cf_from_cfv(cfv: float | ndarray,
                    area_ratio: float | ndarray,
                    Pc: float | ndarray, #noqa: N803
                    Pb: float | ndarray #noqa: N803
                    ) -> float | ndarray:
    """Calculate cf from the vacuum thrust coefficient (cfv).

    Parameters
    ----------
    cfv: [1]
        Vacuum nozzle thrust coefficient
    area_ratio: [1]
        Ae/At Area ratio
    Pc: [Pa]
        Chamber pressure
    Pb: [Pa]
        Back pressure

    Returns
    -------
    cf: [1]
        Actual nozzle thrust coefficient
    """
    cf = cfv - (Pb / Pc) * area_ratio
    return cf

@overload
def get_cf(gamma: float,
           area_ratio: float,
           Pc: float, #noqa N803
           Pb: float, #noqa N803
           Pe_Pc: float | None = ... #noqa N803
           ) -> float: ...
@overload
def get_cf(gamma: FloatArray,
           area_ratio: FloatOrArray,
           Pc: FloatOrArray, #noqa N803
           Pb: FloatOrArray, #noqa N803
           Pe_Pc: FloatOrArray | None = ... #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           area_ratio: FloatArray,
           Pc: FloatOrArray, #noqa N803
           Pb: FloatOrArray, #noqa N803
           Pe_Pc: FloatOrArray | None = ... #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           area_ratio: FloatOrArray,
           Pc: FloatArray, #noqa N803
           Pb: FloatOrArray, #noqa N803
           Pe_Pc: FloatOrArray | None = ... #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           area_ratio: FloatOrArray,
           Pc: FloatOrArray, #noqa N803
           Pb: FloatArray, #noqa N803
           Pe_Pc: FloatOrArray | None = ... #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           area_ratio: FloatOrArray,
           Pc: FloatOrArray, #noqa N803
           Pb: FloatOrArray, #noqa N803
           Pe_Pc: FloatArray #noqa N803
           ) -> FloatArray: ...
def get_cf(gamma: float | ndarray,
           area_ratio: float | ndarray,
           Pc: float | ndarray, #noqa: N803
           Pb: float | ndarray, #noqa: N803
           Pe_Pc: float | ndarray | None = None #noqa: N803
           ) -> float | ndarray:
    """Calculate the actual thrust coefficient (cf) for a given area ratio and pressure ratio.

    Parameters
    ----------
    gamma: [1]
        Specific heat ratio (Cp/Cv)
    area_ratio: [1]
        Area ratio (Ae/At)
    Pc: [Pa]
        Chamber pressure (Pa)
    Pb: [Pa]
        Back pressure (Pa)
    Pe_Pc: [1]
        Exit to chamber pressure ratio (Pe/Pc). If not provided, will be calculated from area_ratio

    Returns
    -------
    cf: [1]
        Actual thrust coefficient
    """
    if Pe_Pc is None:
        Pe_Pc = press_ratio_from_mach(mach_from_area_ratio(area_ratio, gamma, supersonic=True), #noqa: N806
                                  gamma)
    cfv = get_cfv(gamma, area_ratio, Pe_Pc)
    return get_cf_from_cfv(cfv, area_ratio, Pc, Pb)
#endregion Rocket Nozzle Flow
#region DEPRECATED
def normal_shock_P_ratio(mach_1: float | ndarray,  #noqa: N802
                            gamma: float | ndarray) -> float | ndarray:
    """DEPRECATED: Use `normal_shock_press_ratio` instead."""  # noqa: D401
    warn("normal_shock_P_ratio is deprecated. Use `normal_shock_press_ratio` instead.",
         DeprecationWarning,
         stacklevel=2)
    return normal_shock_press_ratio(mach_1, gamma)
def P_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray: #noqa: N802
    """DEPRECATED: Use `press_ratio_from_mach` instead."""  # noqa: D401
    warn("P_ratio_from_mach is deprecated. Use `press_ratio_from_mach` instead.",
         DeprecationWarning,
         stacklevel=2)
    return press_ratio_from_mach(mach, gamma)
def T_ratio_from_mach(mach: float | ndarray, gamma: float | ndarray) -> float | ndarray: #noqa: N802
    """DEPRECATED: Use `temp_ratio_from_mach` instead."""  # noqa: D401
    warn("T_ratio_from_mach is deprecated. Use `temp_ratio_from_mach` instead.",
         DeprecationWarning,
         stacklevel=2)
    return temp_ratio_from_mach(mach, gamma)
def T_recovery(temp_inf: FloatOrArray, #noqa: N802
               prandtl: FloatOrArray,
               gamma: FloatOrArray,
                mach: FloatOrArray,
                flow_type: Literal['turbulent'] = 'turbulent') -> FloatOrArray:
    """
    DEPRECATED: To be replaced by `temp_recovery` instead.

    Calculate the recovery temperature.

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
    """  # noqa: D401
    warn("T_recovery is deprecated. Use `temp_recovery` instead.",
         DeprecationWarning,
         stacklevel=2)
    return temp_recovery(temp_inf, prandtl, gamma, mach, flow_type)
def mach_from_P_ratio(p_rat: float | ndarray, gamma: float | ndarray) -> float | ndarray: #noqa: N802
    """DEPRECATED: Use `mach_from_press_ratio` instead."""  # noqa: D401
    warn("mach_from_P_ratio is deprecated. Use `mach_from_press_ratio` instead.",
         DeprecationWarning,
         stacklevel=2)
    return mach_from_press_ratio(p_rat, gamma)
def normal_shock_P_tot_ratio( #noqa: N802
        mach_1: float | ndarray,
        gamma: float | ndarray
        ) -> float | ndarray:
    """DEPRECATED: Use `normal_shock_press_tot_ratio` instead."""  # noqa: D401
    warn("normal_shock_P_tot_ratio is deprecated. Use `normal_shock_press_tot_ratio` instead.",
         DeprecationWarning,
         stacklevel=2)
    return normal_shock_press_tot_ratio(mach_1, gamma)
