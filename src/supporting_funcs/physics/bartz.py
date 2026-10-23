"""Bartz equation parameters for heat transfer calculations."""
from copy import deepcopy
from types import SimpleNamespace
from typing import overload

import scipy.constants as csp
from numpy import array, empty, linspace
from numpy import exp as np_exp
from numpy import log as np_log
from numpy import polyfit as np_polyfit

from supporting_funcs.coding.type_hints import FloatArray, FloatOrArray
from supporting_funcs.physics.cea_wrapper import CEAOutput, PropellentMixture
from supporting_funcs.physics.compressible_flow import temp_recovery

_G0 = csp.g  # [m/s^2] standard gravity
_G0_FT_S2 = _G0 / csp.foot  # [ft/s^2] standard gravity
_BTU_IN2_S_R = csp.Btu / (csp.inch**2 * csp.degree_Fahrenheit)  # [Btu/(in^2*sec*R)]
_DEFAULT_OMEGA = 0.6  # Default temperature exponent of viscosity equation

# TODO: Blending function for the Bartz equation to handle the Pr & cp_frozen vs equilibrium delta

class BartzParamClass:
    """Parameters for the Bartz equation, generated from CEA results and nozzle geometry.

    Methods
    -------
    heat_flux_with_temp_variables
        Calculate the heat transfer (W/m^2) using the Bartz equation
        with pre-calculated temperature variables.
    bartz_with_temp_variables
        Calculate the heat transfer coefficient (W/m^2*K) using the Bartz equation
        with pre-calculated temperature variables.
    """

    def __init__(self, d_throat: float,
                 mu_0: float,
                 c_p_0: float,
                 pran_0: float,
                 Pc: float, #pylint: disable=C0103 #noqa: N803
                 c_star: float,
                 r_throat_curve: float,
                 T_0: float, #pylint: disable=C0103 #noqa: N803
                 gamma: FloatOrArray,
                 mach: FloatOrArray,
                 area_rat: FloatOrArray,
                 omega: float = _DEFAULT_OMEGA) -> None:
        """
        Generate the temperature variables for the Bartz equation.

        Parameters
        ----------
        d_throat: [m]
            Throat diameter (m)
        mu_0: [Pa*s]
            Stagnation viscosity (Pa*s)
        c_p_0: [J/(kg*K)]
            Stagnation specific heat
        pran_0: [1]
            Stagnation Prandtl number
        Pc: [Pa]
            Chamber pressure
        c_star: [m/s]
            Characteristic velocity
        r_throat_curve: [m]
            Curvature of the nozzle at the throat
        T_0: [K]
            Stagnation temperature
        gamma: [1]
            Specific heat ratio of the gas
        mach: [1]
            Mach number of the gas
        area_rat: [1]
            A/Astar
        omega: [1]
            Temperature exponent of viscosity equation
        """
        (self.sigma_a_mult,
         self.sigma_b_mult,
         self.sigma_c_exp,
         self.hg_term,
         self.Tr_scalar) = bartz_gen_temp_variables(
            d_throat=d_throat,
            mu_0=mu_0,
            c_p_0=c_p_0,
            pran_0=pran_0,
            P_c=Pc,
            c_star=c_star,
            r_throat_curve=r_throat_curve,
            temp_stag=T_0,
            gamma=gamma,
            mach=mach,
            inv_area_ratio=1.0 / area_rat,
            omega=omega
        )

    @overload
    def heat_flux_with_temp_variables(self, temp_gas: float, temp_wall: float
                                      ) -> float: ...
    @overload
    def heat_flux_with_temp_variables(self, temp_gas: FloatOrArray, temp_wall: FloatArray
                                      ) -> FloatArray: ...
    @overload
    def heat_flux_with_temp_variables(self, temp_gas: FloatArray, temp_wall: FloatOrArray
                                      ) -> FloatArray: ...
    def heat_flux_with_temp_variables(self,
                                      temp_gas: FloatOrArray,
                                      temp_wall: FloatOrArray,
                                  ) -> FloatOrArray:
        """
        Calculate the heat transfer using the Bartz equation.

        Uses pre-calculated temperature variables.

        Parameters
        ----------
        temp_wall: [K]
            Wall temperature at the nozzle axial position
        temp_gas: [K]
            Static gas temperature at the nozzle axial position

        Returns
        -------
        hg: [W/m^2]
            Heat transfer into the wall at the nozzle axial position
        """
        return (self.bartz_with_temp_variables(temp_wall=temp_wall)
                * (temp_gas * self.Tr_scalar - temp_wall))

    @overload
    def bartz_with_temp_variables(self, temp_wall: float) -> float: ...
    @overload
    def bartz_with_temp_variables(self, temp_wall: FloatArray) -> FloatArray: ...
    def bartz_with_temp_variables(self, temp_wall: FloatOrArray) -> FloatOrArray:
        """
        Calculate the heat transfer coefficient using the Bartz equation.

        Uses pre-calculated temperature variables.

        Parameters
        ----------
        temp_wall: [K]
            Wall temperature at the nozzle axial position

        Returns
        -------
        hg: [W/(m^2*K)]
            Heat transfer coefficient at the nozzle axial position
        """
        hg_imp = bartz_with_temp_variables_imp(temp_wall_R=temp_wall / csp.degree_Fahrenheit,
                                               sigma_a_mult_imp=self.sigma_a_mult,
                                               sigma_b_mult_imp=self.sigma_b_mult,
                                               sigma_c_exp_imp=self.sigma_c_exp,
                                               hg_term_imp=self.hg_term)
        return hg_imp * _BTU_IN2_S_R

    @overload
    def bartz_with_temp_variables_imp(self,
                                      temp_wall_R: FloatArray #noqa: N803
                                      ) -> FloatArray: ...
    @overload
    def bartz_with_temp_variables_imp(self,
                                      temp_wall_R: float #noqa: N803
                                      ) -> float: ...
    def bartz_with_temp_variables_imp(self,
                                      temp_wall_R: FloatOrArray #noqa: N803
                                      ) -> FloatOrArray:
        """
        Calculate the heat transfer coefficient using the Bartz equation.

        Uses pre-calculated temperature variables.

        Parameters
        ----------
        temp_wall_R: [R]
            Wall temperature

        Returns
        -------
        hg: [Btu/(in^2*sec*R)]
            Heat transfer coefficient
        """
        return bartz_with_temp_variables_imp(temp_wall_R=temp_wall_R,
                                            sigma_a_mult_imp=self.sigma_a_mult,
                                            sigma_b_mult_imp=self.sigma_b_mult,
                                            sigma_c_exp_imp=self.sigma_c_exp,
                                            hg_term_imp=self.hg_term)

def create_bartz_param_class(cea_result: CEAOutput,
                             d_throat: float,
                             r_throat_curve: float,
                             mach: FloatOrArray = 1.0,
                             area_rat: FloatOrArray = 1.0) -> BartzParamClass:
    """
    Create an instance of BartzParamClass using CEA results and nozzle geometry.

    Parameters
    ----------
    cea_result: [-]
        An instance of CEAOutput containing the results from CEA analysis.
    d_throat: [m]
        Throat diameter
    r_throat_curve: [m]
        Curvature of the throat radius
    mach: [1]
        Mach number at location to report (default is 1.0)
    area_rat: [1]
        A/Astar at location to report (default is 1.0)

    Returns
    -------
    bartz_params: BartzParamClass
        An instance of BartzParamClass with the calculated parameters.
    """
    bartz_params = BartzParamClass(
        d_throat = d_throat,
        mu_0 = cea_result.mu[0],
        c_p_0 = cea_result.cp[0],
        pran_0 = cea_result.prandtl[0],
        Pc = cea_result.chamber_pressure,
        c_star = cea_result.c_star[0],
        r_throat_curve = r_throat_curve,
        T_0 = cea_result.temperature[0],
        gamma = cea_result.gamma_s[0],
        mach = mach,
        area_rat = area_rat,
        )
    return bartz_params

def calc_hg_bartz_imp(d_throat_in: float,
                  mu_0_lbm_in_s: float,
                  c_p_0_btu_lbm_R: float, #pylint: disable=C0103 #noqa: N803
                  pran_0: float,
                  P_c_psi: float, #pylint: disable=C0103 #noqa: N803
                  c_star_ft_s: float,
                  r_throat_curve_in: float,
                  temp_wall_R: FloatOrArray, #pylint: disable=C0103 #noqa: N803
                  temp_0_R: float, #pylint: disable=C0103 #noqa: N803
                  gamma: FloatOrArray,
                  mach: FloatOrArray,
                  inv_area_rat: FloatOrArray,
                  omega: float = _DEFAULT_OMEGA
                  ) -> FloatOrArray:
    """
    Calculate the heat transfer coefficient using the Bartz equation.

    https://ntrs.nasa.gov/api/citations/19710011726/downloads/19710011726.pdf

    Parameters
    ----------
    d_throat_in: [in]
        Throat diameter
    mu_0_lbm_in_s: [lb/(in*s)]
        Stagnation viscosity
    c_p_0_btu_lbm_R: [Btu/(lb*R)]
        Stagnation specific heat (Btu/(lb*R))
    pran_0: [dimensionless]
        Stagnation Prandtl number (mu * cp / k)
    P_c_psi: [psi]
        Chamber pressure
    c_star_ft_s: [ft/s]
        Characteristic velocity (ft/s)
    r_throat_curve_in: [in]
        Curvature of the Nozzle radius (in)
    temp_wall_R: [R]
        Wall temperature (R)
    temp_0_R: [R]
        Stagnation temperature (R)
    gamma: [1]
        Specific heat ratio
    mach: [1]
        Mach number
    inv_area_rat: [1]
        Inverse area ratio (Astar / A)
    omega: [1]
        Temperature exponent of viscosity equation

    Returns
    -------
    hg_btu_in2_s_R: [Btu/(in^2*sec*R)]
        Heat transfer coefficient
    """
    (sigma_a_mult_imp,
     sigma_b_mult_imp,
     sigma_c_exp_imp,
     hg_term_imp,
     _) = bartz_gen_temp_variables_imp(d_throat_in=d_throat_in,
                                         mu_0_lbm_in_s=mu_0_lbm_in_s,
                                         c_p_0_btu_lbm_R=c_p_0_btu_lbm_R,
                                         pran_0=pran_0,
                                         P_c_psi=P_c_psi,
                                         c_star_ft_s=c_star_ft_s,
                                         r_throat_curve_in=r_throat_curve_in,
                                         temp_0_R=temp_0_R,
                                         gamma=gamma,
                                         mach=mach,
                                         inv_area_ratio=inv_area_rat,
                                         omega=omega)

    return bartz_with_temp_variables_imp(temp_wall_R=temp_wall_R,
                                         sigma_a_mult_imp=sigma_a_mult_imp,
                                         sigma_b_mult_imp=sigma_b_mult_imp,
                                         sigma_c_exp_imp=sigma_c_exp_imp,
                                         hg_term_imp=hg_term_imp)

def calc_hg_bartz(d_throat: float,
                   mu_0: float,
                   c_p_stag: float,
                   pran_0: float,
                   P_c: float, #pylint: disable=C0103 #noqa: N803
                   c_star: float,
                   r_throat_curve: float,
                   temp_wall: FloatOrArray, #pylint: disable=C0103 #noqa: N803
                   temp_0: float,
                   gamma: FloatOrArray,
                   mach: FloatOrArray,
                   inv_area_rat: FloatOrArray,
                   omega: float = _DEFAULT_OMEGA
                   ) -> FloatOrArray:
    """
    Bartz with SI inputs.

    Parameters
    ----------
    d_throat: [m]
        Throat diameter
    mu_0: [Pa*s]
        Stagnation viscosity
    c_p_stag: [J/(kg*K)]
        Stagnation specific heat
    pran_0: [1]
        Stagnation Prandtl number
    P_c: [Pa]
        Chamber pressure
    c_star: [m/s]
        Characteristic velocity
    r_throat_curve: [m]
        Curvature of the Throat radius
    temp_wall: [K]
        Wall temperature
    temp_0: [K]
        Stagnation temperature
    gamma: [1]
        Specific heat ratio (dimensionless)
    mach: [1]
        Mach number (dimensionless)
    inv_area_rat: [1]
        Inverse area ratio (Astar / A)

    Returns
    -------
    hg: [W/(m^2*K)]
        Heat transfer coefficient
    """
    d_throat_in = d_throat / csp.inch # convert from m to in
    mu_0_lbm_in_s = mu_0 / (csp.lb / csp.inch) # convert from Pa*s to lb/(in*s)
    c_p_stag_btu_lbm_R = c_p_stag / (csp.Btu / (csp.lb * csp.degree_Fahrenheit)) #noqa: N806
    P_c_psi = P_c / csp.psi #noqa: N806

    c_star_ft_s = c_star / csp.foot
    r_throat_curve_in = r_throat_curve / csp.inch
    temp_wall_R = temp_wall / csp.degree_Fahrenheit #noqa: N806
    temp_0_R = temp_0 / csp.degree_Fahrenheit #noqa: N806

    hg_btu_in2_s_R = calc_hg_bartz_imp(d_throat_in=d_throat_in, #pylint: disable=C0103 #noqa: N806
                           mu_0_lbm_in_s=mu_0_lbm_in_s,
                           c_p_0_btu_lbm_R=c_p_stag_btu_lbm_R,
                           pran_0=pran_0,
                           P_c_psi=P_c_psi,
                           c_star_ft_s=c_star_ft_s,
                           r_throat_curve_in=r_throat_curve_in,
                           temp_wall_R=temp_wall_R,
                           temp_0_R=temp_0_R,
                           gamma=gamma,
                           mach=mach,
                           inv_area_rat=inv_area_rat,
                           omega=omega,
                           )
    hg = hg_btu_in2_s_R * _BTU_IN2_S_R
    return hg

def bartz_with_temp_variables_imp(temp_wall_R: FloatOrArray, #noqa: N803
                                  sigma_a_mult_imp: FloatOrArray,
                                  sigma_b_mult_imp: FloatOrArray,
                                  sigma_c_exp_imp: FloatOrArray,
                                  hg_term_imp: FloatOrArray
                                  ) -> FloatOrArray:
    """
    Calculate the heat transfer coefficient using the Bartz equation.

    Uses pre-calculated temperature variables.

    Parameters
    ----------
    temp_wall_R: [R]
        Wall temperature
    sigma_a_mult_imp: [1/R]
        0.5 * (1 / Tstag) * (1 + (gamma-1)/2 * Mach**2)
    sigma_b_mult_imp: [1]
        (1 + (gamma-1)/2 * Mach**2) ** (omega/5)
    sigma_c_exp_imp: [1]
        (0.8 - omega/5)
    hg_term_imp: [Btu/(in^2*sec*R)]
        Pre-calculated term for the heat transfer coefficient

    Returns
    -------
    hg: [Btu/(in^2*sec*R)]
        Heat transfer coefficient
    """
    sigma = 1 / ((temp_wall_R * sigma_a_mult_imp + 0.5) ** sigma_c_exp_imp * sigma_b_mult_imp)
    hg = hg_term_imp * sigma
    return hg

def bartz_with_temp_variables(T_wall: FloatOrArray, #pylint: disable=C0103 #noqa: N803
                              bartz_params: BartzParamClass
                              ) -> FloatOrArray:
    """
    Calculate the heat transfer coefficient using the Bartz equation.

    Uses pre-calculated temperature variables.

    Parameters
    ----------
    T_wall: [K]
        Wall temperature
    bartz_params: [BartzParamClass]
        Instance of BartzParamClass containing Bartz parameters

    Returns
    -------
    hg: [W/(m^2*K)]
        Heat transfer coefficient
    """
    hg_imp = bartz_with_temp_variables_imp(temp_wall_R=T_wall / csp.degree_Fahrenheit,
                                           sigma_a_mult_imp=bartz_params.sigma_a_mult,
                                           sigma_b_mult_imp=bartz_params.sigma_b_mult,
                                           sigma_c_exp_imp=bartz_params.sigma_c_exp,
                                           hg_term_imp=bartz_params.hg_term)
    hg = hg_imp * _BTU_IN2_S_R
    return hg

def bartz_gen_temp_variables(d_throat: float,
                             mu_0: float,
                             c_p_0: float,
                             pran_0: float,
                             P_c: float, #pylint: disable=C0103 #noqa: N803
                             c_star: float,
                             r_throat_curve: float,
                             temp_stag: float, #pylint: disable=C0103 #noqa: N803
                             gamma: FloatOrArray,
                             mach: FloatOrArray,
                             inv_area_ratio: FloatOrArray,
                             omega: float = _DEFAULT_OMEGA
                             ) -> tuple[FloatOrArray,
                                        FloatOrArray,
                                        FloatOrArray,
                                        FloatOrArray,
                                        FloatOrArray]:
    """
    Generate the temperature variables for the Bartz equation.

    Parameters
    ----------
    d_throat: [m]
        Throat diameter
    mu_0: [Pa*s]
        Stagnation viscosity
    c_p_0: [J/(kg*K)]
        Stagnation specific heat
    pran_0: [dimensionless]
        Stagnation Prandtl number
    P_c: [Pa]
        Chamber pressure (Pa)
    c_star: [m/s]
        Characteristic velocity (m/s)
    r_throat_curve: [m]
        Curvature of the Nozzle radius (m)
    temp_stag: [K]
        Stagnation temperature (K)
    gamma: [dimensionless]
        Specific heat ratio
    mach: [dimensionless]
        Mach number
    inv_area_ratio: [1]
        Inverse area ratio (Astar / A)
    omega: [1]
        Temperature exponent of viscosity equation

    Returns
    -------
    sigma_a_mult_imp: [1/R]
        First temperature-dependent term
    sigma_b_mult_imp: [1]
        Second temperature-dependent term
    sigma_c_exp_imp : [1]
        Exponent for the third temperature-dependent term
    hg_term_imp: [btu/(in^2*sec*R)]
        Pre-calculated term for the heat transfer coefficient
    temp_recov_scalar: [1]
        Recovery temperature scalar
    """
    d_throat_in = d_throat / csp.inch  # Convert to inches
    mu_0_lbm_in_s = mu_0 / (csp.lb / (csp.inch))  # Convert to lbm/(in*s)
    c_p_0_btu_lbm_R = c_p_0 / (csp.Btu / (csp.lb * csp.degree_Fahrenheit)) #noqa: N806 # Convert to Btu/(lbm*R)
    # pran_0 is dimensionless
    P_c_psi = P_c / csp.psi  # Convert to psi #noqa: N806
    c_star_ft_s = c_star / csp.foot  # Convert to ft/s
    r_throat_curve_in = r_throat_curve / csp.inch  # Convert to inches
    temp_0_R = temp_stag / csp.degree_Fahrenheit  # Convert to R #noqa: N806
    # gamma is dimensionless
    # mach is dimensionless
    # inv_area_ratio is dimensionless

    return bartz_gen_temp_variables_imp(d_throat_in=d_throat_in,
                                        mu_0_lbm_in_s=mu_0_lbm_in_s,
                                        c_p_0_btu_lbm_R=c_p_0_btu_lbm_R,
                                        pran_0=pran_0,
                                        P_c_psi=P_c_psi,
                                        c_star_ft_s=c_star_ft_s,
                                        r_throat_curve_in=r_throat_curve_in,
                                        temp_0_R=temp_0_R,
                                        gamma=gamma,
                                        mach=mach,
                                        inv_area_ratio=inv_area_ratio,
                                        omega=omega)

@overload
def bartz_gen_temp_variables_imp(d_throat_in: float,
                                 mu_0_lbm_in_s: float,
                                    c_p_0_btu_lbm_R: float, #pylint: disable=C0103 #noqa: N803
                                    pran_0: float,
                                    P_c_psi: float, #pylint: disable=C0103 #noqa: N803
                                    c_star_ft_s: float,
                                    r_throat_curve_in: float,
                                    temp_0_R: float, #pylint: disable=C0103 #noqa: N803
                                    gamma: float,
                                    mach: float,
                                    inv_area_ratio: float,
                                    omega: float = ...
                                    ) -> tuple[float,
                                            float,
                                            float,
                                            float,
                                            float]: ...
@overload
def bartz_gen_temp_variables_imp(d_throat_in: float,
                                 mu_0_lbm_in_s: float,
                                    c_p_0_btu_lbm_R: float, #pylint: disable=C0103 #noqa: N803
                                    pran_0: float,
                                    P_c_psi: float, #pylint: disable=C0103 #noqa: N803
                                    c_star_ft_s: float,
                                    r_throat_curve_in: float,
                                    temp_0_R: float, #pylint: disable=C0103 #noqa: N803
                                    gamma: FloatArray,
                                    mach: FloatOrArray,
                                    inv_area_ratio: FloatOrArray,
                                    omega: float = ...
                                    ) -> tuple[FloatArray,
                                            FloatArray,
                                            FloatArray,
                                            FloatArray,
                                            FloatArray]: ...
@overload
def bartz_gen_temp_variables_imp(d_throat_in: float,
                                 mu_0_lbm_in_s: float,
                                    c_p_0_btu_lbm_R: float, #pylint: disable=C0103 #noqa: N803
                                    pran_0: float,
                                    P_c_psi: float, #pylint: disable=C0103 #noqa: N803
                                    c_star_ft_s: float,
                                    r_throat_curve_in: float,
                                    temp_0_R: float, #pylint: disable=C0103 #noqa: N803
                                    gamma: FloatOrArray,
                                    mach: FloatArray,
                                    inv_area_ratio: FloatOrArray,
                                    omega: float = ...
                                    ) -> tuple[FloatArray,
                                            FloatArray,
                                            FloatArray,
                                            FloatArray,
                                            FloatArray]: ...
@overload
def bartz_gen_temp_variables_imp(d_throat_in: float,
                                 mu_0_lbm_in_s: float,
                                    c_p_0_btu_lbm_R: float, #pylint: disable=C0103 #noqa: N803
                                    pran_0: float,
                                    P_c_psi: float, #pylint: disable=C0103 #noqa: N803
                                    c_star_ft_s: float,
                                    r_throat_curve_in: float,
                                    temp_0_R: float, #pylint: disable=C0103 #noqa: N803
                                    gamma: FloatOrArray,
                                    mach: FloatOrArray,
                                    inv_area_ratio: FloatArray,
                                    omega: float = ...
                                    ) -> tuple[FloatArray,
                                            FloatArray,
                                            FloatArray,
                                            FloatArray,
                                            FloatArray]: ...
def bartz_gen_temp_variables_imp(d_throat_in: float,
                                 mu_0_lbm_in_s: float,
                                    c_p_0_btu_lbm_R: float, #pylint: disable=C0103 #noqa: N803
                                    pran_0: float,
                                    P_c_psi: float, #pylint: disable=C0103 #noqa: N803
                                    c_star_ft_s: float,
                                    r_throat_curve_in: float,
                                    temp_0_R: float, #pylint: disable=C0103 #noqa: N803
                                    gamma: FloatOrArray,
                                    mach: FloatOrArray,
                                    inv_area_ratio: FloatOrArray,
                                    omega: float = _DEFAULT_OMEGA
                                    ) -> tuple[FloatOrArray,
                                            FloatOrArray,
                                            FloatOrArray,
                                            FloatOrArray,
                                            FloatOrArray]:
    """
    Generate the temperature variables for the Bartz equation.

    Parameters
    ----------
    d_throat_in: [in]
        Throat diameter
    mu_0_lbm_in_s: [lb/(in*s)]
        Stagnation viscosity
    c_p_0_btu_lbm_R: [Btu/(lb*R)]
        Stagnation specific heat
    pran_0: [dimensionless]
        Stagnation Prandtl number (mu * cp / k)
    P_c_psi: [psi]
        Chamber pressure
    c_star_ft_s: [ft/s]
        Characteristic velocity
    r_throat_curve_in: [in]
        Curvature of the Nozzle radius
    temp_0_R: [R]
        Stagnation temperature
    gamma: [dimensionless]
        Specific heat ratio (Sometimes Cp/Cv)
    mach: [dimensionless]
        Mach number
    inv_area_ratio: [1]
        Inverse area ratio (Astar / A)
    omega: [1]
        Temperature exponent of viscosity equation

    Returns
    -------
    sigma_a_mult_imp: [1/R]
        First temperature-dependent term
    sigma_b_mult_imp: [1]
        Second temperature-dependent term
    sigma_c_exp_imp : [1]
        Exponent for the third temperature-dependent term
    hg_term_imp: [btu/(in^2*sec*R)]
        Pre-calculated term for the heat transfer coefficient
    temp_recov_scalar: [1]
        Temperature recovery scalar
    """
    sigma_a_mult_imp = 0.5 * (1 / temp_0_R) * (1 + (gamma - 1) / 2 * mach ** 2)
    sigma_b_mult_imp = (1 + (gamma - 1) / 2 * mach ** 2) ** (omega / 5)
    sigma_c_exp_imp = 0.8 - omega / 5
    hg_term_imp = ((0.026 / (d_throat_in ** 0.2))
                * ((mu_0_lbm_in_s ** 0.2) * c_p_0_btu_lbm_R / (pran_0 ** 0.6))
                * ((P_c_psi * _G0_FT_S2 / c_star_ft_s) ** 0.8)
                * (d_throat_in / r_throat_curve_in) ** 0.1
                * inv_area_ratio**0.9
    )
    temp_recov_scalar = temp_recovery(temp_inf=1.,
                                      prandtl=pran_0,
                                      gamma=gamma,
                                      mach=mach,
                                      flow_type='turbulent')

    return sigma_a_mult_imp, sigma_b_mult_imp, sigma_c_exp_imp, hg_term_imp, temp_recov_scalar


@overload
def get_mu(temp: float, a: float, b: float) -> float: ...
@overload
def get_mu(temp: FloatArray, a: float, b: float) -> FloatArray: ...
def get_mu(temp: FloatOrArray, a: float, b: float) -> FloatOrArray:
    """Calculate the dynamic viscosity of the fluid at the specified temperature.

    Parameters
    ----------
    temp: [K]
        Temperature of gases

    a: [kg/(m*s*K^b)]
        Coefficient a for the viscosity curve fit.
    b: [1]
        Coefficient b for the viscosity curve fit.

    Returns
    -------
    mu: [kg/m/s]
        Dynamic viscosity
    """
    return a * temp**b

def calc_mu_coeff(temp_array: FloatArray, viscosity_array: FloatArray) -> SimpleNamespace:
    """Calculate the coefficients for the viscosity curve fit.

    Follows the equation set by get_mu, which is mu = a * T^b.

    Parameters
    ----------
    temp_array: [K]
        Array of temperatures.
    viscosity_array: [kg/m/s]
        Array of viscosities.

    Returns
    -------
    mu_coeff: SimpleNamespace
        A SimpleNamespace containing the coefficients 'a' and 'b' for the viscosity curve fit.
    """
    b, log_a = np_polyfit(np_log(temp_array), np_log(viscosity_array), 1)
    mu_coeff = SimpleNamespace(a=float(np_exp(log_a)), b=float(b))

    return mu_coeff

def calc_mu_coeff_from_cea(cea_obj: PropellentMixture, n_pts: int = 100) -> SimpleNamespace:
    """Calculate the coefficients for the viscosity curve fit from CEA results.

    Parameters
    ----------
    cea_obj: PropellentMixture
        An instance of PropellentMixture containing the results from CEA analysis.
    n_pts: int
        Number of points to use for the curve fit.

    Returns
    -------
    mu_coeff: SimpleNamespace
        A SimpleNamespace containing the coefficients 'a' and 'b' for the viscosity curve fit.
    """
    raise NotImplementedError("This function may have errors, and thus is marked as invalid.")
    cea_obj = deepcopy(cea_obj)
    p_c = cea_obj.solution.chamber_pressure

    enthalpy_limits = array([cea_obj.solution.enthalpy[-1], cea_obj.solution.enthalpy[0]])

    mu_stag_arr = empty([n_pts])
    temp_arr = empty([n_pts])
    for idx, enthalpy in enumerate(linspace(enthalpy_limits[0], enthalpy_limits[1], n_pts)):
        cea_obj.solve(p_c=p_c, h_chamb_kj_kg=enthalpy)
        mu_stag_arr[idx] = cea_obj.solution.mu[0]
        temp_arr[idx] = cea_obj.solution.temperature[0]

    return calc_mu_coeff(temp_arr, mu_stag_arr)
