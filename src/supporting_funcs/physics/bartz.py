""" Bartz equation parameters for heat transfer calculations. """
import scipy.constants as csp
from numpy import ndarray
from CEA_Wrap import Output as CEAOutput

G0 = csp.g  # [m/s^2] standard gravity
G0_FT = G0 / csp.foot  # [ft/s^2] standard gravity

class BartzParamClass:
    """Parameters for the Bartz equation, generated from CEA results and nozzle geometry."""
    def __init__(self, d_throat: float,
                 mu_0: float,
                 c_p_0: float,
                 pran_0: float,
                 Pc: float, #pylint: disable=C0103
                 c_star: float,
                 r_throat_curve: float,
                 T_0: float, #pylint: disable=C0103
                 gamma: ndarray | float,
                 mach: ndarray | float,
                 area_rat: ndarray | float,
                 omega=0.6):
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
            Astar/A
        omega: [1]
            Temperature exponent of viscosity equation
        """
        self.sigma_a_mult, self.sigma_b_mult, self.sigma_c_exp, self.hg_term, self.Tr_scalar = bartz_gen_temp_variables( #pylint: disable=C0103, C0301
            d_throat=d_throat,
            mu_0=mu_0,
            c_p_0=c_p_0,
            pran_0=pran_0,
            P_c=Pc,
            c_star=c_star,
            r_throat_curve=r_throat_curve,
            T_0=T_0,
            gamma=gamma,
            mach=mach,
            inv_area_ratio=1.0 / area_rat,
            omega=omega
        )

    def bartz_with_temp_variables(self,
                                  T_wall: ndarray | float #pylint: disable=C0103
                                  ) -> ndarray | float:
        """
        Calculate the heat transfer coefficient using the Bartz equation.
        Uses pre-calculated temperature variables.

        Parameters
        ----------
        T_wall: [K]
            Wall temperature

        Returns
        -------
        hg: [W/(m^2*K)]
            Heat transfer coefficient
        """
        hg_imp = bartz_with_temp_variables_imp(T_wall_R=T_wall / csp.degree_Fahrenheit,
                                               sigma_a_mult_imp=self.sigma_a_mult,
                                               sigma_b_mult_imp=self.sigma_b_mult,
                                               sigma_c_exp_imp=self.sigma_c_exp,
                                               hg_term_imp=self.hg_term)
        hg = hg_imp * (
            csp.Btu / (csp.inch**2 * csp.degree_Fahrenheit)) # convert from Btu/(in^2*sec*R)
        return hg

    def bartz_with_temp_variables_imp(self,
                                      T_wall_R: ndarray | float #pylint: disable=C0103
                                      ) -> ndarray | float:
        """
        Calculate the heat transfer coefficient using the Bartz equation
        with pre-calculated temperature variables.

        Parameters
        ----------
        T_wall_R: [R]
            Wall temperature

        Returns
        -------
        hg: [Btu/(in^2*sec*R)]
            Heat transfer coefficient
        """
        return bartz_with_temp_variables_imp(T_wall_R=T_wall_R,
                                            sigma_a_mult_imp=self.sigma_a_mult,
                                            sigma_b_mult_imp=self.sigma_b_mult,
                                            sigma_c_exp_imp=self.sigma_c_exp,
                                            hg_term_imp=self.hg_term)

def create_bartz_param_class(cea_result: CEAOutput,
                             d_throat: float,
                             r_throat_curve: float,
                             mach: float | ndarray = 1.0,
                             area_rat: float | ndarray = 1.0) -> BartzParamClass:
    """
    Create an instance of BartzParamClass using CEA results and nozzle geometry.

    Parameters
    ----------
    ceaResult: [-]
        An instance of CEAOutput containing the results from CEA analysis.
    d_throat: [m]
        Throat diameter
    r_throat_curve: [m]
        Curvature of the throat radius
    mach: [1]
        Mach number at location to report (default is 1.0)
    area_rat: 
        area ratio at the throat (default is 1.0)

    Returns
    -------
    bartz_params: bartBartzParamClass
        An instance of BartzParamClass with the calculated parameters.
    """
    bartz_params = BartzParamClass(
        d_throat = d_throat,
        mu_0 = cea_result.c_visc,
        c_p_0 = cea_result.c_cp * csp.kilo, # Convert from kJ/kg-K to J/kg-K
        pran_0 = cea_result.c_pran,
        Pc = cea_result.c_p * csp.bar,
        c_star = cea_result.cstar,
        r_throat_curve = r_throat_curve,
        T_0 = cea_result.c_t,
        gamma = cea_result.c_gamma,
        mach = mach,
        area_rat = area_rat,
        )
    return bartz_params

def calc_hg_bartz_imp(d_throat_in: float | ndarray,
                  mu_0_psi_s: float | ndarray,
                  c_p_0_btu_lbm_R: float | ndarray, #pylint: disable=C0103
                  pran_0: float | ndarray,
                  P_c_psi: float | ndarray, #pylint: disable=C0103
                  c_star_ft_s: float | ndarray,
                  r_throat_curve_in: float | ndarray,
                  T_wall_R: float | ndarray, #pylint: disable=C0103
                  T_0_R: float | ndarray, #pylint: disable=C0103
                  gamma: float | ndarray,
                  mach: float | ndarray,
                  inv_area_rat: float | ndarray,
                  omega: float = 0.6
                  ) -> float | ndarray:
    """
    Calculate the heat transfer coefficient using the Bartz equation.
    https://ntrs.nasa.gov/api/citations/19710011726/downloads/19710011726.pdf

    Parameters
    ----------
        d_throat_in: [in]
            Throat diameter
        mu_0_psi_s: [lb/(in*s)]
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
        T_wall_R: [R]
            Wall temperature (R)
        T_0_R: [R]
            Stagnation temperature (R)
        gamma: [1]
            Specific heat ratio
        mach: [1]
            Mach number
        omega: [1]
            Temperature exponent of viscosity equation
    
    Returns
    -------
        hg_btu_in2_s_R: Heat transfer coefficient (Btu/(in^2*sec*R))
    """

    sigma = 1 / (
        (0.5 * (T_wall_R / T_0_R) * (1 + (gamma - 1) / 2 * mach ** 2) + 0.5) ** (0.8 - omega / 5)
        * (1 + (gamma - 1) / 2 * mach ** 2) ** (omega / 5)
    )

    return (
            (0.026 / (d_throat_in ** 0.2))
            * ((mu_0_psi_s ** 0.2) * c_p_0_btu_lbm_R / (pran_0 ** 0.6))
            * ((P_c_psi * G0_FT / c_star_ft_s) ** 0.8)
            * (d_throat_in / r_throat_curve_in) ** 0.1
            ) * inv_area_rat**0.9 * sigma

def bartz_recovery_temp(T_0: float | ndarray, #pylint: disable=C0103
                        pran_0: float | ndarray,
                        gamma: float | ndarray,
                        mach: float | ndarray
                        ) -> ndarray | float:
    """
    Calculate the recovery temperature using the given parameters.
    https://ntrs.nasa.gov/api/citations/19710011726/downloads/19710011726.pdf

    Parameters
    ----------
    T_0: [K]
        Stagnation temperature
    pran_0: [1]
        Stagnation Prandtl number
    gamma: [1]
        Specific heat ratio
    mach: [1]
        Mach number

    Returns
    -------
        T_recovery: [K]
            Recovery temperature
    """
    r_factor = pran_0 ** 0.5
    return T_0 * (1 + r_factor * (gamma - 1) / 2 * mach ** 2) / (1 + (gamma - 1) / 2 * mach ** 2)

def calc_hg_bartz(d_throat: float | ndarray,
                   mu_0: float | ndarray,
                   c_p_stag: float | ndarray,
                   pran_0: float | ndarray,
                   P_c: float | ndarray, #pylint: disable=C0103
                   c_star: float | ndarray,
                   r_throat_curve: float | ndarray,
                   T_wall: float | ndarray, #pylint: disable=C0103
                   T_0: float | ndarray, #pylint: disable=C0103
                   gamma: float | ndarray,
                   mach: float | ndarray,
                   inv_area_rat: float | ndarray):
    """    
    Bartz with SI inputs
    
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
    T_wall: [K]
        Wall temperature
    T_0: [K]
        Stagnation temperature
    gamma: [1]
        Specific heat ratio (dimensionless)
    mach: [1]
        Mach number (dimensionless)

    Returns:
    hg: [W/(m^2*K)]
        Heat transfer coefficient
    """

    d_throat = d_throat / csp.inch # convert from m to in
    mu_0 = mu_0 / csp.psi # convert from Pa*s to lbf/(in*s)
    c_p_stag = c_p_stag / (csp.Btu / (csp.lb * csp.degree_Fahrenheit))
    P_c = P_c / csp.psi

    c_star = c_star / csp.foot
    r_throat_curve = r_throat_curve / csp.inch
    T_wall = T_wall / csp.degree_Fahrenheit
    T_0 = T_0 / csp.degree_Fahrenheit

    hg_btu_in2_s_R = calc_hg_bartz_imp(d_throat_in=d_throat, #pylint: disable=C0103
                           mu_0_psi_s=mu_0,
                           c_p_0_btu_lbm_R=c_p_stag,
                           pran_0=pran_0,
                           P_c_psi=P_c,
                           c_star_ft_s=c_star,
                           r_throat_curve_in=r_throat_curve,
                           T_wall_R=T_wall,
                           T_0_R=T_0,
                           gamma=gamma,
                           mach=mach,
                           inv_area_rat=inv_area_rat)
    hg = hg_btu_in2_s_R * (
        csp.Btu / (csp.inch**2 * csp.degree_Fahrenheit))
    return hg

def bartz_with_temp_variables_imp(T_wall_R: ndarray | float, #pylint: disable=C0103
                                  sigma_a_mult_imp: ndarray | float,
                                  sigma_b_mult_imp: ndarray | float,
                                  sigma_c_exp_imp: ndarray | float,
                                  hg_term_imp: ndarray | float
                                  ) -> ndarray | float:
    """
    Calculate the heat transfer coefficient using the Bartz equation
    with pre-calculated temperature variables.

    Parameters
    ----------
    T_wall: [R]
        Wall temperature
    sigma_a_mult_imp: [1/R]
        0.5 * (1 / Tstag) * (1 + gamma-1)/2 * Mach**2
    sigma_b_mult_imp: [1]
        (1 + (gamma-1)/2 * Mach**2) ** (omega/5)
    sigma_c_exp_imp: [1]
        (0.8 - omega/5)
    hg_term_imp: [Btu/(in^2*sec*R)]
        Pre-calculated term for the heat transfer coefficient

    Returns:
        hg: [Btu/(in^2*sec*R)]
            Heat transfer coefficient
    """
    sigma = 1 / ((T_wall_R * sigma_a_mult_imp + 0.5) ** sigma_c_exp_imp * sigma_b_mult_imp)
    hg = hg_term_imp * sigma
    return hg

def bartz_with_temp_variables(T_wall: ndarray | float, #pylint: disable=C0103
                              bartz_params: BartzParamClass
                              ) -> ndarray | float:
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
    hg_imp = bartz_with_temp_variables_imp(T_wall_R=T_wall / csp.degree_Fahrenheit,
                                           sigma_a_mult_imp=bartz_params.sigma_a_mult,
                                           sigma_b_mult_imp=bartz_params.sigma_b_mult,
                                           sigma_c_exp_imp=bartz_params.sigma_c_exp,
                                           hg_term_imp=bartz_params.hg_term)
    hg = hg_imp * (csp.Btu / (csp.inch**2 * csp.degree_Fahrenheit)) # convert from Btu/(in^2*sec*R)
    return hg

def bartz_gen_temp_variables(d_throat: float,
                             mu_0: float,
                             c_p_0: float,
                             pran_0: float,
                             P_c: float, #pylint: disable=C0103
                             c_star: float,
                             r_throat_curve: float,
                             T_0: float, #pylint: disable=C0103
                             gamma: ndarray | float,
                             mach: ndarray | float,
                             inv_area_ratio: ndarray | float, omega: ndarray | float = 0.6):
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
        T_0: [K]
            Stagnation temperature (K)
        gamma: [dimensionless]
            Specific heat ratio
        mach: [dimensionless]
            Mach number
        inv_area_ratio: [1]
            Inverse area ratio (Astar / A)
    
    Returns
    -------
    sigma_a_mult_imp: [1/R]
        First temperature-dependent term
    sigma_b_mult_imp: [1]
        Second temperature-dependent term
    sigma_c_exp_imp : [1]
        Exponent for the third temperature-dependent term
    hg_term_imp: [?]
        Pre-calculated term for the heat transfer coefficient
    Tr_scalar: [1]
        Recovery temperature scalar
    """

    d_throat = d_throat / csp.inch  # Convert to inches
    mu_0 = mu_0 / (csp.lbf / (csp.inch**2))  # Convert to lbf/(in*s)
    c_p_0 = c_p_0 / (csp.Btu / (csp.lb * csp.degree_Fahrenheit))  # Convert to Btu/(lbm*R)
    # pran_0 is dimensionless
    P_c = P_c / csp.psi  # Convert to psi
    c_star = c_star / csp.foot  # Convert to ft/s
    r_throat_curve = r_throat_curve / csp.inch  # Convert to inches
    T_0 = T_0 / csp.degree_Fahrenheit  # Convert to R
    # gamma is dimensionless
    # mach is dimensionless
    # inv_area_ratio is dimensionless

    sigma_a_mult_imp = 0.5 * (1 / T_0) * (1 + (gamma - 1) / 2 * mach ** 2)
    sigma_b_mult_imp = (1 + (gamma - 1) / 2 * mach ** 2) ** (omega / 5)
    sigma_c_exp_imp = 0.8 - omega / 5
    hg_term_imp = ((0.026 / (d_throat ** 0.2))
               * ((mu_0 ** 0.2) * c_p_0 / (pran_0 ** 0.6))
               * ((P_c * G0_FT / c_star) ** 0.8)
               * (d_throat / r_throat_curve) ** 0.1
               * (inv_area_ratio)**0.9
    )
    Tr_scalar = 1 + (gamma - 1) / 2 * pran_0 * mach ** 2 #pylint: disable=C0103

    return sigma_a_mult_imp, sigma_b_mult_imp, sigma_c_exp_imp, hg_term_imp, Tr_scalar
