"""# compressible_flow
Equations and functions for compressible flow calculations.
Includes relationships for:
- isentropic flow
- normal shock
- oblique shock
- expansion fan"""
from typing import Literal, Never, Optional, TypeAlias, overload

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
def Psi(gamma: float, mach: float) -> float: ... #pylint: disable=C0103
@overload
def Psi(gamma: FloatArray, mach: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def Psi(gamma: FloatOrArray, mach: FloatArray) -> FloatArray: ... #pylint: disable=C0103

@overload
def P_ratio_from_mach(mach: float, gamma: float) -> float: ... #pylint: disable=C0103
@overload
def P_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def P_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ... #pylint: disable=C0103

@overload
def mach_from_P_ratio(p_rat: float, gamma: float) -> float: ... #pylint: disable=C0103
@overload
def mach_from_P_ratio(p_rat: FloatArray, gamma: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def mach_from_P_ratio(p_rat: FloatOrArray, gamma: FloatArray) -> FloatArray: ... #pylint: disable=C0103

@overload
def T_ratio_from_mach(mach: float, gamma: float) -> float: ... #pylint: disable=C0103
@overload
def T_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def T_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ... #pylint: disable=C0103

@overload
def h_ratio_from_mach(mach: float, gamma: float) -> float: ... #pylint: disable=C0103
@overload
def h_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def h_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ... #pylint: disable=C0103

@overload
def rho_ratio_from_mach(mach: float, gamma: float) -> float: ...
@overload
def rho_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def rho_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...

@overload
def area_ratio_from_mach(mach: float, gamma: float) -> float: ...
@overload
def area_ratio_from_mach(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def area_ratio_from_mach(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...

@overload
def mach_from_stagnation(T0: float, #pylint: disable=C0103 #noqa N803
                         P0: float, #pylint: disable=C0103 #noqa N803
                         mdot: float,
                         gamma: float,
                         R: float, #pylint: disable=C0103 #noqa N803
                         area: float,
                         supersonic: bool = ...) -> float: ...
@overload
def mach_from_stagnation(T0: FloatArray, #pylint: disable=C0103 #noqa N803
                         P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         P0: FloatArray, #pylint: disable=C0103 #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         mdot: FloatArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatArray,
                         R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatArray, #pylint: disable=C0103 #noqa N803
                         area: FloatOrArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...
@overload
def mach_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         mdot: FloatOrArray,
                         gamma: FloatOrArray,
                         R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                         area: FloatArray,
                         supersonic: BoolOrArray = ...) -> FloatArray: ...

@overload
def m_flux_from_stagnation(T0: float, #pylint: disable=C0103 #noqa N803
                           P0: float, #pylint: disable=C0103 #noqa N803
                           gamma: float,
                           R: float, #pylint: disable=C0103 #noqa N803
                           mach: float
                           ) -> float: ...
@overload
def m_flux_from_stagnation(T0: FloatArray, #pylint: disable=C0103 #noqa N803
                           P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           gamma: FloatOrArray,
                           R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           P0: FloatArray, #pylint: disable=C0103 #noqa N803
                           gamma: FloatOrArray,
                           R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           gamma: FloatArray,
                           R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           gamma: FloatOrArray,
                           R: FloatArray, #pylint: disable=C0103 #noqa N803
                           mach: FloatOrArray
                           ) -> FloatArray: ...
@overload
def m_flux_from_stagnation(T0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           P0: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           gamma: FloatOrArray,
                           R: FloatOrArray, #pylint: disable=C0103 #noqa N803
                           mach: FloatArray
                           ) -> FloatArray: ...

@overload
def _D_func(mach: float, gamma: float) -> float: ... #pylint: disable=C0103
@overload
def _D_func(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def _D_func(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ... #pylint: disable=C0103

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

@overload
def T_recovery(temp_inf: float, #pylint: disable=C0103
               prandtl: float,
               gamma: float,
               mach: float,
               flow_type: Literal['turbulent'] = ...
               ) -> float: ...
@overload
def T_recovery(temp_inf: FloatArray, #pylint: disable=C0103
               prandtl: FloatOrArray,
               gamma: FloatOrArray,
               mach: FloatOrArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...
@overload
def T_recovery(temp_inf: FloatOrArray, #pylint: disable=C0103
               prandtl: FloatArray,
               gamma: FloatOrArray,
               mach: FloatOrArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...
@overload
def T_recovery(temp_inf: FloatOrArray, #pylint: disable=C0103
               prandtl: FloatOrArray,
               gamma: FloatArray,
               mach: FloatOrArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...
@overload
def T_recovery(temp_inf: FloatOrArray, #pylint: disable=C0103
               prandtl: FloatOrArray,
               gamma: FloatOrArray,
               mach: FloatArray,
               flow_type: Literal['turbulent'] = ...
               ) -> FloatArray: ...

#region Expansion Fan Relations
@overload
def prandtl_meyer(mach: float, gamma: float) -> float: ...
@overload
def prandtl_meyer(mach: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def prandtl_meyer(mach: FloatOrArray, gamma: FloatArray) -> FloatArray: ...

@overload
def max_prandtl_meyer(gamma: float) -> float: ...
@overload
def max_prandtl_meyer(gamma: FloatArray) -> FloatArray: ...

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

@overload
def expansion_fan_max_turning_angle(mach_1: float, gamma: float) -> float: ...
@overload
def expansion_fan_max_turning_angle(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def expansion_fan_max_turning_angle(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ...
#endregion Expansion Fan Relations
#endregion Isentropic Flow Relations

@overload
def normal_shock_P_ratio(mach_1: float, gamma: float) -> float: ... #pylint: disable=C0103
@overload
def normal_shock_P_ratio(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def normal_shock_P_ratio(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ... #pylint: disable=C0103

@overload
def normal_shock_mach_2(mach_1: float, gamma: float) -> float: ...
@overload
def normal_shock_mach_2(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ...
@overload
def normal_shock_mach_2(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ...

@overload
def normal_shock_P_tot_ratio(mach_1: float, gamma: float) -> float: ... #pylint: disable=C0103
@overload
def normal_shock_P_tot_ratio(mach_1: FloatArray, gamma: FloatOrArray) -> FloatArray: ... #pylint: disable=C0103
@overload
def normal_shock_P_tot_ratio(mach_1: FloatOrArray, gamma: FloatArray) -> FloatArray: ... #pylint: disable=C0103

@overload
def normal_shock(mach_1: float, gamma: float) -> tuple[float, float]: ...
@overload
def normal_shock(mach_1: FloatArray, gamma: FloatOrArray) -> tuple[FloatArray, FloatArray]: ...
@overload
def normal_shock(mach_1: FloatOrArray, gamma: FloatArray) -> tuple[FloatArray, FloatArray]: ...
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

@overload
def oblique_shock_max_angle(mach_1: float, gamma: float) -> float: ...
@overload
def oblique_shock_max_angle(mach_1: FloatArray, gamma: FloatArray | float) -> FloatArray: ...
@overload
def oblique_shock_max_angle(mach_1: FloatArray | float, gamma: FloatArray) -> FloatArray: ...

@overload
def oblique_shock_max_deflection(mach_1: float, gamma: float) -> float: ...
@overload
def oblique_shock_max_deflection(mach_1: FloatArray, gamma: FloatArray | float) -> FloatArray: ...
@overload
def oblique_shock_max_deflection(mach_1: FloatArray | float, gamma: FloatArray) -> FloatArray: ...

#endregion Oblique Shock Equations
#region Nozzle Flow Equations
@overload
def get_cfv(gamma: float,
            Ae_At: float, #pylint: disable=C0103 #noqa N803
            Pe_Pc: Optional[float] = ... #pylint: disable=C0103 #noqa N803
            ) -> float: ...
@overload
def get_cfv(gamma: FloatArray,
            Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
            Pe_Pc: Optional[FloatOrArray] = ... #pylint: disable=C0103 #noqa N803
            ) -> FloatArray: ...
@overload
def get_cfv(gamma: FloatOrArray,
            Ae_At: FloatArray, #pylint: disable=C0103 #noqa N803
            Pe_Pc: Optional[FloatOrArray] = ... #pylint: disable=C0103 #noqa N803
            ) -> FloatArray: ...
@overload
def get_cfv(gamma: FloatOrArray,
            Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
            Pe_Pc: FloatArray #pylint: disable=C0103 #noqa N803
            ) -> FloatArray: ...

@overload
def get_cf_from_cfv(cfv: float,
                    Ae_At: float, #pylint: disable=C0103 #noqa N803
                    Pc: float, #pylint: disable=C0103 #noqa N803
                    Pb: float #pylint: disable=C0103 #noqa N803
                    ) -> float: ...
@overload
def get_cf_from_cfv(cfv: FloatArray,
                    Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
                    Pc: FloatOrArray, #pylint: disable=C0103 #noqa N803
                    Pb: FloatOrArray #pylint: disable=C0103 #noqa N803
                    ) -> FloatArray: ...
@overload
def get_cf_from_cfv(cfv: FloatOrArray,
                    Ae_At: FloatArray, #pylint: disable=C0103 #noqa N803
                    Pc: FloatOrArray, #pylint: disable=C0103 #noqa N803
                    Pb: FloatOrArray #pylint: disable=C0103 #noqa N803
                    ) -> FloatArray: ...
@overload
def get_cf_from_cfv(cfv: FloatOrArray,
                    Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
                    Pc: FloatArray, #pylint: disable=C0103 #noqa N803
                    Pb: FloatOrArray #pylint: disable=C0103 #noqa N803
                    ) -> FloatArray: ...
@overload
def get_cf_from_cfv(cfv: FloatOrArray,
                    Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
                    Pc: FloatOrArray, #pylint: disable=C0103 #noqa N803
                    Pb: FloatArray #pylint: disable=C0103 #noqa N803
                    ) -> FloatArray: ...

@overload
def get_cf(gamma: float,
           Ae_At: float, #pylint: disable=C0103 #noqa N803
           Pc: float, #pylint: disable=C0103 #noqa N803
           Pb: float, #pylint: disable=C0103 #noqa N803
           Pe_Pc: Optional[float] = ... #pylint: disable=C0103 #noqa N803
           ) -> float: ...
@overload
def get_cf(gamma: FloatArray,
           Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pc: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pb: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pe_Pc: Optional[FloatOrArray]= ... #pylint: disable=C0103 #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           Ae_At: FloatArray, #pylint: disable=C0103 #noqa N803
           Pc: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pb: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pe_Pc: Optional[FloatOrArray]= ... #pylint: disable=C0103 #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pc: FloatArray, #pylint: disable=C0103 #noqa N803
           Pb: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pe_Pc: Optional[FloatOrArray]= ... #pylint: disable=C0103 #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pc: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pb: FloatArray, #pylint: disable=C0103 #noqa N803
           Pe_Pc: Optional[FloatOrArray]= ... #pylint: disable=C0103 #noqa N803
           ) -> FloatArray: ...
@overload
def get_cf(gamma: FloatOrArray,
           Ae_At: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pc: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pb: FloatOrArray, #pylint: disable=C0103 #noqa N803
           Pe_Pc: FloatArray #pylint: disable=C0103 #noqa N803
           ) -> FloatArray: ...
#endregion Rocket Nozzle Flow
