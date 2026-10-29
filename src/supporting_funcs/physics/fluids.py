"""Functions for modeling fluid behavior."""
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import overload

import numpy as np
from CoolProp import constants as CPc
from CoolProp.CoolProp import AbstractState, DmassUmass_INPUTS
from scipy.constants import bar as csp_bar
from scipy.constants import hour as csp_hour
from scipy.optimize import root_scalar

from supporting_funcs.coding.type_hints import FloatArray, FloatOrArray

CV_TO_KV = 0.865  # Conversion factor from Cv to Kv (GPM to m^3/hr - IEC 60534)
RHO_H2O_IEC_60534 = 1000 # kg/m^3 (IEC 60534) #noqa: N806

class LiquidClass:
    """A class to represent a liquid with specific properties."""

    def __init__(self, name: str,
                 density: float,
                 specific_heat: float,
                 start_temp: float,
                 ) -> None:
        """Initialize a LiquidClass instance.

        Parameters
        ----------
        name: [str]
            The name of the liquid.
        density: [float]
            The density of the liquid in kg/m^3.
        specific_heat: [float]
            The specific heat capacity of the liquid in J/(kg*K).
        """
        self.name = name
        self.density = density
        self.specific_heat = specific_heat
        self.temp = start_temp

    @property
    def reference_temp(self) -> float:
        """Get the reference temperature of the liquid [K]."""
        return 0. # Absolute zero is the reference temperature for internal energy calculations.
    @property
    def specific_heat(self) -> float:
        """Get the specific heat capacity of the liquid [J/(kg*K)]."""
        return self._specific_heat
    @specific_heat.setter
    def specific_heat(self, value: float) -> None:
        """Set the specific heat capacity of the liquid [J/(kg*K)]."""
        if value <= 0:
            raise ValueError("Specific heat must be positive.")
        self._specific_heat = value
    c_p = specific_heat

    @property
    def internal_energy(self) -> float:
        """Get the internal energy of the liquid [J/(kg*K)]."""
        return self._internal_energy
    @internal_energy.setter
    def internal_energy(self, value: float) -> None:
        """Set the internal energy of the liquid [J/(kg*K)]."""
        if value < 0:
            raise ValueError("Internal energy cannot be negative.")
        self._internal_energy = value
    u = internal_energy

    @property
    def density(self) -> float:
        """Get the density of the liquid [kg/m^3]."""
        return self._density
    @density.setter
    def density(self, value: float) -> None:
        """Set the density of the liquid [kg/m^3]."""
        if value <= 0:
            raise ValueError("Density must be positive.")
        self._density = value

    @property
    def temp(self) -> float:
        """Get the current temperature of the liquid [K]."""
        return self.internal_energy / self.specific_heat + self.reference_temp
    @temp.setter
    def temp(self, value: float) -> None:
        """Set the current temperature of the liquid [K]."""
        if value < 0.:
            raise ValueError("Temperature cannot be below absolute zero.")
        self.internal_energy = (value - self.reference_temp) * self.specific_heat

    def __repr__(self) -> str:
        """Return a string representation of the LiquidClass instance."""
        return (f"LiquidClass(name={self.name}, "
                f"density={self.density}, "
                f"specific_heat={self.specific_heat}, "
                f"temp={self.temp})")

    def update_u(self, internal_energy: float) -> None:
        """Update the internal energy of the liquid.

        Parameters
        ----------
        internal_energy: [J/(kg*K)]
            The internal energy of the liquid.
        """
        self.internal_energy = internal_energy

# region Steady-State Functions
@dataclass
class CopvSizing:
    """Result of an adiabatic COPV blowdown sizing.

    Parameters
    ----------
    copv_volume: [m^3]
        Required COPV internal volume
    mass_copv_start: [kg]
        Pressurant mass in COPV at start
    mass_copv_end: [kg]
        Pressurant mass remaining in COPV at end
    mass_ullage_start: [kg]
        Pressurant mass in tank ullage at start
    mass_ullage_end: [kg]
        Pressurant mass in tank ullage at end
    temp_copv_end: [K]
        COPV gas temperature at end (isentropic expansion)
    temp_ullage_end: [K]
        Tank ullage gas temperature at end
    """

    copv_volume: float
    mass_copv_start: float
    mass_copv_end: float
    mass_ullage_start: float
    mass_ullage_end: float
    temp_copv_end: float
    temp_ullage_end: float
    rho_copv_start: float

    @property
    def mass_start(self) -> float:
        """Total pressurant mass at start."""
        return self.mass_copv_start + self.mass_ullage_start

def size_copv_adiabatic(p1_copv: float, p2_copv: float, temp_start: float,
                        p_tank_start: float,
                        p_tank: float, liquid_volume: float,
                        ullage_volume_start: float,
                        fluid: str = "Helium") -> CopvSizing:
    """Solve for the COPV volume.

    After expelling all liquid at
    constant tank pressure, the COPV ends at p2_copv. The system (COPV gas +
    tank ullage gas) is adiabatic; gas remaining in the COPV expands
    isentropically, and the tank ullage ends at a single mixed state.

    Parameters
    ----------
    p1_copv: [Pa]
        COPV initial pressure
    p2_copv: [Pa]
        COPV final pressure (regulator minimum inlet pressure)
    temp_start: [K]
        Initial temperature of COPV gas and tank ullage gas
    p_tank_start: [Pa]
        The initial pressure of the tank prior to the COPV press
    p_tank: [Pa]
        Regulated tank pressure, constant over the blowdown
    liquid_volume: [m^3]
        Liquid volume expelled from the tank
    ullage_volume_start: [m^3]
        Tank ullage volume before blowdown
    fluid: [-]
        CoolProp fluid name for the pressurant

    Returns
    -------
    sizing: CopvSizing
        CopvSizing with COPV volume, masses, and end temperatures
    """
    state = AbstractState("HEOS", fluid)

    # COPV start state
    state.update(CPc.PT_INPUTS, p1_copv, temp_start)
    rho_copv_start = state.rhomass()
    u_copv_start = state.umass()
    s_copv_start = state.smass()

    # COPV end state: isentropic expansion of the gas that stays behind
    state.update(CPc.PSmass_INPUTS, p2_copv, s_copv_start)
    rho_copv_end = state.rhomass()
    u_copv_end = state.umass()
    temp_copv_end = state.T()

    # Tank ullage start state
    state.update(CPc.PT_INPUTS, p_tank_start, temp_start)
    mass_ullage_start = state.rhomass() * ullage_volume_start
    u_ullage_start = state.umass()

    ullage_volume_end = ullage_volume_start + liquid_volume
    work_out = p_tank * liquid_volume  # boundary work on the liquid [J]

    def _ullage_end(copv_volume: float) -> tuple[float, float, float]:
        """Ullage end mass, density, and specific internal energy.

        Calculated from the adiabatic energy balance of the system (COPV + tank ullage).

        Parameters
        ----------
        copv_volume: [m^3]
            Trial COPV volume

        Returns
        -------
        mass_ullage_end: [kg]
            Ullage mass at end
        rho_ullage_end: [kg/m^3]
            Ullage density at end
        u_ullage_end: [J/kg]
            Ullage specific internal energy required by energy balance
        """
        mass_copv_start = rho_copv_start * copv_volume
        mass_copv_end = rho_copv_end * copv_volume
        mass_ullage_end = mass_ullage_start + mass_copv_start - mass_copv_end

        energy_start = (mass_copv_start * u_copv_start
                        + mass_ullage_start * u_ullage_start)
        energy_end = energy_start - work_out
        u_ullage_end = ((energy_end - mass_copv_end * u_copv_end)
                        / mass_ullage_end)
        return (mass_ullage_end, mass_ullage_end / ullage_volume_end,
                u_ullage_end)

    def _err(copv_volume: float) -> float:
        """Specific internal energy mismatch of the ullage end state.

        Parameters
        ----------
        copv_volume: [m^3]
            Trial COPV volume

        Returns
        -------
        err: [J/kg]
            u from energy balance minus u(rho, p_tank) from EOS
        """
        _, rho_ullage_end, u_required = _ullage_end(copv_volume)
        state.update(CPc.DmassP_INPUTS, rho_ullage_end, p_tank)
        return u_required - state.umass()

    v_lo = 1e-3 * liquid_volume
    v_hi = liquid_volume
    try:
        if _err(v_lo) > 0.0:
            raise RuntimeError("No COPV volume satisfies the end condition; "
                               "check p1/p2/p_tank.")
    except ValueError as e:
        while v_lo < liquid_volume:
            v_lo *= 10.0
            try:
                if _err(v_lo) < 0.0:
                    break
            except ValueError:
                continue
        else:
            raise RuntimeError("No COPV volume satisfies the end condition; "
                               "check p1/p2/p_tank.") from e

    while _err(v_hi) < 0.0:
        v_hi *= 10.0
        if v_hi > 1e4 * liquid_volume:
            raise RuntimeError("No COPV volume satisfies the end condition; "
                               "check p1/p2/p_tank.")

    copv_volume_soln = root_scalar(f=_err,
                                   method='brentq',
                                   bracket=(v_lo, v_hi),
                                   xtol=1e-12,
                                   rtol=1e-12)

    if not copv_volume_soln.converged:
        raise RuntimeError("No COPV volume satisfies the end condition; "
                           "check p1/p2/p_tank.")
    copv_volume = copv_volume_soln.root
    mass_ullage_end, rho_ullage_end, _ = _ullage_end(copv_volume)
    state.update(CPc.DmassP_INPUTS, rho_ullage_end, p_tank)

    return CopvSizing(
        copv_volume=copv_volume,
        mass_copv_start=rho_copv_start * copv_volume,
        mass_copv_end=rho_copv_end * copv_volume,
        mass_ullage_start=mass_ullage_start,
        mass_ullage_end=mass_ullage_end,
        temp_copv_end=temp_copv_end,
        temp_ullage_end=state.T(),
        rho_copv_start=rho_copv_start,
    )

def kv_to_cda(kv: float) -> float:
    """Convert a flow coefficient to an effective area.

    Parameters
    ----------
    kv: [m^3/(hr bar^.5)]
        The flow coefficient.

    Returns
    -------
    cda: [m^2]
        The effective area.
    """
    kv_si = kv / (csp_hour * np.sqrt(csp_bar))
    return kv_si * np.sqrt(RHO_H2O_IEC_60534 / 2)

def cda_to_kv(cda: float) -> float:
    """Convert an effective area to a flow coefficient.

    Parameters
    ----------
    cda: [m^2]
        The effective area.

    Returns
    -------
    kv: [m^3/(hr bar^.5)]
        The flow coefficient.
    """
    cda_si = cda / np.sqrt(RHO_H2O_IEC_60534 / 2)
    return cda_si * (csp_hour * np.sqrt(csp_bar))

def cda_from_dp(mdot: float, rho: float, dp: float) -> float:
    """Calculate the effective area from a mass flow rate, density, and pressure drop.

    Parameters
    ----------
    mdot: [kg/s]
        The mass flow rate.
    rho: [kg/m^3]
        The fluid density.
    dp: [Pa]
        The pressure drop across the orifice.

    Returns
    -------
    cda: [m^2]
        The effective area.
    """
    return mdot / np.sqrt(2 * rho * dp)

@overload
def dp_from_cda(mdot: float, rho: float, cda: float) -> float: ...
@overload
def dp_from_cda(mdot: FloatArray, rho: FloatOrArray, cda: FloatOrArray) -> FloatArray: ...
@overload
def dp_from_cda(mdot: FloatOrArray, rho: FloatArray, cda: FloatOrArray) -> FloatArray: ...
@overload
def dp_from_cda(mdot: FloatOrArray, rho: FloatOrArray, cda: FloatArray) -> FloatArray: ...
def dp_from_cda(mdot: FloatOrArray, rho: FloatOrArray, cda: FloatOrArray) -> FloatOrArray:
    """Calculate the pressure drop from a mass flow rate, density, and effective area.

    Parameters
    ----------
    mdot: [kg/s]
        The mass flow rate.
    rho: [kg/m^3]
        The fluid density.
    cda: [m^2]
        The effective area.

    Returns
    -------
    dp: [Pa]
        The pressure drop across the orifice.
    """
    return (mdot / cda) ** 2 / (2 * rho)

# endregion Steady-State Functions
# region State Space Models
class BlowdownState(IntEnum):
    """Enumeration for the state vector indices in a blowdown process."""

    MASS = 0
    INTERNAL_ENERGY = auto()

def d_dt_blowdown(t: float, state: np.ndarray,
                  gas_state: AbstractState,
                  volume: float,
                  mass_flow_out: float,
                  heat_in: Callable[[float, np.ndarray], float]
                  ) -> np.ndarray:
    """Compute the time derivative of the state vector for a blowdown process.

    Parameters
    ----------
    t: [s]
        The time for the blowdown process.
    state: [per BlowdownState]
        The current state vector.
    gas_state: [AbstractState]
        The state of the gas.
    mass_flow_out: [kg/s]
        The mass flow rate out of the system.
    heat_in: [Callable[[t, state], [W]]]
        A function that computes the heat input to the system.

    Returns
    -------
    d_dt: [per BlowdownState]
        The time derivative of the state vector.
    """
    mass_gas = state[BlowdownState.MASS]
    U_gas = state[BlowdownState.INTERNAL_ENERGY] #noqa: N806
    Q_in = heat_in(t, state) #noqa: N806
    rho_gas = mass_gas / volume

    gas_state.update(DmassUmass_INPUTS, rho_gas, U_gas / mass_gas)
    h_gas = gas_state.hmass()  # Specific enthalpy of the gas

    # Compute the time derivatives
    d_mass_dt = -mass_flow_out
    dU_dt = Q_in - h_gas * mass_flow_out #noqa: N806

    ret_arr = np.empty_like(state)
    ret_arr[BlowdownState.MASS] = d_mass_dt
    ret_arr[BlowdownState.INTERNAL_ENERGY] = dU_dt
    return ret_arr

class UllageLiquidState(IntEnum):
    """Enumeration for the state vector indices in an ullage-liquid process."""

    ULLAGE_MASS = 0
    PROP_MASS = auto()
    ULLAGE_INTERNAL_ENERGY = auto()
    PROP_INTERNAL_ENERGY = auto()

def d_dt_ullage_liquid(t: float,
                       state: np.ndarray,
                       gas_state: AbstractState,
                       liquid_state: LiquidClass,
                       combined_volume: float,
                       mass_flow_in: float,
                       h_flow_in: float,
                       mass_flow_out: float,
                       gas_heat_in: Callable[[float, np.ndarray], float],
                       liquid_heat_in: Callable[[float, np.ndarray], float],
                       gas_to_liquid_heat_transfer: Callable[[float, np.ndarray], float]
                       ) -> np.ndarray:
    """Compute the time derivative of the state vector for an ullage-liquid process.

    Assumes:
    - Rigid tank
    - Incompressible liquid propellant
    - Uniform gas (non-stratified)
    - Uniform liquid (non-stratified)
    - No mass transfer between the ullage and liquid phases
    - Negligible pressure head

    Parameters
    ----------
    t: [s]
        The time for the ullage-liquid process.
    state: [per UllageLiquidState]
        The current state vector.
    gas_state: [AbstractState]
        The state of the gas.
    liquid_state: [LiquidClass]
        The state of the liquid.
    combined_volume: [m^3]
        The combined volume of the ullage and liquid.
    mass_flow_in: [kg/s]
        The mass flow rate into the system.
    h_flow_in: [J/kg]
        The specific enthalpy of the flow into the system.
    mass_flow_out: [kg/s]
        The mass flow rate out of the system.
    gas_heat_in: [Callable[[t, state], [W]]]
        A function that computes the heat input to the gas.
    liquid_heat_in: [Callable[[t, state], [W]]]
        A function that computes the heat input to the liquid.
    gas_to_liquid_heat_transfer: [Callable[[t, state], [W]]]
        A function that computes the heat transfer from the gas to the liquid.

    Returns
    -------
    d_dt: [per UllageLiquidState]
        The time derivative of the state vector.
    """
    mass_ullage = state[UllageLiquidState.ULLAGE_MASS]
    u_ullage = state[UllageLiquidState.ULLAGE_INTERNAL_ENERGY] / mass_ullage
    mass_prop = state[UllageLiquidState.PROP_MASS]
    u_prop = state[UllageLiquidState.PROP_INTERNAL_ENERGY] / mass_prop

    liquid_state.update_u(u_prop)

    v_liquid = mass_prop / liquid_state.density # kg/ (kg/m^3) = m^3
    v_ullage = combined_volume - v_liquid
    gas_state.update(DmassUmass_INPUTS, mass_ullage / v_ullage, u_ullage)
    p_gas = gas_state.p() # Pa
    intra_fluid_heat_transfer = gas_to_liquid_heat_transfer(t, state)

    d_v_gas = mass_flow_out / liquid_state.density # m^3/s
    d_dt_arr = np.empty_like(state)
    d_dt_arr[UllageLiquidState.ULLAGE_INTERNAL_ENERGY] = (gas_heat_in(t, state)
                                                        - intra_fluid_heat_transfer
                                                          + h_flow_in * mass_flow_in
                                                            - p_gas * d_v_gas)
    d_dt_arr[UllageLiquidState.PROP_INTERNAL_ENERGY] = (liquid_heat_in(t, state)
                                                        + intra_fluid_heat_transfer
                                                            - mass_flow_out * u_prop)
    # no pressure work term for liquid since it's incompressible,
    # so +PdV work from the ullage is cancelled by -Pdv work done flowing out of the tank
    d_dt_arr[UllageLiquidState.ULLAGE_MASS] = mass_flow_in
    d_dt_arr[UllageLiquidState.PROP_MASS] = -mass_flow_out
    return d_dt_arr

# def orifice_flow_rate_gas(upstream_state: AbstractState, downstream_pressure: float) -> float:
#     """Calculate the mass flow rate of gas through an orifice.

#     Parameters
#     ----------
#     upstream_state: [AbstractState]
#         The state of the gas upstream of the orifice.
#     downstream_pressure: [Pa]
#         The pressure downstream of the orifice.

#     Returns
#     -------
#     mass_flow_rate: [kg/s]
#         The mass flow rate of gas through the orifice.
#     """
#     upstream_pressure = upstream_state.p()  # Pa
#     upstream_density = upstream_state.rhomass()  # kg/m^3
#     upstream_enthalpy = upstream_state.hmass()  # J/kg

#     if downstream_pressure >= upstream_pressure:
        # raise ValueError("Downstream pressure must be "
        #                  "less than upstream pressure for flow to occur.")

#     # Estimate the critical pressure ratio for choked flow
#     gamma = upstream_state.cpmass() / upstream_state.cvmass()  # Specific heat ratio
#     critical_pressure_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1)) # P*/P0 https://en.wikipedia.org/wiki/Choked_flow
#     if downstream_pressure /
#     if downstream_pressure / upstream_pressure <= critical_pressure_ratio:
#         # Choked flow
#         mass_flow_rate = (upstream_density * np.sqrt(gamma) *
#                           (upstream_pressure / np.sqrt(upstream_enthalpy)) *
#                           ((2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))))
#     else:
#         # Subsonic flow
#         mass_flow_rate = (upstream_density * np.sqrt(2 * gamma / (gamma - 1) *
#                           (upstream_pressure - downstream_pressure) /
#                           upstream_enthalpy))

#     return mass_flow_rate


# class Regulator():
#     def __init__(self, p_set: float, p_max: float, p_min: float, CdA: float) -> None:
#         """Initialize a Regulator instance.

#         Parameters
#         ----------
#         p_set: [Pa]
#             The set pressure of the regulator.
#         p_max: [Pa]
#             The maximum pressure of the regulator.
#         CdA: [m^2]
#             The effective area of the regulator at full open position.
#         """
#         self.p_set = p_set
#         self.p_max = p_max
#         self.p_min = p_min
#         self.CdA = CdA

#     def flow_rate(self, p_in: float, p_out: float, rho: float) -> float:

# endregion State Space Models
