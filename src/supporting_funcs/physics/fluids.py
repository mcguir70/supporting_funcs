"""Functions for modeling fluid behavior."""
from collections.abc import Callable
from enum import IntEnum, auto

import numpy as np
from CoolProp.CoolProp import AbstractState, DmassUmass_INPUTS
from scipy.constants import bar as csp_bar
from scipy.constants import hour as csp_hour

CV_TO_KV = 0.865  # Conversion factor from Cv to Kv (GPM to m^3/hr - IEC 60534)

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
    RHO_H2O = 1000 # kg/m^3 (IEC 60534) F #noqa: N806
    # Kv = Q * sqrt(SG / dP) = Q * sqrt(rho / (rho_ref * dP))
    # ==> Q = Kv * sqrt(rho_ref * dP / rho)
    # mdot = rho * Q = CdA * sqrt(2 * rho * dP)
    # ==> CdA = rho * Q / sqrt(2 * rho * dP) = Q * sqrt(rho / (2 * dP))
    # CdA = Kv * sqrt(rho_ref * dP / rho) * sqrt(rho / (2 * dP)) = Kv * sqrt((rho_ref) * (1 / 2))
    kv_si = kv / (csp_hour * np.sqrt(csp_bar)) # m^3/hr to m^3/s
    return kv_si * np.sqrt(RHO_H2O / 2)

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

if __name__ == "__main__":
    from scipy import constants as csp
    # Check kv to cda conversion
    kv = 1 # m^3/hr
    cda = kv_to_cda(kv)

    #pi d^2 / 4 = CdA
    d_ref = np.sqrt(4 * cda / np.pi) # m
    print(f"Kv: {kv} m^3/hr, CdA: {cda:.6f} m^2, Reference Diameter: {d_ref / csp.milli:.6f} mm")
