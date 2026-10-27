"""A collection of classes and functions to wrap the cea2022 python API in a SI/user-friendly way.

Source Documentation
https://nasa.github.io/cea/interfaces/python_api.html#cea.RocketSolution

Classes
-------
Propellent:
    A propellent instance that can be used in a PropellentMixture to run CEA calculations.
PropellentMixture:
    A mixture of propellents that can be used to run CEA calculations.
CEAOutput:
    A class to format the output of a CEA solution into SI base units, etc.

Functions
---------
output_solution:
    Format the CEA solution into a human-readable string (Match the original/website CEA output).
"""
# TODO: Add Frozen ability
from collections.abc import Sequence
from typing import overload
from warnings import warn

import cea
import numpy as np
import scipy.constants as csp

from supporting_funcs.coding.type_hints import FloatArray, FloatOrArray


class Propellent:
    """A propellent instance that can be used in a PropellentMixture to run CEA calculations."""

    def __init__(self,
                 percent_mass: float,
                 temperature: float = 298.15,
                 *, cea_reactant: cea.Reactant | None = None,
                 name: str | cea.Reactant | None = None
                 ) -> None:

        if cea_reactant is not None:
            if not isinstance(cea_reactant, cea.Reactant):
                raise TypeError("cea_reactant must be a cea.Reactant instance")
            self.cea_reactant = cea_reactant
        elif name is not None:
            if not isinstance(name, str):
                raise TypeError("name must be a string or a cea.Reactant instance")
            self.cea_reactant = cea.Reactant(name=name)
        elif name is None and cea_reactant is None:
            raise ValueError("Either name or cea_reactant must be provided")
        else:
            raise TypeError("Either a name (str) or a cea.Reactant instance must be provided")

        self.temperature = temperature
        self.percent_mass = percent_mass

    def __repr__(self) -> str:
        """Return a string representation of the Propellent instance."""
        return (f"Propellent(name={self.name}, "
               f"percent_mass={self.percent_mass}, "
               f"temperature={self.temperature})")

    def __eq__(self, other) -> bool:
        """Check if two Propellent instances are equal based on their properties."""
        if not isinstance(other, Propellent):
            return False

        return (self.cea_reactant == other.cea_reactant and
                self.percent_mass == other.percent_mass)

    @property
    def temperature(self) -> float | None:
        """Get the temperature of the propellent in Kelvin."""
        return self.cea_reactant.temperature
    @temperature.setter
    def temperature(self, temperature: float) -> None:
        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number")
        self.cea_reactant.temperature = temperature

    @property
    def name(self) -> str:
        """Get the name of the propellent."""
        return self.cea_reactant.name
    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.cea_reactant.name = name

    @property
    def cea_reactant(self) -> cea.Reactant:
        """Get the CEA Reactant instance for this propellent."""
        return self._reactant
    @cea_reactant.setter
    def cea_reactant(self, reactant: cea.Reactant) -> None:
        if not isinstance(reactant, cea.Reactant):
            raise TypeError("reactant must be a cea.Reactant instance")
        self._reactant = reactant

    @property
    def percent_mass(self) -> float:
        """Get the percent mass of the propellent in the mixture."""
        return self._percent_mass
    @percent_mass.setter
    def percent_mass(self, percent_mass: float) -> None:
        if not isinstance(percent_mass, (int, float)):
            raise TypeError("percent_mass must be a number")
        if percent_mass < 0:
            raise ValueError("percent_mass must be non-negative")
        self._percent_mass = percent_mass


class PropellentMixture:
    """A mixture of propellents that can be used to run CEA calculations."""

    def __init__(self, oxidizers: list[Propellent],
                 fuels: list[Propellent],
                 o_f_ratio: float,
                 *,
                 o_f_ratio_is_moles: bool = False,
                 include_ions: bool = False,
                 calculate_transport_properties: bool = True
                 ) -> None:
        """Initialize a PropellentMixture instance.

        Parameters
        ----------
        oxidizers: list[Propellent]
            A list of oxidizer propellents.
        fuels: list[Propellent]
            A list of fuel propellents.
        o_f_ratio: float
            The oxidizer-to-fuel ratio. If o_f_ratio is 0 or np.inf, the mixture is a monopropellant
        o_f_ratio_is_moles: bool, default False
            If True, the o_f_ratio is interpreted as a molar ratio. If False, it is a mass ratio.
        include_ions: bool, default False
            If True, include ions in the CEA calculations. If False, do not include ions.
        """
        self._last_soln: cea.RocketSolution | None = None

        # Abnormal flags for the CEA solver
        self.include_ions = include_ions
        self.calculate_transport_properties = calculate_transport_properties

        if len(oxidizers) == 0 and len(fuels) == 0:
            raise ValueError("At least one oxidizer or fuel must be provided.")

        if len(oxidizers) == 0 and o_f_ratio != 0:
            warn("non-zero o_f_ratio with no oxidizers. Setting o_f_ratio to 0.", stacklevel=2)
            o_f_ratio = 0.0
        self._oxidizers = oxidizers

        if len(fuels) == 0 and o_f_ratio != np.inf:
            warn("non-infinite o_f_ratio with no fuels. Setting o_f_ratio to np.inf.", stacklevel=2)
            o_f_ratio = np.inf
        self._fuels = fuels
        self._o_f_ratio = o_f_ratio

        # Keep the validation in place
        self.fuels = fuels
        self.oxidizers = oxidizers
        if o_f_ratio_is_moles:
            self.o_f_ratio = self.o_f_moles_to_o_f_mass(o_f_ratio)
        else:
            self.o_f_ratio = o_f_ratio
        self._set_cea_solver()

    @property
    def oxidizer_weights(self) -> FloatArray:
        """Get the weights of the oxidizers in the mixture, normalized to sum to 1."""
        return self._oxidizer_weights
    @property
    def fuel_weights(self) -> FloatArray:
        """Get the weights of the fuels in the mixture, normalized to sum to 1."""
        return self._fuel_weights

    @property
    def fuels(self) -> list[Propellent]:
        """Get the list of fuels in the propellent mixture as Propellent instances."""
        return self._fuels
    @fuels.setter
    def fuels(self, fuels: list[Propellent]) -> None:
        self._fuels = fuels
        if len(fuels) == 0:
            if self.o_f_ratio != np.inf:
                warn("non-infinite o_f_ratio with no fuels. Setting o_f_ratio to np.inf.",
                     stacklevel=2)
                self.o_f_ratio = np.inf
            self._fuel_weights = np.array([])
            self._fuel_temperatures = np.array([])
        else:
            self._fuel_weights = np.asarray([fuel.percent_mass for fuel in fuels])
            self._fuel_weights /= sum(self._fuel_weights)
            self._fuel_temperatures = np.asarray([fuel.temperature for fuel in fuels])
        self._set_cea_solver()
    @property
    def cea_fuel_list(self) -> list[cea.Reactant]:
        """Get the list of fuels as cea.Reactant instances."""
        return [fuel.cea_reactant for fuel in self.fuels]
    @property
    def fuel_temperatures(self) -> FloatArray:
        """Get the temperatures of the fuels in Kelvin."""
        return self._fuel_temperatures
    @fuel_temperatures.setter
    def fuel_temperatures(self, temperatures: FloatArray) -> None:
        fuels = self.fuels
        if len(temperatures) != len(fuels):
            raise ValueError("Length of temperatures must match the number of fuels")
        for fuel, temp in zip(fuels, temperatures, strict=True):
            fuel.temperature = temp
        self.fuels = fuels  # Update the fuels list to reflect the new temperatures

    @property
    def oxidizers(self) -> list[Propellent]:
        """Get the list of oxidizers in the propellent mixture."""
        return self._oxidizers
    @oxidizers.setter
    def oxidizers(self, oxidizers: list[Propellent]) -> None:
        self._oxidizers = oxidizers
        if len(oxidizers) == 0:
            if self.o_f_ratio != 0:
                warn("non-zero o_f_ratio with no oxidizers. Setting o_f_ratio to 0.", stacklevel=2)
                self.o_f_ratio = 0.0
            else:
                self._oxidizer_weights = np.array([])
                self._oxidizer_temperatures = np.array([])
        else:
            self._oxidizer_weights = np.asarray([oxidizer.percent_mass for oxidizer in oxidizers])
            self._oxidizer_weights /= sum(self._oxidizer_weights)
            self._oxidizer_temperatures = np.asarray([oxidizer.temperature
                                                      for oxidizer in oxidizers])
        self._set_cea_solver()

    @property
    def cea_oxidizer_list(self) -> list[cea.Reactant]:
        """Get the list of oxidizers as cea.Reactant instances."""
        return [oxidizer.cea_reactant for oxidizer in self.oxidizers]
    @property
    def oxidizer_temperatures(self) -> FloatArray:
        """Get the temperatures of the oxidizers in Kelvin."""
        return self._oxidizer_temperatures
    @oxidizer_temperatures.setter
    def oxidizer_temperatures(self, temperatures: FloatArray) -> None:
        oxidizers = self.oxidizers
        if len(temperatures) != len(oxidizers):
            raise ValueError("Length of temperatures must match the number of oxidizers")
        for oxidizer, temp in zip(oxidizers, temperatures, strict=True):
            oxidizer.temperature = temp
        self.oxidizers = oxidizers  # Update the oxidizers list to reflect the new temperatures

    @property
    def o_f_ratio(self) -> float:
        """Get the oxidizer-to-fuel ratio (mass basis) for the propellent mixture."""
        return self._o_f_ratio
    @o_f_ratio.setter
    def o_f_ratio(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("o_f_ratio must be a number")
        if value < 0:
            raise ValueError("o_f_ratio must be positive")
        self._o_f_ratio = value
        if (self.cea_solver is not None) and (self.last_raw_solution is not None):
            self.resolve()

    @property
    def cea_h_chamb(self) -> float:
        """Get the chamber enthalpy for the propellent mixture [kJ/kg]."""
        return self.get_cea_reactants().calc_property(cea.ENTHALPY,
                                                            self.cea_weights,
                                                            np.append(self.oxidizer_temperatures,
                                                                      self.fuel_temperatures)
                ) / cea.R

    @property
    def cea_weights(self) -> FloatArray:
        """Get the weights of the propellents in the mixture, organized for the CEA solver."""
        return self.get_o_f_weights(o_f_ratio=self.o_f_ratio)

    @property
    def cea_solver(self) -> cea.RocketSolver:
        """Get the CEA solver instance for this propellent mixture."""
        return self._cea_solver

    @property
    def last_raw_solution(self) -> cea.RocketSolution | None:
        """Get the last CEA solution computed for this propellent mixture."""
        return self._last_soln

    @property
    def solution(self) -> CEAOutput:
        """Get the last CEA solution computed for this propellent mixture."""
        if self.last_raw_solution is not None:
            return CEAOutput(solution=self.last_raw_solution, o_f_ratio=self.o_f_ratio)
        else:
            raise ValueError("No solution has been computed yet. Call the 'solve' method first.")

    def _set_cea_solver(self) -> None:
        reac = self.get_cea_reactants()
        prod = self.get_cea_products()
        self._cea_solver = cea.RocketSolver(reactants=reac,
                                            products=prod,
                                            ions=self.include_ions,
                                            transport=self.calculate_transport_properties)

    def get_species(self) -> list[cea.Reactant | str]:
        """Get the list of species in the propellent mixture as cea.Reactant instances / strings."""
        return [*self.cea_oxidizer_list, *self.cea_fuel_list]

    def get_cea_reactants(self) -> cea.Mixture:
        """Get the CEA reactants mixture for this propellent mixture."""
        return cea.Mixture(species=self.get_species(),
                           ions=self.include_ions)

    def get_cea_products(self) -> cea.Mixture:
        """Get the CEA products mixture for this propellent mixture."""
        return cea.Mixture(species=self.get_species(),
                           products_from_reactants=True,
                           ions=self.include_ions)

    def get_o_f_weights(self, o_f_ratio: float) -> FloatArray:
        """Convert an oxidizer-to-fuel mass ratio to a weight distribution for the CEA solver."""
        if o_f_ratio < 0:
            raise ValueError("o_f_ratio must be non-negative")
        if o_f_ratio == 0:
            return self.fuel_weights
        if o_f_ratio == np.inf:
            return self.oxidizer_weights

        # append zeros
        oxidizer_weights = np.append(self.oxidizer_weights, np.zeros_like(self.fuel_weights))
        fuel_weights = np.append(np.zeros_like(self.oxidizer_weights), self.fuel_weights)

        return self.get_cea_reactants().of_ratio_to_weights(
            oxidant_weights=oxidizer_weights, fuel_weights=fuel_weights, of_ratio=o_f_ratio
        )

    def o_f_moles_to_o_f_mass(self,
                          o_f_ratio_moles: float) -> float:
        """Convert an oxidizer-to-fuel ratio in moles to a mass basis.

        Parameters
        ----------
        o_f_ratio_moles: float
            The oxidizer-to-fuel ratio in moles.

        Returns
        -------
        o_f_ratio_mass: float
            The oxidizer-to-fuel ratio in mass.
        """
        if o_f_ratio_moles < 0:
            raise ValueError("o_f_ratio_moles must be non-negative")
        elif o_f_ratio_moles == 0:
            return 0.0  # If there are no oxidizers, the mass ratio is zero
        elif o_f_ratio_moles == np.inf:
            return np.inf  # If there are no fuels, the mass ratio is infinite

        if len(self.fuel_weights) != 1 or len(self.oxidizer_weights) != 1:
            raise NotImplementedError("Conversion from oxidizer-to-fuel ratio in moles"
                                      "to mass is only implemented for single oxidizer "
                                      "and single fuel mixtures.")

        ox_weights = np.concatenate((self.oxidizer_weights, np.zeros_like(self.fuel_weights)))
        fuel_weights = np.concatenate((np.zeros_like(self.oxidizer_weights), self.fuel_weights))

        mws = self.get_cea_reactants().moles_to_weights(np.ones_like(fuel_weights)) # g/mol

        fuel_mw = np.sum(fuel_weights * mws)
        ox_mw = np.sum(ox_weights * mws)

        if fuel_mw == 0:
            return np.inf  # If there are no fuels, the mass ratio is infinite
        if ox_mw == 0:
            return 0.0  # If there are no oxidizers, the mass ratio is zero

        return o_f_ratio_moles * ox_mw / fuel_mw


    def solve(self, p_c: float,
              *,
              pi_p: FloatOrArray | None = None,
              ae_at: FloatOrArray | None = None,
              is_choked: bool = True,
              subar: FloatOrArray | None = None,
              supar: FloatOrArray | None = None,
              h_chamb_kj_kg: float | None = None
              ) -> CEAOutput:
        """Solve the CEA problem for the given propellent mixture.

        Parameters
        ----------
        p_c: [Pa]
            Chamber pressure
        pi_p: [1], optional
            Nozzle pressure ratio (p_c / p_e)
        ae_at: [1], optional
            Nozzle expansion ratio (A_e / A_t)
        is_choked: [bool], default True
            Whether the nozzle is choked or not. If False, the solution will be unchoked.
        subar: [1], optional
            Subsonic expansion ratios (A_e / A_t) for the solution.
            If provided, the solution will be computed for these expansion ratios.
        supar: [1], optional
            Supersonic expansion ratios (A_e / A_t) for the solution.
            If provided along with ae_at, the solution will be computed for all expansion ratios.
        h_chamb_kj_kg: [kJ/kg], optional
            Chamber enthalpy. If None, the chamber enthalpy will be computed.
        """
        solver = self.cea_solver
        solution = cea.RocketSolution(solver)

        kwargs = {}
        if pi_p is not None:
            kwargs["pi_p"] = (pi_p if isinstance(pi_p, (list, np.ndarray)) else [float(pi_p)])
        if ae_at is not None:
            if is_choked:
                key = 'supar'
            else:
                key = 'subar'
            kwargs[key] = np.array(ae_at if isinstance(ae_at, (list, np.ndarray)) else [ae_at])

        if subar is not None:
            old_val: FloatOrArray = kwargs.get('subar', [])
            kwargs['subar'] = np.append(old_val, subar)

        if supar is not None:
            old_val: FloatOrArray = kwargs.get('supar', [])
            kwargs['supar'] = np.append(old_val, supar)

        if h_chamb_kj_kg is None:
            h_chamb_kj_kg = self.cea_h_chamb

        solver.solve(soln=solution,
                     weights=self.cea_weights,
                     pc=p_c / csp.bar, # Pa to bar
                     hc = h_chamb_kj_kg,
                     **kwargs
                     )

        self._last_soln = solution
        return self.solution

    def resolve(self, *,
                p_c: float | None = None,
                o_f_ratio: float | None = None,
                ae_at: FloatOrArray | None = None,
                pi_p: FloatOrArray | None = None
                ) -> CEAOutput:
        """Re-solve the CEA problem for the given propellent mixture with updated parameters.

        Parameters
        ----------
        p_c: [Pa], optional
            Chamber pressure. If None, use the last solved value.
        o_f_ratio: [1], optional
            Oxidizer to fuel ratio (mass basis). If None, use the last solved value.
        ae_at: [1], optional
            Nozzle expansion ratio (A_e / A_t). If both ae_at and pi_p are None, uses the value.
        pi_p: [1], optional
            Nozzle pressure ratio (p_c / p_e).
        """
        if self.last_raw_solution is None:
            raise ValueError("No previous solution exists. Call 'solve' first.")

        # Use last solved values if not provided
        p_c = p_c if p_c is not None else self.solution.chamber_pressure
        kwargs = {}
        if ae_at is None and pi_p is None:
            ae_at = self.solution.ae_at[2:] # Skip the chamber and throat
        elif ae_at is not None and pi_p is not None:
            raise ValueError("Only one of ae_at or pi_p should be provided, not both.")
        elif ae_at is not None:
            kwargs["ae_at"] = ae_at
        elif pi_p is not None:
            kwargs["pi_p"] = pi_p

        if o_f_ratio is not None:
            self.o_f_ratio = o_f_ratio

        return self.solve(p_c=p_c, **kwargs)

    def get_area_specific_vac_thrust(self, eta_isp: float = 1.0) -> FloatOrArray:
        """Get the vacuum thrust for the given throat area.

        Parameters
        ----------
        eta_isp: [1], default 1.0
            The efficiency of the specific impulse. A value of 1.0 means no losses.

        Returns
        -------
        thrust/area: [N/m²]
            Thrust for the given throat area
        """
        mdot = self.solution.gas_velocity[1] * self.solution.density[1]
        isp_v = self.solution.isp_vacuum[2:] # Skip the chamber and throat
        return mdot * isp_v * csp.g * eta_isp

class CEAOutput:
    """A class to format the output of a CEA solution into SI base units, etc."""

    def __init__(self, solution: cea.RocketSolution, o_f_ratio: float) -> None:
        self.raw_solution = solution
        self.o_f_ratio = o_f_ratio

    def __str__(self) -> str:
        """Return a string representation of the CEAOutput instance."""
        return output_solution(self.raw_solution)

    @property
    def raw_solution(self) -> cea.RocketSolution:
        """Get the raw CEA solution."""
        return self._raw_solution
    @raw_solution.setter
    def raw_solution(self, solution: cea.RocketSolution) -> None:
        if not isinstance(solution, cea.RocketSolution):
            raise TypeError("solution must be a cea.RocketSolution instance")
        self._raw_solution = solution

    @property
    def o_f_ratio(self) -> float:
        """Get the oxidizer to fuel ratio (mass basis) for the solution [1]."""
        return self._o_f_ratio
    @o_f_ratio.setter
    def o_f_ratio(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("o_f_ratio must be a number")
        if value < 0:
            raise ValueError("o_f_ratio must be positive")
        self._o_f_ratio = value

    @property
    def is_frozen(self) -> bool:
        """Get whether the solution is frozen or not."""
        return all(self.raw_solution.cv == self.raw_solution.cv_fr)

    @property
    def ae_at(self) -> FloatArray:
        """Get the expansion ratio (A_e / A_t) for the solution [1]."""
        return self.raw_solution.ae_at

    @property
    def effective_exhaust_velocity_vacuum(self) -> FloatArray:
        """Get the effective exhaust velocity in vacuum for the solution [m/s]."""
        return self.raw_solution.Isp_vacuum
    v_eff_v = effective_exhaust_velocity_vacuum # alias

    @property
    def effective_exhaust_velocity_ideal(self) -> FloatArray:
        """Get the effective exhaust velocity in ideal conditions for the solution [m/s]."""
        return self.raw_solution.Isp
    v_eff_i = effective_exhaust_velocity_ideal # alias

    @property
    def isp_vacuum(self) -> FloatArray:
        """Get the vacuum specific impulse for the solution [s]."""
        return self.effective_exhaust_velocity_vacuum / csp.g
    isp_v = isp_vacuum # alias

    @property
    def isp_ideal(self) -> FloatArray:
        """Get the ideal specific impulse for the solution [s]."""
        return self.effective_exhaust_velocity_ideal / csp.g
    isp_i = isp_ideal # alias

    @property
    def temperature(self) -> FloatArray:
        """Get the static temperature for the solution [K]."""
        return self.raw_solution.T
    T = temperature # alias

    @property
    def pressure(self) -> FloatArray:
        """Get the static pressure for the solution [Pa]."""
        return self.raw_solution.P * csp.bar
    P = pressure # alias
    @property
    def chamber_pressure(self) -> float:
        """Get the chamber pressure for the solution [Pa]."""
        return self.raw_solution.P[0] * csp.bar

    @property
    def density(self) -> FloatArray:
        """Get the static density for the solution [kg/m³]."""
        return self.raw_solution.density
    rho = density # alias

    @property
    def enthalpy(self) -> FloatArray:
        """Get the static enthalpy for the solution [J/kg]."""
        return self.raw_solution.enthalpy * csp.kilo
    h = enthalpy # alias

    @property
    def internal_energy(self) -> FloatArray:
        """Get the static internal energy for the solution [J/kg]."""
        return self.raw_solution.energy * csp.kilo
    u = internal_energy # alias

    @property
    def gibbs_energy(self) -> FloatArray:
        """Get the static Gibbs energy for the solution [J/kg]."""
        return self.raw_solution.gibbs_energy * csp.kilo
    g = gibbs_energy # alias

    @property
    def entropy(self) -> FloatArray:
        """Get the static entropy for the solution [J/kg/K]."""
        return self.raw_solution.entropy * csp.kilo
    s = entropy # alias

    @property
    def mach(self) -> FloatArray:
        """Get the Mach number for the solution [1]."""
        return self.raw_solution.Mach
    # no M alias here to prevent confusion with molecular weight
    @property
    def overall_molecular_weight(self) -> FloatArray:
        """Get the overall molecular weight for the solution [kg/mol]."""
        return self.raw_solution.MW / csp.kilo #kg/kmol = 1000 * kg/mol

    @property
    def gaseous_molecular_weight(self) -> FloatArray:
        """Get the gaseous molecular weight for the solution [kg/mol]."""
        return self.raw_solution.M / csp.kilo #kg/kmol = 1000 * kg/mol
    MW_gas = gaseous_molecular_weight # alias

    @property
    def specific_heat_capacity_constant_pressure(self) -> FloatArray:
        """Get the constant pressure specific heat for the solution [J/kg/K]."""
        if self.is_frozen:
            return self.raw_solution.cp_fr * csp.kilo
        else:
            return self.raw_solution.cp_eq * csp.kilo
    cp = specific_heat_capacity_constant_pressure # alias

    @property
    def specific_heat_capacity_constant_volume(self) -> FloatArray:
        """Get the constant volume specific heat for the solution [J/kg/K]."""
        if self.is_frozen:
            return self.raw_solution.cv_fr * csp.kilo
        else:
            return self.raw_solution.cv_eq * csp.kilo
    cv = specific_heat_capacity_constant_volume # alias

    @property
    def ratio_of_specific_heats_sonic(self) -> FloatArray:
        """Get the ratio of specific heats for the solution [1].

        (Not always equal to cp/cv, especially for non-ideal gases).
        """
        return self.raw_solution.gamma_s
    gamma_s = ratio_of_specific_heats_sonic
    gamma = ratio_of_specific_heats_sonic # alias

    @property
    def characteristic_velocity(self) -> FloatArray:
        """Get the characteristic velocity for the solution [m/s]."""
        return self.raw_solution.c_star
    c_star = characteristic_velocity # alias

    @property
    def coefficient_of_thrust(self) -> FloatArray:
        """Get the coefficient of thrust for the solution [1]."""
        return self.raw_solution.coefficient_of_thrust
    c_f = coefficient_of_thrust # alias

    @property
    def sonic_velocity(self) -> FloatArray:
        """Get the sonic velocity for the solution [m/s]."""
        return self.raw_solution.sonic_velocity
    c = sonic_velocity # alias

    @property
    def gas_velocity(self) -> FloatArray:
        """Get the gas velocity for the solution [m/s]."""
        return self.sonic_velocity * self.mach

    @property
    def viscosity(self) -> FloatArray:
        """Get the viscosity for the solution [Pa*s]."""
        return self.raw_solution.viscosity * 1e-4 # Pa*s from millipoise
    mu = viscosity # alias

    @property
    def conductivity_fr(self) -> FloatArray:
        """Get the conductivity for the solution [W/m/K]."""
        return self.raw_solution.conductivity_fr * 0.1 # W/m/K from mW/cm/K
    k_fr = conductivity_fr # alias

    @property
    def conductivity_eq(self) -> FloatArray:
        """Get the conductivity for the solution [W/m/K]."""
        return self.raw_solution.conductivity_eq * 0.1 # W/m/K from mW/cm/K
    k_eq = conductivity_eq # alias

    @property
    def conductivity(self) -> FloatArray:
        """Get the conductivity for the solution [W/m/K]."""
        if self.is_frozen:
            return self.conductivity_fr
        else:
            return self.conductivity_eq
    k = conductivity # alias

    @property
    def prandtl_eq(self) -> FloatArray:
        """Get the Prandtl number for the solution [1]."""
        return self.raw_solution.Pr_eq
    Pr_eq = prandtl_eq # alias

    @property
    def prandtl_fr(self) -> FloatArray:
        """Get the Prandtl number for the solution [1]."""
        return self.raw_solution.Pr_fr
    Pr_fr = prandtl_fr # alias

    @property
    def prandtl(self) -> FloatArray:
        """Get the Prandtl number for the solution [1]."""
        if self.is_frozen:
            return self.prandtl_fr
        else:
            return self.prandtl_eq
    Pr = prandtl # alias

    @overload
    def get_isp_vacuum(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_isp_vacuum(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_isp_vacuum(self, *, ae_at: FloatOrArray | None = None) -> FloatOrArray:
        """Interpolate the vacuum specific impulse for the given expansion ratio.

        Either ae_at or pi_p must be provided, but not both.
        If both are None, returns the chamber value.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber specific impulse

        Returns
        -------
        isp_vacuum: [s]
            Vacuum specific impulse for the given expansion ratio
        """
        if ae_at is None:
            return self.isp_vacuum[0]
        return np.interp(ae_at, self.ae_at, self.isp_vacuum)
    get_isp_v = get_isp_vacuum # alias

    @overload
    def get_isp_ideal(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_isp_ideal(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_isp_ideal(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the ideal specific impulse for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber specific impulse

        Returns
        -------
        isp_ideal: [s]
            Ideal Expansion specific impulse for the given expansion ratio
        """
        if ae_at is None:
            return self.isp_ideal[0]
        return np.interp(ae_at, self.ae_at, self.isp_ideal)
    get_isp_i = get_isp_ideal # alias

    @overload
    def get_temperature(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_temperature(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_temperature(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the static temperature for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber temperature

        Returns
        -------
        temperature: [K]
            Static temperature for the given expansion ratio
        """
        if ae_at is None:
            return self.temperature[0]
        return np.interp(ae_at, self.ae_at, self.temperature)
    get_T = get_temperature #noqa:N815 # alias

    @overload
    def get_pressure(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_pressure(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_pressure(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the static pressure for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber pressure

        Returns
        -------
        pressure: [Pa]
            Static pressure for the given expansion ratio
        """
        if ae_at is None:
            return self.pressure[0]
        return np.interp(ae_at, self.ae_at, self.pressure)
    get_P = get_pressure #noqa:N815 # alias

    @overload
    def get_density(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_density(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_density(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the static density for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber density

        Returns
        -------
        rho: [kg/m^3]
            Static density for the given expansion ratio
        """
        if ae_at is None:
            return self.density[0]
        return np.interp(ae_at, self.ae_at, self.density)
    get_rho = get_density # alias

    @overload
    def get_enthalpy(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_enthalpy(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_enthalpy(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the static enthalpy for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber enthalpy

        Returns
        -------
        enthalpy: [J/kg]
            Static enthalpy for the given expansion ratio
        """
        if ae_at is None:
            return self.enthalpy[0]
        return np.interp(ae_at, self.ae_at, self.enthalpy)
    get_h = get_enthalpy # alias

    @overload
    def get_internal_energy(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_internal_energy(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_internal_energy(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the static internal energy for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber internal energy

        Returns
        -------
        u: [J/kg]
            Static internal energy for the given expansion ratio
        """
        if ae_at is None:
            return self.internal_energy[0]
        return np.interp(ae_at, self.ae_at, self.internal_energy)
    get_u = get_internal_energy # alias

    @overload
    def get_gibbs_energy(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_gibbs_energy(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_gibbs_energy(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the static Gibbs energy for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber Gibbs energy

        Returns
        -------
        g: [J/kg]
            Static Gibbs energy for the given expansion ratio
        """
        if ae_at is None:
            return self.gibbs_energy[0]
        return np.interp(ae_at, self.ae_at, self.gibbs_energy)
    get_g = get_gibbs_energy # alias

    @overload
    def get_entropy(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_entropy(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_entropy(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the static entropy for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber entropy

        Returns
        -------
        s: [J/kg-K]
            Static entropy for the given expansion ratio
        """
        if ae_at is None:
            return self.entropy[0]
        return np.interp(ae_at, self.ae_at, self.entropy)
    get_s = get_entropy # alias

    @overload
    def get_mach(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_mach(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_mach(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the Mach number for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber Mach number

        Returns
        -------
        mach: [1]
            Mach number for the given expansion ratio
        """
        if ae_at is None:
            return self.mach[0]
        return np.interp(ae_at, self.ae_at, self.mach)
    # no M alias here to prevent confusion with molecular weight

    @overload
    def get_overall_molecular_weight(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_overall_molecular_weight(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_overall_molecular_weight(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the overall molecular weight for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber overall molecular weight

        Returns
        -------
        MW: [g/mol]
            Overall molecular weight for the given expansion ratio
        """
        if ae_at is None:
            return self.overall_molecular_weight[0]
        return np.interp(ae_at, self.ae_at, self.overall_molecular_weight)

    @overload
    def get_gaseous_molecular_weight(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_gaseous_molecular_weight(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_gaseous_molecular_weight(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the gaseous molecular weight for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber gaseous molecular weight

        Returns
        -------
        MW_gas: [g/mol]
            Gaseous molecular weight for the given expansion ratio
        """
        if ae_at is None:
            return self.gaseous_molecular_weight[0]
        return np.interp(ae_at, self.ae_at, self.gaseous_molecular_weight)
    get_MW_gas = get_gaseous_molecular_weight # noqa: N815 # alias

    @overload
    def get_constant_pressure_specific_heat(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_constant_pressure_specific_heat(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_constant_pressure_specific_heat(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the constant pressure specific heat for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        cp_eq: [J/kg-K]
            Constant pressure specific heat for the given expansion ratio
        """
        if ae_at is None:
            return self.specific_heat_capacity_constant_pressure[0]
        return np.interp(ae_at, self.ae_at, self.specific_heat_capacity_constant_pressure)
    get_cp = get_constant_pressure_specific_heat # alias

    @overload
    def get_constant_volume_specific_heat(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_constant_volume_specific_heat(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_constant_volume_specific_heat(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the constant volume specific heat for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        cv_eq: [J/kg-K]
            Constant volume specific heat for the given expansion ratio
        """
        if ae_at is None:
            return self.specific_heat_capacity_constant_volume[0]
        return np.interp(ae_at, self.ae_at, self.specific_heat_capacity_constant_volume)
    get_cv = get_constant_volume_specific_heat # alias

    @overload
    def get_ratio_of_specific_heats_sonic(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_ratio_of_specific_heats_sonic(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_ratio_of_specific_heats_sonic(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the ratio of specific heats for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        gamma_s: [1]
            Ratio of specific heats for the given expansion ratio
        """
        if ae_at is None:
            return self.ratio_of_specific_heats_sonic[0]
        return np.interp(ae_at, self.ae_at, self.ratio_of_specific_heats_sonic)
    get_gamma_s = get_ratio_of_specific_heats_sonic # alias

    @overload
    def get_sonic_velocity(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_sonic_velocity(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_sonic_velocity(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the sonic velocity for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        v_sonic: [m/s]
            Sonic velocity for the given expansion ratio
        """
        if ae_at is None:
            return self.sonic_velocity[0]
        return np.interp(ae_at, self.ae_at, self.sonic_velocity)
    get_c = get_sonic_velocity # alias

    @overload
    def get_gas_velocity(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_gas_velocity(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_gas_velocity(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the gas velocity for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        v_gas: [m/s]
            Gas velocity for the given expansion ratio
        """
        if ae_at is None:
            raise ValueError("ae_at must be provided to get the gas velocity.")
        return np.interp(ae_at, self.ae_at, self.gas_velocity)

    @overload
    def get_characteristic_velocity(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_characteristic_velocity(self, *, ae_at: FloatArray) -> FloatArray: ...

    def get_characteristic_velocity(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the characteristic velocity for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        c_star: [m/s]
            Characteristic velocity for the given expansion ratio
        """
        if ae_at is None:
            return self.characteristic_velocity[0]
        return np.interp(ae_at, self.ae_at, self.characteristic_velocity)
    get_c_star = get_characteristic_velocity # alias

    @overload
    def get_coefficient_of_thrust(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_coefficient_of_thrust(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_coefficient_of_thrust(self, *, ae_at: FloatOrArray | None) -> FloatOrArray:
        """Interpolate the coefficient of thrust for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        Cf: [1]
            Coefficient of thrust for the given expansion ratio
        """
        if ae_at is None:
            return self.coefficient_of_thrust[0]
        return np.interp(ae_at, self.ae_at, self.coefficient_of_thrust)
    get_c_f = get_coefficient_of_thrust # alias

    @overload
    def get_viscosity(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_viscosity(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_viscosity(self, *, ae_at: FloatOrArray | None = None) -> FloatOrArray:
        """Interpolate the viscosity for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        mu: [Pa*s]
            Viscosity for the given expansion ratio
        """
        if ae_at is None:
            return self.viscosity[0]
        return np.interp(ae_at, self.ae_at, self.viscosity)
    get_mu = get_viscosity # alias

    @overload
    def get_conductivity(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_conductivity(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_conductivity(self, *, ae_at: FloatOrArray | None = None) -> FloatOrArray:
        """Interpolate the conductivity for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        k: [W/m/K]
            Conductivity for the given expansion ratio
        """
        if ae_at is None:
            return self.conductivity[0]
        return np.interp(ae_at, self.ae_at, self.conductivity)
    get_k = get_conductivity # alias

    @overload
    def get_prandtl(self, *, ae_at: float | None) -> float: ...
    @overload
    def get_prandtl(self, *, ae_at: FloatArray) -> FloatArray: ...
    def get_prandtl(self, *, ae_at: FloatOrArray | None = None) -> FloatOrArray:
        """Interpolate the Prandtl number for the given expansion ratio.

        Parameters
        ----------
        ae_at: [1]
            Nozzle expansion ratio (A_e / A_t), if None, return the chamber value

        Returns
        -------
        pr: [1]
            Prandtl number for the given expansion ratio
        """
        if ae_at is None:
            return self.prandtl[0]
        return np.interp(ae_at, self.ae_at, self.prandtl)

def output_solution(solution: cea.RocketSolution) -> str:
    """Format the CEA solution into a human-readable string."""
    def format_row(label: str, values: Sequence[float] | FloatArray, fmt: str = "10.3f") -> str:
        """Format one labelled row of a fixed-width table.

        Parameters
        ----------
            label: [-]
                Row label, padded to a fixed column width
            values: [varies]
                One value per column, formatted with fmt
            fmt: [-]
                Format spec applied to each value, e.g. "10.3f"

        Returns
        -------
            row: [-]
                Single line, no trailing whitespace or newline
        """
        return f"{label:<15s} " + " ".join(f"{v:{fmt}}" for v in values)

    T = solution.T #noqa: N806
    P = solution.P #noqa: N806
    rho = solution.density
    enthalpy = solution.enthalpy
    energy = solution.energy
    gibbs = solution.gibbs_energy
    entropy = solution.entropy
    M_1n = solution.M #noqa: N806
    MW = solution.MW #noqa: N806
    cp_eq = solution.cp_eq
    cp_fr = solution.cp_fr
    cv_eq = solution.cv_eq
    cv_fr = solution.cv_fr
    Mach = solution.Mach #noqa: N806
    gamma_s = solution.gamma_s
    v_sonic = solution.sonic_velocity
    ae_at = solution.ae_at
    c_star = solution.c_star
    Cf = solution.coefficient_of_thrust #noqa: N806
    Isp = solution.Isp #noqa: N806
    Isp_vac = solution.Isp_vacuum #noqa: N806

    state_rows = [
        ("P, bar", P, "10.3f"),
        ("T, K", T, "10.3f"),
        ("Density, kg/m^3", rho, "10.3f"),
        ("H, kJ/kg", enthalpy, "10.2f"),
        ("U, kJ/kg", energy, "10.2f"),
        ("G, kJ/kg", gibbs, "10.1f"),
        ("S, kJ/kg-K", entropy, "10.3f"),
        ("M, (1/n)", M_1n, "10.3f"),
        ("MW", MW, "10.3f"),
        ("Cp_eq, kJ/kg-K", cp_eq, "10.3f"),
        ("Cp_fr, kJ/kg-K", cp_fr, "10.3f"),
        ("Cv_eq, kJ/kg-K", cv_eq, "10.3f"),
        ("Cv_fr, kJ/kg-K", cv_fr, "10.3f"),
        ("Gamma_s", gamma_s, "10.3f"),
        ("Son. vel., m/s", v_sonic, "10.2f"),
        ("Mach", Mach, "10.3f"),
    ]

    performance_rows = [
        ("Ae/At", ae_at, "10.3f"),
        ("C*, m/s", c_star, "10.2f"),
        ("Cf", Cf, "10.4f"),
        ("Isp, vac., m/s", Isp_vac, "10.3f"),
        ("Isp, m/s", Isp, "10.3f"),
    ]

    output: str = ""
    output += "\nSTATE PROPERTIES\n\n"
    for label, values, fmt in state_rows:
        output += format_row(label, values, fmt) + "\n"

    output += "\nPERFORMANCE PARAMETERS\n\n"
    for label, values, fmt in performance_rows:
        output += format_row(label, values, fmt) + "\n"

    output += "\nMOLE FRACTIONS\n\n"
    trace_species = []
    for prod in solution.mole_fractions:
        if np.any(solution.mole_fractions[prod] > 5e-6):
            output += f"{prod:15s}" + " "
            for j in range(len(solution.mole_fractions[prod])):
                if j < len(solution.mole_fractions[prod])-1:
                    output += f"{solution.mole_fractions[prod][j]:10.5g}" + " "
                else:
                    output += f"{solution.mole_fractions[prod][j]:10.5g}" + "\n"
        else:
            trace_species.append(prod)

    output += "\nTRACE SPECIES\n\n"
    max_cols = 8
    nrows = (len(trace_species) + max_cols - 1) // max_cols
    for i in range(nrows):
        output += " ".join(f"{trace_species[j]:15s}"
                    for j in range(i * max_cols, min((i + 1) * max_cols, len(trace_species))
                                    )) + "\n"

    return output
