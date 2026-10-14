from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import override

import numpy as np
from scipy.constants import Stefan_Boltzmann as sigma
from scipy.optimize import root_scalar

from supporting_funcs.physics.bartz import (
    create_bartz_param_class,
)
from supporting_funcs.physics.cea_wrapper import PropellentMixture


class ThermalCases(Enum):
    """Enum for different thermal cases."""

    CYLINDRICAL = auto()
    SPHERICAL = auto()
    PLANAR = auto()

class BoundaryConditionType(Enum):
    """Enum for different boundary conditions."""

    CONSTANT_TEMPERATURE = auto()
    CONSTANT_HEAT_FLUX = auto()
    CONVECTION = auto()
    RADIATION = auto()

class Material:
    """Class representing a material with thermal properties."""

    def __init__(self, name: str,
                 thermal_conductivity: float,
                 specific_heat: float,
                 density: float):
        self.name = name
        self.thermal_conductivity = thermal_conductivity  # W/(m·K)
        self.specific_heat = specific_heat  # J/(kg·K)
        self.density = density  # kg/m^3

class BoundaryCondition:
    """Base class for boundary conditions."""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def infinite_temperature(self) -> float:
        """Return the temperature at which the heat flux would be zero.

        Returns
        -------
        infinite_temperature: [K]
            The temperature at which the heat flux would be zero.
        """

    @abstractmethod
    def compute_heat_flux(self, wall_temperature: float) -> float:
        """Compute the heat flux out of the wall.

        Parameters
        ----------
        wall_temperature: [K]
            The temperature of the wall.

        Returns
        -------
        heat_flux: [W/m^2]
            The heat flux out of the wall.
        """
# class ConstantTemperatureBoundaryCondition(BoundaryCondition):
#     def __init__(self, temperature: float):
#         super().__init__()
#         self.temperature = temperature

#     @property
#     def infinite_temperature(self):
#         return self.temperature

#     def compute_heat_flux(self, wall_temperature: float):
#         """Computes the heat flux out of the wall."""
#         return 0.0  # No heat flux for constant temperature boundary condition

class ConstantHeatFluxBoundaryCondition(BoundaryCondition):
    """Boundary condition with a constant heat flux."""

    def __init__(self, heat_flux: float):
        super().__init__()
        self.heat_flux = heat_flux

    @property
    @override
    def infinite_temperature(self) -> float:
        """Doesn't have a physical meaning for constant heat flux."""
        return float('inf')  # Represents an infinite temperature for heat flux boundary condition

    @override
    def compute_heat_flux(self, wall_temperature: float):
        """Compute the heat flux out of the wall."""
        return self.heat_flux

class ConvectionBoundaryCondition(BoundaryCondition):
    """Boundary condition with convective heat transfer."""

    def __init__(self, heat_transfer_coefficient: float, ambient_temperature: float):
        super().__init__()
        self.heat_transfer_coefficient = heat_transfer_coefficient
        self.ambient_temperature = ambient_temperature

    @property
    @override
    def infinite_temperature(self) -> float:
        """Return the ambient temperature for convective boundary condition."""
        return self.ambient_temperature

    @override
    def compute_heat_flux(self, wall_temperature: float):
        """Compute the heat flux out of the wall."""
        return self.heat_transfer_coefficient * (wall_temperature - self.ambient_temperature)

class RadiationBoundaryCondition(BoundaryCondition):
    """Boundary condition with radiative heat transfer."""

    def __init__(self, emissivity: float, ambient_temperature: float):
        super().__init__()
        self.emissivity = emissivity
        self.ambient_temperature = ambient_temperature

    @property
    @override
    def infinite_temperature(self) -> float:
        """Return the temperature at which the heat flux would be zero."""
        return self.ambient_temperature

    @override
    def compute_heat_flux(self, wall_temperature: float) -> float:
        """Compute the heat flux out of the wall."""
        return self.emissivity * sigma * (wall_temperature**4 - self.ambient_temperature**4)

class BartzBoundaryCondition(BoundaryCondition):
    """Boundary condition using the Bartz correlation for heat transfer (rocket nozzles)."""

    def __init__(self, cea_obj: PropellentMixture, d_throat: float, r_throat_curve: float):
        super().__init__()
        self.cea_obj = cea_obj
        self.d_throat = d_throat
        self.r_throat_curve = r_throat_curve
        self.bartz_param_obj = create_bartz_param_class(cea_result=cea_obj.solution,
                                                    d_throat=d_throat,
                                                    r_throat_curve=r_throat_curve
                                                    )

    def compute_heat_transfer_coefficient(self, wall_temperature: float) -> float:
        """Compute the heat transfer coefficient using the Bartz correlation."""
        return self.bartz_param_obj.bartz_with_temp_variables(temp_wall=wall_temperature)

    def compute_heat_flux(self, wall_temperature: float, *, gas_temperature: float | None = None) -> float:
        """Compute the heat flux out of the wall using the Bartz correlation."""
        h = self.compute_heat_transfer_coefficient(wall_temperature)
        if gas_temperature is None:
            gas_temperature = float(self.cea_obj.solution.temperature[1]) # Throat
        return h * (wall_temperature - gas_temperature)
class ThermalLayer:
    """Class representing a thermal layer in the model."""

    def __init__(self, material: Material, thickness: float, start: float, shape: ThermalCases):
        self.material = material
        self.thickness = thickness
        self.start = start
        self.shape = shape

class ThermalModel:
    """Class representing a thermal model composed of multiple layers."""

    def __init__(self, start_temperature: float, layers: list[ThermalLayer]):
        self.start_temperature = start_temperature
        self.layers = layers

    @property
    def layers(self):
        """Return the list of thermal layers."""
        return self._layers
    @layers.setter
    def layers(self, value):
        if not isinstance(value, list):
            raise TypeError("Layers must be a list of ThermalLayer objects.")
        shape = None
        last_end = 0
        for idx_layer, layer in enumerate(value):
            if not isinstance(layer, ThermalLayer):
                raise TypeError("Each layer must be an instance of ThermalLayer.")
            if shape is None:
                shape = layer.shape
            elif layer.shape != shape:
                raise ValueError("All layers must have the same shape.")
            else:
                pass

            if layer.start < 0 and shape != ThermalCases.PLANAR:
                raise ValueError("Layer start position must be non-negative for non-planar shapes.")

            if layer.start < last_end:
                raise ValueError("Layer start position must be greater than or equal to the" \
                                " previous layer's end position.")
            elif layer.start > last_end and idx_layer != 0:
                raise ValueError("There is a gap between layers. Layer start position must be"
                                " equal to the previous layer's end position.")
            last_end = layer.start + layer.thickness

        if shape is None:
            raise ValueError("At least one layer must be provided"
                             "to determine the shape of the thermal model.")
        self._shape = shape
        self._layers = value

    @property
    def shape(self) -> ThermalCases:
        """The shape of the thermal model."""
        return self._shape

    @property
    def layer_count(self) -> int:
        """How many layers are in the thermal model."""
        return len(self.layers)

    def compute_steady_state_temperature_distribution(self,
                                                      inner_boundary_condition: BoundaryCondition,
                                                      outer_boundary_condition: BoundaryCondition,
                                                      temp_limits: tuple[float, float]
                                                      ) -> tuple[float, float]:
        """Calculate the steady-state temperature distribution across the thermal layers.

        Parameters
        ----------
        inner_boundary_condition : BoundaryCondition
            The boundary condition at the inner surface of the thermal model.
        outer_boundary_condition : BoundaryCondition
            The boundary condition at the outer surface of the thermal model.
        temp_limits : [K, K]
            A tuple containing the lower and upper limits for the inner boundary temperature.

        Returns
        -------
        temp_inner: [K]
            The computed inner boundary temperature that satisfies the heat flux balance.
        heat_flow: [W/m^n]
            The heat flow through the thermal layers, where n depends on the shape of the layers
            (2 for planar, 1 for cylindrical, 2 for spherical).
        """
        resistors = []
        for layer in self.layers:
            if layer.shape == ThermalCases.CYLINDRICAL:
                resistors.append(np.log((layer.start + layer.thickness) / layer.start)
                                / (2 * np.pi * layer.material.thermal_conductivity))
            elif isinstance(layer.shape, ThermalCases):
                raise NotImplementedError(f"Shape {layer.shape} not implemented yet.")
            else:
                raise TypeError(f"Invalid shape type: {type(layer.shape)}")

        if self.shape == ThermalCases.CYLINDRICAL:
            area_inner = 2 * np.pi * self.layers[0].start
            area_outer = 2 * np.pi * (self.layers[-1].start + self.layers[-1].thickness)
        elif self.shape == ThermalCases.SPHERICAL:
            area_inner = 4 * np.pi * self.layers[0].start**2
            area_outer = 4 * np.pi * (self.layers[-1].start + self.layers[-1].thickness)**2
        elif self.shape == ThermalCases.PLANAR:
            area_inner = 1.0  # Arbitrary unit area for planar case
            area_outer = 1.0  # Arbitrary unit area for planar case
        elif isinstance(self.shape, ThermalCases):
            raise NotImplementedError(f"Shape {self.shape} not implemented yet.")
        else:
            raise TypeError(f"Invalid shape type: {type(self.shape)}")

        total_resistance = sum(resistors)

        def _err_func(temp_inner):
            """Compute the difference in heat flux between the inner and outer boundaries."""
            heat_inner = area_inner *inner_boundary_condition.compute_heat_flux(temp_inner)
            temp_outer = temp_inner - heat_inner * total_resistance
            heat_outer = area_outer * outer_boundary_condition.compute_heat_flux(temp_outer)
            return (heat_inner - heat_outer)

        soln = root_scalar(_err_func, bracket=temp_limits, method='brentq')
        if not soln.converged:
            raise ValueError("Failed to converge on a solution for the inner boundary temperature.")
        return soln.root, area_inner * inner_boundary_condition.compute_heat_flux(soln.root)
