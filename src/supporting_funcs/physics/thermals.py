"""
Thermal analysis tools for modeling heat transfer.

Classes
-------
Material
    A class representing a material with thermal properties.
BoundaryCondition
    A base class for boundary conditions.
ConstantHeatFluxBoundaryCondition
    A boundary condition with a constant heat flux.
ConvectionBoundaryCondition
    A boundary condition with convective heat transfer.
RadiationBoundaryCondition
    A boundary condition with radiative heat transfer.
BartzBoundaryCondition
    A boundary condition using the Bartz correlation for heat transfer (rocket nozzles).
ThermalCases
    An enum for different thermal cases (cylindrical, spherical, planar).
ThermalLayer
    A class representing a thermal layer in the model.
ThermalModel
    A class representing a thermal model composed of multiple layers.

Functions
---------
None
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum, auto
from typing import override

import numpy as np
from scipy.constants import Stefan_Boltzmann as sigma
from scipy.optimize import root_scalar

from supporting_funcs.coding.type_hints import OneDFloatArray
from supporting_funcs.math.geometry import circle_circumference, sphere_area
from supporting_funcs.physics.bartz import (
    create_bartz_param_class,
)
from supporting_funcs.physics.cea_wrapper import PropellentMixture


class ThermalCases(Enum):
    """Enum for different thermal cases."""

    CYLINDRICAL = auto()
    SPHERICAL = auto()
    PLANAR = auto()


class Material:
    """Class representing a material with thermal properties."""

    def __init__(self, name: str,
                 thermal_conductivity: float,
                 specific_heat: float,
                 density: float) -> None:
        """Initialize a Material instance.

        Parameters
        ----------
        name: str
            The name of the material.
        thermal_conductivity: [W/(m*K)]
            The thermal conductivity of the material.
        specific_heat: [J/(kg*K)]
            The specific heat of the material.
        density: [kg/m^3]
            The density of the material.
        """
        self.name = name
        self.thermal_conductivity = thermal_conductivity
        self.specific_heat = specific_heat
        self.density = density


class BoundaryCondition(ABC):
    """Base class for boundary conditions."""

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


class ConstantHeatFluxBoundaryCondition(BoundaryCondition):
    """Boundary condition with a constant heat flux."""

    def __init__(self, heat_flux: float) -> None:
        super().__init__()
        self.heat_flux = heat_flux

    @property
    @override
    def infinite_temperature(self) -> float:
        """Doesn't have a physical meaning for constant heat flux."""
        return float('nan')

    @override
    def compute_heat_flux(self, wall_temperature: float) -> float:
        """Compute the heat flux out of the wall."""
        return self.heat_flux


class ConvectionBoundaryCondition(BoundaryCondition):
    """Boundary condition with convective heat transfer."""

    def __init__(self, heat_transfer_coefficient: float, ambient_temperature: float) -> None:
        super().__init__()
        self.heat_transfer_coefficient = heat_transfer_coefficient
        self.ambient_temperature = ambient_temperature

    @property
    @override
    def infinite_temperature(self) -> float:
        """Return the ambient temperature for convective boundary condition."""
        return self.ambient_temperature

    @override
    def compute_heat_flux(self, wall_temperature: float) -> float:
        """Compute the heat flux out of the wall."""
        return self.heat_transfer_coefficient * (wall_temperature - self.ambient_temperature)


class RadiationBoundaryCondition(BoundaryCondition):
    """Boundary condition with radiative heat transfer."""

    def __init__(self, emissivity: float, ambient_temperature: float) -> None:
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

    def __init__(self, cea_obj: PropellentMixture,
                 d_throat: float,
                 r_throat_curve: float,
                 heat_transfer_scalar: float = 1.0,
                 thermal_idx: int = 1,  # Throat default
                 ) -> None:
        """Initialize a BartzBoundaryCondition instance.

        Parameters
        ----------
        cea_obj: PropellentMixture
            An instance of the PropellentMixture class containing CEA results.
        d_throat: [m]
            The throat diameter of the nozzle.
        r_throat_curve: [m]
            The radius of curvature at the throat of the nozzle.
        heat_transfer_scalar: float, optional
            A scalar to adjust the heat transfer coefficient (default is 1.0).
        thermal_idx: int, default 1
            The index in the CEA results corresponding to the location of interest.
        """
        super().__init__()
        self.cea_obj = cea_obj
        self.d_throat = d_throat
        self.r_throat_curve = r_throat_curve
        self.heat_transfer_scalar = heat_transfer_scalar
        self._thermal_idx = thermal_idx
        self.bartz_param_obj = create_bartz_param_class(cea_result=cea_obj.solution,
                                                    d_throat=d_throat,
                                                    r_throat_curve=r_throat_curve,
                                                    mach=cea_obj.solution.mach[thermal_idx],
                                                    area_rat=cea_obj.solution.ae_at[thermal_idx]
                                                    )

    @property
    @override
    def infinite_temperature(self) -> float:
        """Return the temperature at which the heat flux would be zero."""
        return float(self.cea_obj.solution.temperature[self._thermal_idx]
                     * self.bartz_param_obj.Tr_scalar)

    @override
    def compute_heat_flux(self,
                          wall_temperature: float,
                          *,
                          gas_temperature: float | None = None) -> float:
        """Compute the heat flux out of the wall using the Bartz correlation.

        Parameters
        ----------
        wall_temperature: [K]
            The temperature of the wall.
        gas_temperature: [K], optional pulls from CEA if not provided
            The static temperature of the gas at that station.

        Returns
        -------
        heat_flux: [W/m^2]
            The heat flux out of the wall.
        """
        if gas_temperature is None:
            gas_temperature = float(self.cea_obj.solution.temperature[self._thermal_idx])
        # Negative sign because heat flux is defined as positive out of the wall
        return -(self.bartz_param_obj.heat_flux_with_temp_variables(temp_wall=wall_temperature,
                                                                  temp_gas=gas_temperature)
                                                                  * self.heat_transfer_scalar)

class ThermalElement(ABC):
    """Element of a 1-D conduction stack (single layer or nested model)."""

    @property
    @abstractmethod
    def start(self) -> float:
        """The start position [m] of the thermal element."""

    @property
    @abstractmethod
    def thickness(self) -> float:
        """The thickness [m] of the thermal element."""

    @property
    @abstractmethod
    def shape(self) -> ThermalCases:
        """The shape of the thermal element."""

    @property
    @abstractmethod
    def resistance(self) -> float:
        """The thermal resistance [K * m^n/W] of the thermal element."""

    @property
    def end(self) -> float:
        """Outer position [m]."""
        return self.start + self.thickness

    @property
    def n(self) -> int:
        """Return the exponent n for the thermal resistance based on the shape."""
        if self.shape == ThermalCases.CYLINDRICAL:
            return 1
        elif self.shape == ThermalCases.SPHERICAL:
            return 0
        elif self.shape == ThermalCases.PLANAR:
            return 2
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

    def calc_heat_flow(self, inner_temperature: float, outer_temperature: float) -> float:
        """Heat flow through the element, positive in +r / +x.

        Parameters
        ----------
        inner_temperature: [K]
            Temperature at the inner surface.
        outer_temperature: [K]
            Temperature at the outer surface.

        Returns
        -------
        heat_flow: [W/m^n]
            n = 2 planar, 1 cylindrical, 0 spherical.
        """
        return (inner_temperature - outer_temperature) / self.resistance

class ThermalLayer(ThermalElement):
    """Class representing a thermal layer in the model."""

    def __init__(self, material: Material,
                 thickness: float,
                 start: float,
                 shape: ThermalCases
                 ) -> None:
        self._set_material(material)
        self._set_thickness(thickness)
        self._set_shape(shape)
        self._set_start(start)

    @property
    def material(self) -> Material:
        """The material of the thermal layer."""
        return self._material
    def _set_material(self, value: Material) -> None:
        if not isinstance(value, Material):
            raise TypeError("Material must be an instance of Material.")
        self._material = value

    @property
    @override
    def thickness(self) -> float:
        """The thickness [m] of the thermal layer."""
        return self._thickness
    def _set_thickness(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Thickness must be positive.")
        self._thickness = value

    @property
    @override
    def start(self) -> float:
        """The start position [m] of the thermal layer."""
        return self._start
    def _set_start(self, value: float) -> None:
        if value <= 0 and self.shape != ThermalCases.PLANAR:
            raise ValueError("Start position must be positive for non-planar shapes.")
        self._start = value

    @property
    @override
    def shape(self) -> ThermalCases:
        """The shape of the thermal layer."""
        return self._shape
    def _set_shape(self, value: ThermalCases) -> None:
        if not isinstance(value, ThermalCases):
            raise TypeError("Shape must be an instance of ThermalCases.")
        self._shape = value

    @property
    @override
    def resistance(self) -> float:
        """Calculate the thermal resistance of the layer based on its shape."""
        if self.shape == ThermalCases.CYLINDRICAL:
            return (np.log(self.end / self.start)
                    / (2 * np.pi * self.material.thermal_conductivity))
        elif self.shape == ThermalCases.SPHERICAL:
            return ((1 / (4 * np.pi * self.material.thermal_conductivity))
                    * (1 / self.start - 1 / self.end))
        elif self.shape == ThermalCases.PLANAR:
            return self.thickness / self.material.thermal_conductivity
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")


class ThermalModel(ThermalElement):
    """Class representing a thermal model composed of multiple layers."""

    def __init__(self, layers: Sequence[ThermalElement]) -> None:
        self._set_layers(layers)

    @property
    def layers(self) -> tuple[ThermalElement, ...]:
        """Return the list of thermal layers."""
        return self._layers
    def _set_layers(self, value: Sequence[ThermalElement]) -> None:
        if not isinstance(value, Sequence):
            raise TypeError("Layers must be a list or tuple of ThermalElement objects.")

        if len(value) == 0:
            raise ValueError("Layers list cannot be empty.")
        for layer in value:
            if isinstance(layer, ThermalElement):
                if layer.shape != value[0].shape:
                    raise ValueError("All layers must have the same shape.")
            else:
                raise TypeError("Each layer must be an instance of ThermalLayer or ThermalModel.")

        ends = np.array([layer.end for layer in value])
        starts = np.array([layer.start for layer in value])
        if not np.allclose(ends[:-1], starts[1:], rtol=1e-9, atol=1e-12):
            raise ValueError("There is a gap between layers. Layer start position must be"
                             " equal to the previous layer's end position.")

        self._shape = value[0].shape
        self._layers = tuple(value)  # make layers immutable after initialization

    @property
    def leaf_layers(self) -> list[ThermalElement]:
        """Flattened list of ThermalElements, ordered inner to outer."""
        leaves: list[ThermalElement] = []
        for layer in self.layers:
            if isinstance(layer, ThermalModel):
                leaves.extend(layer.leaf_layers)
            elif isinstance(layer, ThermalLayer):
                leaves.append(layer)
            elif isinstance(layer, ThermalElement):
                raise NotImplementedError(f"{type(layer).__name__} is not implemented.")
            else:
                raise TypeError("Each layer must be an instance of ThermalLayer or ThermalModel.")
        return leaves

    @property
    @override
    def shape(self) -> ThermalCases:
        """The shape of the thermal model."""
        return self._shape

    @property
    def layer_count(self) -> int:
        """How many layers are in the thermal model."""
        return len(self.layers)

    @property
    @override
    def thickness(self) -> float:
        """Calculate the total thickness [m] of all layers in the thermal model."""
        return self.layers[-1].end - self.layers[0].start

    @property
    @override
    def resistance(self) -> float:
        """Calculate the total thermal resistance [K * m^n/W] of all layers in the thermal model."""
        return sum(layer.resistance for layer in self.layers)

    @property
    @override
    def start(self) -> float:
        """The start position [m] of the thermal model."""
        return self.layers[0].start

    @property
    @override
    def end(self) -> float:
        """The end position [m] of the thermal model."""
        return self.layers[-1].end

    @property
    def inner_area(self) -> float:
        """Calculate the inner surface area [m^n] of the thermal model based on its shape."""
        if self.shape == ThermalCases.CYLINDRICAL:
            return circle_circumference(radius=self.layers[0].start)
        elif self.shape == ThermalCases.SPHERICAL:
            return sphere_area(radius=self.layers[0].start)
        elif self.shape == ThermalCases.PLANAR:
            return 1.0  # Arbitrary unit area for planar case
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

    @property
    def outer_area(self) -> float:
        """Calculate the outer surface area [m^n] of the thermal model based on its shape."""
        if self.shape == ThermalCases.CYLINDRICAL:
            return circle_circumference(radius=self.layers[-1].end)
        elif self.shape == ThermalCases.SPHERICAL:
            return sphere_area(radius=self.layers[-1].end)
        elif self.shape == ThermalCases.PLANAR:
            return 1.0  # Arbitrary unit area for planar case
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

    def compute_steady_state_temperature_distribution(self,
                                                      inner_boundary_condition: BoundaryCondition,
                                                      outer_boundary_condition: BoundaryCondition,
                                                      temp_limits: tuple[float, float],
                                                      *,
                                                      leaf_output: bool = False
                                                      ) -> tuple[OneDFloatArray, float]:
        """Calculate the steady-state temperature distribution across the thermal layers.

        Parameters
        ----------
        inner_boundary_condition: BoundaryCondition
            The boundary condition at the inner surface of the thermal model.
        outer_boundary_condition: BoundaryCondition
            The boundary condition at the outer surface of the thermal model.
        temp_limits: [K, K]
            A tuple containing the lower and upper limits for the inner boundary temperature.
        leaf_output: bool, optional
            If True, return the temperature distribution for each leaf layer;
            otherwise, return for each top-level layer (default is False).

        Returns
        -------
        temp_distribution: [K]
            The computed temperature on the inner surface of the thermal layers.
        heat_flow: [W/m^n]
            The heat flow out of the thermal layers (Positive for hot inner, cold outer).
            n depends on the shape of the layers
        """
        total_resistance = self.resistance  # Cache
        inner_area = self.inner_area  # Cache
        outer_area = self.outer_area  # Cache

        def _err_func(temp_inner_wall: float) -> float:
            """Compute the difference in heat flux between the inner and outer boundaries."""
            heat_inner = inner_area * inner_boundary_condition.compute_heat_flux(temp_inner_wall)
            temp_outer = temp_inner_wall + heat_inner * total_resistance
            heat_outer = outer_area * outer_boundary_condition.compute_heat_flux(temp_outer)
            return heat_inner + heat_outer

        if temp_limits[0] >= temp_limits[1]:
            raise ValueError("Lower temperature limit must be less than upper temperature limit.")
        if temp_limits[0] < 0 or temp_limits[1] < 0:
            raise ValueError("Temperature limits must be non-negative.")

        soln = root_scalar(_err_func, bracket=temp_limits, method='brentq')
        if not soln.converged:
            raise ValueError("Failed to converge on a solution.")

        heat_flow = inner_area * (-inner_boundary_condition.compute_heat_flux(soln.root))

        elements = self.leaf_layers if leaf_output else self.layers
        resistances = np.array([element.resistance for element in elements])
        temp_distribution: OneDFloatArray = np.empty(resistances.size + 1)
        temp_distribution[0] = soln.root
        temp_distribution[1:] = soln.root - heat_flow * np.cumsum(resistances)
        return temp_distribution, heat_flow
