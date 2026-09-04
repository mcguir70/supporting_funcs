"""#units.py
Module for managing units and output formatting in physics calculations.
This module provides the OutputFormatting class, which allows users to specify
the desired output units for various physical quantities, including pressure,
temperature, length, mass, force, time, and volume flow. It also provides methods
for converting between different units and for formatting output strings."""
from typing import Optional
from pint import Unit, UnitRegistry
from pint.facets.plain import PlainUnit
from scipy import constants as csp

UREG = UnitRegistry()
Q_ = UREG.Quantity #type: ignore

class OutputFormatting:
    """Class to manage output formatting and unit conversions for physics calculations."""
    def __init__(self,
                 output_SI: bool = True, #pylint: disable=C0103
                 vol_flow_unit: Optional[Unit] = None,
                 mass_unit: Optional[Unit] = None,
                 pressure_unit: Optional[Unit] = None,
                 vol_flow_unit_str: Optional[str] = None):
        self.output_SI = output_SI #pylint: disable=C0103
        if mass_unit is not None:
            if isinstance(mass_unit, Unit):
                self.mass_unit = mass_unit #pylint: disable=C0103
            elif isinstance(mass_unit, float):
                match mass_unit:
                    case csp.pound:
                        self.mass_unit = UREG.pound
                    case csp.gram:
                        self.mass_unit = UREG.gram
                    case _:
                        raise ValueError("massUnit float value must be either pound or gram.")
            else:
                raise TypeError("massUnit must be a pint Unit object.")

        if pressure_unit is not None:
            if isinstance(pressure_unit, Unit):
                self.pressure_unit = pressure_unit
            elif isinstance(pressure_unit, float):
                match pressure_unit:
                    case csp.psi:
                        self.pressure_unit = UREG.psi
                    case csp.atm:
                        self.pressure_unit = UREG.atm
                    case csp.bar:
                        self.pressure_unit = UREG.bar
                    case csp.torr:
                        self.pressure_unit = UREG.torr
                    case _:
                        raise ValueError(
                        "pressureUnit float value must be either psi, atm, bar, or torr.")
            else:
                raise TypeError("pressureUnit must be a pint Unit object.")

        if vol_flow_unit is not None:
            if isinstance(vol_flow_unit, Unit):
                self.vol_flow_unit = vol_flow_unit
            else:
                raise TypeError("volFlowUnit must be a pint Unit object.")
        if vol_flow_unit_str is not None:
            if isinstance(vol_flow_unit_str, str):
                self.vol_flow_unit_str = vol_flow_unit_str
            else:
                raise TypeError("volFlowUnitStr must be a string.")

    @property
    def output_SI(self): #pylint: disable=C0103
        """Whether to output results in SI units (True) or any other units (False)."""
        return self._output_SI
    @output_SI.setter
    def output_SI(self, value: bool): #pylint: disable=C0103
        """Set the output unit system. If True, use SI units; if False, use specified units."""
        self._output_SI = value #pylint: disable=C0103
        if value:
            self.pressure_unit = UREG.Pa
            self.temperature_unit = UREG.K
            self.length_unit = UREG.m
            self.mass_unit = UREG.kilogram
            self.force_unit = UREG.newton
            self.time_unit = UREG.second
        else:
            self.pressure_unit = UREG.psi
            self.temperature_unit = UREG.degR
            self.length_unit = UREG.inch
            self.mass_unit = UREG.pound
            self.force_unit = UREG.pound_force
            self.time_unit = UREG.second

        self.vol_flow_unit = self.length_unit ** 3 / self.time_unit
    @property
    def pressure_unit(self) -> Unit:
        """Pressure unit as a pint Unit."""
        return self._pressure_unit
    @pressure_unit.setter
    def pressure_unit(self, value: Unit):
        self._pressure_unit_dbl = Q_(1, value).to(UREG.Pa).magnitude
        self._pressure_unit_str = str(value)
        self._pressure_unit = value
    @property
    def temperature_unit(self) -> Unit:
        """Temperature unit as a pint Unit."""
        return self._temperature_unit
    @temperature_unit.setter
    def temperature_unit(self, value: Unit):
        self._temperature_unit_dbl = Q_(1, value).to(UREG.K).magnitude
        self._temperature_unit_str = str(value)
        self._temperature_unit = value
    @property
    def length_unit(self) -> Unit:
        """Length unit as a pint Unit."""
        return self._length_unit
    @length_unit.setter
    def length_unit(self, value: Unit):
        self._length_unit_dbl = Q_(1, value).to(UREG.m).magnitude
        self._length_unit_str = str(value)
        self._length_unit = value
    @property
    def mass_unit(self) -> Unit:
        """Mass unit as a pint Unit."""
        return self._mass_unit
    @mass_unit.setter
    def mass_unit(self, value: Unit):
        self._mass_unit_dbl = Q_(1, value).to(UREG.kilogram).magnitude
        self._mass_unit_str = str(value)
        self._mass_unit = value
    @property
    def force_unit(self) -> Unit:
        """Force unit as a pint Unit."""
        return self._force_unit
    @force_unit.setter
    def force_unit(self, value: Unit):
        self._force_unit_dbl = Q_(1, value).to(UREG.newton).magnitude
        self._force_unit_str = str(value)
        self._force_unit = value
    @property
    def time_unit(self) -> Unit:
        """Time unit as a pint Unit."""
        return self._time_unit
    @time_unit.setter
    def time_unit(self, value: Unit):
        self._time_unit_dbl = Q_(1, value).to(UREG.second).magnitude
        self._time_unit_str = str(value)
        self._time_unit = value

    @property
    def pressure_unit_dbl(self) -> float:
        """Pressure unit conversion factor to SI (Pa)."""
        return self._pressure_unit_dbl
    @property
    def temperature_unit_dbl(self) -> float:
        """Temperature unit conversion factor to SI (K)."""
        return self._temperature_unit_dbl
    @property
    def length_unit_dbl(self) -> float:
        """Length unit conversion factor to SI (m)."""
        return self._length_unit_dbl
    @property
    def mass_unit_dbl(self) -> float:
        """Mass unit conversion factor to SI (kg)."""
        return self._mass_unit_dbl
    @property
    def force_unit_dbl(self) -> float:
        """Force unit conversion factor to SI (N)."""
        return self._force_unit_dbl
    @property
    def time_unit_dbl(self) -> float:
        """Time unit conversion factor to SI (s)."""
        return self._time_unit_dbl

    @property
    def vol_flow_unit(self) -> Unit | PlainUnit:
        """Volume flow unit as a pint Unit or PlainUnit."""
        if self._vol_flow_unit is None:
            return self.length_unit ** 3 / self.time_unit
        return self._vol_flow_unit
    @vol_flow_unit.setter
    def vol_flow_unit(self, value: Optional[Unit | PlainUnit]):
        if value is None:
            self._vol_flow_unit = self.length_unit ** 3 / self.time_unit
            self._vol_flow_unit_dbl = self.length_unit_dbl ** 3 / self.time_unit_dbl
            self._vol_flow_unit_str = f"{self.length_unit_dbl}^3 / {self.time_unit_dbl}"
        else:
            self._vol_flow_unit_dbl = Q_(1, value).to(UREG.meter ** 3 / UREG.second).magnitude
            self._vol_flow_unit_str = str(value)
            self._vol_flow_unit = value

    @property
    def vol_flow_unit_dbl(self) -> float:
        """Volume flow unit conversion factor to SI (m^3/s)."""
        return self._vol_flow_unit_dbl
    @property
    def vol_flow_unit_str(self) -> str:
        """Volume flow unit as a string."""
        return self._vol_flow_unit_str
    @vol_flow_unit_str.setter
    def vol_flow_unit_str(self, value: Optional[str]):
        if value is None:
            self._vol_flow_unit_str = str(self.vol_flow_unit)
        elif isinstance(value, str):
            self._vol_flow_unit_str = value
        else:
            raise ValueError("volFlowUnitStr must be a string or None.")
