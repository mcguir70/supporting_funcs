"""# atmosphere.py
Simulation of atmospheric properties.

## Functions and Classes
`AirProperties`: (Class)
    class uses USSA1976 to compute standard atmospheric properties.
`get_cp_air_properties`: (function)
    uses CoolProp to compute real-gas properties
    at a given pressure and temperature.
`plot_air_properties`: (function)
    function plots air properties vs altitude.
`interp_air_properties`: (function)
    function interpolates air properties from an array of air properties.
"""
from enum import IntEnum
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
from CoolProp.HumidAirProp import HAPropsSI
from ussa1976 import compute as ussa1976_compute

from supporting_funcs.coding.type_hints import OneDFloatArray


class AirPropertiesIndex(IntEnum):
    """Enum for indexing air properties in an array"""
    ALTITUDE = 0
    DENSITY = 1
    KINEMATIC_VISCOSITY = 2
    SPECIFIC_HEAT_PRESSURE = 3
    HUMIDITY_RELATIVE = 4
    TEMPERATURE = 5
    PRESSURE = 6
    SPECIFIC_HEAT_VOLUME = 7
    SPEED_OF_SOUND = 8

class AirProperties:
    """Class for atmospheric properties using USSA1976"""
    def __init__(self):
        self.atmosphere = ussa1976_compute()

    @property
    def atmosphere(self):
        """Get the atmosphere data

        Returns
        -------
        atmosphere: dict
            dictionary with keys 'z', 't', 'p', 'rho', 'cs', 'mu'
        """
        return self._atmosphere
    @atmosphere.setter
    def atmosphere(self, value):
        self._atmosphere = value
        self._altitudes = value['z']
        self._t = value['t']
        self._p = value['p']
        self._rho = value['rho']
        self._son = value['cs']
        self._mu = value['mu']

    @overload
    def get_properties(self, altitude: float,
                       property_index: None=...
                       ) -> dict: ...
    @overload
    def get_properties(self, altitude: float,
                       property_index: AirPropertiesIndex
                       ) -> float: ...
    def get_properties(self, altitude: float,
                       property_index: AirPropertiesIndex | None = None
                       ) -> dict[str, float] | float:
        """Get atmospheric properties at a given altitude using USSA1976

            Parameters
            ----------
            altitude: [m]
                altitude

            Returns
            -------
            properties: [SI units]
                dictionary with keys 't', 'p', 'rho', 'cs', 'mu'
        """
        if property_index is not None:
            if property_index == AirPropertiesIndex.ALTITUDE:
                return altitude
            elif property_index == AirPropertiesIndex.DENSITY:
                return np.interp(altitude, self._altitudes, self._rho)
            elif property_index == AirPropertiesIndex.KINEMATIC_VISCOSITY:
                return np.interp(altitude, self._altitudes, self._mu)
            # elif property_index == AirPropertiesIndex.SpecificHeatPressure:
            #     cp = np.interp(altitude, self._altitudes, self._cp)
            #     return {'SpecificHeatPressure': cp}
            # elif property_index == AirPropertiesIndex.RelativeHumidity:
                # relHumidity = np.interp(altitude, self._altitudes, self._relHumidity)
                # return {'RelativeHumidity': relHumidity}
            elif property_index == AirPropertiesIndex.TEMPERATURE:
                return np.interp(altitude, self._altitudes, self._t)
            elif property_index == AirPropertiesIndex.PRESSURE:
                return np.interp(altitude, self._altitudes, self._p)
            # elif property_index == AirPropertiesIndex.SpecificHeatVolume:
            #     cv = np.interp(altitude, self._altitudes, self._cv)
            #     return {'SpecificHeatVolume': cv}
            elif property_index == AirPropertiesIndex.SPEED_OF_SOUND:
                return np.interp(altitude, self._altitudes, self._son)
        t = np.interp(altitude, self._altitudes, self._t)
        p = np.interp(altitude, self._altitudes, self._p)
        rho = np.interp(altitude, self._altitudes, self._rho)
        cs = np.interp(altitude, self._altitudes, self._son)
        mu = np.interp(altitude, self._altitudes, self._mu)
        return {'t': t, 'p': p, 'rho': rho, 'cs': cs, 'mu': mu}

    def get_density(self, altitude: float) -> float:
        """Get air density at a given altitude using USSA1976

        Parameters
        ----------
        altitude: [m]
            altitude

        Returns
        -------
        rho: [kg/m^3]
            density
        """
        return np.interp(altitude, self._altitudes, self._rho)

    def get_temperature(self, altitude: float) -> float:
        """Get air temperature at a given altitude using USSA1976

            Parameters
            ----------
            altitude: [m]
                altitude

            Returns
            -------
            temp: [K]
                temperature
        """
        return np.interp(altitude, self._altitudes, self._t)

    def get_speed_of_sound(self, altitude: float) -> float:
        """Get speed of sound at a given altitude using USSA1976

            Parameters
            ----------
            altitude: [m]
                altitude

            Returns
            -------
            cs: [m/s]
                speed of sound
        """
        return np.interp(altitude, self._altitudes, self._son)

def get_cp_air_properties(pressure: float,
                          temperature: float,
                          mass_ratio: float = 0.0
                          ) -> np.ndarray:
    """Get air properties using CoolProp

        Parameters
        ----------
        pressure: [Pa]
            pressure
        temperature: [K]
            temperature
        mass_ratio: [kg water/kg dry air]
            kg water/kg dry air

        Returns
        -------
        return_array: [See AirPropertiesIndex]
            Property array with columns set by AirPropertiesIndex
    """
    rho = HAPropsSI('D', 'P', pressure, 'T', temperature, 'W', mass_ratio)
    kine_visc = HAPropsSI('V', 'P', pressure, 'T', temperature, 'W', mass_ratio) / rho
    cp = HAPropsSI('C', 'P', pressure, 'T', temperature, 'W', mass_ratio)
    rel_humidity = HAPropsSI('R', 'P', pressure, 'T', temperature, 'W', mass_ratio)
    return_array = np.empty(len(AirPropertiesIndex))
    return_array[AirPropertiesIndex.ALTITUDE] = np.nan  # Altitude is not known here
    return_array[AirPropertiesIndex.DENSITY] = rho
    return_array[AirPropertiesIndex.KINEMATIC_VISCOSITY] = kine_visc
    return_array[AirPropertiesIndex.TEMPERATURE] = temperature
    return_array[AirPropertiesIndex.PRESSURE] = pressure
    return_array[AirPropertiesIndex.SPECIFIC_HEAT_PRESSURE] = cp
    return_array[AirPropertiesIndex.HUMIDITY_RELATIVE] = rel_humidity
    return return_array

def plot_air_properties(air_array: np.ndarray) -> None:
    """Plot air properties vs altitude

        Parameters
        ----------
        air_array: [See AirPropertiesIndex]
            array of shape (N, len(AirPropertiesIndex)) with columns set by AirPropertiesIndex
        """
    plt.figure()
    plt.plot(air_array[:,AirPropertiesIndex.DENSITY],
             air_array[:,AirPropertiesIndex.ALTITUDE],
             label="Rho (kg/m^3)")
    plt.ylabel("Altitude (m)")
    plt.xlabel("Rho (kg/m^3)")
    plt.title("Atmospheric Properties vs Altitude")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    plt.figure()
    plt.plot(air_array[:,AirPropertiesIndex.TEMPERATURE],
             air_array[:,AirPropertiesIndex.ALTITUDE],
             label="Temperature (K)")
    plt.ylabel("Altitude (m)")
    plt.xlabel("Temperature (K)")
    plt.title("Atmospheric Properties vs Altitude")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    plt.figure()
    plt.plot(air_array[:,AirPropertiesIndex.PRESSURE],
             air_array[:,AirPropertiesIndex.ALTITUDE],
             label="Pressure (Pa)")
    plt.ylabel("Altitude (m)")
    plt.xlabel("Pressure (Pa)")
    plt.title("Atmospheric Properties vs Altitude")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

def interp_air_properties(air_array: np.ndarray, altitude: float) -> OneDFloatArray:
    """Interpolate air properties at a given altitude from airArray

        Parameters
        ----------
        air_array: [See AirPropertiesIndex]
            array of shape (N, len(AirPropertiesIndex)) with columns set by AirPropertiesIndex
        altitude: [m]
            altitude

        Returns
        -------
        air_properties: [See AirPropertiesIndex]
            array of interpolated properties at the given altitude
    """
    air_properties = np.empty(len(AirPropertiesIndex))
    for i in AirPropertiesIndex:
        air_properties[i] = np.interp(altitude,
                                      air_array[:,AirPropertiesIndex.ALTITUDE],
                                      air_array[:,i])
    return air_properties

#region Archive
# def get_wind_speed(altitude: float, wind_profile: np.ndarray | float) -> float:
#     """Get wind speed at a given altitude from windProfile
#         Args:
#             altitude: [m]
#                 altitude in meters
#             windProfile: [m, m/s]
#                 array of shape (N, 2) with columns [altitude, wind speed]
#
#         Returns:
#             wind_speed: wind speed in m/s at the given altitude
#     """
#     if type(wind_profile) == float or type(wind_profile) == int:
#         return wind_profile
#     elif type(wind_profile) == np.ndarray:
#         windSpeed = float(np.interp(altitude, wind_profile[:,0], wind_profile[:,1]))
#         return windSpeed
#     else:
#         raise TypeError("windProfile must be a float or a numpy array")
#
# def createRampedWindProfile(velocity: float,
#                             alt_start: float,
#                             alt_end: float,
#                             n_points: int = 1000) -> np.ndarray:
#     """Create a ramped wind profile from 0 m/s at alt_start
#     to velocity m/s at alt_end
#
#
#     Parameters
#     ----------
#         velocity: [m/s]
#             Wind speed in m/s at alt_end
#         alt_start: [m]
#             altitude in meters where wind starts to ramp up
#         alt_end: [m]
#             altitude in meters where wind reaches velocity
#         n_points: [-]
#             number of points in the wind profile
#
#     Returns
#     -------
#         wind_profile: [m, m/s]
#             array of shape (n_points, 2) with columns [altitude, wind speed]
#     """
#     alt_ramp = np.linspace(alt_start, alt_end, n_points//2)
#     wind_speed_ramp = np.linspace(0.0, velocity, n_points//2)
#     wind_profile_ramp = np.column_stack((alt_ramp, wind_speed_ramp))
#     alt_const = np.logspace(alt_end, 100e3, n_points//2)
#     wind_speed_const = velocity * np.ones_like(alt_const)
#     wind_profile_const = np.column_stack((alt_const, wind_speed_const))
#     wind_profile = np.vstack((wind_profile_ramp, wind_profile_const))
#     return wind_profile
#endregion Archive
