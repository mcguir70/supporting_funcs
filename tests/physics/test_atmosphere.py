import warnings

import numpy as np
import pytest
import scipy.optimize as scipy_optimize
from tests.physics.conftest import RTOL_SOLVER, RTOL_TABLE

from supporting_funcs.physics import atmosphere


class TestAtmosphere:
    air_properties = atmosphere.AirProperties()
    #TODO: Add more tests for the atmosphere module, including edge cases and error handling.

    def test_get_altitude_equivalent(self):
        """Check that the get_altitude_equivalent function returns the correct altitude for a given pressure."""
        # Test with a known pressure value
        pressure = 101325.0  # 1 atm in Pa
        expected_altitude = 0.0  # Sea level

        actual_altitude = self.air_properties.get_altitude_equivalent(pressure)

        assert np.isclose(actual_altitude, expected_altitude, rtol=RTOL_TABLE)

    def test_get_altitude_equivalent_invalid_pressure(self):
        """Check that the get_altitude_equivalent function returns NaN for invalid values."""
        invalid_pressures = [-100.0, 0.0]  # Negative and zero pressures
        mixed_invalid_pressures = np.array([-100.0, 0.0, 101_325.0, 200_000.0, 50_000.0])  # Mixed valid and invalid pressures
        mixed_invalid_pressure_mask = ((mixed_invalid_pressures <= 0)
                                       | (mixed_invalid_pressures > 101_325.0))

        for pressure in invalid_pressures:
            assert np.isnan(self.air_properties.get_altitude_equivalent(pressure))

        assert np.all(np.isnan(self.air_properties.get_altitude_equivalent(mixed_invalid_pressures))
                == mixed_invalid_pressure_mask)


    def check_round_trip(self, pressure):
        """Check that the get_altitude_equivalent function returns the correct altitude for a given pressure."""
        altitude = self.air_properties.get_altitude_equivalent(pressure)
        round_trip_pressure = self.air_properties.get_pressure(altitude)

        assert np.isclose(round_trip_pressure, pressure, rtol=RTOL_SOLVER)
