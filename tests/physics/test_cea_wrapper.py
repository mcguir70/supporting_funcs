"""Test the cea_wrapper module."""
#TODO: Verify that all cea_wrapper.py properties, methods, and functions are in cea_wrapper.pyi and vice versa. # noqa: E501
#TODO: Add Pi_p and Ae/At tests
import re
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.constants as csp
from numpy.typing import NDArray
from pint import UnitRegistry

from supporting_funcs.coding.type_hints import FloatArray
from supporting_funcs.physics.cea_wrapper import (
    Propellent,
    PropellentMixture,
)

ureg = UnitRegistry()
Q_ = ureg.Quantity # type: ignore

_PLAIN_FLOAT_ROW_RE = re.compile(r"(?:\s+\d+\.\d+)+\s*")
_FLOAT_REGEX = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

@pytest.fixture(scope="session")
def area_ratios() -> NDArray[np.float64]:
    """Nozzle expansion area ratios for vectorised tests.

    Returns
    -------
    area_ratios: [1]
        Area ratio (Ae/At), 1.0 to 100.0 in steps of 1.0
    """
    area_ratios = np.linspace(1, 100, 100)
    area_ratios.flags.writeable = False  # Session-shared; block in-place mutation
    return area_ratios

def _error_search(regex_pattern: str, line: str, inst: int = 0) -> str:
    val = re.search(regex_pattern, line)
    if val:
        return val.group(inst)
    return ""

class TestCEAComputations:
    """Test the calculations of the CEA wrapper against the CEA website output."""

    input_path = r"tests/physics/inputs/mon_mmh_p150.txt"
    good_val = ""

    with open(input_path) as f:
        good_val = f.read()

    good_lines = good_val.splitlines()

    p_c = float(_error_search(r"p,psia= (\d+)", good_lines[19], 1)) * csp.psi  # Convert psia to Pa
    pi_p = float(_error_search(r"pi/p= (\d+\.?\d*)", good_lines[21], 1))
    o_f_ratio = float(_error_search(r"o/f = (\d+\.?\d*)", good_lines[24], 1))

    fuel_ox_row = next(i for i,
                       line in enumerate(good_lines)
                       if line.startswith(" # You selected the following fuels and oxidizers:"))
    output_row = next(i for i,
                      line in enumerate(good_lines)
                      if line.startswith(" # You selected these options for output:"))

    cea_website = SimpleNamespace()
    cea_website.fuels = []
    cea_website.oxidizers = []

    @pytest.mark.skip(reason="Helper Function to parse the CEA output file. Not a test.")
    @staticmethod
    def parse_cea_line(line: str,
                       key: str,
                       expected_qty: int
                       ) -> tuple[None | FloatArray,
                                  float | None]:
        """Parse a line from the CEA output file for a given key and expected quantity of values."""
        replace_str = {
            "SEC": "second",
            "M": "meter",
            "BAR": "bar",
            "KJ": "kilojoule",
            "KG": "kilogram",
            "CU meter": "meter**3",
        }

        def re_replace_fortran_exponents(s: str) -> str:
            return re.sub(r"([+-]\d+)", r'e\1', s)
        def parse_unit(line: str) -> float | None:
            if ',' not in line:
                return 1.0 # Dimensionless
            head = line.split(key + ',')[-1].strip()
            if _PLAIN_FLOAT_ROW_RE.match(head): # if there is no unit
                return 1.0
            try:
                float(head.split("  ")[0].strip())
                return 1.0
            except ValueError:
                pass
            s = head.split("  ")[0].strip()
            for unit, factor in replace_str.items():
                s = s.replace(unit, factor)
            return round(Q_(s).to_base_units().magnitude, 12) if s else None

        if key not in line:
            return None, None

        vals = re.findall(_FLOAT_REGEX, re_replace_fortran_exponents(line))
        return_val = None
        scalar = None
        if len(vals) == expected_qty:
            return_val =  np.asarray([float(v) for v in vals])
        elif len(vals) > expected_qty:
            if "(1/n)" in line and vals[0] == "1":
                # remove the (1/n) from the line
                line = line.replace("(1/n)", "")
                vals = re.findall(_FLOAT_REGEX, line)
                scalar = 1 / 1000 # kg/kmol -> kg/mol
            if len(vals) > expected_qty:
                raise ValueError(f"Found more values than expected for {key}: {vals}")

        if return_val is None:
            rev_return_arr = np.full(expected_qty, np.nan)
            rev_texts = line.split('   ')[::-1]
            vals = vals[::-1]
            vals_idx = 0

            # Working from the back of the list, any empty strings should be filled in with zeros
            for i in range(expected_qty):
                if rev_texts[i].isspace():
                    rev_return_arr[i] = np.nan
                    continue
                try:
                    if float(rev_texts[i]) == float(vals[vals_idx]):
                        rev_return_arr[i] = float(vals[vals_idx])
                        vals_idx += 1
                    else:
                        raise ValueError(f"Value mismatch for {key}: " \
                                         f"{vals[vals_idx]} vs {rev_texts[i]}")
                except ValueError as e:
                    if "could not convert string to float" in str(e):
                        continue
                    raise ValueError(f"Could not convert value to float for {key}: "
                                    f"{vals[vals_idx]} vs {rev_texts[i]}") from e
            return_val = rev_return_arr[::-1]
        if scalar is None:
            scalar = parse_unit(line)
        return return_val, scalar

    for i in range(fuel_ox_row + 1, output_row):
        line = good_lines[i]

        if line.startswith(" reac"):
            continue
        elif line.startswith(" fuel"):
            is_fuel = True
        elif line.startswith(" oxid"):
            is_fuel = False
        else:
            continue

        chemical_name = line.split()[1]
        if "," in chemical_name:
            chemical_name = chemical_name.split(",")[0] # "CH6N2(L),MMH" "CH6N2(L)"
        weight_percent = float(_error_search(r"wt%=\s*(\d+\.?\d*)", line, 1))
        temperature = float(_error_search(r"t,k=\s*(\d+\.?\d*)", line, 1))

        (cea_website.fuels if is_fuel else cea_website.oxidizers).append({
            "chemical_name": chemical_name,
            "weight_percent": weight_percent,
            "temperature": temperature
        })

    online_keys = {
        "Isp": ("Isp", 193),
        "Isp_vacuum": ("Ivac", 192),
        "M": ("M", 179),
        "Mach": ("MACH NUMBER", 185),
        "P": ("P", 171),
        "T": ("T", 172),
        "ae_at": ("Ae/At", 189),
        "c_star": ("CSTAR", 190),
        "coefficient_of_thrust": ("CF", 191),
        "cp": ("Cp", 182),
        "density": ("RHO", 173),
        "energy": ("U", 175), # May be incorrect
        "enthalpy": ("H", 174),
        "entropy": ("S", 177),
        "gamma_s": ("GAMMAs", 183),
        "gibbs_energy": ("G", 176),
        "sonic_velocity": ("SON VEL", 184)
    }

    for key, (label, line_num) in online_keys.items():
        value = None
        scalar = None
        if key in good_lines[line_num] or label in good_lines[line_num]:
            value, scalar = parse_cea_line(good_lines[line_num], label, 3)
        if value is None:
            for i in range(line_num - 5, line_num + 5):
                if i < 0 or i >= len(good_lines):
                    continue
                value, scalar = parse_cea_line(good_lines[i], label, 3)
                if value is not None:
                    break
            else:
                for _, line in enumerate(good_lines):
                    value, scalar = parse_cea_line(line, label, 3)
                    if value is not None:
                        break
                else:
                    raise ValueError(f"Could not find value for {key} in CEA output file.")

        if scalar is None:
            raise ValueError(f"Could not find scalar for {key} in CEA output file.")
        setattr(cea_website, key, value)
        setattr(cea_website, key + "_scalar", scalar)

    def test_raw_computation_matches_website(self):
        """Check that the CEA wrapper's raw solution matches the CEA website directly."""
        fuels = [Propellent(name=fuel["chemical_name"],
                            percent_mass=fuel["weight_percent"],
                            temperature=fuel["temperature"])
                            for fuel in self.cea_website.fuels]
        oxidizers = [Propellent(name=oxidizer["chemical_name"],
                                percent_mass=oxidizer["weight_percent"],
                                temperature=oxidizer["temperature"])
                                for oxidizer in self.cea_website.oxidizers]

        propellents = PropellentMixture(oxidizers=oxidizers, fuels=fuels, o_f_ratio=self.o_f_ratio)
        propellents.solve(p_c=self.p_c, pi_p=self.pi_p)
        raw_soln = propellents.solution.raw_solution
        # Check each attribute in the solution against the expected value from the CEA website
        failed_keys = []

        for key in self.online_keys.keys():
            expected_value = getattr(self.cea_website, key)
            actual_value = getattr(raw_soln, key)

            zero_mask = (expected_value == 0) & (actual_value == 0)
            expected_value = expected_value[~zero_mask]
            actual_value = actual_value[~zero_mask]

            if any(abs(expected_value - actual_value) / abs(expected_value) >= 1e-3):
                failed_keys.append((key, expected_value, actual_value))
        if failed_keys:
            raise AssertionError(f"Failed: {len(failed_keys)}\n"
                                 + "\n".join([f"- Key: {key}, Expected: {expected}, Got: {actual}"
                                              for key, expected, actual in failed_keys]))

    def test_filtered_computation_matches_website(self):
        """Check that the CEA wrapper's SI base solution matches the CEA website."""
        key_dict = {
            "Isp": "effective_exhaust_velocity_ideal",
            "Isp_vacuum": "effective_exhaust_velocity_vacuum",
            "M":'gaseous_molecular_weight',
            "MW": 'overall_molecular_weight',
            "Mach": "mach",
            "energy": "internal_energy",
            "c_star": "characteristic_velocity",
            "Cf": "coefficient_of_thrust",
            "gamma_s": "ratio_of_specific_heats_sonic",
        }
        fuels = [Propellent(name=fuel["chemical_name"],
                            percent_mass=fuel["weight_percent"],
                            temperature=fuel["temperature"])
                            for fuel in self.cea_website.fuels]
        oxidizers = [Propellent(name=oxidizer["chemical_name"],
                                percent_mass=oxidizer["weight_percent"],
                                temperature=oxidizer["temperature"])
                                for oxidizer in self.cea_website.oxidizers]

        propellents = PropellentMixture(oxidizers=oxidizers, fuels=fuels, o_f_ratio=self.o_f_ratio)
        propellents.solve(p_c=self.p_c, pi_p=self.pi_p)

        # Check each attribute in the solution against the expected value from the CEA website
        failed_keys = []

        for key in self.online_keys.keys():
            expected_value = getattr(self.cea_website, key)
            expected_value *= getattr(self.cea_website, key + "_scalar")

            if key in key_dict:
                custom_key = key_dict[key]
            else:
                custom_key = key
            actual_value = getattr(propellents.solution, custom_key)
            zero_mask = (expected_value == 0) & (actual_value == 0)
            expected_value = expected_value[~zero_mask]
            actual_value = actual_value[~zero_mask]
            if any(abs(expected_value - actual_value) / abs(expected_value) >= 1e-3):
                failed_keys.append((key, expected_value, actual_value))
        if failed_keys:
            raise AssertionError(f"Failed: {len(failed_keys)}\n"
                                 + "\n".join([f"- Key: {key}, Expected: {expected}, Got: {actual}"
                                              for key, expected, actual in failed_keys]))

class TestCEAOutput:
    """Test the implementation of the CEA wrapper for consistency."""

    def test_aliases(self, area_ratios):
        """Check that aliases in the CEA wrapper are consistent with their primary attributes."""
        mixture = PropellentMixture(
            oxidizers=[Propellent(name="N2O4(L)", percent_mass=0.97),
                       Propellent(name="NO", percent_mass=0.03)],
            fuels=[Propellent(name="CH6N2(L)", percent_mass=1.0)],
            o_f_ratio=1.65
        )
        mixture.solve(p_c=150 * csp.psi, pi_p=150 * csp.psi / 0.05)

        np.testing.assert_allclose(mixture.solution.v_eff_i,
                                   mixture.solution.effective_exhaust_velocity_ideal)
        np.testing.assert_allclose(mixture.solution.v_eff_v,
                                   mixture.solution.effective_exhaust_velocity_vacuum)
        np.testing.assert_allclose(mixture.solution.isp_i,
                                   mixture.solution.isp_ideal)
        np.testing.assert_allclose(mixture.solution.isp_v,
                                   mixture.solution.isp_vacuum)
        np.testing.assert_allclose(mixture.solution.T,
                                   mixture.solution.temperature)
        np.testing.assert_allclose(mixture.solution.P,
                                   mixture.solution.pressure)
        np.testing.assert_allclose(mixture.solution.rho,
                                   mixture.solution.density)
        np.testing.assert_allclose(mixture.solution.h,
                                   mixture.solution.enthalpy)
        np.testing.assert_allclose(mixture.solution.u,
                                   mixture.solution.internal_energy)
        np.testing.assert_allclose(mixture.solution.g,
                                   mixture.solution.gibbs_energy)
        np.testing.assert_allclose(mixture.solution.s,
                                   mixture.solution.entropy)
        # No Mach alias
        # no total_MW alias
        np.testing.assert_allclose(mixture.solution.MW_gas,
                                   mixture.solution.gaseous_molecular_weight)
        np.testing.assert_allclose(mixture.solution.cp,
                                   mixture.solution.specific_heat_capacity_constant_pressure)
        np.testing.assert_allclose(mixture.solution.cv,
                                   mixture.solution.specific_heat_capacity_constant_volume)
        np.testing.assert_allclose(mixture.solution.gamma_s,
                                   mixture.solution.ratio_of_specific_heats_sonic)
        np.testing.assert_allclose(mixture.solution.c_star,
                                   mixture.solution.characteristic_velocity)
        np.testing.assert_allclose(mixture.solution.c_f,
                                   mixture.solution.coefficient_of_thrust)
        np.testing.assert_allclose(mixture.solution.c,
                                   mixture.solution.sonic_velocity)
        # No gas velocity alias

        np.testing.assert_allclose(mixture.solution.get_isp_v(ae_at=area_ratios),
                                   mixture.solution.get_isp_vacuum(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_isp_i(ae_at=area_ratios),
                                   mixture.solution.get_isp_ideal(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_T(ae_at=area_ratios),
                                   mixture.solution.get_temperature(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_P(ae_at=area_ratios),
                                   mixture.solution.get_pressure(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_rho(ae_at=area_ratios),
                                   mixture.solution.get_density(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_h(ae_at=area_ratios),
                                   mixture.solution.get_enthalpy(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_u(ae_at=area_ratios),
                                   mixture.solution.get_internal_energy(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_g(ae_at=area_ratios),
                                   mixture.solution.get_gibbs_energy(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_s(ae_at=area_ratios),
                                   mixture.solution.get_entropy(ae_at=area_ratios))
        # no get_Mach alias
        np.testing.assert_allclose(mixture.solution.get_MW_gas(ae_at=area_ratios),
                                   mixture.solution.get_gaseous_molecular_weight(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_cp(ae_at=area_ratios),
                                   mixture.solution.get_constant_pressure_specific_heat(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_cv(ae_at=area_ratios),
                                   mixture.solution.get_constant_volume_specific_heat(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_gamma_s(ae_at=area_ratios),
                                   mixture.solution.get_ratio_of_specific_heats_sonic(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_c(ae_at=area_ratios),
                                   mixture.solution.get_sonic_velocity(ae_at=area_ratios))
        # no get_gas_velocity alias
        np.testing.assert_allclose(mixture.solution.get_c_star(ae_at=area_ratios),
                                   mixture.solution.get_characteristic_velocity(ae_at=area_ratios))
        np.testing.assert_allclose(mixture.solution.get_c_f(ae_at=area_ratios),
                                   mixture.solution.get_coefficient_of_thrust(ae_at=area_ratios))


        np.testing.assert_allclose(mixture.solution.effective_exhaust_velocity_ideal,
                                   mixture.solution.isp_ideal * csp.g)
        np.testing.assert_allclose(mixture.solution.effective_exhaust_velocity_vacuum,
                                   mixture.solution.isp_vacuum * csp.g)

class TestPropellent:
    """Test the Propellent class for correct behavior."""

    def test_repr(self):
        """Test the __repr__ method of the Propellent class."""
        prop = Propellent(name="CH6N2(L)", percent_mass=1.0, temperature=298.15)
        expected_repr = "Propellent(name=CH6N2(L), percent_mass=1.0, temperature=298.15)"
        assert repr(prop) == expected_repr

    def test_invalid_name_type(self):
        """Test that setting an invalid name type raises a TypeError."""
        with pytest.raises(TypeError):
            Propellent(name=123, percent_mass=1.0, temperature=298.15) #type:ignore # Invalid name type

    def test_invalid_percent_mass_type(self):
        """Test that setting an invalid percent_mass type raises a TypeError."""
        with pytest.raises(TypeError):
            Propellent(name="CH6N2(L)", percent_mass="1.0", temperature=298.15)  #type:ignore # Invalid percent_mass type
        with pytest.raises(TypeError):
            Propellent(name="CH6N2(L)", percent_mass=None, temperature=298.15)  #type:ignore # Invalid percent_mass type
        with pytest.raises(TypeError):
            Propellent(name="CH6N2(L)", percent_mass=[1.0], temperature=298.15)  #type:ignore # Invalid percent_mass type
        with pytest.raises(ValueError):
            Propellent(name="CH6N2(L)", percent_mass=-1.0, temperature=298.15)  #type:ignore # Invalid percent_mass value

    def test_monopropellant(self):
        """Test that a monopropellant mixture is correctly identified."""
        monopropellant = [Propellent(name="H2O2(L)", percent_mass=0.90, temperature=298.15),
                          Propellent(name="H2O(L)", percent_mass=0.10, temperature=298.15)]
        mixture_fu = PropellentMixture(oxidizers=[], fuels=monopropellant, o_f_ratio=0.0)
        mixture_ox = PropellentMixture(oxidizers=monopropellant, fuels=[], o_f_ratio=np.inf)

        mixture_fu.solve(p_c=Q_(500, "psi").to("Pa").magnitude, ae_at=50.0)
        mixture_ox.solve(p_c=Q_(500, "psi").to("Pa").magnitude, ae_at=50.0)
