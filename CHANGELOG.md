# Changelog
Note for the 0.X.X branch, semantic versioning is not followed.

## 0.1.1
Connor McGuire
### 2026-10-23
Corrected further Bartz function unit errors.

## 0.1.0
Connor McGuire
### 2026-10-18
Corrected some issues with Bartz function calculations (Unit errors).
Updated thermals to be able to return the temperature distribution for each leaf layer instead of just the top-level layers.
Steady State Temperature Distribution was corrected for sign errors.

## 0.0.A
Connor McGuire
### 2026-10-17
Changed compute_steady_state_temperature_distribution to return all surfaces instead of just the inner boundary temperature.
Formatting changes to many files to bring into compliance with ruff.
Removed TODO: XXX from ruff checks for git commit checks, but left them in the .toml. This allows for the TODOs to be seen in the code, but not block commits.

## 0.0.9
Connor McGuire
### 2026-10-15
Updated formatting on a number of files
Changed Ae_At to area_ratio in compressible_flow.py for consistency with other functions

## 0.0.8
Connor McGuire
### 2026-10-14
bartz.py: Added a new function set to calculate the viscosity coefficient for a given CEAOutput object mu(T) = a * T^b.
cea_wrapper.py: made h_chamb_kj_kg an optional argument for the solve function, and gave PropellentMixture.cea_h_chamb so that iteration is easier.
nozzle.py: added a new function to calculate a parabolic nozzle
thermals.py: added a new function to calculate the inner boundary temperature and heat flux for a 1-D resistor network

## 0.0.7
Connor McGuire
### 2026-10-14
Updated bartz.py to utilize cea_wrapper.py (instead of the old CEA2 wrapper). 
Updated cea_wrapper.py to allow calculation of transport properties (enabled by default) and added some mole ratio capabilities.
Added some support to test_cea_wrapper for new functions (not full coverage yet).

## 0.0.6
Connor McGuire
### 2026-10-12
Removed the cea_wrapper.pyi file. All of the overload type hints have been added directly to the cea_wrapper.py file, so the pyi file is no longer necessary (.pyi was missing some .py functions so this way there is no longer a mismatch).

## 0.0.5
Connor McGuire
### 2026-10-05
Added cea_wrapper.py under physics. This allows for the use of the CEA solver in Python.

## 0.0.4
Connor McGuire
### 2026-10-04
Added a matplotlib type alias to the type_hints module, and updated the format_plot function to use this more accurate type hint.

## 0.0.3
Connor McGuire
### 2026-10-02
Added a few type aliases to the type_hints module.

## 0.0.2
Connor McGuire
### Sep 2026
Added a get_altitude_equivalent function to the atmosphere model. This function returns the altitude equivalent of a given pressure value.
Minor edits to the docstrings

## 0.0.1
Connor McGuire
### Sep 2026
Corrected Dynamic vs Kinematic properties in the atmosphere model.

## 0.0.0
Connor McGuire
### Aug 2026
Start of versioning. This is the first version of the changelog.