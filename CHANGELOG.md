# Changelog
Note for the 0.X.X branch, semantic versioning is not followed.

## 0.0.7
### 2026-10-14
Updated bartz.py to utilize cea_wrapper.py (instead of the old CEA2 wrapper). 
Updated cea_wrapper.py to allow calculation of transport properties (enabled by default) and added some mole ratio capabilities.
Added some support to test_cea_wrapper for new functions (not full coverage yet).

## 0.0.6
### 2026-10-12
Removed the cea_wrapper.pyi file. All of the overload type hints have been added directly to the cea_wrapper.py file, so the pyi file is no longer necessary (.pyi was missing some .py functions so this way there is no longer a mismatch).

## 0.0.5
### 2026-10-05
Added cea_wrapper.py under physics. This allows for the use of the CEA solver in Python.

## 0.0.4
### 2026-10-04
Added a matplotlib type alias to the type_hints module, and updated the format_plot function to use this more accurate type hint.

## 0.0.3
### 2026-10-02
Added a few type aliases to the type_hints module.

## 0.0.2
### Sep 2026
Added a get_altitude_equivalent function to the atmosphere model. This function returns the altitude equivalent of a given pressure value.
Minor edits to the docstrings

## 0.0.1
### Sep 2026
Corrected Dynamic vs Kinematic properties in the atmosphere model.

## 0.0.0
### Aug 2026
Start of versioning. This is the first version of the changelog.