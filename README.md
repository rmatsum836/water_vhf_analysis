# Water Van Hove Function Analysis

This package contains Python functions and raw data to analyze the Van Hove function.

## Project Overview
The goal of this project is to analyze various water models through the calculation of the
van hove function.  

The following models are studied:
- SPC/E
- TIP3P\_EW
- BK3
- ReaxFF CHON-2017\_weak
- DFTB 3obw
- AIMD optB88

## Simulation Details
All simulations have been run in the canoncial (constant temperature, volume, and molecules)
ensemble.

## Package Details
Raw van hove function data is contained in the `data` directory.  
The total van hove data is contained within the `nvt_total_data` directories and the partial van
hove data is contained within the `nvt_partial_data` directories.
Plotting and analysis functions are contained in the `analysis` directory.  
Unit tests are contained within `tests` to verify the results of the analysis functions.

## Required Packages:
- [MDTraj](https://github.com/mdtraj/mdtraj)
- [scattering](https://github.com/mattwthompson/scattering)
- NumPy
- SciPy
- matplotlib
- Seaborn
- PyTest

## Docker Image
Docker image is in progress to setup an environment to use the package.

## Jupyter Notebook
For an overview of analysis functions in this package, a Jupyter notebook is provided that
produces the plots shown in the paper.
