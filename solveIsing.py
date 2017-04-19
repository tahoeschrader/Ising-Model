### Tahoe Schrader
### Computational Physics
### Spring, 2017

################################################################################
### This code will use the functions from functions.py to solve the 2D Ising
### Model
################################################################################

# ------------------------------------------------------------------------------

from functions import InitializeSpins, MeasureSystem
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

################################################################################
### Define parameters
################################################################################

J = 1.5                 # nearest neighbor interaction strength
B = 0.0                 # no external magnetic field

lattice_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 200, 500]
