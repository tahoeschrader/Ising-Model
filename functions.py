### Tahoe Schrader
### Computational Physics
### Spring, 2017

################################################################################
### This code will generate the functions needing to solve the 2D Ising Model.
### Our problem uses an n x n lattice with periodic boundary conditions. There
### is a nearest neighbor interactions strength of J = 1.5 and no external
### magnetic field. The metropolis algorithm is used to relax the system of spins
### to the desired temperature(s).
###         a) Plots magnetization as a function of temperature to extract the
###             critical temperature.
###         b) Calculates specific heat per spin for multiple lattices, then
###             verifies the finite-size scaling relation
################################################################################

# ------------------------------------------------------------------------------

import random
import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------------------------

# Begin a time tracker
start = time.time()
print "Code Initiated"
################################################################################
### Initialize a 2D Lattice with random up/down spins
################################################################################

def InitializeSpins(n) :
    lattice = np.zeros((n,n))

    # Randomly assign each lattice site a spin up or down (1 vs -1)
    random.seed()
    lattice = [[1 if random.random() >= .5 else -1 for row in range(n)] for col in range(n)]

    return lattice

################################################################################
### Measuring the Energy of the system
### For Energy: E = - J SUM (S_i*S_j) -- the sum is of nearest neighbor products
### Note: We want periodic boundary conditions for the lattice
################################################################################

def MeasureEnergy(lattice, n, J) :
    # Initialize a list to store s_i * s_j values of all nearest neighbors
    spin_products = []

    for row in range(n) :
        for col in range(n) :
            spin = lattice[row][col]
            # ---------------------------------------------------------------
            # Periodic boundary conditions
            # ---------------------------------------------------------------
            if row == 0 : # top boundary
                spin_products.append(lattice[n - 1][col] * spin)
            elif row != 0 :
                spin_products.append(lattice[row - 1][col] * spin)
            # ---------------------------------------------------------------
            if row == n - 1 : # bottom boundary
                spin_products.append(lattice[0][col] * spin)
            elif row != n - 1 :
                spin_products.append(lattice[row + 1][col] * spin)
            # ---------------------------------------------------------------
            if col == 0 : # left boundary
                spin_products.append(lattice[row][n - 1] * spin)
            elif col != 0 :
                spin_products.append(lattice[row][col - 1] * spin)
            # ---------------------------------------------------------------
            if col == n - 1 : # right boundary
                spin_products.append(lattice[row][0] * spin)
            elif col != n - 1 :
                spin_products.append(lattice[row][col + 1] * spin)
            # ---------------------------------------------------------------
    # Sum all of the spin products
    neighbor_spin_sums = np.sum(spin_products)

    # Summing every nearest neighbor results in each nearest neighbor pair being
    # double counted, so the sum must be divided by two to get the energy
    energy = - J * (neighbor_spin_sums / 2.0)
    return energy

################################################################################
### Measure the magnetization of the system
### For Magnetization: M = SUM (s_j) -- the sum is of individual points
################################################################################

def MeasureMagnetization(lattice) :
    magnetization = np.sum(lattice)
    return magnetization

########## TESTS
n = 500
lattice = InitializeSpins(n)
energy = MeasureEnergy(lattice, n, 1.5)
magnetization = MeasureMagnetization(lattice)

print energy
print magnetization

# Finish counting time
end = time.time()
print "This code finished after ", (end-start), " seconds"
