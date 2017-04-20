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
### Flip a random spin on a lattice
################################################################################

def SpinFlip(n, lattice) :
    random.seed()
    row = int(random.random()*n)                # random row
    col = int(random.random()*n)                # random column
    lattice[row][col] = - lattice[row][col]     # negative sign switches flip
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
### For Magnetization in the presence of a heatbath: M = SUM (M_a * P_a)
################################################################################

def MeasureMagnetization(magnetization, lattice, energy_microstate, kB, temperature) :
    # M_a is the microstate magnetization -- equivalent to the sum of all spins
    magnetization_microstate = np.sum(lattice)

    # Now, multiply by the Metropolis Probability
    magnetization = (magnetization_microstate * np.exp(- energy_microstate / (kB * temperature)))
    return magnetization

################################################################################
### Metropolis Algorithm
################################################################################

def MetropolisAlgorithm(energy_microstate, hypothetical_energy_microstate, kB, temperature, n, lattice, evolved_lattice) :
    # Calculate the change in energy_microstate
    dE = (hypothetical_energy_microstate - energy_microstate)

    # Check values
    if dE < 0 :
        # Keep the flipped spin
        return evolved_lattice, hypothetical_energy_microstate
    elif dE > 0 :
        # Compute the exponential... somehow this can be sped up by tabulating?
        exponential = np.exp(- dE / (kB * temperature))

        # Generate a random number r
        r = random.random()

        # Compare to the exponential
        if r <= exponential :
            # Keep the flipped spin
            return evolved_lattice, hypothetical_energy_microstate
        else :
            # Don't keep the flipped spin
            return lattice, energy_microstate

# INCOMPLETE
