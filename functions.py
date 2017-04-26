### Tahoe Schrader
### Computational Physics
### Spring, 2017

################################################################################
### This code will generate the functions needing to solve the 2D Ising Model.
### Our problem uses an n x n lattice with periodic boundary conditions. There
### is a nearest neighbor interactions strength of J = 1.5 and no external
### magnetic field. 
###         a) Plots magnetization as a function of temperature to extract the
###             critical temperature.
###         b) Calculates specific heat per spin for multiple lattices, then
###             verifies the finite-size scaling relation
################################################################################

# ------------------------------------------------------------------------------

import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

################################################################################
### Initialize the Ising Model and compute its energy
### A 2D Lattice with random up/down spins and periodic boundary conditions
################################################################################

def InitializeIsingModel(J, n) :
    lattice = np.zeros((n,n))

    # a) Randomly assign each lattice site a spin up or down (1 vs -1)
    random.seed()
    lattice = [[1 if random.random() > .5 else -1 for row in np.arange(n)] for col in np.arange(n)]

    # b) Grab the initial energy of this system
    # Initialize a list to store s_i * s_j values of all nearest neighbors
    spin_products = []

    for row in np.arange(n) :
        for col in np.arange(n) :
            spin = lattice[row][col]
            # ---------------------------------------------------------------
            # Periodic boundary conditions -- Toroidal Symmetry
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

    return lattice, energy

################################################################################
### Flip a random spin on a lattice
################################################################################

def SpinFlip(n, lattice, J) :
    # 1. Duplicate the lattice
    hypothetical_lattice = np.copy(lattice)

    # 2. Pick a random position
    random.seed()
    row = int(random.random()*n)                # random row
    col = int(random.random()*n)                # random column

    # 3. Flip the spin
    hypothetical_lattice[row][col] = - hypothetical_lattice[row][col]

    # 4. Get the energy change involved in this flip... only need neighbors
    # around this single flip, rather than all neighbors in entire lattice
    spin_products = []
    hypothetical_spin_products = []

    spin = lattice[row][col]
    hypothetical_spin = hypothetical_lattice[row][col]

    # ---------------------------------------------------------------
    # Periodic boundary conditions -- Toroidal Symmetry
    # ---------------------------------------------------------------
    if row == 0 : # top boundary
        spin_products.append(lattice[n - 1][col] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[n - 1][col] * hypothetical_spin)
    elif row != 0 :
        spin_products.append(lattice[row - 1][col] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[row - 1][col] * hypothetical_spin)
    # ---------------------------------------------------------------
    if row == n - 1 : # bottom boundary
        spin_products.append(lattice[0][col] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[0][col] * hypothetical_spin)
    elif row != n - 1 :
        spin_products.append(lattice[row + 1][col] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[row + 1][col] * hypothetical_spin)
    # ---------------------------------------------------------------
    if col == 0 : # left boundary
        spin_products.append(lattice[row][n - 1] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[row][n - 1] * hypothetical_spin)
    elif col != 0 :
        spin_products.append(lattice[row][col - 1] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[row][col - 1] * hypothetical_spin)
    # ---------------------------------------------------------------
    if col == n - 1 : # right boundary
        spin_products.append(lattice[row][0] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[row][0] * hypothetical_spin)
    elif col != n - 1 :
        spin_products.append(lattice[row][col + 1] * spin)
        hypothetical_spin_products.append(hypothetical_lattice[row][col + 1] * hypothetical_spin)
    # ---------------------------------------------------------------

    # Sum all of the spin products
    neighbor_spin_sums = np.sum(spin_products)
    hypothetical_neighbor_spin_sums = np.sum(hypothetical_spin_products)

    # Multiply by - J to get energy
    energy = - J * (neighbor_spin_sums)
    hypothetical_energy = - J * (hypothetical_neighbor_spin_sums)

    # Subtract to get dE
    hypothetical_dE = hypothetical_energy - energy

    return hypothetical_lattice, hypothetical_dE

################################################################################
### Measure the magnetization of the system
### For Magnetization: M_tot = N_spins * <s> = N_spins * (1/N_spins)SUM (s_i)
################################################################################

def MeasureMagnetization(lattice) :
    # Total magnetization is just the sum of all spins in the lattice
    magnetization= np.sum(lattice)

    return magnetization

################################################################################
### Metropolis Algorithm
################################################################################

def MetropolisAlgorithm(dE_exp, hypothetical_dE, lattice, hypothetical_lattice, energy) :
    # Generate a random number r
    r = random.random()

    # Compare to the exponential
    if r <= dE_exp :
        # Keep the flipped spin by accepting the hypothetical lattice and energy
        # change
        lattice = np.copy(hypothetical_lattice)
        energy = energy + hypothetical_dE
        return lattice, energy
    else :
        # Don't keep the flipped spin and return the old lattice and energy
        return lattice, energy

################################################################################
### Energy exponential
### Based on neighbors, the total sum can change by 0 or +/- 4, 8
### However, the metropolis algorithm only cares about positive energies, which
### correspond to negative sums (not 0 or positive sums... so we store 2 values)
################################################################################

def EnergyExponential(t, J) :
    # Negative sums
    possible_negative_sums = [-4, -8]

    # Resulting energies
    dEs = [-J * i for i in possible_negative_sums]

    # Resulting exponentials for each, in the form of a dictionary
    if t != 0 :
        dE_exp = {dEs[0]:np.exp(-dEs[0]/t), dEs[1]:np.exp(-dEs[1]/t)}
    elif t == 0 :
        dE_exp = {dEs[0]:0, dEs[1]:0}
    return dE_exp

################################################################################
### Measure Energy
### This function will flip spins, find hypothetical energies, and run the
### Metropolis Algorithm if needed. It should spit back out energy and lattice.
################################################################################

def MeasureEnergy(n, lattice, J, energy, exponentials) :
    # 1. Flip a random spin and find the hypothetical change in energy, dE
    hypothetical_lattice, hypothetical_dE = SpinFlip(n, lattice, J)

    # 2. Check dE to see if we need to run the Metropolis algorithm
    if hypothetical_dE <= 0 :
        # 3a. Keep the flipped spin
        lattice = np.copy(hypothetical_lattice)
        energy = energy + hypothetical_dE
        return lattice, energy
    elif hypothetical_dE > 0 :
        # 3b. Run the metropolis algorithm to check if we keep the spin
        exponential = exponentials[hypothetical_dE]
        lattice, energy = MetropolisAlgorithm(exponential, hypothetical_dE, lattice, hypothetical_lattice, energy)
        return lattice, energy

# COMPLETE
#---------------------------------------------------------------------------------------
