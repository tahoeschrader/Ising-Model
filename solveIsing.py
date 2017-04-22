### Tahoe Schrader
### Computational Physics
### Spring, 2017

################################################################################
### This code will use the functions from functions.py to solve the 2D Ising
### Model
################################################################################

# ------------------------------------------------------------------------------

from functions import InitializeIsingModel, MeasureMagnetization
from functions import SpinFlip, MetropolisAlgorithm, EnergyExponential
import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------------------------

# Begin a time tracker
start = time.time()
print "Code Initiated"

################################################################################
### Define parameters
################################################################################

J = 1.5                     # nearest neighbor interaction strength
B = 0.0                     # no external magnetic field
kB = 1                      # Boltzmann constant is 1 for this code

lattice_sizes = [3, 5]
#lattice_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 200, 500]

runs = 50                   # number of runs

T = np.linspace(5, 0, runs) # start at a high temperature and cool down

Nm = 50                     # number of times times we calculate energy before
                            # it relaxes

################################################################################
### Evolve the Model as a function of temperature for multiple lattice sizes
################################################################################

# Loop across all lattice sizes
for n in range(0, len(lattice_sizes)) :
    # Create empty vectors to save energy values needed for specific heat
    energies = []
    energies2 = []
    c = []

    # Create empty vector for magnetizations
    magnetizations = []

    # Initialize the lattice and grab its energy
    lattice, energy = InitializeIsingModel(J, lattice_sizes[n])

    # Cool down the lattice
    for temp in range(runs) :
        # Create dictionary of possible energy exponentials
        dE_exp = EnergyExponential(kB, T[temp], J)

        # Run each temperature enough times to relax the system
        energy_runs = np.zeros(Nm)
        for i in range(relaxer) :
            # Flip a random spin and find dE
            hypothetical_lattice, hypothetical_dE = SpinFlip(lattice_sizes[n], lattice, J)

            # Check dE to see if we need to run the metropolis algorithm
            if hypothetical_dE < 0 :
                # Keep the flipped spin
                lattice = hypothetical_lattice
                energy = energy + hypothetical_dE
                energy_runs[i] = energy
            elif hypothetical_dE > 0 :
                # Run the metropolis algorithm
                lattice, energy_runs[i] = MetropolisAlgorithm(dE_exp[hypothetical_dE], hypothetical_dE, lattice, hypothetical_lattice, energy)

        # Save the energy microstate, which is the relaxed energy value average
        energy = np.sum(energy_runs) / Nm

        # Calculate various energies and specific heat
        energies.append(energy)
        energies2.append(np.sum(energy_runs**2) / Nm)
        dE2 = energies2 - (energies**2)
        c.append(dE2/(kB*(T[temp])))

        # Calculate magnetizations
        magnetization = MeasureMagnetization(lattice)
        magnetizations.append(magnetization)

# Finish counting time
end = time.time()
print "This code finished after ", (end-start), " seconds"
