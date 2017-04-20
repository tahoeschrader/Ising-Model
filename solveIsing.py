### Tahoe Schrader
### Computational Physics
### Spring, 2017

################################################################################
### This code will use the functions from functions.py to solve the 2D Ising
### Model
################################################################################

# ------------------------------------------------------------------------------

from functions import InitializeSpins, MeasureEnergy, MeasureMagnetization
from functions import SpinFlip, MetropolisAlgorithm
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

J = 1.5                 # nearest neighbor interaction strength
B = 0.0                 # no external magnetic field
kB = 1                  # Boltzmann constant is 1 for this code

lattice_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 200, 500]

temperature = 5         # start at a high temperature and cool down

################################################################################
### Evolve the Model
################################################################################

# 1. Initialize the lattice
lattice = InitializeSpins(lattice_sizes[0])

# Create a loop that flips spins until the system is relaxed
#relax = False
#while relax is False :
counter = 0
while counter < 1000 :
    # 2. Measure the energy and magnetization of the initial microstate
    energy_microstate = MeasureEnergy(lattice, lattice_sizes[0], J)

    # 3. Flip a random spin
    evolved_lattice = SpinFlip(lattice_sizes[0], lattice)

    # 4. Calculate the hypothetical energy
    hypothetical_energy_microstate = MeasureEnergy(evolved_lattice, lattice_sizes[0], J)

    # 5. Run the metropolis algorithm to update the lattice and energy microstate
    lattice, energy_microstate = MetropolisAlgorithm(energy_microstate, hypothetical_energy_microstate, kB, temperature, lattice_sizes[0], lattice, evolved_lattice)

    # 6. The system reaches equilibrium once ???????
    #if something happens :
    #    relax = True
    counter += 1

print energy_microstate, magnetization_microstate

# Finish counting time
end = time.time()
print "This code finished after ", (end-start), " seconds"
