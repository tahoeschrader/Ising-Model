### Tahoe Schrader
### Computational Physics
### Spring, 2017

################################################################################
### This code will use the functions from functions.py to solve the 2D Ising
### Model
################################################################################

# ------------------------------------------------------------------------------

# Import necessary functions
from functions import InitializeIsingModel, MeasureMagnetization
from functions import MeasureEnergy, EnergyExponential
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ------------------------------------------------------------------------------

################################################################################
### Define parameters... I have set kB = 1 and removed it from the code
################################################################################

J = 1.5              # nearest neighbor interaction strength
Nm = 1000            # times to calculate energy to get a good average
relaxer = 10000      # number of times to flip a spin before taking
                     # measurements to make sure it has relaxed

T = np.linspace(5,0,50)
num_temps = len(T)   # the number of temperature measurements

lattice_size = 50

################################################################################
### Evolve the Model as a function of temperature for multiple lattice sizes
################################################################################

# Loop across all lattice sizes
print "Running the ", lattice_size, "x", lattice_size, " lattice"

# Initialize magnetizations list
magnetizations = []

# --------------------------------------------------------------------------
# The following code is where the Ising Model is actually solved.
# IMPORTANT steps are numbered.
# --------------------------------------------------------------------------

# 1. Generate a nxn lattice and find its initial energy
lattice, energy = InitializeIsingModel(J, lattice_size)

# Cools down the lattice
for temp in range(num_temps) :
    print 50 - temp
    # Create a dictionary of exponentials
    dE_exp = EnergyExponential(T[temp], J)

    # ----------------------------------------------------------------------
    # 2. Relax the system before I start taking measurements
    # ----------------------------------------------------------------------
    for flip in range(relaxer) :
        lattice, energy = MeasureEnergy(lattice_size, lattice, J, energy, dE_exp)

    print "relaxed"
    # Save a magnetization
    magnetization_runs = np.zeros(Nm)
    magnetization = MeasureMagnetization(lattice)
    magnetization_runs[0] = magnetization

    # ----------------------------------------------------------------------
    # 3. Take Nm measurements of energy now that it has relaxed
    # ----------------------------------------------------------------------
    for run in range(1, Nm) :
        # Due to strong correlation between steps, only save after another 100 flips
        for i in range(100) :
            lattice, energy = MeasureEnergy(lattice_size, lattice, J, energy, dE_exp)

        magnetization = MeasureMagnetization(lattice)
        magnetization_runs[run] = magnetization

    # ----------------------------------------------------------------------
    # 4. Save parameters that will be used for plotting and data analysis
    # ----------------------------------------------------------------------
    magnetization = np.sum(magnetization_runs) / Nm
    magnetizations.append(magnetization)

# Update that the lattice has cooled down
print "This lattice has cooled down."

################################################################################
### Plot the remaining data
### I want to plot the following relations:
###         1) For n = 50: M vs. T and find Tc (critical temperature)
################################################################################

# Plot one (still needs to find Tc)
magnetizations = np.array(magnetizations)
plt.figure()
plt.title('Magnetization for a $50\\times 50$ lattice', fontsize = 20 )
plt.plot(T,np.absolute(magnetizations)/(lattice_size**2), 'bo', label = '$M = N \langle s \\rangle = \sum_i\; \sigma_i$')
plt.ylabel('magnetization, $M$', fontsize = 15)
plt.xlabel('temperature, $T$', fontsize = 15)
plt.legend()
plt.grid()

plt.show()
