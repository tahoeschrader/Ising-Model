### Tahoe Schrader
### Computational Physics
### Spring, 2017

################################################################################
### This code will use the functions from functions.py to solve the 2D Ising
### Model
################################################################################

# Begin a time tracker
import time
start = time.time()
print "Code Initiated"

# ------------------------------------------------------------------------------

# Import necessary functions
from functions import InitializeIsingModel, MeasureMagnetization
from functions import SpinFlip, MetropolisAlgorithm, EnergyExponential
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ------------------------------------------------------------------------------

################################################################################
### Define parameters
################################################################################

J = 1.5                     # nearest neighbor interaction strength
kB = 1                      # Boltzmann constant is 1 for this code

lattice_sizes = [5, 10]
#lattice_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 200, 500]

runs = 50                   # number of runs

T = np.linspace(5, 0, runs) # start at a high temperature and cool down

Nm = 50                     # number of times times we calculate energy before
                            # it relaxes

################################################################################
### Evolve the Model as a function of temperature for multiple lattice sizes
################################################################################

# Initialize a list to store the time it takes to run the code
runtime = []

# Initialize a list to save c_max values
c_maxs = []

# Loop across all lattice sizes
for n in range(len(lattice_sizes)) :
    # Save the time it starts
    start_time = time.time()
    print "Running the ", lattice_sizes[n], "x", lattice_sizes[n], " lattice"

    # Create empty vectors to specific heat per spin, specific heat
    c_per_N = []
    cs = []

    # Create a list to save magnetizations if n = 50
    if lattice_sizes[n] == 5 :
        magnetizations = []

    # Initialize the lattice and grab its energy
    lattice, energy = InitializeIsingModel(J, lattice_sizes[n])

    # Cool down the lattice
    for temp in range(runs) :
        # Create dictionary of possible energy exponentials
        dE_exp = EnergyExponential(kB, T[temp], J)

        # Run each temperature enough times to relax the system
        energy_runs = np.zeros(Nm)
        for i in range(Nm) :
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

        # Save the energy microstates, which are the relaxed energy value average
        energy = np.sum(energy_runs) / Nm
        energy2 = np.sum(energy_runs**2) / Nm
        dE2 = energy2 - (energy**2)
        if T[temp] != 0 :
            c = dE2 / (kB * (T[temp]**2))
        elif T[temp] == 0 :
            c = dE2 / (kB * (T[temp - 1]**2))

        # Save specific heats
        cs.append(c)
        c_per_N.append(c / (lattice_sizes[n]**2))

        # Save magnetizations if n = 50
        if lattice_sizes[n] == 5 :
            magnetization = MeasureMagnetization(lattice)
            magnetizations.append(magnetization)

    # Save the time it ended
    end_time = time.time()
    time_elapsed = end_time - start_time
    runtime.append(time_elapsed)
    print "This lattice has cooled down."

    # Now that the lattice has cooled down and the data has been stored, I must
    # find the maximum c value and store it in a list
    c_maxs.append(max(c_per_N))

    # I want to show a couple plots of c(T) for various n
    if lattice_sizes[n] == 5 or lattice_sizes[n] == 50 or lattice_sizes[n] == 500 :
        fig = plt.subplot()
        plt.title('Specific heat for a $%i\\times%i$ lattice' %(lattice_sizes[n],lattice_sizes[n]), fontsize = 20 )
        plt.plot(T, cs, color = 'dodgerblue', linestyle = 'solid', linewidth = 3, label = '$C = \\frac{(\\Delta E^2)}{kB\; T^2}$')
        plt.ylabel('specific heat, $C$', fontsize = 15)
        plt.xlabel('temperature, $T$', fontsize = 15)
        plt.legend()
        plt.grid()
        plt.show()

################################################################################
### Plot the data
### I want to plot the following relations:
###         1) For n = 50: M vs. T and find Tc (critical temperature)
###         2) c_max/N vs n and fit log(n) to the plot
###         3) Time relation of how long it takes to code this
################################################################################

# Starting with plot one (still needs to find Tc)
fig = plt.subplot()
plt.title('Magnetization for a $50\\times 50$ lattice', fontsize = 20 )
plt.plot(T, magnetizations, color = 'dodgerblue', linestyle = 'solid', linewidth = 3, label = '$M = N \langle s \\rangle = \sum_i\; \\rho_i$')
plt.ylabel('magnetization, $M$', fontsize = 15)
plt.xlabel('temperature, $T$', fontsize = 15)
plt.legend()
plt.grid()
plt.show()

# Plot two (still needs to fit log(n))
fig = plt.subplot()
plt.title('Verification of the finite-size scaling relation', fontsize = 20 )
plt.plot(lattice_sizes, c_maxs, color = 'dodgerblue', linestyle = 'solid', linewidth = 3, label = 'data')
plt.ylabel('specific heat per spin, $C/N$', fontsize = 15)
plt.xlabel('$1D$ lattice dimension, $n$', fontsize = 15)
plt.legend()
plt.grid()
plt.show()

# Plot three
fig = plt.subplot()
plt.title('Code runtime', fontsize = 20 )
plt.plot(lattice_sizes, runtime, color = 'dodgerblue', linestyle = 'solid', linewidth = 3)
plt.ylabel('time, $s$', fontsize = 15)
plt.xlabel('$1D$ lattice dimension, $n$', fontsize = 15)
plt.grid()
plt.show()
