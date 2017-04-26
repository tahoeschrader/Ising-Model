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
from scipy.optimize import curve_fit

# ------------------------------------------------------------------------------

################################################################################
### Define parameters... I have set kB = 1 and removed it from the code
################################################################################

J = 1.5              # nearest neighbor interaction strength
Nm = 1000            # times to calculate energy to get a good average
relaxer = 10000      # number of times to flip a spin before taking
                     # measurements to make sure it has relaxed

T=np.linspace(5,.1,20)
num_temps = len(T)   # the number of temperature measurements

lattice_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 200, 500]
lattices = len(lattice_sizes)

################################################################################
### Evolve the Model as a function of temperature for multiple lattice sizes
################################################################################

# Initialize lattice-dependent lists
c_maxes = []         # stores C_max/N for each lattice size

# Loop across all lattice sizes
for n in np.arange(lattices) :
    print "Running the ", lattice_sizes[n], "x", lattice_sizes[n], " lattice"

    # Initialize temperature-dependent lists
    c_per_N = []                  # stores C/N for a given temp to locate C_max
    cs = []                       # stores all specific heats for plotting reasons

    # --------------------------------------------------------------------------
    # The following code is where the Ising Model is actually solved.
    # IMPORTANT steps are numbered.
    # --------------------------------------------------------------------------

    # 1. Generate a nxn lattice and find its initial energy
    lattice, energy = InitializeIsingModel(J, lattice_sizes[n])

    # Cools down the lattice
    for temp in np.arange(num_temps) :
        print 20-temp
        # Create a dictionary of exponentials
        dE_exp = EnergyExponential(T[temp], J)

        # ----------------------------------------------------------------------
        # 2. Relax the system before I start taking measurements
        # ----------------------------------------------------------------------
        for flip in np.arange(relaxer) :
            lattice, energy = MeasureEnergy(lattice_sizes[n], lattice, J, energy, dE_exp)

        # Create a list to save energies and save the first one
        energy_runs = np.zeros(Nm)
        energy_runs[0] = energy

        print "Relaxed"

        # ----------------------------------------------------------------------
        # 3. Take Nm measurements of energy now that it has relaxed
        # ----------------------------------------------------------------------
        for run in np.arange(1, Nm) :
            # Due to strong correlation between steps, only save energy after
            # another 100 or so flips
            for i in np.arange(100) :
                lattice, energy = MeasureEnergy(lattice_sizes[n], lattice, J, energy, dE_exp)

            energy_runs[run] = energy

        # ----------------------------------------------------------------------
        # 4. Save parameters that will be used for plotting and data analysis
        # ----------------------------------------------------------------------

        # Calculate specific heat (be careful of division by 0!!!)
        energy = np.sum(energy_runs) / Nm
        energy2 = np.sum(energy_runs**2) / Nm
        dE2 = energy2 - (energy**2)
        c = dE2 / (T[temp]**2)

        # Store specific heat and specific heat per N -- (N = n^2)
        cs.append(c)
        c_per_N.append(c / (lattice_sizes[n]**2))

    # Update that the lattice has cooled down
    print "This lattice has cooled down."

    # --------------------------------------------------------------------------
    # 5. Now that the lattice has cooled, find the maximum c value and plot a
    #    couple cases of c(T).
    # --------------------------------------------------------------------------
    c_maxes.append(max(c_per_N))

    # I want to show a couple plots of c(T) for various n
    if lattice_sizes[n] == 5 or lattice_sizes[n] == 10 or lattice_sizes[n] == 20 :
        plt.figure()
        plt.title('Specific heat for a $%i\\times%i$ lattice' %(lattice_sizes[n],lattice_sizes[n]), fontsize = 20 )
        plt.plot(T, cs, color = 'dodgerblue', linestyle = 'solid', linewidth = 3, label = '$C = \\frac{(\\Delta E^2)}{kB \; T^2}$')
        plt.ylabel('specific heat, $C$', fontsize = 15)
        plt.xlabel('temperature, $T$', fontsize = 15)
        plt.legend()
        plt.grid()
        plt.show()


################################################################################
### Plot the remaining data
### I want to plot the following relation:
###         a) c_max/N vs n and fit log(n) to the plot
################################################################################

# Log fit func....
def logfunc(x, a, b):
    return a*np.log(x) + b
best_vals, pcov = curve_fit(logfunc, lattice_sizes[0:4], c_maxes[0:4])
print(best_vals)

# Plot two
nmany = np.arange(5,100,.1)
plt.figure()
plt.title('Verification of the finite-size scaling relation', fontsize = 20 )
plt.plot(lattice_sizes, c_maxes, color = 'dodgerblue', linestyle = 'solid', linewidth = 3, label = 'data')
plt.plot(nmany, logfunc(nmany, best_vals[0], best_vals[1]), color='red',linestyle='dashed', linewidth = 3, label='best fit')
plt.ylabel('specific heat per spin, $C/N$', fontsize = 15)
plt.xlabel('$1D$ lattice dimension, $n$', fontsize = 15)
plt.legend()
plt.grid()

plt.show()
