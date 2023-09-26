import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy import units
from astropy import constants as const
from astropy.visualization import quantity_support
from atoms import Atom
quantity_support() 

plt.rcParams.update({
    'lines.linewidth': 1,
    'font.family': 'serif'
})

def read_NIST_species(input_file):
    """
    Reads data from a text file downloaded from NIST database with
    level energies (4th column and statistical weights (2nd column).
    """
    data = open(input_file, 'r').readlines()[1:]  # skip header
    nlevels = len(data)
    g = np.zeros(nlevels, dtype='i') - 1
    chi = np.zeros(nlevels, dtype='f') - 1
    chi_ion = 0.
    
    for i, line in enumerate(data):
        entries = line.split('\t')
        if entries[1] != '':
            g[i] = int(entries[1])
            chi[i] = float(entries[3].strip('"').strip('[').strip(']'))
        if entries[0].strip('"').lower() == "limit":
            chi_ion = float(entries[3].strip('"').strip('[').strip(']'))
            break
    # clean up missing values
    mask = (g >= 0) & (chi >= 0)
    return g[mask], chi[mask], chi_ion

e_press = 100 * units.Pa
temps = np.linspace(3.5e3, 2e6, 500) * units.K

g, chi, chi_ion_float = read_NIST_species('Mg-1.txt')
chi_ion = np.ones(chi.shape) * 0
chi_ion[-1] = chi_ion_float
stage = np.arange(len(chi))

with open('MgI-XI.txt', 'w') as outfile:
        outfile.write(f"#\tE (cm^-1){'':<10}\tg\tstage\tlevels\n")

count = 0
ground_count = 0

for i in range(1, 12):
    filename = 'Mg-' + f'{i}' + '.txt'

    g, chi, chi_ion_float = read_NIST_species(filename)
    chi_ion = np.ones(chi.shape) * i
    chi_ion[-1] = chi_ion_float
    chi += ground_count
    stage = np.ones(chi.shape, dtype=int) * (i-1)
    stage[-1] = i
    
    with open('MgI-XI.txt', 'a') as outfile:

        for j in range(len(g)):
            levels = j + count
            line = f'\t{chi[j]:<18}\t{g[j]}\t\t{stage[j]}\t\t{levels}\n'
            outfile.write(line)

    count += len(g)
    # ground_count += chi[-1]