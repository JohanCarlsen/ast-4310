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

g, chi, chi_ion = read_NIST_species('MgI-XI.txt')
chi = (chi / units.cm).to('aJ', equivalencies=units.spectral())
chi_ion = (chi_ion / units.cm).to('aJ', equivalencies=units.spectral())

part = np.sum(g[:, np.newaxis] * np.exp(-chi[:, np.newaxis] / (const.k_B * temps[np.newaxis, np.newaxis])), axis=1)
exi = g[:, np.newaxis] / part[:, np.newaxis] * np.exp(-chi[:, np.newaxis] / (const.k_B * temps[np.newaxis, np.newaxis]))
e_dens = e_press / (const.k_B * temps)
saha_const = ((2 * np.pi * const.m_e * const.k_B * temps) / (const.h**2))**(3/2)
nstage = np.zeros_like(part) / units.m**3
nstage[0] = 1. / units.m**3
print(nstage.shape)

for r in range(len(nstage) - 1):
    nstage[r+1] = (nstage[r] / e_dens * 2 * saha_const * part[r+1] / part[r] * np.exp(chi_ion[r+1, np.newaxis] / (const.k_B * temps[np.newaxis])))
    print(nstage[r+1])
# nstage /= np.sum(nstage, axis=0)
# print(part.shape)

# mg = Atom('MgI-III.txt')
# e_press = 100 * units.kPa
# temps = np.linspace(1e2, 1e5, 500) * units.K
# mg.plot_payne(temps, e_press)
# plt.show()