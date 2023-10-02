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

with open('MgI-XI.txt', 'w') as outfile:
        outfile.write(f"#\tE (cm^-1){'':<10}\tg\tstage\tlevels\n")

count = 0
ion = 0

for i in range(1, 12):
    filename = 'Mg-' + f'{i}' + '.txt'

    g, chi, chi_ion = read_NIST_species(filename)
    chi += ion
    stage = np.ones(chi.shape, dtype=int) * (i-1)
    stage[-1] = i
    
    with open('MgI-XI.txt', 'a') as outfile:

        for j in range(len(g)):
            levels = j + count
            line = f'\t{chi[j]:<18}\t{g[j]}\t\t{stage[j]}\t\t{levels}\n'
            outfile.write(line)

    count += len(g)
    ion = chi_ion

e_press = 100 * units.Pa
temps = np.linspace(3.5e3, 2e6, 10000) * units.K

MgI_III = Atom('MgI-III.txt')
wavelength_aJ = MgI_III.chi[1, :]
wavelength_nm = wavelength_aJ.to('nm', equivalencies=units.spectral())

h_line = np.logical_and(wavelength_nm.value >= 279, wavelength_nm.value <= 280)
k_line = np.logical_and(wavelength_nm.value >= 280, wavelength_nm.value <= 281)

print(f'Energy of Mg II h_line: {wavelength_aJ[h_line][0]:.3f}')
print(f'Energy of Mg II k_line: {wavelength_aJ[k_line][0]:.3f}')

MgI_XI = Atom('MgI-XI.txt')
MgI_XI.plot_payne(temps, e_press, compute_excitation=False)

CIE = np.loadtxt('Mg_CIE.txt')
temp = CIE[:,0]
plt.subplots()
plt.plot(temp, CIE[:, 1:], 'r')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6, 1.1)

plt.show()