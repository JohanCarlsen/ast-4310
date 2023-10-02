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

def compute_electron_density_hydrogen(temperature, atom, fraction=0.5):
    T = temperature
    U_r, U_rplus1 = atom.compute_partition_function(temperature)
    chi_ion = atom.chi_ion[1]
    saha_const = (2 * np.pi * const.m_e * const.k_B * T / (const.h**2))**(3/2)
    
    N_e = (fraction * 2 * U_rplus1 / U_r * saha_const * np.exp(-chi_ion / (const.k_B * T))).to('m-3')

    print(f"Electron density of Hydrogen at {T} for {fraction*100:.1f} % ionization: {N_e.value[0]:.3e}")
    
    return N_e

h = Atom('H_atom.txt')
T = 9500 * units.K 
N_e = compute_electron_density_hydrogen(T, h)
print(h.chi.shape)
exit()
e_press = (N_e * const.k_B * T).to('Pa')

temps = np.linspace(3e3, 3e4, 500) * units.K 
h.plot_payne(temps, e_press)
plt.title(f'Payne diagram, H, {e_press[0]:.3g}')
plt.savefig('figures/payne_H.png', bbox_inches='tight')

plt.show()