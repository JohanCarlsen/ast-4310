import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        'figure.figsize': (10, 5),
        'lines.linewidth': 1,
        'font.size': 12
})

def compute_slab_intensity(tau, I0, S):
    result = I0 * np.exp(-tau) + S * (1 - np.exp(-tau))

    return result

tau = np.logspace(-1, 0.75, 50)
S = 1
fig, ax = plt.subplots()

for i in np.arange(0, 2, 0.5):
    intensity = compute_slab_intensity(tau, i, S)
    ax.plot(tau, intensity)

ax.plot(tau, np.ones_like(tau) * S, 'k--')
ax.plot(tau, tau * S, 'k--')
ax.set_ylim(0.0, 1.5)

plt.show()
