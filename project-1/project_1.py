import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import cumtrapz

plt.rcParams.update({
    'font.family': 'serif',
    'lines.markersize': 4,
    'lines.linewidth': 1
})

def gaussian_profile(frequency, mult_peak=False, baseline=0.1, slope=0.01, scale=100):
    continuum = slope * frequency + baseline

    a, c = (1, 1) if not mult_peak else (3, 0.2)

    exp1 = a * (np.exp(-frequency**2 / c) + continuum) / scale
    exp2 = 0.3 * (np.exp(-(frequency - 2)**2 / 0.05) + continuum) / scale
    exp3 = 0.1 * (np.exp(-(frequency - 3)**2 / 0.1) + continuum) / scale
    mult = exp1 + exp2 + exp3 

    return exp1 if not mult_peak else mult

def make_panels(profile_func='gaussian', source_func='exp'):

    n = 101

    freq = np.linspace(-5, 5, n)
    depth = np.logspace(0, 4, 200)
    z = -depth + depth.max()

    if callable(profile_func):
        profile = profile_func(freq)
    
    else:
        if isinstance(profile_func, str):
            profile = gaussian_profile(freq)
        
        elif isinstance(profile_func, tuple):
            mult_peak, baseline, slope, scale = profile_func
            profile = gaussian_profile(
                freq,
                mult_peak=mult_peak,
                baseline=baseline,
                slope=slope,
                scale=scale
            )
    
    alpha = np.tile(profile, (len(z), 1))

    tau = cumtrapz(alpha, -z, axis=0, initial=1e-3)
    tau_z0 = np.argwhere(tau[:, 0] >= 1)[0]
    tau_zhalf = np.argwhere(tau[:, 50] >= 1)[0]

    exp = np.exp(-(depth - 500)**2 / 1.5e5) + 0.65
    exp2 = np.exp((depth - 500)**2 / 1e9) - 0.65
    linear = depth / 100 * 2 + 0.65

    S_func = {
        'exp': exp,
        'exp2': exp2,
        'linear': linear
    }

    try: 
        S = S_func[source_func]
    
    except KeyError:
        print('The key ' + source_func + ' is not valid. Possible keys are:\n')
        [print(key) for key in S_func]
        exit()

    I = np.trapz(S[:, np.newaxis] * np.exp(-tau), tau, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.set_title(r'Extinction $\alpha(\nu)$')
    ax1.plot(freq, alpha[0], color='black')
    ax1.plot(freq[0], alpha[0, 0], 'bo')
    ax1.plot(freq[50], alpha[0, 50], 'ro')

    ax2.set_title(r'Optical depth $\tau(z)$')
    ax2.plot(z, tau[:, 0], color='blue')
    ax2.plot(z, tau[:, 50], color='red')
    ax2.plot(z[tau_z0], 1, 'bo')
    ax2.plot(z[tau_zhalf], 1, 'ro')

    ax3.set_title(r'Intensity $I(\nu)$')
    ax3.plot(freq, I, color='black')
    ax3.plot(freq[0], I[0], 'bo')
    ax3.plot(freq[50], I[50], 'ro')

    ax4.set_title(r'Source function $S(z)$')
    ax4.plot(z, S, color='black')
    ax4.plot(z[tau_z0], S[tau_z0], 'bo')
    ax4.plot(z[tau_zhalf], S[tau_zhalf], 'ro')

    fig.tight_layout()

# make_panels()
# make_panels(profile_func=(True, 0.2, 0, 1000), source_func='exp2')
make_panels(profile_func=(True, 0.2, 0, 1000), source_func='linear')

plt.show()