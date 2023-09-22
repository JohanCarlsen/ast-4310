import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import cumtrapz

plt.rcParams.update({
    'font.family': 'serif',
    'lines.markersize': 4,
    'lines.linewidth': 1
})

def gaussian_profile(frequency, mult_peaks=False, baseline=0.1, slope=0.01, scale=100):
    r'''
    Compute a Gaussian profile as function of frequency, on the form:

        .. math::
            f(\nu) = ae^{-\nu^2/c} + \mathrm{continuum}

    Parameters:
    -----------
    frequency : ``array_like``
        Array containing the frequencies.

    mult_peaks : ``bool``, default=``False`` 
        If set to ``True``, a Gaussian profile with 3 peaks will be returned.

    baseline : ``float``, default=0.1
        The continuum is considered to be on the form:
            
        .. math::
            y = a + bx,
        
        where a is the baseline.

    slope : ``float``, default=0.01
        Slope of the continuum (see baseline).

    scale : ``int`` or ``float``, default=100
        The returned profile will be scaled down with this value, ie. if G is the Gaussian, the 
        returned value is G/scale.

    Returns:
    --------
    ``array_like``
        The Gaussian profile with 1 or 3 peaks plus the baseline.
    
    Example:
    --------
    Default call to :any:`gaussian_profile`:

    .. code-block:: python 

        import numpy as np 

        nu = np.linspace(-5, 5, 101)
        alpha = gaussian_profile(nu)
    
    Or, with multiple peaks and no continuum (b = 0):

    .. code-block:: python 

        alpha = gaussian_profile(alpha, mult_peaks=True, slope=0)

        plt.plot(nu, alpha)

    '''
    cont = baseline + slope * frequency # Continuum

    # Selecting a, c depending on the number of peaks
    a, c = (1, 1) if not mult_peaks else (3, 0.2)

    exp1 = a * (np.exp(-frequency**2 / c) + cont) / scale
    exp2 = 0.3 * (np.exp(-(frequency - 2)**2 / 0.05) + cont) / scale
    exp3 = 0.1 * (np.exp(-(frequency - 3)**2 / 0.1) + cont) / scale
    
    profile = exp1 + exp2 + exp3 if mult_peaks else exp1

    return profile   

def make_panels(profile_func='gaussian', source_func='gaussian', mult_mu=False):
    r'''
    Create four panels for visualising the extinction, optical depth, intensity, and source function.

    Parameters:
    -----------
    profile_func : ``callable`` or ``tuple``, optional
        The function describing the extinction profile. User can provide their own if they want.
        Default is the :any:`gaussian_profile`. If a ``tuple``, the elements are (bool, float, float, float),
        representing the arguments of the :any:`gaussian_profile`.

    source_func : ``string``, optional 
        Form of the source function. The two forms are:

            * 'gaussian' (default): Gaussian function on the form:
                .. math::
                    S(\nu) = e^{-\frac{(\nu-500)^2}{1.5\cdot10^5}} + 65

            * 'linear': Linear function on the form:
                .. math::
                    S(\nu) = \frac{\nu}{50} + 65

    mult_mu : ``bool``, default=``False``
        If ``True``, the intensity panel will display two curves,
        one for :math:`\mu = 1`, and one for :math:`\mu = 0.2`.
    
    Example
    _______

    An example of a four panel image created with default settings:

    .. code-block:: python 

        make_panels()
    
    .. image:: ./gaussian.png
        :width: 400
    '''
    # Number of frequency elements.
    n = 101

    # Frequency from -5 to 5.
    freq = np.linspace(-5, 5, n)

    # Optical depth from 1 to 10^4 with 200 elements.
    # Height z is the negative height plus the highest depth.
    depth = np.logspace(0, 4, 200)
    z = -depth + depth.max()

    if callable(profile_func):
        profile = profile_func

    else:
        if profile_func == 'gaussian':
            profile = gaussian_profile(freq)

        elif isinstance(profile_func, tuple):
            mult_peaks, baseline, slope, scale = profile_func
            profile = gaussian_profile(
                freq,
                mult_peaks=mult_peaks,
                baseline=baseline,
                slope=slope,
                scale=scale
            )

    # Extinction and optical depth
    alpha = np.tile(profile, (len(z), 1))
    tau   = cumtrapz(alpha, -z, axis=0, initial=1e-3)

    # Get the indicies where tau = 1
    tau_z0    = np.argwhere(tau[:, 0] >= 1)[0]
    tau_zhalf = np.argwhere(tau[:, 50] >= 1)[0]

    # Source functions
    gauss = np.exp(-(depth - 500)**2 / 1.5e5) + 0.65
    lin   = 2 * depth / 100 + 0.65

    S_func = {
        'gaussian': gauss,
        'linear'  : lin
    }
    
    S = S_func[source_func]

    # Intensity
    I = np.trapz(S[:, np.newaxis] * np.exp(-tau), tau, axis=0)

    # Create the figure 
    fig, axes = plt.subplots(2, 2)
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

    ax3.set_title(r'Intensity I$(\nu)$')
    ax3.plot(freq, I, color='black', label=r'I$(\mu=1)$')
    ax3.plot(freq[0], I[0], 'bo')
    ax3.plot(freq[50], I[50], 'ro')

    if mult_mu:
        I2 = np.trapz(S[:, np.newaxis] * np.exp(-tau / 0.2), tau, axis=0)
        ax3.plot(freq, I2, color='saddlebrown', label=r'I$(\mu=0.2)$')
        ax3.plot(freq[0], I2[0], 'bo')
        ax3.plot(freq[50], I2[50], 'ro')
        ax3.legend()

    ax4.set_title(r'Source function $S(z)$')
    ax4.plot(z, S, color='black')
    ax4.plot(z[tau_z0], S[tau_z0], 'bo')
    ax4.plot(z[tau_zhalf], S[tau_zhalf], 'ro')

    fig.tight_layout()
    figname = source_func + 'mult' if mult_mu else source_func
    fig.savefig('figures/' + figname, bbox_inches='tight')

if __name__ == '__main__':

    make_panels()
    make_panels(profile_func=(True, 0.2, 0, 1000), source_func='linear')
    make_panels(profile_func=(False, 0.2, 0, 1000), source_func='linear', mult_mu=True)

    plt.show()