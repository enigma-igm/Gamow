
import os
import numpy as np
import astropy.units as u
import astropy.constants as const


import numpy as np
import os
from collections import deque
import itertools
from bisect import insort, bisect_left


from matplotlib import pyplot as plt

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import resample
import scipy
from astropy import convolution


def fast_running_median(seq, window_size):
    """

    Compute the median of sequence of numbers with a running window. The
    boundary conditions are identical to the scipy 'reflect' boundary
    codition:

    'reflect' (`d c b a | a b c d | d c b a`)

    The input is extended by reflecting about the edge of the last pixel.

    This code has been confirmed to produce identical results to
    scipy.ndimage.filters.median_filter with the reflect boundary
    condition, but is ~ 100 times faster.

    Args:
        seq (list or 1-d numpy array of numbers):
        window_size (int): size of running window.

    Returns:
        ndarray: median filtered values

    Code contributed by Peter Otten, made to be consistent with
    scipy.ndimage.filters.median_filter by Joe Hennawi.

    See discussion at:
    http://groups.google.com/group/comp.lang.python/browse_thread/thread/d0e011c87174c2d0
    """
    # Enforce that the window_size needs to be smaller than the sequence, otherwise we get arrays of the wrong size
    # upon return (very bad). Added by JFH. Should we print out an error here?

    if (window_size > (len(seq)-1)):
        raise ValueError('window_size > len(seq)-1. Truncating window_size to len(seq)-1, but something is probably wrong....')
    if (window_size < 0):
        raise ValueError('window_size is negative. This does not make sense something is probably wrong. Setting window size to 1')

    window_size = int(np.fmax(np.fmin(int(window_size), len(seq)-1),1))
    # pad the array for the reflection
    seq_pad = np.concatenate((seq[0:window_size][::-1],seq,seq[-1:(-1-window_size):-1]))

    seq_pad = iter(seq_pad)
    d = deque()
    s = []
    result = []
    for item in itertools.islice(seq_pad, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq_pad:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])

    # This takes care of the offset produced by the original code deducec by trial and error comparison with
    # scipy.ndimage.filters.medfilt

    result = np.roll(result, -window_size//2 + 1)
    return result[window_size:-window_size]




def get_delta_wave(wave, gpm, frac_spec_med_filter=0.03):
    """
    Given an input wavelength vector and an input good pixel mask, this code computes the delta_wave defined
    to be wave_old[i]-wave_old[i-1]. The missing point at the end of the array is just appended to have an
    equal size array.

    Args:
        wave (float `numpy.ndarray`_): shape = (nspec,)
            Array of input wavelengths. Muse be a one dimensional array.
        gpm (bool `numpy.ndarray`_): shape = (nspec)
            Boolean good pixel mask defining where the wave_old are good.
        frac_spec_med_filter (float, optional):
            Fraction of the nspec to use to median filter the delta wave by to ensure that it is smooth. Deafult is
            0.03. In other words, a running median filter will be applied with window equal to 0.03*nspec

    Returns
    -------
    delta_wave (float `numpy.ndarray`_): shape = (nspec,)
            Array of wavelength differences using np.diff, where the last pixel has been extrapolated

    """

    if wave.ndim > 1:
        raise ValueError('This routine can only be run on one dimensional wavelength arrays')
    nspec = wave.size
    # This needs to be an odd number
    nspec_med_filter = 2*int(np.round(nspec*frac_spec_med_filter/2.0)) + 1
    delta_wave = np.zeros_like(wave)
    wave_diff = np.diff(wave[gpm])
    wave_diff = np.append(wave_diff, wave_diff[-1])
    wave_diff_filt = fast_running_median(wave_diff, nspec_med_filter)

    # Smooth with a Gaussian kernel
    sig_res = np.fmax(nspec_med_filter/10.0, 3.0)
    gauss_kernel = convolution.Gaussian1DKernel(sig_res)
    wave_diff_smooth = convolution.convolve(wave_diff_filt, gauss_kernel, boundary='extend')
    delta_wave[gpm] = wave_diff_smooth
    return delta_wave


#  For (1), if you could look at the observing time for a range of GRB redshift at z=(4, 5, 6, 7, 8, 9, 10) at
#  the wavelengths corresponding to rest-frame Lya and a range of flux density at say f=(1, 5, 10, 20, 50) uJy,
#  that’d be super helpful. Currently we are baselining having a SNR=20 per R at R>=3000 for the reionization constraint.


# For reference convert the fluxes to AB magnitude

f_nu = np.array([1.0,5.0, 10.0, 20.0, 50.0])*u.microjansky

f_AB = f_nu.to('mag(AB)')

# Flamingos 2 with the 4-pixel slit (0.72") has R = 450. Spectrograph has 167 km/s pixels.

# GMOS with a 1.0" slit (6.2 pixels) has R=2200 and 24 km/s pixels when binned by 2 spectrally.

# NIRES has as 0.55" slit (3.7 pixels) and 39 km/s pixels

# GNIRS with the 1.0" (6.7 pixels) slit has R = 510. Spectrograph has 88 km/s pixels



# exptime = 120*300 = 36,000s



# To convert FLAMINGOS-2 S/N to NIRES, divide SNR by sqrt(167/39) = 2

# GMOS will be roughly comparalbe to Keck LRIS


z = 7.0
R = 3000
t_exp = 10.0
f_nu_J = 10.0
#def snr_gamow(z, f_nu_J, t_exp, R):


t_exp_ref = 10.0 # units hr
f_nu_J_ref = 10.0 # units of uJy, in J-band observed frame
file_gmos_7100 = os.path.join('data', 'GMOS_N_831_7100_10hr_SNR.txt')
data = np.loadtxt(file_gmos_7100)
lam_gmos_7100, snr_gmos_7100 = 10*data[:,0], data[:,1]
dlam_gmos_7100 = get_delta_wave(lam_gmos_7100, np.ones_like(lam_gmos_7100,dtype=bool))
Rlam_gmos_7100 = lam_gmos_7100/dlam_gmos_7100

file_gmos_9300 = os.path.join('data', 'GMOS_N_831_9300_10hr_SNR.txt')
data = np.loadtxt(file_gmos_9300)
lam_gmos_9300, snr_gmos_9300 = 10*data[:,0], data[:,1]
dlam_gmos_9300 = get_delta_wave(lam_gmos_9300, np.ones_like(lam_gmos_9300,dtype=bool))
Rlam_gmos_9300 = lam_gmos_9300/dlam_gmos_9300

file_flamingos2_JH = os.path.join('data', 'FLAMINGOS_2_JH_10hr_SNR.txt')
data = np.loadtxt(file_flamingos2_JH)
lam_flamingos2_JH, snr_flamginso2_JH = 10*data[:, 0], data[:, 1]
dlam_flamingos2_JH = get_delta_wave(lam_flamingos2_JH, np.ones_like(lam_flamingos2_JH,dtype=bool))
Rlam_flamingos2_JH = lam_flamingos2_JH/dlam_flamingos2_JH

# This assumes you are background limited, i.e. objects much fainter than sky. In this regime scales roughly as
# SNR = f_nu_J/f_nu_J_ref*SNR_ref*sqrt(t_exp/t_exp_ref)*sqrt(R_itc/R)
snr_gamow_gmos_7100 = np.sqrt(Rlam_gmos_7100/R)*np.sqrt(t_exp/t_exp_ref)*(f_nu_J/f_nu_J_ref)*snr_gmos_7100
snr_gamow_gmos_9300 = np.sqrt(Rlam_gmos_9300/R)*np.sqrt(t_exp/t_exp_ref)*(f_nu_J/f_nu_J_ref)*snr_gmos_9300
snr_gamow_flamingos2_JH = np.sqrt(Rlam_flamingos2_JH/R)*np.sqrt(t_exp/t_exp_ref)*(f_nu_J/f_nu_J_ref)*snr_flamginso2_JH


plt.plot(lam_gmos_7100,snr_gamow_gmos_7100,color='blue', label='GMOS-7100', alpha=0.7)
plt.plot(lam_gmos_9300,snr_gamow_gmos_9300,color='green', label='GMOS-9300', alpha=0.7)
plt.plot(lam_flamingos2_JH,snr_flamginso2_JH,color='red', label='FLAMINGOS-2-JH', alpha=0.7)
plt.axvline((1.0 + z)*1215.67, linestyle='--', color='black', label=r'$(1 +z_{{GRB}})*1216{{\rm \AA}}$')
plt.legend()
plt.ylabel('S/N per R')
plt.xlabel('Wavelength  ' +  r'[${{\AA}}$]')
plt.show()



