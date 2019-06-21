import numpy as np
import scipy.fftpack as spf

def fourier(f, A):
    """Approximate the Fourier transform of f truncated on [-A,A] using the discrete Fourier transform.

    Args:
        f -- target function evaluated at -A+2A*k/N, k=0..N-1, N should be even.
        A -- truncation level.

    Returns:
        freqs -- an array of N frequencies.
        F -- the array of corresponding approximate values of the Fourier transform.

    """
    N = len(f)
    if not int(N/2) == N/2:
        raise("N should be even")
    B = A/2/np.pi

    F = spf.fft(spf.ifftshift(f))
    F = spf.fftshift(F)         # shift F to align indices and [-A,A]
    freqs = np.arange(-N/2,N/2)/(2*B)
    F *= 2*A/N
    F /= np.sqrt(2*np.pi)         # arbitrary normalization
    return freqs, F

def fourierInverse(F, freqMax):
    """Approximate the inverse Fourier transform of f truncated on [-freqMax,freqMax] using the discrete Fourier transform.

    Args:
        F -- target function evaluated at -freqMax+2freqMax*k/N, k=0..N-1, N should be even.
        freqMax -- truncation level.

    Returns:
        times -- an array of N times.
        f -- the array of correspdongin approximate values of the inverse Fourier transform.

    """
    times, f = fourier(F, freqMax)
    return times, f[::-1]

def extr2minth(M, th):
    """Extract zeros as minima over a small patch that are additionally below a threshold.

    Args:
        M -- array of values from which to extract minima
        th -- threshold

    Returns:
        x -- array of first coordinates of zeros
        y -- array of second coordinates of zeros
    """

    C,R = M.shape
    maxM = np.max(M)
    if th > maxM/2:
        print("Threshold seems too high, we reset it to th =", maxM/2) # reset threshold for minima if needed
        th = maxM/2

    Mid_Mid = np.zeros((C,R), dtype=bool)

    for c in range(1, C-1):
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) < th)

    x, y = np.where(Mid_Mid)
    return x, y

def autoscales(N, dt, dj, wf, p):
     """Compute wavelet scales as fractional power of two.

     Args:
        N -- integer
           number of data samples
        dt -- float
           time step
        dj -- float
           scale resolution (smaller values of dj give finer resolution)
        wf -- string
           wavelet function ('morlet', 'paul', 'dog')
        p -- float
           omega0 ('morlet') or order ('paul', 'dog')

     Returns:
        scales -- 1d numpy array
           scales
     """

     if wf == 'dog':
         s0 = (dt * np.sqrt(p + 0.5)) / np.pi
     elif wf == 'paul':
         s0 = (dt * ((2 * p) + 1)) / (2 * np.pi)
     elif wf == 'morlet':
         s0 = (dt * (p + np.sqrt(2 + p**2))) / (2 * np.pi)
     else:
         raise ValueError('wavelet function not available')

     #  See (9) and (10) at page 67.

     J = np.floor(dj**-1 * np.log2((N * dt) / s0))
     s = np.empty(int(J) + 1)

     for i in range(s.shape[0]):
         s[i] = s0 * 2**(i * dj)

     return s
