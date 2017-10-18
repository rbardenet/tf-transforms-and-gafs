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
    F /= np.sqrt(2*pi)         # arbitrary normalization
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

    Mid_Mid = np.zeros((C,R), dtype=bool)

    for c in range(1, C-1):   
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) > th)
            
    x, y = np.where(Mid_Mid)
    return x, y
