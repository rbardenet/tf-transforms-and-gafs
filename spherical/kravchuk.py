import numpy as np
import scipy.special as sps

def mu(N, p, values):
    """
    Evaluate pdf of Kravchuk base measure, equivalently the binomial B(N,p) pdf
    """
    if not np.all(np.ceil(values)==values):
        raise("IncorrectType")
    if not np.shape(values):
        values = np.array(values)
    res = np.zeros(values.shape, dtype='float')
    ind = (values>=0)
    res[ind] = np.exp( sps.gammaln(N+1) - sps.gammaln(values[ind]+1) -
        sps.gammaln(N-values[ind]+1) + values[ind]*np.log(p) + (N-values[ind])*np.log(1-p) )
    return res

def cmb(N, p, values, normalize=True):
    """
    Evaluate Conway-Maxwell-binomial CMB(N,p) pdf
    """
    if not np.all(np.ceil(values)==values):
        raise("IncorrectType")
    if not np.shape(values):
        values = np.array(values)
    res = np.zeros(values.shape, dtype='float')
    ind = (values>=0)
    res[ind] = np.exp( .5*sps.gammaln(N+1) - .5*sps.gammaln(values[ind]+1) -
        .5*sps.gammaln(N-values[ind]+1) + values[ind]*np.log(p) + (N-values[ind])*np.log(1-p) )
    if normalize:
        res /= np.max(res)
    return res

class KravchukPolynomials:
    """
    Iteratively evaluate Kravchuk polynomials using a three-term recurrence.

    Args:
        n -- the number of increasing degrees. Maximum degree will thus be n-1.
        N, p -- parameters of Kravchuk base measure B(N,p)
        values -- an array of values where to evaluate the polynomials.
    """

    def __init__(self, N, p, n, values):
        self.k = 1
        self.n = n
        self.N = N
        self.p = p
        self.values = values

        # Fill in the values of the first two unnormalized polynomials
        # Note we store two consecutive values since recurrence is of order 2
        self.res = np.ones((2, len(values)))
        self.res[1,:] += (values - p*N -1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.k < self.n:
            k = self.k
            N = self.N
            p = self.p
            resTmp = self.res[0,:].copy()
            self.res[0,:] = self.res[1,:]
            self.res[1,:] = 1./(k+1)*( (self.values-
                (k+p*(N-2*k)))*self.res[1,:] -
                (N-k+1)*p*(1-p)*resTmp )
            self.k += 1
            return resTmp / np.exp(.5*( sps.gammaln(N+1)-sps.gammaln(k)-
                sps.gammaln(N-k+2)+(k-1)*np.log(p)+(k-1)*np.log(1-p) ))
        else:
            raise StopIteration()
