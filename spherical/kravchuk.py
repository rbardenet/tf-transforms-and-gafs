import numpy as np
import scipy.special as sps

def mu(llambda, values):
    """Evaluate pdf of Charlier base measure"""
    if not np.all(np.ceil(values)==values):
        raise("IncorrectType")
    if not np.shape(values):
        values = np.array(values)
    res = np.zeros(values.shape, dtype='float')
    ind = (values>=0)
#    res[ind] = llambda**values[ind]/sps.gamma(values[ind]+1)*np.exp(-llambda)
    res[ind] = np.exp(values[ind]*np.log(llambda) - sps.gammaln(values[ind]+1) - llambda)
    return res

class CharlierPolynomials:
    """Iteratively evaluate Charlier polynomials using a three-term recurrence.

    Args:
        n -- the number of increasing degrees. Maximum degree will thus be n-1.
        llambda -- parameter of the Chalier polynomials (mean of Poisson base measure)
        values -- an array of values where to evaluate the polynomials.

    """

    def __init__(self, n, llambda, values):
        self.k = 1
        self.n = n
        self.llambda = llambda
        self.values = values

        # Fill in the values of the first two unnormalized polynomials
        self.res = np.ones((2, len(values)))   # store two consecutive values since recurrence is of order 2
        self.res[1,:] += (-values/llambda)

    def __iter__(self):
        return self

    def __next__(self):
        if self.k < self.n:
            k = self.k
            llambda = self.llambda
            resTmp = self.res[0,:].copy()
            self.res[0,:] = self.res[1,:]
            self.res[1,:] = 1./(llambda) * ( (k+llambda-self.values)*self.res[1,:] - k*resTmp )
            self.k += 1
            return resTmp / np.sqrt(sps.gamma(k)/llambda**(k-1))
        else:
            raise StopIteration()
