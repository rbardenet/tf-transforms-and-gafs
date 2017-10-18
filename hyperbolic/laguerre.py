import numpy as np
import scipy.special as sps

def psi(alpha, t):
    """Evaluate Paul's wavelet"""
    return 1./(t+1J)**(alpha+1)

def mu(alpha, t):
    """Evaluate pdf of Laguerre base measure"""
    if not np.shape(t):
        t = np.array(t)
    res = np.zeros(t.shape)
    ind = (t>0)
    res[ind] = t[ind]**alpha * np.exp(-t[ind])
    return res

def laguerreFunction(n, alpha, t, normalized=True):
    """Evaluate Laguerre function using scipy.special"""
    if normalized:
        Z = np.exp( .5*sps.gammaln(n+1) - .5*sps.gammaln(n+alpha+1) )
    else:
        Z = 1
    return Z * np.sqrt(mu(alpha,t)) * sps.eval_genlaguerre(n, alpha, t)

class LaguerrePolynomials:
    """Iteratively evaluate Laguerre polynomials using a three-term recurrence.

    Args:
        n -- the number of increasing degrees. Maximum degree will thus be n-1.
        alpha -- parameter of the generalized Laguerre polynomials.
        values -- an array of values where to evaluate the polynomials.

    """

    def __init__(self, n, alpha, values):
        self.k = 1
        self.n = n
        self.alpha = alpha
        self.values = values
        
        # Fill in the values of the first two polynomials
        self.res = np.ones((2, len(values)))        # store two consecutive values since recurrence is of order 2
        self.res[1,:] += (alpha - values)
        
        self.Z = 1/sps.gamma(alpha+1)        # squared normalization of res[0,:]

    def __iter__(self):
        return self

    def __next__(self):
        if self.k < self.n:
            k = self.k
            alpha = self.alpha
            resTmp = self.res[0,:].copy()
            ZTmp = self.Z
            self.res[0,:] = self.res[1,:]
            self.res[1,:] = 1./(k+1) * ( (2*k+1+alpha-self.values)*self.res[1,:] - (k+alpha)*resTmp )
            self.Z *= k/(k+alpha)
            self.k += 1
            return np.sqrt(ZTmp)*resTmp
        else:
            raise StopIteration()

class IFLaguerreFunctions:
    """Iteratively evaluate the inverse Fourier transform of Laguerre functions using a three-term recurrence.

    Args:
        n -- the number of increasing degrees. Maximum degree will thus be n-1.
        alpha -- parameter of the generalized Laguerre polynomials.
        values -- an array of values where to evaluate the polynomials.

    """

    def __init__(self, n, alpha, values):
        self.k = 1
        self.n = n
        self.a = alpha/2
        a = self.a
        self.z = (2*values-1J)/(2*values+1J)
        self.cst = sps.gamma(a+1) / np.sqrt(2*np.pi*sps.gamma(2*a+1)) *(1-self.z)**(a+1) # term that doesn't depend on n
        self.kappa = np.array([1., (a+1)/np.sqrt(2*a+1),
                               (a+1)*(a+2)/np.sqrt( 2*(2*a+1)*(2*a+2) )]) # store three consecutive values n-1,n,n+1
        if a==0:
            self.g0 = np.array([1,0,0])
        else:
            self.g0 = self.kappa * a/(a+np.arange(3)) # store three values for consistency with kappa
        self.res = self.cst * np.ones((2, len(values)),
                                      dtype="complex") # store two consecutive values since recurrence is of order 2
        self.res[1,:] = self.cst * (self.g0[1] + self.kappa[1]*self.z)

    def __iter__(self):
        return self

    def __next__(self):
        if self.k < self.n:
            k = self.k
            a = self.a
            z = self.z

            if a==0:
                tmp = self.cst*z**(k-1)
            else:
                kappa = self.kappa
                g0 = self.g0
                tmp = self.res[0,:].copy()
                self.res[0,:] = self.res[1,:]
                # recurrence
                self.res[1,:] = 1./(kappa[1]*g0[1]) * ((kappa[1]*g0[2]+kappa[2]*g0[1]*z)*self.res[1,:] 
                                                   - kappa[0]*g0[2]*z*tmp)

            # update coefficients
            self.kappa[0] = self.kappa[1]
            self.kappa[1] = self.kappa[2]
            self.kappa[2] *= (a+k+2)/np.sqrt((k+2)*(2*a+k+2))
            self.g0[0] = self.g0[1]
            self.g0[1] = self.g0[2]
            self.g0[2] = a/(k+2+a) * self.kappa[2]
            self.k += 1
            return tmp
            # return np.sqrt(sps.gamma(2*a+1+k+1)/sps.gamma(2*a+1)/sps.gamma(k+1)) * tmp # Shen's normalization
        else:
            raise StopIteration()
