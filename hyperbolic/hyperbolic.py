import numpy as np
import numpy.random as npr
import mlpy.wavelet as wave
import utils

class Experiment:
    """Sample an analytic white noise, compute its AWT, and look up its zeros.

    Args:
        expId -- a string identifying a particular experiment, used for saving results.
        N -- the length of the discretized signal.
        M -- the truncation level for the GAF.
        alpha -- parameter for Paul's wavelet.
        A -- bound of the observation window [-A,A].

    """

    def __init__(self, expId, N, M, alpha, A):
        self.expId = expId
        self.N = N
        self.M = M
        self.alpha = alpha
        self.A = A
        self.tArray = np.linspace(-A,A,N)
        self.dt = 2*A/N

    def sampleWhiteNoise(self):
        """Sample realization of white noise."""
        M = self.M
        alpha = self.alpha
        tArray = self.tArray

        # Compute truncated random series in the time domain
        sd = npr.seed()
        self.signal = np.sum([1/np.sqrt(2)*(npr.randn()+1J*npr.randn())*s
                    for s in laguerre.IFLaguerreFunctionsIsmail(M,alpha,tArray)], 0)

        # Compute the same series in the frequency domain, and avoid Fourier
        npr.seed(sd)
        freqs, _ = utils.fourier
        wArray = np.linspace(0, freqs[-1], len(freqs))
        self.spectrum = np.sum([1/np.sqrt(2)*(npr.randn()+1J*npr.randn())*l*np.sqrt(laguerre.mu(alpha,wArray))
                    for l in laguerre.LaguerrePolynomial(M,alpha,tArray)], 0)

    def performAWT(self):
        """Perform analytic wavelet transform."""
        dt = 2*self.A/N
        self.scales = wave.autoscales(N=self.N, dt=dt, dj=0.1, wf='paul', p=self.alpha+1e-5)
        self.awt = wave.cwt(x=self.signal, dt=dt, scales=self.scales, wf='paul', p=self.alpha+1e-5)

    def findZeros(self, th=0.01):
        """Find zeros as local minima that are below a threshold"""
        zeros = extr2minth(np.abs(self.awt), th)
        self.zerosComplex = np.array([[self.tArray[zeros[1][i]] + 1J *self.scales[zeros[0][i]]]
                                      for i in range(len(zeros[0]))])

    def plotResults(self, boolShow=False):
        """Plot and save spectrum, signal, and scalogram"""

        # plot the spectrum of our white noise
        plt.figure(figsize=20,12)
        plt.plot(self.wArray, self.spectrum)
        plt.set_xlabel(r"$\omega$")
        plt.savefig("spectrum_"+self.expId+"_alpha="+str(self.alpha)+".pdf")
        plt.savefig("spectrum_"+self.expId+"_alpha="+str(self.alpha)+".eps")
        if boolShow:
            plt.show()

        # plot the white noise itself
        plt.figure(figsize=20,12)
        plt.plot(self.tArray, self.signal)
        plt.set_xlabel(r"$t$")
        plt.savefig("signal_"+self.expId+"_alpha="+str(self.alpha)+".pdf")
        plt.savefig("signal_"+self.expId+"_alpha="+str(self.alpha)+".eps")
        if boolShow:
            plt.show()

        # plot the scalogram
        plt.figure(figsize=20,12)
        tArray = self.tArray
        scales = self.scales
        extent = [tArray[0],tArray[-1],np.log10(scales[-1]), np.log10(scales[0])]
        ax = plt.gca()
        pl = ax.imshow(np.abs(self.awt)/self.M, interpolation='nearest', aspect="auto", cmap="viridis", extent=extent)

        # add the zeros
        for i in range(len(zeros[0])):
            ax.plot(tArray[zeros[1][i]], np.log10(scales[zeros[0][i]]), 'o', color="white")
        ax.set_xlim(tArray[0], tArray[-1])
        ax.set_ylim(np.log10(scales[0]), np.log10(scales[-1]))
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\log s$")
        plt.colorbar(pl, orientation='horizontal')
        plt.savefig("scalogram_"+self.expId+"_alpha="+str(self.alpha)+".pdf")
        plt.savefig("scalogram_"+self.expId+"_alpha="+str(self.alpha)+".eps")
        if boolShow:
            plt.show()
