import numpy as np
import numpy.random as npr
#import mlpy.wavelet as wave
import utils
import laguerre
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal

class Experiment:
    """Sample an analytic white noise, compute its AWT, and look up its zeros.

    Args:
    -----
        expId -- a string identifying a particular experiment, used for saving results.

        N -- the length of the discretized signal.

        M -- the truncation level for the GAF.

        alpha -- parameter for Paul's wavelet.

        A -- bound of the observation window [-A,A].

    """

    def __init__(self, expId="demo", N=2**10, M=10**5, alpha=0., A=5):
        self.expId = expId
        self.N = N
        self.M = M
        self.alpha = alpha
        self.A = A
        self.tArray = np.linspace(-A,A,N)
        self.dt = 2*A/N

        print("Figures will be saved in the current folder; file names will contain the id \""+expId+"\".")
        print("Signals are discretized with N="+str(N), "values, equidistributed along [-A,A], where A="+str(A)+'.')
        print("The wavelet parameter is alpha="+str(alpha)+'.')
        print("The truncation level for random series is M="+'%.2E' % Decimal(str(M))+'.')

    def sampleWhiteNoise(self):
        """Sample realization of white noise."""
        M = self.M
        alpha = self.alpha
        tArray = self.tArray

        # Compute truncated random series in the time domain
        print("### Computing truncated random series in the time domain.")
        sd = npr.seed()
        self.signal = np.sum([1/np.sqrt(2)*(npr.randn()+1J*npr.randn())*s
                    for s in laguerre.IFLaguerreFunctions(M,alpha,tArray)], 0)

        # Compute the same series in the frequency domain, and avoid Fourier
        print("### Computing spectrum.")
        npr.seed(sd)
        freqs, _ = utils.fourier(self.signal, self.A)
        self.wArray = np.linspace(0, freqs[-1], len(freqs))
        self.spectrum = np.sum([1/np.sqrt(2)*(npr.randn()+1J*npr.randn())*l*np.sqrt(laguerre.mu(alpha,self.wArray))
                    for l in laguerre.LaguerrePolynomials(M,alpha,self.wArray)], 0)

    def performAWT(self):
        """Approximate Paul's wavelet transform at scale s of f truncated on [-A,A]

        Args:
            f -- target function evaluated at -A+2A*k/N, k=0..N-1, N should be even.
            A -- truncation level.

        Returns:
            freqs -- an array of N frequencies.
            F -- the array of corresponding approximate values of the Fourier transform.
        """
        freqs, F = utils.fourier(self.signal, self.A)
        dt = 2*self.A/self.N
        s0 = 2*dt*(2*self.alpha+1)/4/np.pi # 4pi s/(2*alpha+1) should be approx 2dt following [ToCo97]
        dj = 0.1
        J = int(1/dj*np.log2(self.N*dt/s0))+1
        self.scales = [s0*2**(j*dj) for j in range(J)]

        def psiHat(s, omega):
            """Fourier transform of wavalet psi_alpha evaluated at s*omega"""
            res = np.zeros(omega.shape)
            res[omega>=0] = (s*omega[omega>=0])**self.alpha*np.exp(-s*omega[omega>=0])
            return res

        self.awt = np.array([utils.fourierInverse(F*psiHat(s,freqs), np.max(freqs))[1]
                for s in self.scales])
        self.awt /= np.sum(np.abs(self.awt))

#    def performAWT(self):
#        """Perform analytic wavelet transform."""
#        dt = 2*self.A/self.N
#        self.scales = utils.autoscales(N=self.N, dt=dt, dj=0.1, wf='paul', p=self.alpha+1e-5)
#        self.awt = wave.cwt(x=self.signal, dt=dt, scales=self.scales, wf='paul', p=self.alpha+1e-5)

    def findZeros(self, th=0.01):
        """Find zeros as local minima that are below a threshold"""
        zeros = utils.extr2minth(np.abs(self.awt), th)
        self.zerosComplex = np.array([[self.tArray[zeros[1][i]] + 1J *self.scales[zeros[0][i]]]
                                      for i in range(len(zeros[0]))])

    def plotResults(self, boolShow=False, boolDemo=False):
        """Plot and save spectrum, signal, and scalogram"""

        # set plotting options
        mpl.rcParams['xtick.labelsize'] = 22;
        mpl.rcParams['ytick.labelsize'] = 22;
        plt.rc('axes', labelsize=22);
        plt.rc('legend', fontsize=18);
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0));
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0));
        mpl.rcParams['ps.useafm'] = True;
        mpl.rcParams['pdf.use14corefonts'] = True;
        mpl.rcParams['text.usetex'] = True;

        # plot the spectrum of our white noise
        print("### Plotting the spectrum of the realization of white noise.")
        plt.figure(figsize=(22,12))
        plt.subplot(2,1,1)
        plt.plot(self.wArray, np.real(self.spectrum), label="Re")
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(self.wArray, np.imag(self.spectrum), color='g', label="Im")
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("spectrum")
        plt.savefig("spectrum_"+self.expId+"_alpha="+str(self.alpha)+".pdf")
        plt.savefig("spectrum_"+self.expId+"_alpha="+str(self.alpha)+".eps")
        if boolShow:
            plt.show()

        # plot the white noise itself
        print("### Plotting the corresponding realization of white noise.")
        plt.figure(figsize=(22,12))
        if not boolDemo:
            plt.subplot(2,1,1)
            plt.plot(self.tArray, np.real(self.signal), label="Re")
            plt.xlim([-self.A, self.A])
            plt.legend()
            plt.subplot(2,1,2) # prepare for scalogram
            plt.plot(self.tArray, np.imag(self.signal), color='g', label="Im")
            plt.xlim([-self.A, self.A])
            ax = plt.gca()
            ax.set_xlabel(r"$t$")
            ax.set_ylabel("signal")

            plt.savefig("signal_"+self.expId+"_alpha="+str(self.alpha)+".pdf")
            plt.savefig("signal_"+self.expId+"_alpha="+str(self.alpha)+".eps")
            if boolShow:
                plt.show()

        # plot the scalogram
        print("### Plotting the scalogram.")
        #plt.figure(figsize=(22,12))
        plt.figure(figsize=(28,10))
        tArray = self.tArray
        scales = self.scales
        extent = [tArray[0],tArray[-1],np.log10(scales[-1]), np.log10(scales[0])]
        ax = plt.gca()
        pl = ax.imshow(np.sqrt(np.abs(self.awt)), interpolation='nearest', aspect="auto", cmap="viridis", extent=extent)

        # Add the zeros
        print("There are ", len(self.zerosComplex), "zeros.")
        for i in range(len(self.zerosComplex)):
            z = self.zerosComplex[i]
            x, y = np.real(z), np.imag(z)
            ax.plot(x, np.log10(y),'o', markersize=7, color="white")

        # Adjust window
        ax.set_xlim(tArray[0], tArray[-1])
        ax.set_ylim(np.log10(scales[0]), np.log10(scales[-1]))

        if not boolDemo:
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\log s$")
            plt.colorbar(pl, orientation='horizontal')
        else:
            ax.set_axis_off()  # Don't show the axes

        plt.savefig("scalogram_"+self.expId+"_alpha="+str(self.alpha)+".pdf")
        plt.savefig("scalogram_"+self.expId+"_alpha="+str(self.alpha)+".eps")
        if boolShow:
            plt.show()
