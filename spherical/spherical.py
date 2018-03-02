import numpy as np
import numpy.random as npr
import utils
import charlier
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal
import scipy.fftpack as spf
import cmath as cm
from scipy.interpolate import interp2d

class PlanarExperiment:
    """Sample a discrete white noise, compute its Chalier transfom, and look up its zeros.

    Args:
    -----
        expId -- a string identifying a particular experiment, used for saving results.

        N -- the length of the discretized signal.

        M -- the truncation level for the GAF.

        lambda -- parameter for Charlier base measure.

        A -- bound of the observation window [-A,A].

    """

    def __init__(self, expId="demo", N=2**8, llambda=1.):
        self.expId = expId
        self.N = N
        self.llambda = llambda

        print("Figures will be saved in the current folder; file names will contain the id \""+expId+"\".")
        print("The Poisson parameter is lambda="+str(llambda)+'.')

    def sampleWhiteNoise(self):
        """Sample realization of white noise."""
        N = self.N
        self.signal = 1/np.sqrt(2)*npr.randn(N)+1J*npr.randn(N)

    def transform(self):
        """Approximate the Charlier transform at different radii $r$

        Args:
        Returns:
        """
        N = len(self.signal)
        self.rArray = np.linspace(1e-3, N/4, 200) # N/4 chosen so that Poisson has support within cut
        self.spectrogram = np.array([spf.fftshift(spf.fft(self.signal*
                        np.sqrt(charlier.mu(r, np.arange(N))))) for r in self.rArray]).T
        np.flip(self.spectrogram, axis=0) # put small thetas at the bottom for later display

    def findZeros(self, th=0.01):
        """Find zeros as local minima that are below a threshold"""
        zeros = utils.extr2minth(np.abs(self.spectrogram), th)
        self.thetaArray = 2*np.pi*np.arange(self.N)/self.N
        self.zerosPolar = [[self.rArray[zeros[1][i]], self.thetaArray[zeros[0][i]]]
                                      for i in range(len(zeros[0]))]

    def plotResults(self, boolShow=False):
        """Plot and save spectrum, signal, and scalogram"""

        # set plotting options
        mpl.rcParams['xtick.labelsize'] = 26;
        mpl.rcParams['ytick.labelsize'] = 26;
        plt.rc('axes', labelsize=26);
        plt.rc('legend', fontsize=18);
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0));
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0));
        mpl.rcParams['ps.useafm'] = True;
        mpl.rcParams['pdf.use14corefonts'] = True;
        mpl.rcParams['text.usetex'] = True;

        # plot the scalogram
        print("### Plotting the spectrogram.")
        plt.figure(figsize=(22,12))
        rArray = self.rArray
        thetaArray = self.thetaArray
        extent = [rArray[0], rArray[-1], thetaArray[-1], thetaArray[0]]
        ax = plt.gca()
        pl = ax.imshow(np.abs(self.spectrogram), interpolation='nearest', aspect="auto",
                                    cmap="viridis", extent=extent)

        # add the zeros
        for i in range(len(self.zerosPolar)):
            x, y = self.zerosPolar[i]
            ax.plot(x, y, 'o', color="white")
        ax.set_xlim(rArray[0], rArray[-1])
        ax.set_ylim(thetaArray[0], thetaArray[-1])
        plt.yticks(np.pi*np.arange(5)/2., ["0", "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"])
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\theta$")
        plt.colorbar(pl, orientation='horizontal')
        plt.savefig("spectrogram_charlier_"+self.expId+"_lambda="+str(self.llambda)+".pdf")
        plt.savefig("spectrogram_charlier_"+self.expId+"_lambda="+str(self.llambda)+".eps")
        if boolShow:
            plt.show()

    def plotTransformedResults(self):
        rs = self.rArray
        thetas = self.thetaArray
        fr = interp2d(rs, thetas, np.real(self.spectrogram), kind="cubic")
        fi = interp2d(rs, thetas, np.imag(self.spectrogram), kind="cubic")
        f = lambda z: fr(np.abs(z), cm.phase(z))+1J*fi(np.abs(z),cm.phase(z))
        self.phi = lambda z: np.sqrt(self.llambda)-z
        X, Y = np.meshgrid(rs, rs)
        Z = np.zeros(X.shape)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = np.abs(f(self.phi(X[i,j]+1J*Y[i,j])))
        plt.contourf(X, Y, Z, cmap="viridis", levels=np.linspace(np.min(Z),np.max(Z),100))
        plt.colorbar()

        # add the zeros
        for i in range(len(self.zerosPolar)):
            x, y = self.phi(cm.rect(self.zerosPolar[i]))
            ax.plot(x, y, 'o', color="white")
