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
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PlanarExperiment:
    """
    Sample a discrete white noise, compute its Chalier transfom, and look up its zeros.

    Args:
    -----
        expId -- a string identifying a particular experiment, used for saving results.
        N -- the length of the discretized signal.
        lambda -- parameter for Charlier base measure.
    """

    def __init__(self, expId="demo", N=2**8, llambda=1.):
        self.expId = expId
        self.N = N
        self.llambda = llambda
        self.phi = lambda z: np.sqrt(self.llambda)-z # phi is its own inverse

        print("Figures will be saved in the current folder; file names will contain the id \""+expId+"\".")
        print("The Poisson parameter is lambda="+str(llambda)+'.')

    def sampleWhiteNoise(self, seed=1):
        npr.seed(seed)
        """Sample realization of white noise."""
        N = self.N
        self.signal = 1/np.sqrt(2)*(npr.randn(N)+1J*npr.randn(N))

    def transform(self):
        """
        Approximate the Charlier transform at different radii $r$
        """
        N = len(self.signal)
        self.rArray = np.linspace(1e-3, np.sqrt(N), 1000) # N/4 chosen so that Poisson has support within cut
        self.spectrogram = np.array([spf.fftshift(spf.ifft(np.conjugate(self.signal)*
                        charlier.cmp(r, np.arange(N)))) for r in self.rArray]).T
        np.flip(self.spectrogram, axis=0) # put small thetas at the bottom for later display
        self.spectrogram /= np.max(np.abs(self.spectrogram))

    def findZeros(self, th=0.01):
        """
        Find zeros as local minima that are below a threshold
        """
        zeros = utils.extr2minth(np.abs(self.spectrogram), th)
        self.thetaArray = -np.pi+2*np.pi*np.arange(self.N)/self.N
        self.zerosPolar = [[self.rArray[zeros[1][i]], self.thetaArray[zeros[0][i]]]
                                      for i in range(len(zeros[0]))]

    def findMaxima(self, patchSize=3):
        """
        Find maxima, patchSize should be odd
        """
        maxima = utils.extr2max(np.abs(self.spectrogram), patchSize)
        self.thetaArray = -np.pi+2*np.pi*np.arange(self.N)/self.N
        self.maximaPolar = [[self.rArray[maxima[1][i]], self.thetaArray[maxima[0][i]]]
                                      for i in range(len(maxima[0]))]

    def getZerosInCartesianCoordinates(self):
        """
        Return a realization of the list of zeros of the planar GAF
        """
        rArray = self.rArray
        thetaArray = self.thetaArray
        zeros = []
        for i in range(len(self.zerosPolar)):
            r, theta = self.zerosPolar[i]
            z = self.phi(cm.rect(r, theta))
            zeros.append([np.real(z), np.imag(z)])
        return zeros

    def plotResults(self, boolShow=False, boolSave=1):
        """
        plot and save spectrogram, in its computationally-friendly coordinates
        """

        # Set plotting options
        mpl.rcParams['xtick.labelsize'] = 26;
        mpl.rcParams['ytick.labelsize'] = 26;
        plt.rc('axes', labelsize=26);
        plt.rc('legend', fontsize=18);
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0));
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0));
        mpl.rcParams['ps.useafm'] = True;
        mpl.rcParams['pdf.use14corefonts'] = True;
        mpl.rcParams['text.usetex'] = True;

        # Plot the sprectrogram
        print("### Plotting the spectrogram.")
        plt.figure(figsize=(22,12))
        rArray = self.rArray
        thetaArray = self.thetaArray
        extent = [rArray[0], rArray[-1], thetaArray[-1], thetaArray[0]]
        ax = plt.gca()
        pl = ax.imshow(np.abs(self.spectrogram), interpolation='nearest', aspect="auto",
                                    cmap="viridis", extent=extent)

        # Add the zeros
        for i in range(len(self.zerosPolar)):
            x, y = self.zerosPolar[i]
            ax.plot(x, y, 'o', color="white")
        # Add the maxima
        for i in range(len(self.maximaPolar)):
            x, y = self.maximaPolar[i]
            ax.plot(x, y, 'o', color="red")

        ax.set_xlim(rArray[0], rArray[-1])
        ax.set_ylim(thetaArray[0], thetaArray[-1])
        plt.yticks(-np.pi+np.pi*np.arange(5)/2., ["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\theta$")
        plt.colorbar(pl, orientation='horizontal')

        # Add the maximaPolar

        # Save and display
        if boolSave:
            plt.savefig("spectrogram_charlier_"+self.expId+"_lambda="+str(self.llambda)+".pdf")
            plt.savefig("spectrogram_charlier_"+self.expId+"_lambda="+str(self.llambda)+".eps")
        if boolShow:
            plt.show()

    def plotTransformedResults(self, boolShow=False, boolDemo=False):
        """
        plot and save the spectrogram in its natural coordinates
        """

        # Interpolate the spectrogram
        rs = self.rArray
        thetas = self.thetaArray
        fr = interp2d(rs, thetas, np.real(self.spectrogram), kind="cubic")
        fi = interp2d(rs, thetas, np.imag(self.spectrogram), kind="cubic")
        f = lambda z: fr(np.abs(z), cm.phase(z))+1J*fi(np.abs(z),cm.phase(z))

        # Compute the spectrogram in natural coordinates
        m = np.max(self.rArray)+1
        xx = np.linspace(-m, m, 500)
        X, Y = np.meshgrid(xx, xx)
        Z = np.zeros(X.shape)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = np.abs(f(self.phi(X[i,j]+1J*Y[i,j])))
        zmin = np.min(Z)
        zmax = np.max(Z)
        Z = (Z-zmin)/(zmax-zmin) # normalize spectrogram for plotting purposes

        # Plot the spectrogram
        fig = plt.figure(figsize=3*plt.figaspect(1.))
        ax = fig.add_subplot(111)
        extent = [xx[0], xx[-1], xx[-1], xx[0]]
        ax = plt.gca()
        pl = ax.imshow(Z, interpolation='nearest', aspect="auto",
                                            cmap="viridis", extent=extent)
        if not boolDemo:
            # Add colorbar while maintaining aspect ratio
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(pl, cax=cax)

        # Add the zeros
        for i in range(len(self.zerosPolar)):
            r, theta = self.zerosPolar[i]
            z = self.phi(cm.rect(r, theta))
            ax.plot(np.real(z), np.imag(z), 'o', markersize=10, color="white")

        # Add maxima
        #for i in range(len(self.maximaPolar)):
        #    r, theta = self.maximaPolar[i]
        #    z = self.phi(cm.rect(r, theta))
        #    ax.plot(np.real(z), np.imag(z), 'o', markersize=10, color="red")


        if boolDemo:
            # Crop the plot to give the illusion of the full planar GAF
            plt.xlim(-10,10)
            plt.ylim(-10,10)
            ax.set_axis_off()  # Turn off the axis planes

        # Save and display
        plt.savefig("spectrogram_charlier_natural_"+self.expId+"_lambda="+str(self.llambda)+".pdf")
        plt.savefig("spectrogram_charlier_natural_"+self.expId+"_lambda="+str(self.llambda)+".eps")
        if boolShow:
            plt.show()
