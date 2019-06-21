import numpy as np
import numpy.random as npr
import utils
import kravchuk
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal
import scipy.fftpack as spf
import cmath as cm
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib import cm as cmaps
from mpl_toolkits.mplot3d import Axes3D

class SphericalExperiment:
    """
    Sample a discrete white noise, compute its Kravchuk transfom, and look up its zeros.

    Args:
    -----
        expId -- a string identifying a particular experiment, used for saving results.
        N -- the length of the discretized signal.
        q -- parameter for base measure.
    """

    def __init__(self, expId="demo", N=2**8, q=.5):
        self.expId = expId
        self.N = N
        self.p = 1/(1+q**2)
        self.q = q

        print("Figures will be saved in the current folder; file names will contain the id \""+expId+"\".")
        print("The binomial parameters are N,p=", N, self.p)

    def sampleWhiteNoise(self):
        """
        Sample realization of white noise.
        """
        N = self.N
        self.signal = 1/np.sqrt(2)*(npr.randn(N)+1J*npr.randn(N))

    def transform(self):
        """
        Approximate the Charlier transform at different radii $r$
        """
        N = len(self.signal)
        self.rArray = np.linspace(1e-3, N/4, 1000) # N/4 chosen so that Poisson has support within cut
        self.spectrogram = np.array([spf.fftshift(spf.ifft(np.conjugate(self.signal)*
                        kravchuk.cmb(N, r/(r+1), np.arange(N)))) for r in self.rArray]).T
        np.flip(self.spectrogram, axis=0) # put small rs at the bottom for later display
        self.spectrogram /= np.max(np.abs(self.spectrogram))

    def findZeros(self, th=0.01):
        """
        Find zeros as local minima that are below a threshold
        """
        zeros = utils.extr2minth(np.abs(self.spectrogram), th)
        self.thetaArray = -np.pi+2*np.pi*np.arange(self.N)/self.N
        self.zerosPolar = [[self.rArray[zeros[1][i]], self.thetaArray[zeros[0][i]]]
                                      for i in range(len(zeros[0]))]

    def plotResults(self, boolShow=False):
        """
        plot and save spectrogram, in its computationally-friendly coordinates
        """

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

        # plot the spectrogram
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

        # Embellish, save and display
        ax.set_xlim(rArray[0], rArray[-1])
        ax.set_ylim(thetaArray[0], thetaArray[-1])
        plt.yticks(-np.pi+np.pi*np.arange(5)/2., ["$-\pi$", "$-\pi/2$", "$0$", "$\pi/2$", "$\pi$"])
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\theta$")
        plt.colorbar(pl, orientation='horizontal')
        plt.savefig("spectrogram_kravchuk_"+self.expId+"_p="+str(self.p)+".pdf")
        plt.savefig("spectrogram_kravchuk_"+self.expId+"_p="+str(self.p)+".eps")
        if boolShow:
            plt.show()

    def plotTransformedResults(self):
        """
        plot and save the spectrogram in its natural coordinates
        """
        q = self.q
        rs = self.rArray
        thetas = self.thetaArray
        fr = interp2d(rs, thetas, np.real(self.spectrogram), kind="cubic")
        fi = interp2d(rs, thetas, np.imag(self.spectrogram), kind="cubic")
        f = lambda z: fr(np.abs(z), cm.phase(z))+1J*fi(np.abs(z),cm.phase(z))
        self.phi = lambda z: (1-q*z)/(q+z)
        x = np.linspace(-3,3,200)
        X, Y = np.meshgrid(x, x)
        Z = np.zeros(X.shape)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = np.abs(f(self.phi(X[i,j]+1J*Y[i,j])))
        plt.contourf(X, Y, Z, cmap="viridis", levels=np.linspace(np.min(Z),np.max(Z),100))
        plt.colorbar()

        # add the zeros
        ax = plt.gca()
        for i in range(len(self.zerosPolar)):
            z = self.phi(cm.rect(*self.zerosPolar[i]))
            ax.plot(np.real(z), np.imag(z), 'o', color="white")

        # Embellish and plot
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(x), np.max(x))
        plt.show()

    def plotSphericalResults(self):
        """
        plot the spectrogram in spherical coordinates, and on the sphere
        """
        rs = self.rArray
        thetas = self.thetaArray
        q = self.q
        fr = interp2d(rs, thetas, np.real(self.spectrogram), kind="cubic")
        fi = interp2d(rs, thetas, np.imag(self.spectrogram), kind="cubic")
        f = lambda z: fr(np.abs(z), cm.phase(z))+1J*fi(np.abs(z),cm.phase(z))
        self.phi = lambda z: (1-q*z)/(q+z)
        phis = np.linspace(1e-3,np.pi,400)
        #phis = np.linspace(1e-3-np.pi,np.pi,400)
        #phis = np.linspace(np.pi/2,np.pi+np.pi/2,400)

        Phi, Theta = np.meshgrid(phis, thetas)
        Z = np.zeros(Phi.shape)
        phiOffset = 0. # Adding an offset to see the monomials behind the sphere
        thetaOffset = 0. #3*np.pi/2
        for i in range(Phi.shape[0]):
            for j in range(Phi.shape[1]):
                Z[i,j] = np.abs(f(self.phi(cm.rect(np.sin(phiOffset+Phi[i,j])/(1-np.cos(phiOffset+Phi[i,j])), Theta[i,j]+thetaOffset))))
        plt.contourf(Phi, Theta, Z, cmap="viridis", levels=np.linspace(np.min(Z),np.max(Z),100))
        plt.colorbar()

        # Plot the zeros
        for i in range(len(self.zerosPolar)):
            z = self.phi(cm.rect(*self.zerosPolar[i]))
            phi, theta = 2*np.arctan(1/np.abs(z)), cm.phase(z)
            plt.plot(np.mod(phi+phiOffset,np.pi),theta,'o')
#            plt.plot(np.mod(phi+phiOffset,2*np.pi),theta,'s')
#            plt.plot(phi,theta,'t')


        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$\theta$")
        plt.show()

        # The Cartesian coordinates of the unit sphere
        XX = np.sin(Phi) * np.cos(Theta)
        YY = np.sin(Phi) * np.sin(Theta)
        ZZ = np.cos(Phi)

        # Normalize to [0,1]
        fmax, fmin = Z.max(), Z.min()
        Z = (Z - fmin)/(fmax - fmin)

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=4*plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, zorder=-1, facecolors=cmaps.viridis(Z), rasterized=True)
        ax.set_axis_off()  # Turn off the axis planes

        # Rotate the axes to see the monomial
        elev = -60
        azim = 145
        ax.view_init(elev, azim)
        ta = np.radians(azim)
        # rotationMatrix = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        # R_azim = np.eye(3)
        # R_azim[:2,:2] = rotationMatrix(np.radians(azim))
        # print("Ra", R_azim)
        # R_elev = np.eye(3)
        # R_elev[[0,2],[0,2]] = rotationMatrix(np.radians(elev))
        # print("Re", R_elev)
        # normalVector =  R_azim.dot(R_elev).dot(np.array([1,-1,1]))
        print("hello", ax.format_coord(0,0))

        # Add zeros
        xs = []
        ys = []
        zs = []
        eps=1e-3 # used to shift zeros towards the viewer, and bypass a rendering bug on zorder

        for i in range(len(self.zerosPolar)):
            z = self.phi(cm.rect(*self.zerosPolar[i]))
            phi, theta = 2*np.arctan(1/np.abs(z)), cm.phase(z)
            phi -= phiOffset
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            for i in range(2,3):
                npr.seed(i)
                n = npr.randn(3)
                if np.dot(n, np.array([x,y,z]))>=0: # that is, if the zero is on the hemisphere we can see
                #            if x-y+z>=0: # that is, if the zero is on the hemisphere we can see
                    t = np.array([x,y,z]) + eps*n
                    xs.append(t[0]) # shift the zeros towards the viewer
                    ys.append(t[1])
                    zs.append(t[2])
            ax.scatter(xs, ys, zs, s=30, zorder=1, color="white")

        plt.savefig("spectrogram_kravchuk_natural_"+self.expId+"_p="+str(self.p)+".pdf")
        plt.savefig("spectrogram_kravchuk_natural_"+self.expId+"_p="+str(self.p)+".eps")
        plt.show()
