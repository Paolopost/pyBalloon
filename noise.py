"""Variuos noise models."""

import numpy as N


class Noise(object):
    """Base class of noise generators.
    """
    def __init__(self, size):
        """Contructor

        size = amount of noise to produce.
        """
        self.size = size
        return
    
    pass


class GaussianWhiteNoise(Noise):
    """Gaussian White Noise.
    """
    def __init__(self, sigma=1.0, size=100):
        Noise.__init__(self,size)
        self.sigma = sigma
        self.generate()
        return

    def generate(self):
        self.data = N.random.randn(self.size)*self.sigma
        return self.data
    
    pass


class GaussianNoise(Noise):
    """Gaussian (non white) Noise generator.
    """
    def __init__(self,covariance, size=None):
        """Size can be any arbitrary size, including a tuple for the case of
        multimensional noise generation.
        
        The noise generated has the same dimensions as size (in the general
        case) or it is just one random vector of size covariance.shape[0]
        (when size is not specified).

        >>> N.random.seed(1)
        >>> cov=N.eye(3)
        >>> gn=GaussianNoise(cov)
        >>> gn.data
        array([ 1.62434536, -0.61175641, -0.52817175])
        """
        if size==None:
            size = covariance.shape[0]
            pass
        Noise.__init__(self,size)
        self.covariance = covariance
        self.generate()
        return
    
    def generate(self):
        size = []
        if not N.isscalar(self.size):
            size = self.size[1:]
            pass
        self.data = N.random.multivariate_normal( \
            N.zeros(self.covariance.shape[0]),
            self.covariance, size)
        # roll axes to match self.size.shape (in the general case), as
        # it is expected by this application (but not from
        # N.random.multivariate_normal, see its docstring for
        # details about why this operation is necessary):
        self.data = N.rollaxis(self.data,-1)
        return self.data
    
    pass


class GaussianAR1Noise(GaussianNoise):
    """Gaussian Auto Regressive (p=1) temporal Noise.
    """
    def __init__(self,rho=0.9,sigma1=0.5,size=100):
        self.size = size
        if not N.isscalar(size):
            self.size = N.array(size)
            size = size[0]
            pass
        sigma_squared = sigma1**2/(1.0-rho**2)
        self.covariance = rho**N.abs(N.subtract.outer(N.arange(size),N.arange(size)))*sigma_squared
        GaussianNoise.__init__(self, covariance=self.covariance, size=self.size)
        pass

    pass


class SpatialGaussianAR1Noise(object):
    """Gaussian Auto Regressive (p=1) Spatial (2D) noise.
    """
    def __init__(self, noise3D, rho_spatial=0.5, sigma_spatial=0.1):
        self.noise3D = noise3D
        self.rho_spatial = rho_spatial
        self.sigma_spatial = sigma_spatial
        self.generate()
        return

    def generate(self):
        self.data = N.zeros(self.noise3D.shape)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                tmp = N.roll(self.noise3D,i,1)
                tmp = N.roll(tmp,j,2)
                self.data += tmp
                pass
            pass
        self.data /= 9.0
        # print self.data.shape
        self.data = self.rho_spatial*self.data + GaussianNoise(covariance=N.eye(self.data.shape[0])*self.sigma_spatial**2,size=self.data.shape).data
        return self.data

    pass

        
class SpatialGaussianAR1Noise2(object):
    """Gaussian Auto Regressive (p=1) Spatial (2D) noise.

    Iterative agorithm that computes voxels in a given sequence
    (and using updated values 
    """
    def __init__(self, noise3D, rho_spatial=0.5, sigma_spatial=0.1):
        self.noise3D = noise3D # .copy()?
        self.rho_spatial = rho_spatial
        self.sigma_spatial = sigma_spatial
        self.generate()
        return

    def generate(self):
        # 1. generate a (random) path to navigate the 2D matrix,
        #    which is a random permutation of the indexes of the 2D matrix.
        path = zip(*N.where(self.noise3D[0,:,:]!=N.nan)) # trick to generate an ordered list of (X,Y) pairs (path). Assumes there is no NaN in noise3D
        N.random.shuffle(path) # shuffle path (inplace) 
        # 2. for each voxel in the path compute the AR value according to the usual formula:
        #    X_ij = rho * mean(X_neighborhood(i,j)) + N(0,sigma)
        #    remember to update the dataset with the new compute value!!
        # Note: instead of mean it can be a weighted mean.
        radius = 1 # radius
        print self.noise3D.shape
        for i,j in path:
            print i,j
            for k in range(i-radius,i+radius+1):
                for l in range(j-radius,j+radius+1):
                    if i!=k and j!=l:
                        self.noise3D[:,i,j] += self.noise3D[:,k % self.noise3D.shape[1],l % self.noise3D.shape[2]]
                        pass
                    pass
                pass
            self.noise3D[:,i,j] = self.rho_spatial*self.noise3D[:,i,j]/float((radius*2+1)**2) + N.random.randn(self.noise3D.shape[0])*self.sigma_spatial
            pass
        self.data = self.noise3D
        return self.noise3D
    pass

        
def GWN(size,sigma=1.0):
    """Gaussian White Noise.

    sigma = std of the Gaussian
    size = size of the result
    """
    return N.random.randn(size)*sigma


def GN(covariance):
    """Colored Gaussian Noise.

    covariance = the covariance structure of the noise
    """
    return N.random.multivariate_normal(N.zeros(covariance.shape[0]),covariance)


def GN_AR1(rho=0.9,sigma1=0.5,size=100):
    """Autoregressive AR(1) Gaussian noise.

    rho = temporal correlation (-1 < rho < 1)
    sigma1 = amplitude of iid Gaussian noise
    size = number of timesteps to generate
    """
    sigma_squared = sigma1**2/(1.0-rho**2)
    cova = rho**N.abs(N.subtract.outer(N.arange(size),N.arange(size)))*sigma_squared
    return GN(cova)


def GN_AR1_recurrent(rho=0.9,sigma1=0.5,size=100):
    """Autoregressive AR(1) Gaussian noise. Recurrent algoritm.

    rho = temporal correlation (-1 < rho < 1)
    sigma1 = amplitude of iid Gaussian noise
    size = number of timesteps to generate
    """
    noise = N.zeros(size)
    for i in range(size):
        noise[i] = rho*noise[i-1] + N.random.randn()*sigma1
        pass
    return noise


def GN_ARp_recurrent(rho=N.array([0.9,0.5,-0.1,0.001]),sigma1=0.5,size=100):
    """Autoregressive AR(p) Gaussian noise. Recurrent algoritm.

    rho = temporal correlation vector (-1 < rho[i] < 1)
    sigma1 = amplitude of iid Gaussian noise
    size = number of timesteps to generate
    """
    noise = N.zeros(size)
    p = rho.size
    tmp = -N.arange(p)
    for i in range(size):
        s = range(i-1,i-p-1,-1)
        noise[i] = (rho*noise[s]).sum() + N.random.randn()*sigma1
        pass
    return noise


def purdon1998(rho=0.9,sigma1=0.5,sigma2=0.3,size=100):
    """Gaussian AR(1) noise plus a second source of white noise.
    This model wan introduced by Purdon et al. (1998) to model
    fMRI noise and take into account white noise from scanner.

    rho = temporal correlation (-1 < rho < 1)
    sigma1 = amplitude of iid Gaussian noise
    sigma2 = amplitude of iid Gaussian scanner noise
    size = number of timesteps to generate
    """
    return GN_AR1_recurrent(rho=rho,sigma1=sigma1,size=size)+N.random.randn(size)*sigma2


if __name__=="__main__":

    import pylab

    dim = 500

    noise_GWN = GWN(dim,1.0)
    pylab.plot(noise_GWN+0,label="GWN")
    
    cova = N.eye(dim)
    noise_GN = GN(cova)
    pylab.plot(noise_GN+3,label="GN (white)")

    rho = 0.9
    sigma1 = 0.4
    noise_GN_AR1 = GN_AR1(rho=rho, sigma1=sigma1, size=dim)
    pylab.plot(noise_GN_AR1+6,label="GN AR(1)")

    rho = 0.9
    sigma1 = 0.4
    noise_GN_AR1_r = GN_AR1_recurrent(rho=rho, sigma1=sigma1, size=dim)
    pylab.plot(noise_GN_AR1_r+9,label="GN AR(1) rec.")

    rho = N.array([0.9,0.1,-0.05,0.001])
    sigma1 = 0.4
    noise_GN_ARp_r = GN_ARp_recurrent(rho=rho, sigma1=sigma1, size=dim)
    pylab.plot(noise_GN_ARp_r+12,label="GN AR(p) rec.")

    rho = 0.9
    sigma1 = 0.4
    sigma2 = 0.3
    noise_purdon1998 = purdon1998(rho=rho, sigma1=sigma1, sigma2=0.3, size=dim)
    pylab.plot(noise_purdon1998+15,label="Purdon (1998)")

    pylab.legend()
    pylab.title("Examples of noise")
