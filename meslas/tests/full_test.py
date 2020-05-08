""" Try to test all functionalities together.

"""
import torch
from meslas.means import ConstantMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.sampling import GRF

# Specifiy Covariance function.
dim = 2

# Spatial Covariance.
lmbda = torch.Tensor([0.1])
sigma = torch.Tensor([1.0])
matern_cov = Matern32(lmbda, sigma)

# Cross covariance.
gamma0 = torch.Tensor([0.3])
sigmas = torch.sqrt(torch.Tensor([0.25, 0.6]))
cross_cov = UniformMixing(gamma0, sigmas)

covariance = FactorCovariance(matern_cov, cross_cov)

# Specify mean function
mean = ConstantMean([1.0, -2.0])

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create a regular square gird in 2 dims.
from meslas.grid import square_grid, get_isotopic_generalized_location
my_grid = square_grid(100, dim)

# Create an index vector for isotopic sampling.
S_iso, L_iso = get_isotopic_generalized_location(my_grid, dim)

# Sample all components at all locations.
sample = myGRF.sample(S_iso, L_iso)

# Plot.
import matplotlib.pyplot as plt

# Separate indices.
sample = sample.reshape((2, 100*100))

plt.subplot(121)
plt.imshow(sample[0, :].reshape((100, 100)).numpy(), vmin=-3.5, vmax=3.5, cmap='jet')
plt.colorbar()
plt.subplot(122)
plt.imshow(sample[1, :].reshape((100, 100)).numpy(), vmin=-3.5, vmax=3.5, cmap='jet')
plt.colorbar()
plt.show()
