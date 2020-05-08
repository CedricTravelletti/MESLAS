""" Try to test all functionalities together.

"""
import torch
from meslas.means import ConstantMean
from meslas.covariance.heterotopic import matern32, uniform_mixing_crosscov, Covariance
from meslas.sampling import GRF

# Specifiy Covariance function.
dim = 2

lmbda = torch.Tensor([0.1])
sigma = torch.Tensor([1.0])
# Cross covariance parameters.
gamma0 = torch.Tensor([0.6])
sigmas = torch.sqrt(torch.Tensor([0.25, 0.6]))

def my_factor_cov(H, L1, L2):
    cov_spatial = matern32(H, lmbda, sigma)
    cross_cov = uniform_mixing_crosscov(L1, L2, gamma0, sigmas)
    return cross_cov * cov_spatial

my_covariance = Covariance(my_factor_cov)

# Specify mean function
# Constant mean of each component.
means = torch.Tensor([1.0, -2.0])
my_mean = ConstantMean(means)

# Create the GRF.
myGRF = GRF(my_mean, my_covariance)

# Create a regular square gird in 2 dims.
from meslas.grid import square_grid, get_isotopic_generalized_location
my_grid = square_grid(100, dim)

# Create an index vector for isotopic sampling.
S_iso, L_iso = get_isotopic_generalized_location(my_grid, dim)

# Test the sampling.
print(myGRF.sample(S_iso, L_iso))
