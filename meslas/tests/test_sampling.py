""" Test script for meslas.sampling

"""
import torch
from meslas.means import ConstantMean
from meslas.covariance.heterotopic import matern32, uniform_mixing_crosscov, Covariance
from meslas.sampling import GRF

# Specifiy Covariance function.
dim = 4

lmbda = torch.Tensor([1.0])
sigma = torch.Tensor([1.0])
# Cross covariance parameters.
gamma0 = torch.Tensor([0.6])
sigmas = torch.sqrt(torch.Tensor([0.25, 0.3, 0.4, 0.5]))

def my_factor_cov(H, L1, L2):
    cov_spatial = matern32(H, lmbda, sigma)
    cross_cov = uniform_mixing_crosscov(L1, L2, gamma0, sigmas)
    return cross_cov * cov_spatial

my_covariance = Covariance(my_factor_cov)

# Specify mean function
# Constant mean of each component.
means = torch.Tensor([1.0, -2.0, 4.0, 33.0])
my_mean = ConstantMean(means)

# Create the GRF.
myGRF = GRF(my_mean, my_covariance)

# Array of locations.
S1 = torch.Tensor([[0, 0], [0, 1], [0, 2], [3, 0]]).float()
S2 = torch.Tensor([[0, 0], [3, 0], [5, 4]]).float()

# Corresponding response indices.
L1 = torch.Tensor([0, 0, 0, 1]).long()
L2 = torch.Tensor([0, 3, 0]).long()

# Test the sampling.
print(myGRF.sample(S1, L1))
