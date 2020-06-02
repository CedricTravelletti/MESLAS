""" Test script for meslas.sampling

"""
import torch
import numpy as np
from meslas.means import ConstantMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid, SquareGrid
from meslas.sampling import GRF
from meslas.excursion import coverage_fct_fixed_location

# Dimension of the response.
n_out = 4

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.0, sigmas=[np.sqrt(0.25), np.sqrt(0.3),
        np.sqrt(0.4), np.sqrt(0.5)])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function
mean = ConstantMean([1.0, -2.0, 4.0, 33.0])

# Create the GRF.
myGRF = GRF(mean, covariance)





# Array of locations.
S1 = torch.Tensor([[0, 0], [0, 1], [0, 2], [3, 0]]).float()
S2 = torch.Tensor([[0, 0], [3, 0], [5, 4]]).float()

# Corresponding response indices.
L1 = torch.Tensor([0, 0, 0, 1]).long()
L2 = torch.Tensor([0, 3, 0]).long()

# Test the sampling.
print(myGRF.sample(S1, L1))
