""" Tests the coverage function capabilities.
Here we only test if it works. For corectness checks, see test_mvnorm.py

"""
import numpy as np
import torch
from meslas.means import ConstantMean
from meslas.covariance.spatial_covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid, SquareGrid
from meslas.random_fields import GRF
from meslas.excursion import coverage_fct_fixed_location


# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.0, sigmas=[np.sqrt(1.0), np.sqrt(1.5)])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function
mean = ConstantMean([0.0, 0.0])

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create an equilateral triangular grid in 2 dims.
# Number of respones.
my_grid = TriangularGrid(40)
print(my_grid.n_points)

# Observe some data.
S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0],
        [0.6, 0.5]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0])
y = torch.tensor(11*[-6]).float()

# Predict at some points.
S2 = torch.Tensor([[0.2, 0.1], [0, 0], [3, 0], [5, 4]]).float()
L2 = torch.Tensor([0, 0, 1, 0]).long()

mu_cond_list, var_cond_list = myGRF.krig(
        S2, L2, S_y, L_y, y,
        noise_std=0.05,
        compute_post_var=True)
"""
# Plot.
from meslas.plotting import plot_grid_values
plot_grid_values(my_grid, mu_cond_iso, S_y, L_y)
plot_grid_values(my_square_grid, mu_cond_iso_sq, S_y, L_y)
"""
