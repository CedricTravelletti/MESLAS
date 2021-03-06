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
my_square_grid = SquareGrid(50, 2)
print(my_grid.n_points)

# Observe some data.
S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0],
        [0.6, 0.5]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0])
y = torch.tensor(11*[-6]).float()

mu_cond, K_cond = myGRF.krig_isotopic(
        my_grid.points, S_y, L_y, y,
        noise_std=0.05,
        compute_post_cov=True)

mu_cond_sq, K_cond_sq = myGRF.krig_isotopic(
        my_square_grid.points, S_y, L_y, y,
        noise_std=0.05,
        compute_post_cov=True)

# Plot.
from meslas.plotting import plot_grid_values, plot_grid_probas
plot_grid_values(my_grid, mu_cond, S_y, L_y)
plot_grid_values(my_square_grid, mu_cond_sq, S_y, L_y)

# Try the variance part.
mu_cond, var_cond = myGRF.krig_isotopic(
        my_grid.points, S_y, L_y, y,
        noise_std=0.05,
        compute_post_var=True)

plot_grid_values(my_grid, var_cond, S_y, L_y, cmap="proba")


variance_reduction = myGRF.variance_reduction_isotopic(
        my_grid.points, S_y, L_y,
        noise_std=0.05)

plot_grid_values(my_grid, variance_reduction, S_y, L_y, cmap="proba")

K_cond_diag = torch.diagonal(K_cond.isotopic, dim1=0, dim2=1).T
lower = torch.tensor([-1.0, -1.0]).double()

coverage = coverage_fct_fixed_location(mu_cond.isotopic, K_cond_diag, lower, upper=None)
plot_grid_probas(my_grid, coverage, title="Excursion Probability")
