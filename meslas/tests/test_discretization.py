""" Tests the DiscreteGRF class.

"""
import numpy as np
import torch
from meslas.means import ConstantMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid, SquareGrid
from meslas.random_fields import GRF, DiscreteGRF
from meslas.excursion import coverage_fct_fixed_location
from meslas.plotting import plot_grid_values, plot_grid_probas


# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.4, sigmas=[np.sqrt(1.0), np.sqrt(5.5)])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function
mean = ConstantMean([0.0, 5.0])

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create an equilateral triangular grid in 2 dims.
# Number of respones.
my_grid = TriangularGrid(40)
my_square_grid = SquareGrid(50, 2)

my_discrete_grf = DiscreteGRF.from_model(myGRF, my_grid)

# Sample and plot.
sample = my_discrete_grf.sample()
plot_grid_values(my_grid, sample)

# Sample and plot.
sample = my_discrete_grf.sample()
plot_grid_values(my_grid, sample)

# Sample the continuous version and compare.
sample_cont = myGRF.sample_isotopic(my_grid.points)
plot_grid_values(my_grid, sample_cont)

# Observe some data.
S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0],
        [0.6, 0.5]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0])
y = torch.tensor(11*[-6]).float()

# Since we are working with a discrete GRF, we can only observa data at grid
# nodes. Hence get the ones corresponding to the measured data.
S_y_inds = my_grid.get_closest(S_y)

my_discrete_grf.update(S_y_inds, L_y, y, noise_std=0.05)
plot_grid_values(my_grid, my_discrete_grf.mean_vec, S_y, L_y)

# -----------------------------------------
# Now compare with the non-discrete version.
# -----------------------------------------
mu_cond, K_cond = myGRF.krig_isotopic(
        my_grid.points, S_y, L_y, y,
        noise_std=0.05,
        compute_post_cov=True)

plot_grid_values(my_grid, mu_cond, S_y, L_y)

# -----------------------------------------
# Try the variance part.
plot_grid_values(my_grid, my_discrete_grf.variance, S_y, L_y, cmap="proba")
