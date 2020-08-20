""" Demonstrates how to define a multivariate Gaussian Random Field,
sample a realization and plot it.

"""
import numpy as np
import torch

from meslas.means import LinearMean
from meslas.covariance.spatial_covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance

from meslas.geometry.grid import TriangularGrid
from meslas.random_fields import GRF, DiscreteGRF
from meslas.sensor_plotting import DiscreteSensor

from meslas.plotting import plot_grid_values, plot_grid_probas

from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky


# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.9, sigmas=[np.sqrt(0.25), np.sqrt(0.6)])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function, here it is a linear trend that decreases with the
# horizontal coordinate.
beta0s = np.array([5.8, 24.0])
beta1s = np.array([
        [0, -4.0],
        [0, -3.8]])
mean = LinearMean(beta0s, beta1s)

# Create the GRF.
myGRF = GRF(mean, covariance)

# ------------------------------------------------------
# DISCRETIZE EVERYTHING
# ------------------------------------------------------
# Create a regular equilateral triangular grid in 2 dims.
# The argument specified the number of cells along 1 dimension, hence total
# size of the grid is roughly the square of this number.
my_grid = TriangularGrid(21)
print("Working on an equilateral triangular grid with {} nodes.".format(my_grid.n_points))

# Discretize the GRF on a grid and be done with it.
# From now on we only consider locations on the grid.
my_discrete_grf = DiscreteGRF.from_model(myGRF, my_grid)

# ------------------------------------------------------
# Sample and plot
# ------------------------------------------------------
# Sample all components at all locations.
sample = my_discrete_grf.sample()
plot_grid_values(my_grid, sample)


# ------------------------------------------------------
# Observe some data.
# ------------------------------------------------------

# Data observations must be specified by a so-called generalized location.
# A generalized location is a couple of vectors (S, L). The first vector
# specifies WHERE the observations have been made, whereas the second vector
# indicates WHICH component was measured at that location (remember we are
# considering multivariate GRFs).
#
# Example consider S = [[0, 0], [-1, -1]] and L = [0, 8].
# Then, the generalized location (S, L) describes a set of observations that
# consists of one observation of the 0-th component of the field at the points
# (0,0) and one observation of the 8-th component of the field at (-1, -1).
# WARNING: python used 0-based indices, to 0-th component translates to first
# component in math language.

S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0],
        [0.6, 0.5]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0]).long()

# Now define which value were observed. Here we arbitrarily set all values to
# -5.
y = torch.tensor(11*[-6])

# Now integrate this data to the model (i.e. compute the conditional
# distribution of the GRF.
#
# SUBTLETY: we are here working with a discrete GRF, i.e. a GRF on a grid, that
# maintains a mean vector and covariance matrix for ALL grid points and only
# understands data that is on the grid.
# This means that this GP takes as spatial inputs integers that define the
# index of the given point in the grid array.
#
# Hence, in this setting, coordinates should be projected back to the grid in
# order for the GP to understand them.
S_y_inds = my_grid.get_closest(S_y) # Get grid index of closest nodes.

# Note that working on a grid an never stepping outside of it is the usual
# setting in adaptive designs.
# One can work with the non-discretized GP class otherwise.

#
noise_std = torch.tensor([0.1, 0.1])

# Condition the model.
my_discrete_grf.update(S_y_inds, L_y, y, noise_std=noise_std)

# Plot conditional mean.
plot_grid_values(my_grid, my_discrete_grf.mean_vec)
