import numpy as np
import torch
from meslas.means import LinearMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid
from meslas.random_fields import GRF, DiscreteGRF
from meslas.excursion import coverage_fct_fixed_location
from meslas.plotting import plot_grid_values, plot_grid_probas
from meslas.sensor import DiscreteSensor
from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky
import matplotlib.pyplot as plt


# ------------------------------------------------------
# DEFINITION OF THE MODEL
# ------------------------------------------------------
# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.5, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.2, sigmas=[2.25, 2.25])
covariance = FactorCovariance(
        spatial_cov=matern_cov,
        cross_cov=cross_cov,
        n_out=n_out)

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
# Create a regular square grid in 2 dims.
my_grid = TriangularGrid(40)
print("Working on an equilateral triangular grid with {} nodes.".format(my_grid.n_points))

# Discretize the GRF on a grid and be done with it.
# From now on we only consider locatoins on the grid.
my_discrete_grf = DiscreteGRF.from_model(myGRF, my_grid)

# ------------------------------------------------------
# Sample and plot
# ------------------------------------------------------
# Sample all components at all locations.
sample = my_discrete_grf.sample()
plot_grid_values(my_grid, sample)

# From now on, we will consider the drawn sample as ground truth.
# ---------------------------------------------------------------
ground_truth = sample

# Use it to declare the data feed.
noise_std = torch.tensor([0.1, 0.1])
# Noise distribution
lower_chol = psd_safe_cholesky(torch.diag(noise_std**2))
noise_distr = MultivariateNormal(
    loc=torch.zeros(n_out),
    scale_tril=lower_chol)

def data_feed(node_ind):
    noise_realization = noise_distr.sample()
    return ground_truth[node_ind] + noise_realization

my_sensor = DiscreteSensor(my_discrete_grf)

# Excursion threshold.
lower = torch.tensor([2.3, 22.0]).float()

# Get the real excursion set and plot it.
excursion_ground_truth = (sample.isotopic > lower).float()
plot_grid_values(my_grid, excursion_ground_truth.sum(dim=1), cmap="proba")

# Plot the prior excursion probability.
excu_probas = my_sensor.compute_exursion_prob(lower)
plot_grid_probas(my_grid, excu_probas)
print(my_sensor.grf.mean_vec.isotopic.shape)

# Start from lower left corner.
my_sensor.set_location([0.0, 0.0])
my_sensor.run_myopic_stragegy(n_steps=35, data_feed=data_feed, lower=lower,
        noise_std=noise_std)

# At the end, print true excursion, with visited points overlaid.
plot_grid_values(my_grid, excursion_ground_truth.sum(dim=1),
        my_grid.points[my_sensor.visited_node_inds],
        cmap="proba")

# Also plot plug_in estimate of excursion.
plot_grid_values(my_grid, my_discrete_grf.mean_vec.isotopic > lower,
        my_grid.points[my_sensor.visited_node_inds],
        cmap="proba")

# Plot the excursion probability in the end.
excu_probas = my_sensor.compute_exursion_prob(lower)
plot_grid_probas(my_grid, excu_probas)
print(my_sensor.grf.mean_vec.isotopic.shape)
