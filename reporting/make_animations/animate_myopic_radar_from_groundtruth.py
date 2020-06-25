""" Run the myopic startegy and plot progress each time.

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
from plotting_functions import plot_myopic_radar

from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky


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
my_grid = TriangularGrid(51)
print("Working on an equilateral triangular grid with {} nodes.".format(my_grid.n_points))

# Discretize the GRF on a grid and be done with it.
# From now on we only consider locatoins on the grid.
my_discrete_grf = DiscreteGRF.from_model(myGRF, my_grid)

# ------------------------------------------------------
# Load sample and plot
# ------------------------------------------------------
sample = torch.from_numpy(np.load("./ground_truth.npy")).float()
plot_grid_values(my_grid, sample)

# From now on, we will consider the drawn sample as ground truth.
# ---------------------------------------------------------------
ground_truth = sample


# -------------------------------------------
# TODO: To make things more reproducible,
# this should be implemented in a module. 
# Manual specification is not nice.
# -------------------------------------------
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
# -------------------------------------------
# -------------------------------------------

my_sensor = DiscreteSensor(my_discrete_grf)

# Excursion threshold.
lower = torch.tensor([2.3, 22.0]).float()

# Get the real excursion set and plot it.
excursion_ground_truth = (sample.isotopic > lower).float()
plot_grid_values(my_grid, excursion_ground_truth.sum(dim=1), cmap="excu")


# Start from lower middle corner.
my_sensor.set_location([0.0, 0.5])

# First initialize the myopic strategy.
my_sensor.neighbors_eibv, my_sensor.neighbors_inds = my_sensor.get_neighbors_isotopic_eibv(
        noise_std, lower)

# Run the myopic strategy one step at a time.
n_steps = 600
for i in range(n_steps):
    my_sensor.run_myopic_stragegy(n_steps=1, data_feed=data_feed, lower=lower,
            noise_std=noise_std)

    # Plot progress.
    plot_myopic_radar(
            my_sensor, lower, excursion_ground_truth, output_filename="gif{}.png".format(i))
