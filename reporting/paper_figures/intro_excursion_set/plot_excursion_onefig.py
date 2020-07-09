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
from meslas.excursion import coverage_fct_fixed_location
from meslas.plotting import plot_grid_values, plot_grid_probas, plot_grid_values_ax

#from meslas.sensor import DiscreteSensor
from meslas.sensor_plotting import DiscreteSensor

from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

from meslas.vectors import GeneralizedVector


sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})


def plot(sensor, lower, excursion_ground_truth, output_filename=None):
    # Generate the plot array.
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111)


    # 1) Get the real excursion set and plot it.
    plot_grid_values_ax(fig, ax1,
            # "Temperature",
            sensor.grid,
            sample.isotopic[:, 0],
            cbar_label=r"$[^{\circ}\mathrm{C}]$")
    ax1.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_xlim([0.0, 0.98])

    plt.savefig("intro_temp.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.show()

    fig = plt.figure(figsize=(3, 3))
    ax2 = fig.add_subplot(111)
    plot_grid_values_ax(fig, ax2,
            # "Salinity",
            sensor.grid,
            sample.isotopic[:, 1],
            cbar_label=r"$[g/kg]$")
    ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_xlim([0.0, 0.98])

    plt.savefig("intro_sal.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.show()

    fig = plt.figure(figsize=(3, 3))
    ax3 = fig.add_subplot(111)
    plot_grid_values_ax(fig, ax3,
            # "Regions of interest",
            sensor.grid,
            excursion_ground_truth,
            cmap="excu",
            disable_cbar=True)
    ax3.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax3.set_xlim([0.0, 0.98])


    # Legend for the patches.
    # Get cmap.
    from matplotlib.colors import ListedColormap
    CMAP_EXCU = ListedColormap(sns.color_palette("Reds", 300))

    from matplotlib.patches import Patch

    legend_elements = [
            Patch(facecolor=CMAP_EXCU(0.0),
                         label='No excursion',
                         edgecolor="black"),
            Patch(facecolor=CMAP_EXCU(0.5),
                         label='Single excursion',
                         edgecolor="black"),
            Patch(facecolor=CMAP_EXCU(1.0),
                         label='Joint excursion',
                         edgecolor="black")]

    ax3.legend(handles=legend_elements, loc='upper right')

    plt.savefig("intro_excu.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.show()

    return


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
beta0s = np.array([6.8, 24.0])
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
my_grid = TriangularGrid(61)
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
# Save for reproducibility.
np.save("ground_truth.npy", ground_truth.numpy())

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
lower = torch.tensor([3.3, 22.0]).float()

# Get the real excursion set and plot it.
excursion_ground_truth = (sample.isotopic > lower).float()
plot_grid_values(my_grid, excursion_ground_truth.sum(dim=1), cmap="proba")

# Plot the prior excursion probability and realization.
excu_probas = my_sensor.compute_exursion_prob(lower)
plot(my_sensor, lower, excursion_ground_truth.sum(1))
