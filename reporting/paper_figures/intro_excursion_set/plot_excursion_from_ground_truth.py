""" Generate plot of excursion set (Figure 2), but loads ground truth instead
of generating it.

"""
import numpy as np
import torch
from meslas.means import LinearMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid
from meslas.random_fields import GRF, DiscreteGRF
from meslas.excursion import coverage_fct_fixed_location
from meslas.vectors import GeneralizedVector
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
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Fix spacings between plots.
    plt.subplots_adjust(wspace=0.16)

    # 1) Get the real excursion set and plot it.
    plot_grid_values_ax(fig, ax1, "Temperature", sensor.grid,
            sample.isotopic[:, 0],
            cbar_label=r"$[^{\circ}\mathrm{C}]$")
    plot_grid_values_ax(fig, ax2, "Salinity", sensor.grid,
            sample.isotopic[:, 1],
            cbar_label=r"$[g/kg]$")
    plot_grid_values_ax(fig, ax3, "Regions of interest", sensor.grid,
            excursion_ground_truth,
            cmap="excu",
            disable_cbar=True)

    # Disable yticks for all but first.
    ax2.set_yticks([])
    ax3.set_yticks([])

    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])

    ax1.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax3.set_xticks([0.2, 0.4, 0.6, 0.8])

    # Cut the part that doesn't get interpolated.
    ax1.set_xlim([0.0, 0.98])
    ax2.set_xlim([0.0, 0.98])
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


    if output_filename is not None:
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.close(fig)
    else:
        plt.savefig("out.png", bbox_inches='tight', pad_inches=0, dpi=400)
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
my_grid = TriangularGrid(61)
print("Working on an equilateral triangular grid with {} nodes.".format(my_grid.n_points))

# Discretize the GRF on a grid and be done with it.
# From now on we only consider locatoins on the grid.
my_discrete_grf = DiscreteGRF.from_model(myGRF, my_grid)

# ------------------------------------------------------
# Load ground truth from file.
sample = torch.from_numpy(np.load("./ground_truth.npy")).float()
sample = GeneralizedVector.from_isotopic(sample)

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

# Plot the prior excursion probability and realization.
excu_probas = my_sensor.compute_exursion_prob(lower)
plot(my_sensor, lower, excursion_ground_truth.sum(1))
