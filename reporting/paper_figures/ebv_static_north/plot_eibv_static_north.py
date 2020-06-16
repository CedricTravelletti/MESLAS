""" Plots the EIBV at each point if we were to measure S, T, or both
simultaneously.

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
from meslas.plotting import plot_grid_values, plot_grid_probas

#from meslas.sensor import DiscreteSensor
from meslas.sensor_plotting import DiscreteSensor

from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky



import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from matplotlib.colors import Normalize

from meslas.vectors import GeneralizedVector


sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

# Color palettes
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
from matplotlib.colors import ListedColormap
CMAP_PROBA = ListedColormap(sns.color_palette("RdBu_r", 400))
CMAP = ListedColormap(sns.color_palette("BrBG", 100))

CMAP_EXCU = ListedColormap(sns.color_palette("RdBu_r", 3))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 27})
plt.rcParams.update({'font.style': 'oblique'}) 

# Scientific notation in colorbars
import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format


def plot(sensor, ebv_north, ebv_east,
        current_proba,
        S_inds_north, S_inds_east,
        output_filename=None):
    # Generate the plot array.
    fig = plt.figure(figsize=(15, 10))
    # gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    """
    ax3 = fig.add_subplot(223, projection="polar")
    """

    # Set the ticks for all.
    """
    plot_grid.axes_llc.set_xticks([0.2, 0.4, 0.6, 0.8])
    plot_grid.axes_llc.set_yticks([0.2, 0.4, 0.6, 0.8])
    """

    # Normalize EBV color range.
    norm = Normalize(vmin=0.0, vmax=0.005, clip=False)
    # 1) Get the real excursion set and plot it.
    _plot_helper2(fig, ax1, "Excursion probability", sensor.grid,
            current_proba,
            cmap="proba")
    _plot_helper2(fig, ax2, "Static north", sensor.grid,
            ebv_north, cmap="proba",
            S_y = sensor.grid[S_inds_north],
            cbar_format=OOMFormatter(-2, mathText=False))
    _plot_helper2(fig, ax3, "Static east", sensor.grid,
            ebv_east, cmap="proba",
            S_y = sensor.grid[S_inds_east],
            cbar_format=OOMFormatter(-2, mathText=False))

    # 2) Plot excursion.

    if output_filename is not None:
        plt.savefig(output_filename)
        plt.close(fig)
    else: plt.show()

    return

def _plot_helper2(fig, axis, title, grid, vals, S_y=None, L_y=None, cmap=None,
        norm=None,
        disable_cbar=False,
        cbar_format=None):
    """ Helper for the neighbors EIBV version of the plotting function.

    Parameters
    ----------
    axis

    """
    if isinstance(vals, GeneralizedVector):
        vals = vals.isotopic
    if cmap == "proba":
        cmap = CMAP_PROBA
        color="lime"
    elif cmap == "excu":
        cmap = CMAP_EXCU
        color="lime"
    else:
        cmap = CMAP
        color = "red"
    reshaped_vals = grid.interpolate_to_image(vals[:])
    axis.set_title(title)
    im = axis.imshow(
            reshaped_vals[:, :].numpy(),
            origin="lower",
            extent=[0,1,0,1],
            norm=norm,
            cmap=cmap)

    if not disable_cbar:
        # Colorbar.
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                format=cbar_format)
        # cax = plot_grid.cbar_axes[i]
        cbar.ax.tick_params(labelsize=7)

    if (S_y is not None):
        # Add the location of the measurement points on top.
        locs = S_y.numpy()
        axis.scatter(locs[:, 1], locs[:, 0], marker="^", s=9.5, color=color)

    # Restore plot borders, which might be deformed by the scatter.
    axis.set_xlim([0, 0.95])
    axis.set_ylim([0, 0.95])
    axis.set_xticks([])
    axis.set_yticks([])
    axis.yaxis.set_label_position("left")
    
    """
    # Colorbar
    ax = plot_grid[i]
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    """

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
my_grid = TriangularGrid(31)
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


# -------------------------------------------------------------
# Compare two designs: Static North and Static East (a priori).
# -------------------------------------------------------------

# First get the static north line.
y_north = torch.linspace(0.5, 0.5, 100)
x_north = torch.linspace(0, 1, 100)
design_inds_north = torch.unique(my_grid.get_closest(torch.stack([x_north, y_north], dim=1)))
S_inds_north, L_north = my_grid.get_isotopic_generalized_location_inds(
        my_grid[design_inds_north], p=2)

y_east = torch.linspace(0.0, 1.0, 100)
x_east = torch.linspace(0.515, 0.515, 100)
design_inds_east = torch.unique(my_grid.get_closest(torch.stack([x_east, y_east], dim=1)))
S_inds_east, L_east = my_grid.get_isotopic_generalized_location_inds(
        my_grid[design_inds_east], p=2)

# Get the current Bernoulli variance.
p = my_sensor.compute_exursion_prob(lower)
current_bernoulli_variance = p * (1 - p)

ebv_north = my_sensor.grf.ebv(
        S_inds_north, L_north, lower, noise_std=noise_std)

# Second component.
ebv_east = my_sensor.grf.ebv(
        S_inds_east, L_east, lower, noise_std=noise_std)

plot(my_sensor, current_bernoulli_variance - ebv_north,
        current_bernoulli_variance - ebv_east,
        p,
        S_inds_north, S_inds_east, output_filename=None)
