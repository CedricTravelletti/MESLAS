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
from meslas.plotting import plot_grid_values, plot_grid_probas, plot_grid_values_ax

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


def plot(sensor, ebv_1, ebv_2, ebv_full, excursion_ground_truth, output_filename=None):
    # Generate the plot array.
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.2)

    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    # Normalize EBV color range.
    norm = Normalize(vmin=0.0, vmax=0.005, clip=False)
    # 1) Get the real excursion set and plot it.
    plot_grid_values_ax(fig, ax1, "Excursion set", sensor.grid,
            excursion_ground_truth,
            S_y = sensor.grid[sensor.current_node_ind], cmap="excu",
            disable_cbar=True)
    plot_grid_values_ax(fig, ax2, "Temperature", sensor.grid,
            ebv_1, cmap="proba", norm=norm,
            cbar_format=OOMFormatter(-2, mathText=False))
    plot_grid_values_ax(fig, ax3, "Salinity", sensor.grid,
            ebv_2, cmap="proba", norm=norm,
            cbar_format=OOMFormatter(-2, mathText=False))
    plot_grid_values_ax(fig, ax4, "Both", sensor.grid,
            ebv_full, cmap="proba", norm=norm,
            cbar_format=OOMFormatter(-2, mathText=False))

    # Disable yticks for all but first.
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])

    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])

    ax1.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax3.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax4.set_xticks([0.2, 0.4, 0.6, 0.8])

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
lower = torch.tensor([2.3, 22.0]).float()

# Get the real excursion set and plot it.
excursion_ground_truth = (sample.isotopic > lower).float()
plot_grid_values(my_grid, excursion_ground_truth.sum(dim=1), cmap="proba")

# Plot the prior excursion probability.
excu_probas = my_sensor.compute_exursion_prob(lower)
plot_grid_probas(my_grid, excu_probas)
print(my_sensor.grf.mean_vec.isotopic.shape)

# Start from lower middle corner.
my_sensor.set_location([0.0, 0.5])

# First initialize the myopic strategy.
my_sensor.neighbors_eibv, my_sensor.neighbors_inds = my_sensor.get_neighbors_isotopic_eibv(
        noise_std, lower)

# Run the myopic strategy for a while
n_steps = 4
for i in range(n_steps):
    my_sensor.run_myopic_stragegy(n_steps=1, data_feed=data_feed, lower=lower,
            noise_std=noise_std)

"""
# Plot progress.
plot_myopic_with_neighbors(
        my_sensor, lower, excursion_ground_truth)
"""
# Plot progress.

# Get the next point.
next_point_ind, _ = my_sensor.choose_next_point_myopic(noise_std, lower)
# next_point_ind = next_point_ind.reshape(1,1)

next_point_ind = next_point_ind.reshape(1)

# Get the current Bernoulli variance.
p = my_sensor.compute_exursion_prob(lower)
current_bernoulli_variance = p * (1 - p)

# Get EBVs for the next points.
# First component.
ebv_1 = my_sensor.grf.ebv(next_point_ind, torch.tensor([0]).long(), lower, noise_std=noise_std)

# Second component.
ebv_2 = my_sensor.grf.ebv(next_point_ind, torch.tensor([1]).long(), lower, noise_std=noise_std)

# Both.
ebv_full = my_sensor.grf.ebv(next_point_ind.repeat(2,1).reshape(2), torch.tensor([0, 1]).long(),
        lower, noise_std=noise_std)

plot(my_sensor, current_bernoulli_variance - ebv_1,
        current_bernoulli_variance - ebv_2,
        current_bernoulli_variance - ebv_full, excursion_ground_truth.sum(1), output_filename=None)
