""" Plotting functions for animation of the myopic strategy.

"""
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from meslas.plotting import plot_grid_values_ax
from meslas.vectors import GeneralizedVector

# Colormap for the radar.
from matplotlib.colors import ListedColormap
CMAP_RADAR = ListedColormap(sns.color_palette("inferno_r", 30))


def plot_myopic_radar(sensor, lower, excursion_ground_truth, output_filename=None):
    # Generate the plot array.
    fig = plt.figure(figsize=(15, 10))
    widths = [3, 3, 1]
    heights = [3, 3, 3]
    gs = fig.add_gridspec(
            ncols=3, nrows=3, width_ratios=widths,
            height_ratios=heights)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[0, 2], projection="polar")


    # 1) Get the real excursion set and plot it.
    plot_grid_values_ax(fig, ax1, "Regions of interest", sensor.grid,
            excursion_ground_truth.sum(dim=1),
            cmap="excu",
            disable_cbar=True)

    # 2) Plot coverage function.
    excu_probas = sensor.compute_exursion_prob(lower)
    plot_grid_values_ax(fig, ax2, "Excursion probability.", sensor.grid,
            excu_probas,
            S_y=sensor.grid.points[sensor.visited_node_inds],
            cmap="proba", vmin=0, vmax=1)
    
    # 3) Plot the EIBVS of the neighbors in a radar.
    # Get the polar coordinates of the neighbors.
    r, phi = to_polar(sensor.location, sensor.grid[sensor.neighbors_inds])

    # Replicate radius, so we have a line instead of a single point.
    phiS = np.repeat(phi.numpy(), 100)
    rS = torch.tensor(np.linspace(0.1, 1.3, 100))
    rS = rS.repeat(phi.shape[0])
    cs = np.repeat(sensor.neighbors_eibv.numpy(), 100)

    im = ax3.scatter(phiS, rS, c=cs, s=50, alpha=0.02, cmap=CMAP_RADAR)

    # Plot the best direction with a thicker line.
    min_ind = np.argmin(sensor.neighbors_eibv.numpy())
    phi_min = phi.numpy()[min_ind]
    phiS_min = np.repeat(phi_min, 100)
    rS_min = torch.tensor(np.linspace(0.1, 1.3, 100))
    cs_min = np.repeat(sensor.neighbors_eibv.numpy()[min_ind], 100)

    im = ax3.scatter(phiS_min, rS_min, c=cs_min, s=250, alpha=0.9, cmap=CMAP_RADAR)

    # Add a big black one at the middle.
    ax3.scatter([0.0], [0.0], c=[0.0], s=1400, cmap="gist_gray")
    ax3.set_yticks([])

    if output_filename is not None:
        plt.savefig(output_filename)
        plt.close(fig)
    else: plt.show()

    return

def to_polar(center_coords, neighbors_coords):
    """ Converts neighbors coordinates to polar, wrt the current location.

    """
    return cart2pol((neighbors_coords - center_coords)[:, 0],
            (neighbors_coords - center_coords)[:, 1])

# WARNING: The argument are switched on purpose.
def cart2pol(y, x):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
