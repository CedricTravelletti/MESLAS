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
# CMAP_RADAR = ListedColormap(sns.color_palette("inferno_r", 30))
CMAP_RADAR = ListedColormap(sns.cubehelix_palette(n_colors=500))


def plot_myopic_radar(sensor, lower, excursion_ground_truth, bv_reduction, output_filename=None):

    # 1) Get the real excursion set and plot it.
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111)
    plot_grid_values_ax(fig, ax1, sensor.grid,
            excursion_ground_truth.sum(dim=1),
            cmap="excu",
            disable_cbar=True)

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

    ax1.legend(handles=legend_elements, loc='upper right')

    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_xlim([0.0, 0.98])

    plt.savefig("ebv_myopic_excu.png", bbox_inches='tight', pad_inches=0, dpi=400)

    # 2) Plot Bernoulli variance reduction.
    fig = plt.figure(figsize=(3, 3))
    ax2 = fig.add_subplot(111)

    plot_grid_values_ax(fig, ax2, sensor.grid,
            bv_reduction,
            S_y=sensor.grid.points[sensor.visited_node_inds],
            cmap="proba", vmin=0, vmax=1)
    ax2.scatter(sensor.grid[sensor.current_node_ind][0, 1],
            sensor.grid[sensor.current_node_ind][0, 0],
            marker="^", s=8.0, color="cyan")

    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_xlim([0.0, 0.98])

    plt.savefig("ebv_myopic_ebv.png", bbox_inches='tight', pad_inches=0, dpi=400)
    
    # 3) Plot the EIBVS of the neighbors in a radar.
    # Get the polar coordinates of the neighbors.
    fig = plt.figure(figsize=(3, 3))
    ax3 = fig.add_subplot(111, projection="polar")

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

    plt.savefig("ebv_myopic_radar.png", bbox_inches='tight', pad_inches=0, dpi=400)

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
