""" Plotting capabilities.

"""
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid


sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

# Color palettes
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
from matplotlib.colors import ListedColormap
cmap_proba = ListedColormap(sns.color_palette("RdBu_r", 30))
cmap = ListedColormap(sns.color_palette("BrBG", 100))



def plot_2d_slice(sliced_sample, title=None, cmin=None, cmax=None):
    """ Plots 2 dimensional slice.

    For each of the n_out responses, plot a two dimensional image.

    Parameters
    ----------
    sliced_sample: (n1, n2, n_out) Tensor
        Two dimensional slice of a sample or prediction.

    """

    # Number of responses.
    # Special case if only one, since do not need grid of plots.
    if len(sliced_sample.shape) <= 2:
        if title is None:
            title = r"$Z^1$"
        plt.title(title)
        im = plt.imshow(
                sliced_sample[:, :].numpy(),
                vmin=cmin, vmax=cmax,
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap)
        plt.colorbar(im)
        # plt.toggle_label(True)
        ptl.xticks([0.2, 0.4, 0.6, 0.8])
        plt.yticks([0.2, 0.4, 0.6, 0.8])
        plt.show()
        return

    n_out = sliced_sample.shape[-1]
    # Dimensions of the plotting area.
    n_col = int(np.ceil(np.sqrt(n_out)))
    n_row = int(n_out/n_col)

    fig = plt.figure()
    plot_grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_col),
            axes_pad=0.45, share_all=True,
            cbar_location="right", cbar_mode="each", cbar_size="7%", cbar_pad=0.15,)

    for i in range(n_out):
        # fig.axes[i].set_title(r"$Z^" + str(i+1) + "$")
        plot_grid[i].set_title(r"$Z^" + str(i+1) + "$")
        im = plot_grid[i].imshow(
                sliced_sample[:, :, i].numpy(),
                # vmin=0.8*sample_min, vmax=0.8*sample_max,
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap)
        cax = plot_grid.cbar_axes[i]
        cax.colorbar(im)
    # Hide the unused plots.
    for i in range(n_out, len(plot_grid)):
        plot_grid[i].axis("off")

    plot_grid.axes_llc.set_xticks([0.2, 0.4, 0.6, 0.8])
    plot_grid.axes_llc.set_yticks([0.2, 0.4, 0.6, 0.8])

    plt.show()

def plot_krig_slice(sliced_sample, S_y, L_y):
    """ TEMP: plot kriging mean with observations.
    
    """
    sample_min = torch.min(sliced_sample).item()
    sample_max = torch.max(sliced_sample).item()

    # Number of responses.
    # Special case if only one, since do not need grid of plots.
    if len(sliced_sample.shape) <= 2:
        plt.title(r"$Z^1$")
        im = plt.imshow(
                sliced_sample[:, :].numpy(),
                # vmin=0.8*sample_min, vmax=0.8*sample_max,
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap)
        # Add the location of the measurement points on top.
        locs = S_y[L_y == i].numpy()
        plt.scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color="red")
        plt.colorbar(im)
        ptl.xticks([0.2, 0.4, 0.6, 0.8])
        plt.yticks([0.2, 0.4, 0.6, 0.8])
        plt.show()
        return

    n_out = sliced_sample.shape[-1]

    # Dimensions of the plotting area.
    n_col = int(np.ceil(np.sqrt(n_out)))
    
    n_row = int(n_out/n_col)

    fig = plt.figure()
    plot_grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_col),
            axes_pad=0.45, share_all=True,
            cbar_location="right", cbar_mode="each", cbar_size="7%", cbar_pad=0.15,)

    for i in range(n_out):
        # fig.axes[i].set_title(r"$Z^" + str(i+1) + "$")
        plot_grid[i].set_title(r"$Z^" + str(i+1) + "$")
        im = plot_grid[i].imshow(
                sliced_sample[:, :, i].numpy(),
                # vmin=0.8*sample_min, vmax=0.8*sample_max,
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap)
        cax = plot_grid.cbar_axes[i]
        cax.colorbar(im)

        # Add the location of the measurement points on top.
        locs = S_y[L_y == i].numpy()
        plot_grid[i].scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color="red")

    # Hide the unused plots.
    for i in range(n_out, len(plot_grid)):
        plot_grid[i].axis("off")

    # Colorbar
    ax = plot_grid[i]
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    plot_grid.axes_llc.set_xticks([0.2, 0.4, 0.6, 0.8])
    plot_grid.axes_llc.set_yticks([0.2, 0.4, 0.6, 0.8])

    plt.show()

def plot_proba(coverage_image, title=None):
    """ Plots excursion probability.

    Parameters
    ----------
    coverage_image: (n1, n2) Tensor
    title: string

    """
    if title is None:
        title = r"$Excursion Probability$"
    plt.title(title)
    im = plt.imshow(
                coverage_image[:, :].numpy(),
                vmin=0.0, vmax=1.0,
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap_proba)
    plt.colorbar(im)
    # plt.toggle_label(True)
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    plt.yticks([0.2, 0.4, 0.6, 0.8])
    plt.show()
    return
