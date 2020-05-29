""" Plot physical variable, i.e. Temperature and Salinity.
The only change compared to the plotting script is in the plot titles.

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
        plt.xticks([0.2, 0.4, 0.6, 0.8])
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
        locs = S_y[L_y == 0].numpy()
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

def plot_2D_triangular_grid(grid_coords, grid_vals, S_y=None, L_y=None):
    # Special case if only one, since do not need grid of plots.
    if len(grid_vals.shape) <= 1:
        plt.title(r"$Z^1$")
        """
        im = plt.scatter(
                grid_coords[:, 0].numpy(), grid_coords[:, 1].numpy(),
                c=grid_vals.numpy(),
                cmap=cmap)
        """
        im = plt.tricontourf(
                grid_coords[:, 0].numpy(), grid_coords[:, 1].numpy(),
                grid_vals.numpy(),
                cmap=cmap)
        # Add the location of the measurement points on top.
        if S_y is not None and L_y is not None:
            locs = S_y[L_y == 0].numpy()
            plt.scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color="red")

        plt.colorbar(im)
        plt.xticks([0.2, 0.4, 0.6, 0.8])
        plt.yticks([0.2, 0.4, 0.6, 0.8])
        plt.show()
        return

def plot_grid_values(grid, vals, S_y=None, L_y=None):
    """ Plot values defined on a grid. Values can me multidimensional
    responses. In this case, one plot per response will be produced.
    One can also provide a generalized location vector, those will be added as
    points on top of the corresponding plot (can be used to plot observation
    locations in kriging for example).

    Parameters
    ----------
    grid: IrregularGrid
        Grid on which the values are defined.
    vals: (grid.n_points, p)
        Values to plot, defined at the grid nodes. The number of response p can
        be arbitrary.
    S_y: (n, grid.dim) Tensor (optional)
        Vector of spatial locations to plot as points.
    L_y: (n) Tensor (optional)
        Vector of response indices. Will specify on which plot the points will
        be added.

    """
    # Special case if only one, since do not need grid of plots.
    if len(vals.shape) <= 1:
        # Interpolate to regular grid
        reshaped_vals = grid.interpolate_to_image(vals)

        plt.title(r"$Z^1$")
        im = plt.imshow(
                reshaped_vals[:, :].numpy(),
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap)

        if (S_y is not None) and (L_y is not None):
            # Add the location of the measurement points on top.
            locs = S_y[L_y == 0].numpy()
            plt.scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color="red")

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plot_grid[i].set_ylim([0, 1])
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=6)
        ptl.xticks([0.2, 0.4, 0.6, 0.8])
        plt.yticks([0.2, 0.4, 0.6, 0.8])
        plt.show()
        return

    else: 
        n_out = vals.shape[-1]

        # Dimensions of the plotting area.
        n_col = int(np.ceil(np.sqrt(n_out)))
        n_row = int(n_out/n_col)
    
        # Generate the plot array
        fig = plt.figure()
        plot_grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_col),
                axes_pad=0.45, share_all=True,
                cbar_location="right", cbar_mode="each", cbar_size="7%", cbar_pad=0.15,)
    
        for i in range(n_out):
            reshaped_vals = grid.interpolate_to_image(vals[:, i])
            if i == 0:
                plot_grid[i].set_title(r"Temperature [$^\circ$C]")
            if i == 1:
                plot_grid[i].set_title(r"Salinity [g/kg]")
            im = plot_grid[i].imshow(
                    reshaped_vals[:, :].numpy(),
                    origin="lower",
                    extent=[0,1,0,1],
                    cmap=cmap)
            cax = plot_grid.cbar_axes[i]
            cbar = cax.colorbar(im)
            cbar.ax.tick_params(labelsize=8)
    
            if (S_y is not None) and (L_y is not None):
                # Add the location of the measurement points on top.
                locs = S_y[L_y == i].numpy()
                plot_grid[i].scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color="red")

            # Restore plot borders, which might be deformed by the scatter.
            plot_grid[i].set_xlim([0, 1])
            plot_grid[i].set_ylim([0, 1])
    
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

def plot_grid_probas(grid, probas, points=None, title=None):
    """ Plots excursion probability.

    Parameters
    ----------
    grid: IrregularGrid
        Grid on which the values are defined.
    probas: (grid.n_points)
        Probalilities to plot, defined at the grid nodes.
    points: (N, d) Tensor
        Discrete points to overlay on top of the plot. Useful to plot positions
        where data has been collected.
    title: string

    """
    if title is None:
        title = r"$Excursion Probability$"
    plt.title(title)

    # Interpolate to regular grid
    reshaped_probas = grid.interpolate_to_image(probas)

    im = plt.imshow(
                reshaped_probas[:, :].numpy(),
                vmin=0.0, vmax=1.0,
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap_proba)
    plt.colorbar(im)

    if points is not None:
        # Add the location of the measurement points on top.
        plt.scatter(points.numpy()[:, 1], points.numpy()[:, 0], marker="x",
                s=1.8, color="lightgreen")

    # plt.toggle_label(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    plt.yticks([0.2, 0.4, 0.6, 0.8])
    plt.show()
    return
