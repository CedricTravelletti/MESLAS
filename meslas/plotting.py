""" Plotting capabilities.

"""
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from meslas.vectors import GeneralizedVector

# Trygve's plot parameters.
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 27, 'font.style': 'oblique',
        # 'xtick.labelsize': 'x-small',
        'axes.labelsize': 'x-small',
        'axes.titlesize':'x-small',
        'xtick.major.pad': '1',
        'xtick.minor.pad': '1',
        'ytick.major.pad': '1'}
plt.rcParams.update(plot_params)

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

# Color palettes
from matplotlib.colors import ListedColormap
CMAP_PROBA = ListedColormap(sns.cubehelix_palette(n_colors=500))
# CMAP_PROBA = ListedColormap(sns.color_palette("RdBu_r", 30))
# CMAP_EXCU = ListedColormap(sns.color_palette("Set1", 3))
# CMAP_EXCU = ListedColormap(sns.color_palette("RdGy", 3))
CMAP_EXCU = ListedColormap(sns.color_palette("Reds", 300))

CMAP_RADAR = ListedColormap(sns.color_palette("cool", 30))

# CMAP = ListedColormap(sns.color_palette("BrBG", 100))
CMAP = ListedColormap(sns.color_palette("GnBu_d", 100))


def plot_grid_values(grid, vals, S_y=None, L_y=None, cmap=None):
    """ Plot values defined on a grid. Values can me multidimensional
    responses. In this case, one plot per response will be produced.
    One can also provide a generalized location vector, those will be added as
    points on top of the corresponding plot (can be used to plot observation
    locations in kriging for example).

    Parameters
    ----------
    grid: IrregularGrid
        Grid on which the values are defined.
    vals: (grid.n_points, p) or GeneralizedVector
        Values to plot, defined at the grid nodes. The number of response p can
        be arbitrary.
        If GeneralizedVector, then the reshaping is handled automatically.
    S_y: (n, grid.dim) Tensor (optional)
        Vector of spatial locations to plot as points.
    L_y: (n) Tensor (optional)
        Vector of response indices. Will specify on which plot the points will
        be added.
    cmap: string
        If set to "proba", will use red-blue.

    """
    if isinstance(vals, GeneralizedVector):
        vals = vals.isotopic
    if cmap == "proba":
        cmap = CMAP_PROBA
        color="lime"
    elif cmap == "excu":
        cmap = CMAP_EXCU
        color="lime"
        # When we plot excursions, we want to increase the range so that it
        # looks better with the colormap. Thats why we square the values.
        vals = vals**(2.2)
    else:
        cmap = CMAP
        color = "red"

    # Special case if only one, since do not need grid of plots.
    if len(vals.shape) <= 1:
        # Interpolate to regular grid
        reshaped_vals = grid.interpolate_to_image(vals)

        # Generate the plot array
        fig = plt.figure()
        plt.title(r"$Z^1$")
        im = plt.imshow(
                reshaped_vals[:, :].numpy(),
                origin="lower",
                extent=[0,1,0,1],
                cmap=cmap)

        if (S_y is not None) and (L_y is not None):
            # Add the location of the measurement points on top.
            locs = S_y[L_y == 0].numpy()
            plt.scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color=color)

        # If only one dim, then can also allow L_y to be unspecified.
        elif (S_y is not None):
            # Add the location of the measurement points on top.
            locs = S_y.numpy()
            plt.scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color=color)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=6)
        plt.xticks([0.2, 0.4, 0.6, 0.8])
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
            plot_grid[i].set_title(r"$Z^" + str(i+1) + "$")
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
                plot_grid[i].scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5,
                color=color)

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

def plot_grid_probas(grid, probas, points=None, title=None,
        output_filename=None):
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
    plt.figure()
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
                cmap=CMAP_PROBA)
    plt.colorbar(im)

    if points is not None:
        # Add the location of the measurement points on top.
        plt.scatter(points.numpy()[:, 1], points.numpy()[:, 0], marker="x",
                s=2.5, color="lightgreen")

    # plt.toggle_label(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    plt.yticks([0.2, 0.4, 0.6, 0.8])
    
    # If output filename provided, then save, else show.
    if output_filename is not None:
        plt.savefig(output_filename)
    else: plt.show()
    return

def plot_grid_values_ax(fig, axis, grid, vals,
        title=None,
        S_y=None, cmap=None,
        vmin=None, vmax=None, norm=None,
        disable_cbar=False, cbar_format=None, cbar_label=None):
    """ Plots an image corresponding to values at points of a grid.
    This function takes an axis as input, so can be used to produce subplots.

    Parameters
    ----------
    axis: matplotlib.axis
        The axis instance on which to draw.
    grid: meslas.Geometry.Grid
        Grid on which to plot.
    vals: (n_points) Tensor
        The values at the grid nodes.
    title: string
        Title for the plot.
    S_y: (n_points, n_dim) Tensor, optional
        Allows to add points on the plot.
    cmap: string
        Either "proba" of "excu". The second one is used to plot excursion sets
        (i.e. discrete).
    vmin: float
        Minimal value for the color range.
        Defaults to minimal data value.
    vmax: float
        Maximal value for the color range.
        Defaults to maximal data value.
    disable_cbar: bool
        If set to True, disables the colorbar next to the plot.
    norm: Normalizer
        Can be used to normalize colors once and for all.
    cbar_format: matplotlib.ticker.ScalarFormatter
        Allows one to format colorbar ticks in scientific notation.
    cbar_label: string, optional
        Text to display near the colorbar.

    """
    if isinstance(vals, GeneralizedVector):
        vals = vals.isotopic
    if cmap == "proba":
        cmap = CMAP_PROBA
        color="lime"
    elif cmap == "excu":
        cmap = CMAP_EXCU
        color="lime"
        # When we plot excursions, we want to increase the range so that it
        # looks better with the colormap. Thats why we square the values.
        vals = vals**(2.2)
    else:
        cmap = CMAP
        color = "red"
    reshaped_vals = grid.interpolate_to_image(vals[:])
    axis.set_title(title)
    im = axis.imshow(
            reshaped_vals[:, :].numpy(),
            origin="lower",
            extent=[0,1,0,1],
            vmin=vmin, vmax=vmax,
            norm=norm,
            cmap=cmap)

    # Colorbar
    # Even if disabled, add the divider so all plots have the same size.
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_visible(False)
    if not disable_cbar:
        cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                format=cbar_format, label=cbar_label)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.yaxis.get_offset_text().set(size=8)
        cax.set_visible(True)

    if (S_y is not None):
        # Add the location of the measurement points on top.
        locs = S_y.numpy()
        axis.scatter(locs[:, 1], locs[:, 0], marker="^", s=6.5, color=color)

    # Restore plot borders, which might be deformed by the scatter.
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1])

    axis.tick_params(axis='both', which='major', labelsize=9)
    axis.tick_params(axis='both', which='minor', labelsize=9)

    return
