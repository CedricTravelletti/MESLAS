""" Plotting functions for animation of the myopic strategy.

"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from meslas.vectors import GeneralizedVector


sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

# Color palettes
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
from matplotlib.colors import ListedColormap
CMAP_PROBA = ListedColormap(sns.color_palette("RdBu_r", 30))
CMAP = ListedColormap(sns.color_palette("BrBG", 100))


def plot_myopic(sensor, lower, excursion_ground_truth, output_filename=None):
    # Generate the plot array.
    n_row = 1
    n_col = 2
    fig = plt.figure()
    plot_grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_col),
            axes_pad=0.45, share_all=True,
            cbar_location="right", cbar_mode="each", cbar_size="7%", cbar_pad=0.15,)

    # Set the ticks for all.
    plot_grid.axes_llc.set_xticks([0.2, 0.4, 0.6, 0.8])
    plot_grid.axes_llc.set_yticks([0.2, 0.4, 0.6, 0.8])

    # 1) Get the real excursion set and plot it.
    _plot_helper(plot_grid, 0, "Excursion set: Ground truth", sensor.grid,
            excursion_ground_truth.sum(dim=1),
            cmap="proba")

    """
    # 2) Plot plug_in estimate of excursion.
    _plot_helper(plot_grid, 1, "Excursion set: Plug-in estimate", sensor.grid,
            (sensor.grf.mean_vec.isotopic > lower).sum(1),
            S_y=sensor.grid.points[sensor.visited_node_inds],
            cmap="proba")
    """

    # 3) Plot coverage function.
    excu_probas = sensor.compute_exursion_prob(lower)
    _plot_helper(plot_grid, 1, "Excursion probability.", sensor.grid,
            excu_probas,
            S_y=sensor.grid.points[sensor.visited_node_inds],
            cmap="proba")

    """
    # 4) Plot pointwise variance.
    pw_var = sensor.grf.variance
    _plot_helper(plot_grid, 2, "Excursion probability.", sensor.grid,
            pw_var[:, 0])
    _plot_helper(plot_grid, 3, "Excursion probability.", sensor.grid,
            pw_var[:, 1])
    """

    # 5) Plot neigbors EIBV.

    if output_filename is not None:
        plt.savefig(output_filename)
    else: plt.show()

    return


def _plot_helper(plot_grid, i, title, grid, vals, S_y=None, L_y=None, cmap=None):
    """

    Parameters
    ----------
    plot_grid
    i: int
        Index in the plot grid at which to plot.

    """
    if isinstance(vals, GeneralizedVector):
        vals = vals.isotopic
    if cmap == "proba":
        cmap = CMAP_PROBA
        color="lightgreen"
    else:
        cmap = CMAP
        color = "red"
    reshaped_vals = grid.interpolate_to_image(vals[:])
    plot_grid[i].set_title(title)
    im = plot_grid[i].imshow(
            reshaped_vals[:, :].numpy(),
            origin="lower",
            extent=[0,1,0,1],
            cmap=cmap)
    cax = plot_grid.cbar_axes[i]
    cbar = cax.colorbar(im)
    cbar.ax.tick_params(labelsize=8)

    if (S_y is not None):
        # Add the location of the measurement points on top.
        locs = S_y.numpy()
        plot_grid[i].scatter(locs[:, 1], locs[:, 0], marker="x", s=1.5, color=color)

    # Restore plot borders, which might be deformed by the scatter.
    plot_grid[i].set_xlim([0, 1])
    plot_grid[i].set_ylim([0, 1])

    
    # Colorbar
    ax = plot_grid[i]
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    return
