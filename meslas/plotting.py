""" Plotting capabilities.

"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_2d_slice(sliced_sample):
    """ Plots 2 dimensional slice.

    For each of the n_out responses, plot a two dimensional image.

    Parameters
    ----------
    sliced_sample: (n1, n2, n_out) Tensor
        Two dimensional slice of a sample or prediction.

    """
    # Number of responses.
    n_out = sliced_sample.shape[-1]

    # Dimensions of the plotting area.
    n_col = int(np.ceil(np.sqrt(n_out)))
    
    n_row = int(n_out/n_col)

    sample_min = torch.min(sliced_sample).item()
    sample_max = torch.max(sliced_sample).item()

    fig = plt.figure()
    plot_grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_col),
            axes_pad=0.15, share_all=True,
            cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad=0.15,)

    for i in range(n_out):
        # fig.axes[i].set_title(r"$Z^" + str(i+1) + "$")
        plot_grid[i].set_title(r"$Z^" + str(i+1) + "$")
        im = plot_grid[i].imshow(
                sliced_sample[:, :, i].numpy(),
                vmin=0.8*sample_min, vmax=0.8*sample_max,
                extent=[0,1,0,1], cmap='plasma')

    # Hide the unused plots.
    for i in range(n_out, len(plot_grid)):
        plot_grid[i].axis("off")

    # Colorbar
    ax = plot_grid[i]
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    plt.show()
