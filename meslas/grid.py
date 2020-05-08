""" Tools for handling regular grids of points.

"""
import torch


class Grid():
    """ Create a regular square grid of given dimension and size.
    The grid will contain size^dim points. We alway grid over the unit cube in
    dimension dim.

    Parameters
    ----------
    size: int
        Number of point along one axis.
    dim: int
        Dimension of the space to grid.

    Returns
    -------
    coords: (size^dim, dim)
        List of coordinate of each point in the grid.

    """
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        self.grid = create_square_grid(size, dim)

    @property
    def shape(self):
        return tuple(self.dim * [self.size])

    @property
    def coordinate_vector(self):
        """ Returns the grid coordinates in a list.

        """
        return self.grid.reshape((self.size**self.dim, self.dim))

def create_square_grid(size, dim):
    """ Create a regualar square grid of given dimension and size.
    The grid will contain size^dim points. We alway grid over the unit cube in
    dimension dim.

    Parameters
    ----------
    size: int
        Number of point along one axis.
    dim: int
        Dimension of the space to grid.

    Returns
    -------
    coords: (size^dim, dim)
        List of coordinate of each point in the grid.

    """
    # Creat one axis.
    x = torch.linspace(0, 1, steps=size)

    # Mesh with itself dim times. Stacking along dim -1 mean create new dim at
    # the end.
    grid = torch.stack(torch.meshgrid(dim * [x]), dim=-1)

    return grid

def get_isotopic_generalized_location(S, p):
    """ Given a list of spatial location, create the generalized measurement
    vector that corresponds to measuring ALL responses at those locations.

    Parameters
    ----------
    S: (M, d) Tensor
        List of spatial locations.
    p: int
        Number of responses.

    Returns
    -------
    S_iso: (M * p, d) Tensor
        Location tensor repeated p times.
    L_iso: (M * p) Tensor
        Response index vector for isotopic measurement.

    """
    # Generate index list
    inds = torch.Tensor([list(range(p))]).long()
    # Repeat it by the number of cells.
    inds_iso = inds.repeat_interleave(S.shape[0]).long()

    # Repeat the cells.
    S_iso = S.repeat((p, 1))

    return S_iso, inds_iso
