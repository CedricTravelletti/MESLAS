""" Tools for handling regular grids of points.

"""
import torch
import numpy as np
from scipy.spatial import KDTree
from meslas.geometry.tilings import generate_triangles


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
        self.n_cells = self.size**self.dim

    @property
    def shape(self):
        return tuple(self.dim * [self.size])

    @property
    def coordinate_vector(self):
        """ Returns the grid coordinates in a list.

        """
        return self.grid.reshape((self.size**self.dim, self.dim))

    def isotopic_vector_to_grid(self, vector, n_out):
        """ Given  an isotopic measurement vector, reshape it to grid form.
        I.e., the input vector is a list of sites, which are duplicated because
        there is one instance per reponse index. We reshape it to a grid.

        Parameter
        ---------
        vector: (n_out * n_cells) Tensor
            Vector corresponding to isotopic measurement at every grid
            coordinate.
        n_out: int
            Number of repsonses.

        Returns
        -------
        grid_vector: (grid dim, n_out)
            Vector projected back to grid.

        """
        grid_vector = vector.reshape((n_out, self.n_cells)).t()
        grid_vector = grid_vector.reshape((*self.shape, n_out))

        return grid_vector

    def get_closest(self, points):
        """ Given a list of points, for each of them return the closest grid
        point. Also returns its index in the grid.

        Parameters
        ----------
        points: (N, dim) Tensor
            List of point coordinates.

        Returns
        -------
        closests: (N, dim) Tensor
            Coordinates of closes grid points.
        closests_inds: (N, dim) Tensor
            Grid indices of the closest points.

        """
        # Note that KDTree takes a list of points, so have to reshape.
        tree = KDTree(np.reshape(self.grid, (self.n_cells, self.dim)))
        closests, closests_inds = tree.query(points)

        # The tree returns one dimensional indices, we turn them back to
        # multidim.
        closests_inds = np.unravel_index(closests_inds, self.shape)
        # closests_inds = np.array(closests_inds).T

        return closests, closests_inds

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

def create_triangular_grid(size):
    """ Create a triagular gridding of the unit square (only available in 2D).

    Parameters
    ----------
    size: int
        Number of point along one axis.

    Returns
    -------
    coords: (n_points, 2)
        List of coordinate of each point in the grid.
        The points correspond to the upper left corner of the corresponding
        triangle.

    """
    coords = []
    for triang in generate_triangles(1, 1, 1/size):
        # Extract upper left corner
        point = triang[0]
        # Exclude if not in grid
        if (0 <= point[0] <= 1) and (0 <= point[1] <= 1):
            coords.append(point)
    return np.array(coords)

class TriangularGrid():
    """ Create a grid of triangular cells. Only valid for two dimensions.

    Parameters
    ----------
    size: int
        Number of point along one axis.

    """
    def __init__(self, size):
        self.size = size
        self.grid = create_triangular_grid(size)
        self.n_cells = self.grid.shape[0]

    @property
    def shape(self):
        return self.grid.shape

    def isotopic_vector_to_grid(self, vector, n_out):
        """ Given  an isotopic measurement vector, reshape it to grid form.
        I.e., the input vector is a list of sites, which are duplicated because
        there is one instance per reponse index. We reshape it to a grid.

        !! to list here

        Parameter
        ---------
        vector: (n_out * n_cells) Tensor
            Vector corresponding to isotopic measurement at every grid
            coordinate.
        n_out: int
            Number of repsonses.

        Returns
        -------
        grid_vector: (n_cells, n_out)
            Vector projected back to grid.

        """
        grid_vector = vector.reshape((n_out, self.n_cells)).t()

        return grid_vector

    def get_closest(self, points):
        """ Given a list of points, for each of them return the closest grid
        point. Also returns its index in the grid.

        Parameters
        ----------
        points: (N, dim) Tensor
            List of point coordinates.

        Returns
        -------
        closests: (N, dim) Tensor
            Coordinates of closes grid points.
        closests_inds: (N, dim) Tensor
            Grid indices of the closest points.

        """
        tree = KDTree(self.grid,)
        closests, closests_inds = tree.query(points)

        # The tree returns one dimensional indices, we turn them back to
        # multidim.

        return closests, closests_inds

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
