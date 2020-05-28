""" Tools for handling regular grids of points.

"""
import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from meslas.geometry.tilings import generate_triangles


torch.set_default_dtype(torch.float32)


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

class IrregularGrid():
    """ Gridding of space that is just a collection of points, with no
    pre-defined regularity.

    Attributes
    ----------
    points: (M, d) Tensors
        List of points belonging to the grid.
    n_points: int
        Number of points in the grid.

    """
    def __init__(self, size):
        raise NotImplementedError

    # TODO: Inspect these methods. Ther are used by sampling for some
    # reshaping. Should be delegated to the grid.
    @property
    def shape(self):
        """ Shape of the grid.

        """
        return self.points.shape

    # TODO: Currently only used to plot posterior, but should be delegated to
    # the GRF posterior sampling procedure.
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
        grid_vector = vector.reshape((n_out, self.n_points)).t()

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
            Coordinates of closest grid points.
        closests_inds: (N, dim) Tensor
            Grid indices of the closest points.

        """
        closests, closests_inds = self.tree.query(points)

        # TODO: How should we handle this? This is used by the sampling
        """
        # The tree returns one dimensional indices, we turn them back to
        # multidim.
        closests_inds = np.unravel_index(closests_inds, self.shape)
        """

        return closests, closests_inds

    def interpolate_to_image(self, vals, IM_HEIGHT=30, IM_WIDTH=30, method="linear"):
        """ Given a list of values at each point of the grid, interpolate it to
        a regular square grid. Used to plot values as images.

        Parameters
        ----------
        vals: (n_points) Tensor
            Values of a field defined at each point of the grid.
        IM_HEIGHT: int
            Number of pixels in the resulting image.
        IM_WIDTH: int
            Number of pixels in the resulting image.
        method: string
            Method to use for interpolation. Default is 'linear', which can
            cause blanks where no points are present. Can chage to 'nearest' if
            want to remove blanks, but the image will be more bumpy.

        Returns
        -------
        image (IM_HEIGHT, IM_WIDTH) Tensor
            Values interpolated to a regular square grid (image).

        """
        xi = np.linspace(0, 1, IM_HEIGHT)
        yi = np.linspace(0, 1, IM_WIDTH)

        zi = griddata((self.points[:, 1], self.points[:, 0]),
                vals, (xi[None,:], yi[:,None]), method=method)
        return torch.from_numpy(zi.reshape(xi.shape[0], yi.shape[0]))
            

class TriangularGrid(IrregularGrid):
    """ Create a grid of triangular cells. Only valid for two dimensions.

    Parameters
    ----------
    size: int
        Number of point along one axis.

    """
    def __init__(self, size):
        self.size = size
        # WARNING: Really needs to be Tensor in order to be compatible with the
        # rest.
        self.points = torch.from_numpy(create_triangular_grid(size)).float()
        self.n_points = self.points.shape[0]

        # Initialize a KDTree once and for all for neighbor search.
        self.tree = KDTree(self.points,)

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

class SquareGrid(IrregularGrid):
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
        # TODO: I think it would be more economic
        self.points = create_square_grid(size, dim).reshape((self.size**self.dim, self.dim))
        self.n_points = self.size**self.dim

        # Initialize a KDTree once and for all for neighbor search.
        self.tree = KDTree(self.points,)

    # TODO: Inspect these methods. Ther are used by sampling for some
    # reshaping. Should be delegated to the grid.
    @property
    def shape(self):
        return tuple(self.dim * [self.size])
    @property
    def grid(self):
        """ Returns the grid coordinates in grid shape, i.e. one axis per
        dimension.

        """
        return self.points.reshape((self.dim * [self.size] + self.dim))

    # TODO: Currently only used to plot posterior, but should be delegated to
    # the GRF posterior sampling procedure.
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
        grid_vector = vector.reshape((n_out, self.n_points)).t()
        grid_vector = grid_vector.reshape((*self.shape, n_out))

        return grid_vector

    def interpolate_to_image(self, vals):
        """ Given a list of values at each point of the grid, interpolate it to
        a regular square grid. Used to plot values as images.

        On an already regular grid, we just reshape, no interpolation.

        Parameters
        ----------
        vals: (n_points) Tensor
            Values of a field defined at each point of the grid.

        Returns
        -------
        image (IM_HEIGHT, IM_WIDTH) Tensor
            Values interpolated to a regular square grid (image).

        """
        return vals.reshape(self.shape)
