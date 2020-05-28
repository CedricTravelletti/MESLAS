""" 

TODO: Maybe add grid, grf, n_ouptuts to the sensor class.
"""
import torch
from meslas.excursion import coverage_fct_fixed_location


torch.set_default_dtype(torch.float32)


class Sensor():
    """ Implements the data collection process.
    Will be responsible for querying the GP for mean and variances conditional
    on the already collected data.

    Attributes
    ----------
    S_y_tot: (N, d) Tensor
        Spatial locations of the already collected data.
    L_y_tot: (N) Tensor
        Corresponding response indices.
    location: (d) Tensor
        Current position of the sensor.
    grid: IrregularGrid
        A discretization of space that defines the locations the sensor can
        move to.
    grf: GRF
        Gaussian Random Field used to model the unknown phenomenon of interest.
    current_node_ind: int
        Index of the point in the grid that is closest to the sensor location.
    visited_nodes_inds: Tensor
        Grid indices of the visited locations.
    noise_std: float
        Standard deviation of the sensor noise.
        TODO: allow different noises for each component.

    """
    def __init__(self, grid, grf):
        # Empty tensors to hold the already collected information.
        self.S_y_tot = torch.Tensor()
        self.L_y_tot = torch.Tensor().long()
        self.y_tot = torch.Tensor()

        # Current location of the sensor.
        self.location = torch.Tensor()
        self.visited_node_inds = torch.Tensor().long()

        self.grid = grid
        self.grf = grf


        self.noise_std = 0.05

    def set_location(self, location):
        """ Setter for the location.

        Parameters
        ----------
        location: (d) array_like

        """
        if not torch.is_tensor(location):
            location = torch.tensor(location)
        self.location = location
        self.current_node_ind = self.grid.get_closest(self.location)

        # 0-dim tensor cannot be concatenated, so have to unsqueeze.
        self.visited_node_inds = torch.cat(
                [self.visited_node_inds, self.current_node_ind.unsqueeze(0)], dim=0)

    def add_data(self, S_y, L_y, y):
        """ Add new data to the already collected one.
        Can also handle batches.
        This will just concatenate the new data vectors with the current ones.

        Parameters
        ----------
        S_y: (n, d) Tensor
            Spatial locations of the new measurements.
        L_y :(n) Tensor
            Corresponding response indices.
        y :(n, p) Tensor
            Measured data.

        """
        # 0-dim tensor cannot be concatenated. So throw if happens.
        if L_y.dim() == 0:
            raise ValueError("Shouldn't have Scalar tensor, wrap it inside [].")
        self.S_y_tot = torch.cat([self.S_y_tot, S_y], dim=0)
        self.L_y_tot = torch.cat([self.L_y_tot, L_y], dim=0)
        self.y_tot = torch.cat([self.y_tot, y], dim=0)
        return

    def get_neighbors(self):
        # First find the grid node at which we are sitting
        # (the closest one).
        current_node_ind = self.grid.get_closest(self.location)

        neighbors_inds = self.grid.get_neighbors(current_node_ind)
        return neighbors_inds

    def compute_exursion_prob(self, points, lower, upper=None):
        """ Compute the excursion probability at a set of points given the
        currently available data.

        Note this is a helper function that take an index in the grid as input.

        Parameters
        ----------
        points: (N, d) Tensor
            List of points (coordinates) at which to compute the excursion probability.
        lower: (p) Tensor
            List of lower threshold for each response. The excursion set is the set
            where responses are above the specified threshold.
            Note that np.inf is supported.
        upper: (p) Tensor
            List of upper threshold for each response. The excursion set is the set
            where responses are above the specified threshold.
            If not provided, defaults to + infinity.

        Returns
        -------
        excursion_proba: (N) Tensor
            Excursion probability at each point.

        """
        # First step: compute kriging predictors at locations of interest
        # based on the data acquired up to now.
        # Compute the prediction for all responses (isotopic).
        mu_cond_list, mu_cond_iso , K_cond_list, K_cond_iso = self.grf.krig_isotopic(
                points,
                self.S_y_tot, self.L_y_tot, self.y_tot,
                noise_std=self.noise_std,
                compute_post_cov=True)

        # Extract the variances only.
        # TODO: Since we always compute the full covariance matrix at the
        # moment, this extraction should be delegated somewhere, or better,
        # have a (conditional) distribution object that has methods to extract
        # the diagonal.
        K_cond_diag = torch.diagonal(K_cond_iso, dim1=0, dim2=1).T

        excursion_proba = coverage_fct_fixed_location(
                mu_cond_iso, K_cond_diag, lower,
                upper=None)
        return excursion_proba

    def _compute_neighbors_exursion_prob(self, ind, lower, upper=None):
        """ Compute the excursion probability of the neighbors of a given cell.

        Note this is a helper function that take an index in the grid as input.

        Parameters
        ----------
        ind: int
            Index of the current node in the points list of the grid.
        lower: (p) Tensor
            List of lower threshold for each response. The excursion set is the set
            where responses are above the specified threshold.
            Note that np.inf is supported.
        upper: (p) Tensor
            List of upper threshold for each response. The excursion set is the set
            where responses are above the specified threshold.
            If not provided, defaults to + infinity.

        """
        # Get the neighboring locations.
        neighbors_inds = self.grid.get_neighbors(ind)
        neighbors_coords = self.grid.points[neighbors_inds]
        
        excursion_proba = compute_exursion_prob(neighbors_coords, lower, upper)
        return excursion_proba
        
    def compute_neighbors_exursion_prob(self, lower, upper=None):
        """ Compute the excursion probability of the neighbors of the current
            location.

            Parameters
            ----------
            lower: (p) Tensor
                List of lower threshold for each response. The excursion set is the set
                where responses are above the specified threshold.
                Note that np.inf is supported.
            upper: (p) Tensor
                List of upper threshold for each response. The excursion set is the set
                where responses are above the specified threshold.
                If not provided, defaults to + infinity.
    
        """
        neighbors_excu_prob = self._compute_neighbors_exursion_prob(
                self.current_node_ind, lower, upper)
        return neighbors_excu_prob
