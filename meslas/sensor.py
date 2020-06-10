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
        # In order to append correctly, we want the location to ba a line
        # vector.
        self.location = location.reshape(1, -1).float()
        self.current_node_ind = self.grid.get_closest(self.location)

        # 0-dim tensor cannot be concatenated, so have to unsqueeze.
        self.visited_node_inds = torch.cat(
                [self.visited_node_inds, self.current_node_ind])

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

    def get_current_neighbors(self):
        """ Get the neighbouring grid nodes of the current sensor location.

        This is done by first finding the node closes to the sensor location,
        and then returning its neighbors.

        Returns
        -------
        neighbors_inds: (n_neighbors)
            Grid indices of the neighbors.

        """
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

class DiscreteSensor(Sensor):
    """ Sensor on a fixed discretization.

    """
    def __init__(self, discrete_grf):
        super(DiscreteSensor, self).__init__(discrete_grf.grid, discrete_grf)

    def update_design(self, S_y_inds, L_y, y, noise_std=None):
        """ Updates the full design grid by computing current conditional mean and covariance.
        Note that this updates the internal of the discrete GRF.

        Returns
        -------
        mu_cond_iso: (self.grid.n_points, self.grf.n_out)
            Conditional mean.
        K_cond_iso: (self.grid.n_points, self.grid.n_points, self.grf.n_out, self.grf.n_out) Tensor

            Conditional covariance matrix in isotopic ordered form.
            It means that the covariance matrix at cell i can be otained by
            subsetting K_cond_iso[i, i, :, :].

        """
        S_y = self.grid.points[S_y_inds]
        self.add_data(S_y, L_y, y)
        # self.grf.update(S_y_inds, L_y, y, noise_std)
        self.grf.update_from_scratch(S_y_inds, L_y, y, noise_std)

    def compute_exursion_prob(self, lower, upper=None):
        """ Compute the excursion probability on the whole grid, given the
        currently available data.

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

        Returns
        -------
        excursion_proba: (self.grid.n_points) Tensor
            Excursion probability at each point.

        """
        # Extract covariance matrix at every point.
        pointwise_cov = self.grf.pointwise_cov

        excursion_proba = coverage_fct_fixed_location(
                self.grf.mean_vec.isotopic, pointwise_cov, lower,
                upper=None)
        return excursion_proba

    def get_neighbors_isotopic_eibv(self, noise_std, lower, upper=None):
        """ For each neighbouring node of the current sensor location, compute
        the eibv if we were to measure every response (isotopic) at that node.
        Returns a list containing the EIBV for each neighbor.

        Parameters
        ----------
        noise_std: float
            Standar deviation of measurement noise.
        lower: (p) Tensor
            List of lower threshold for each response. The excursion set is the set
            where responses are above the specified threshold.
            Note that np.inf is supported.
        upper: (p) Tensor
            If not provided, defaults to +infty (excursion set above
            threshold).

        Returns
        -------
        neighbors_eibv: (n_neighbors) Tensor
            EIBV for each neighbouring cell.
        neighbors_inds: (n_neighbors) Tensor
            Grid indices of the neighbouring cells.

        """
        neighbors_inds = self.get_current_neighbors()
        neighbors_coords = self.grid.points[neighbors_inds]

        neighbors_eibv = torch.Tensor()
        for S_y_ind in neighbors_inds:
            S_inds, L = self.grid.get_isotopic_generalized_location_inds(
                    self.grid.points[S_y_ind], self.grf.n_out)
            eibv = self.grf.eibv(S_inds, L, lower, upper, noise_std)
            neighbors_eibv = torch.cat([neighbors_eibv, eibv.unsqueeze(0)])

        return neighbors_eibv, neighbors_inds

    def choose_next_point_myopic(self, noise_std, lower, upper=None):
        """ Choose the next observation location (given the current one)
        using the myopic strategy.

        Parameters
        ----------
        noise_std: float
            Standar deviation of measurement noise.
        lower: (p) Tensor
            List of lower threshold for each response. The excursion set is the set
            where responses are above the specified threshold.
            Note that np.inf is supported.
        upper: (p) Tensor
            If not provided, defaults to +infty (excursion set above
            threshold).

        Returns
        -------
        next_point_ind: (1) Tensor
            Grid index of next observation location chosen according to the
            myopic strategy.
        next_point_eibv: (1) Tensor
            EIBV corresponding to the next chosen point.

        """
        neighbors_eibv, neighbors_inds = self.get_neighbors_isotopic_eibv(
                noise_std, lower, upper)
        min_ind = torch.argmin(neighbors_eibv)

        return neighbors_inds[min_ind], neighbors_eibv[min_ind]

    def run_myopic_stragegy(self, n_steps, data_feed, noise_std, lower,
            upper=None):
        """ Run the myopic strategy for n_steps, starting from the current
        location. That is, at each point, pick the neighbors with the smallest
        EIBV, move there, observe, update model, repeat.

        Parameters
        ----------
        n_steps: int
            Number of steps (observations) to run the strategy for.
        data_feed: function(int)
            Function that, given a node index, returns the measured data
            (isotopic) at that node.

        """
        for i in range(n_steps):
            # Choose next point.
            next_point_ind, next_point_eibv = self.choose_next_point_myopic(
                    noise_std, lower, upper)

            # Move there.
            print("Moving to next best point {}".format(
                    self.grid.points[next_point_ind]))
            self.set_location(self.grid.points[next_point_ind])

            # Get the measurement vector corresponding to isotopic data
            S_ind, L = self.grid.get_isotopic_generalized_location_inds(
                    self.grid.points[next_point_ind], self.grf.n_out)

            # Acquire data.
            y = data_feed(next_point_ind)
            
            # Update model.
            self.update_design(S_ind, L, y, noise_std)
