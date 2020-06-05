""" Implement multivariate random fields.

Convention is that p is the number of responses.
Tensors may be returned either in *heterotopic* or in *isotopic* form.

Heterotopic form means that a response index must be specified for each
element. For example, given a vector of locations S and a vector of response indices
L, both of size M, the heteretopic form of the mean vector at these (generalized) locations is
a vector of size M such that its i-th element is the mean of component L[i] at
spatial location S[i].

When *all* responsed indices are considered, we use the word isotopic.
Since under the hood, a response index vector always has to be specified, in
the isotopic case we use L = (1, ..., p, 1, ..., p, 1, ..., p, ....) and
S = (s1, ..., s1, s2, ..., s2, ...). That is, for each spatial location s_i, we
repeat it p-times, and the response index vector is just made of the list 1,
..., p, repeated n times, n being the number of spatial locations.

Now, in this situation, it makes sense to reshape the resulting mean vector
such that the repsonse dimensions have their own axis.
This is what is meant by *isotopic form*. In isotopic form , the corresponding
means vector has shape (n. p).

"""
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from meslas.geometry.grid import get_isotopic_generalized_location
from meslas.vectors import GeneralizedVector, GeneralizedMatrix
from meslas.excursion import coverage_fct_fixed_location
from meslas.external_dependencies.numpytorch import kron as kronecker
from gpytorch.utils.cholesky import psd_safe_cholesky

torch.set_default_dtype(torch.float32)


class GRF():
    """ 
    GRF(mean, covariance)

    Gaussian Random Field with specified mean function and covariance function.

    Parameters
    ----------
    mean: function(s, l)
        Function returning l-th component of  mean at location s.
        Should be vectorized.
    covariance: function(s1, s2, l1, l2)
        Function returning the covariance matrix between the l1-th component at
        s1 and the l2-th component at l2.
        Should be vectorized.

    """
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        self.n_out = covariance.n_out

    def __repr__(self):
        out_string = ("Gaussian Random Field model with {} output "
                "dimensions.\n").format(self.n_out)
        sep_string = "------------------------------------------------------\n"
        mean_string = self.mean.__repr__()
        return out_string + sep_string + mean_string + sep_string + cov_string

    def variance(self, S, L):
        """ Compute the (pointwise) variances at generalized location (S, L).

        Parameters
        ----------
        S: (M, d) Tensor
            List of spatial locations.
        L: (M) Tensor
            List of response indices.

        Returns
        -------
        vars: (M) Tensor
            The variances Z_{s_i} component l_i.

        """
        raise NotImplementedError

    def sample(self, S, L):
        """ Sample the GRF at generalized location (S, L).

        Parameters
        ----------
        S: (M, d) Tensor
            List of spatial locations.
        L: (M) Tensor
            List of response indices.

        Returns
        -------
        Z: (M) Tensor
            The sampled value of Z_{s_i} component l_i.

        """
        K = self.covariance.K(S, S, L, L)
        # chol = torch.cholesky(K)
        mu = self.mean(S, L)

        # Sample M independent N(0, 1) RVs.
        # TODO: Determine if this is better than doing Cholesky ourselves.
        lower_chol = psd_safe_cholesky(K, jitter=1e-3)
        distr = MultivariateNormal(
                loc=mu,
                scale_tril=lower_chol)
        sample = distr.sample()

        #sample = mu + chol @ v 

        return sample.float()

    def sample_isotopic(self, points):
        """ Sample the GRF (all components) on a list of points.

        Parameters
        ----------
        points: (N, d) Tensor
            Spatial locations where to sample.

        Returns
        -------
        sample_list: (N, p) Tensor
            The sampled values.

        """
        S_iso, L_iso = get_isotopic_generalized_location(
                points, self.n_out)
        sample = self.sample(S_iso, L_iso)

        # Separate indices.
        sample_list = sample.reshape((points.shape[0], self.n_out))
        return sample_list

    # TODO: delegate stuff to grid to allow it to return regular only if square
    # grid.
    def sample_grid(self, grid):
        """ Sample the GRF (all components) on a grid.

        Parameters
        ----------
        grid: Grid

        Returns
        -------
        sample_grid: (n1, ..., n_d, ,p) Tensor
            The sampled field on the grid. Here p is the number of output
            components and n1, ..., nd are the number of cells along each axis.
        sample_list: (n_points, p) Tensor
            Same as above, but in list form.

        """
        sample = self.sample_isotopic(grid.points)

        # Separate indices.
        sample_list = sample.reshape((grid.n_points, self.n_out))
        # Put back in grid form.
        sample_grid = sample_list.reshape((*grid.shape, self.n_out))

        return sample_grid, sample_list

    def krig(self, S, L, S_y, L_y, y, noise_std=0.0,
            compute_post_var = False, compute_post_cov=False):
        """ Predict field at some generalized locations, based on some measured data at other
        generalized locations.

        This is the most general possible form of kriging, since it takes
        measurements at generalized locations and predicts at generalized
        locations.

        Parameters
        ----------
        S: (N, d)
            Spatial locations at which to predict
        L: (N) Tensor
            Response indices to predict.
        S_y: (M, d) Tensor
            Spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        y: (M) Tensor
            Measured values.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.
        compute_post_var: bool
            If true, compute and return posterior variance (matrix) at each
            point.
        compute_post_cov: bool
            If true, compute and return posterior covariance. Only one of
            compute_post_var or compute_post_var may be avtivated.
    
        Returns
        -------
        mu_cond: (N) Tensor
            Kriging means at each generalized location.
        var_cond: (N)
            Conditional varance at each generalized location.
            Only returned if specified.
        K_cond: (N, N) Tensor
            Conditional covariance matrix between the generalized locations.    
            Only returned if specified.

        """
        # We need y to be a single dimensional vector.
        y = y.reshape(-1)

        mu_pred = self.mean(S, L)
        mu_y = self.mean(S_y, L_y)
        K_pred_y = self.covariance.K(S, S_y, L, L_y)
        K_yy = self.covariance.K(S_y, S_y, L_y, L_y)

        noise = noise_std**2 * torch.eye(y.shape[0])

        weights = K_pred_y @ torch.inverse(K_yy + noise)
        mu_cond = mu_pred + weights @ (y - mu_y)
        if compute_post_var:
            K = self.covariance.K(S, S, L, L)
            var_cond = torch.diag(K) - torch.einsum('ik,ik->i', weights, K_pred_y)
            return mu_cond, var_cond

        elif compute_post_cov:
            K = self.covariance.K(S, S, L, L)
            K_cond = K - weights @ K_pred_y.t()
            return mu_cond, K_cond

        return mu_cond

    def krig_isotopic(self, points, S_y, L_y, y, noise_std=0.0,
            compute_post_var=False, compute_post_cov=False):
        """ Predict field at some points, based on some measured data at other
        points. Predicts all repsonses (isotopic).
    
        Parameters
        ----------
        points: (N, d) Tensor
            List of points at which to predict.
        S_y: (M, d) Tensor
            Spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        y: (M) Tensor
            Measured values.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.
        compute_post_var: bool
            If true, compute and return posterior variance (matrix) at each
            point.
        compute_post_cov: bool
            If true, compute and return posterior covariance. Only one of
            compute_post_var or compute_post_var may be avtivated.
    
        Returns
        -------
        mu_cond_list: (N*p) Tensor
            Kriging mean, but in list form.
        mu_cond_iso: (N, p) Tensor
            Kriging means in isotopic list form.
        K_cond_list: (N * p, N * p) Tensor
            Conditional covariance matrix in heterotopic form.
        K_cond_iso: (N, N, p, p) Tensor
            Conditional covariance matrix in isotopic ordered form.
            It means that the covariance matrix at cell i can be otained by
            subsetting K_cond_iso[i, i, :, :].
        var_cond_list: (N*p) Tensor
            Conditional varance at each generalized location.
            Only returned if specified.
        var_cond_iso: (N, p) Tensor
            Same as above, but in list form.
    
        """
        n_pts = points.shape[0]
        # Generate prediction locations corrresponding to the full grid.
        S, L = get_isotopic_generalized_location(
                points, self.n_out)

        if compute_post_var:
            mu_cond_list, var_cond_list = self.krig(
                    S, L, S_y, L_y, y, noise_std=noise_std,
                    compute_post_var=True)
        elif compute_post_cov:
            mu_cond_list, K_cond_list = self.krig(
                    S, L, S_y, L_y, y, noise_std=noise_std,
                    compute_post_cov=True)
        else:
            mu_cond_list = self.krig(
                    S, L, S_y, L_y, y, noise_std=noise_std)

        # Wrap in a GeneralizedVector.
        mu_cond = GeneralizedVector.from_list(mu_cond_list, n_pts, self.n_out)

        if compute_post_var: 
            var_cond = GeneralizedVector.from_list(var_cond_list, n_pts, self.n_out)
            return mu_cond, var_cond

        elif compute_post_cov: 
            K_cond = GeneralizedMatrix(
                    K_cond_list,
                    n_pts, self.n_out, n_pts, self.n_out)
            return mu_cond, K_cond

        return mu_cond

    # TODO: See if we can deprecate this.
    # It was mainly used for plotting, but since now reordering has been
    # defered to grid class, we might want to get rid of this.
    def krig_grid(self, grid, S_y, L_y, y, noise_std=0.0, compute_post_cov=False):
        """ Predict field at some points, based on some measured data at other
        points.
    
        Parameters
        ----------
        grid: Grid
            Regular grid of size (n1, ..., nd).
        S_y: (M, d) Tensor
            Spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        y: (M) Tensor
            Measured values.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.
        compute_post_cov: bool
            If true, compute and return posterior covariance.
    
        Returns
        -------
        mu_cond_grid: (grid.shape, p) Tensor
            Kriging means at each grid node.
        mu_cond_list: (grid.n_points*p) Tensor
            Kriging mean, but in list form.
        mu_cond_iso: (grid.n_points, p) Tensor
            Kriging means in isotopic list form.
        K_cond_list: (grid.n_points * p, grid.n_points * p) Tensor
            Conditional covariance matrix in heterotopic form.
        K_cond_iso: (grid.n_points, grid.n_points, p, p) Tensor
            Conditional covariance matrix in isotopic ordered form.
            It means that the covariance matrix at cell i can be otained by
            subsetting K_cond_iso[i, i, :, :].
    
        """
        # Generate prediction locations corrresponding to the full grid.
        S, L = get_isotopic_generalized_location(
                grid.points, self.n_out)

        if compute_post_cov:
            mu_cond_list, K_cond_list = self.krig(
                    S, L, S_y, L_y, y, noise_std=noise_std,
                    compute_post_cov=compute_post_cov)
        else:
            mu_cond_list = self.krig(
                    S, L, S_y, L_y, y, noise_std=noise_std,
                    compute_post_cov=compute_post_cov)

        # Reshape to regular grid form. Begin by adding a dimension for the
        # response indices.
        mu_cond_iso = mu_cond_list.reshape((grid.n_points, self.n_out))
        # Put back in grid form.
        mu_cond_grid = mu_cond_iso.reshape((*grid.shape, self.n_out))

        if compute_post_cov: 
            # Reshape to isotopic form by adding dimensions for the response
            # indices.
            K_cond_iso = K_cond_list.reshape(
                    (grid.n_points, self.n_out, grid.n_points,
                            self.n_out)).transpose(1,2)
            return mu_cond_grid, mu_cond_list, mu_cond_iso, K_cond_list, K_cond_iso

        return mu_cond_grid

    def variance_reduction(self, S, L, S_y, L_y, noise_std=0.0):
        """ Computes the reduction in variance at generalized
        locations (S, L) that would be caused by observing data at generalized
        locations (S_y, L_y).
        Note that this doesn't depend on the measured data.

        Parameters
        ----------
        S: (N, d)
            Spatial locations at which to predict
        L: (N) Tensor
            Response indices to predict.
        S_y: (M, d) Tensor
            Spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.

        Returns
        -------
        variance_reduction: (N) Tensor
            Reduction in variance at each generalized location.

        """
        K_pred_y = self.covariance.K(S, S_y, L, L_y)
        K_yy = self.covariance.K(S_y, S_y, L_y, L_y)

        noise = noise_std**2 * torch.eye(L_y.shape[0])
        weights = K_pred_y @ torch.inverse(K_yy + noise)
        variance_reduction = torch.einsum('ik,ik->i', weights, K_pred_y)

        return variance_reduction

    def variance_reduction_isotopic(self, points, S_y, L_y, noise_std=0.0):
        """ Computes the reduction in variance (all components) at location S
        that would be caused by observing data at generalized locations (S_y, L_y).
        Note that this doesn't depend on the measured data.

        Parameters
        ----------
        points: (N, d) Tensor
            List of points at which to predict.
        S_y: (M, d) Tensor
            Spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.

        Returns
        -------
        variance_reduction: (N) Tensor
            Reduction in variance at each generalized location.

        """
        n_pts = points.shape[0]
        # Generate prediction locations corrresponding to the full grid.
        S, L = get_isotopic_generalized_location(
                points, self.n_out)
        
        variance_reduction_list = self.variance_reduction(
                S, L, S_y, L_y, noise_std=noise_std)

        # Reshape to isotopic form by adding dimensions for the response
        # indices.
        variance_reduction = GeneralizedVector.from_list(
                variance_reduction_list, n_pts, self.n_out)
        return variance_reduction

class DiscreteGRF(GRF):
    """ Gaussian Random Field that is discretized over a static grid. This
    means that values can only be computed at grid nodes and observations may
    also only be performed at grid nodes.

    Since it is not known a piori at which locations we will observe or
    predict, the GRF computes and stores the mean vector and covariance matrix
    for the whole design at each stage.

    Parameters
    ----------
    grid: Grid
    mean_vec: GeneralizedVector
    covariance_mat: GeneralizedMatrix

    """
    def __init__(self, grid, mean_vec, covariance_mat):
        self.grid = grid
        self.mean_vec = mean_vec
        self.covariance_mat = covariance_mat
        self.n_points = self.grid.n_points
        self.n_out = mean_vec.shape[1]

    @classmethod
    def from_model(cls, grf, grid):
        """ Create a discrete GRF by discretizing a GRF model. One specifies
        the mean and covariance funcions and the model is then discretized on a
        grid.

        Parameters
        ----------
        grf: GRF
            Gaussian Random Field to discretize.
        grid: Grid
            Grid on which to discretize.

        """
        n_out = grf.covariance.n_out

        # Generate the measurment vector that corresponds to all components.
        S, L = get_isotopic_generalized_location(
                grid.points, n_out)

        mean_vec = grf.mean(S, L)
        mean_vec = GeneralizedVector.from_list(mean_vec, grid.n_points, n_out)

        covariance_mat = grf.covariance.K(S, S, L, L)
        covariance_mat = GeneralizedMatrix.from_list(covariance_mat,
                grid.n_points, n_out, grid.n_points, n_out)

        return cls(grid, mean_vec, covariance_mat)

    @property
    def pointwise_cov(self):
        """ Pointwise covariance matrix, i.e. covariance matrix of the field at
        each points (i.e. no cross-location covariance).

        Returns
        -------
        pointwise_cov: (n_points, n_out, n_out) Tensor

        """
        pointwise_cov = torch.diagonal(self.covariance_mat.isotopic, dim1=0, dim2=1).T
        return pointwise_cov

    @property
    def variance(self):
        """ Returns the (pointwise) variances.

        Returns
        -------
        variances: Generalized vector
            Variance of each component of the random field at each point.

        """
        # First extract the covariance matrices at each points
        pointwise_cov = self.pointwise_cov

        # Now extract their diagonal.
        variances = torch.diagonal(pointwise_cov, dim1=1, dim2=2)

        variances = GeneralizedVector.from_isotopic(variances)
        return variances

    def update(self, S_y_inds, L_y, y, noise_std=None):
        """ Observe some data and update the field. This will compute the new
        values of the mean vector and covariance matrix.

        Parameters
        ----------
        S_y_inds: (M) Tensor
            Indices (in the grid) of the spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        y: (M) Tensor
            Measured values.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.

        """
        # We need y to be a single dimensional vector.
        y = y.reshape(-1)

        # Create the generalized measurement vector corresponding to prediction
        # on the whole grid.
        S_inds, L = self.grid.get_isotopic_generalized_location_inds(
                self.grid.points, self.n_out)

        mu_y = self.mean_vec[S_y_inds, L_y]

        # Subsetting of covariance matrices has to be done in two steps.
        K_pred_y = self.covariance_mat[S_inds, :, L, :]
        K_pred_y = K_pred_y[:, S_y_inds, L_y]

        # WARNING: It is very important to do the indexing in two steps.
        # If not, then torch will return an object of the wrong dimension, but
        # then silently convert it once it is used. This can (and will) produce
        # nasyt bugs.
        K_yy = self.covariance_mat[S_y_inds, :, L_y, :]
        K_yy = K_yy[:, S_y_inds, L_y]

        noise = noise_std**2 * torch.eye(y.shape[0])

        weights = K_pred_y @ torch.inverse(K_yy + noise)

        # Directly update the one dimensional list of values for the mean
        # vector.
        self.mean_vec.set_vals(self.mean_vec.list + weights @ (y - mu_y))
            
        self.covariance_mat.set_vals(self.covariance_mat.list - weights @ K_pred_y.t())

        return self.mean_vec, self.covariance_mat

    def compute_cov_reduction(self, S_y_inds, L_y, noise_std=None):
        """ Compute the covariance reduction that would result from observing
        some hypothetic data.

        Parameters
        ----------
        S_y_inds: (M) Tensor
            Indices (in the grid) of the spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.

        Returns
        -------
        cov_reduction: (n_points, n_out) GeneralizedMatrix

        """
        # Create the generalized measurement vector corresponding to prediction
        # on the whole grid.
        S_inds, L = self.grid.get_isotopic_generalized_location_inds(
                self.grid.points, self.n_out)

        # Subsetting of covariance matrices has to be done in two steps.
        K_pred_y = self.covariance_mat[S_inds, :, L, :]
        K_pred_y = K_pred_y[:, S_y_inds, L_y]

        # WARNING: It is very important to do the indexing in two steps.
        # If not, then torch will return an object of the wrong dimension, but
        # then silently convert it once it is used. This can (and will) produce
        # nasyt bugs.
        K_yy = self.covariance_mat[S_y_inds, :, L_y, :]
        K_yy = K_yy[:, S_y_inds, L_y]

        noise = noise_std**2 * torch.eye(K_yy.shape[0])

        weights = K_pred_y @ torch.inverse(K_yy + noise)
        cov_reduction = weights @ K_pred_y.t()

        # Wrap in GeneralizedMatrix for convenience.
        cov_reduction = GeneralizedMatrix(cov_reduction,
                    self.n_points, self.n_out, self.n_points, self.n_out)

        return cov_reduction

    def sample(self):
        """ Sample the discretized GRF on the whole grid.


        Returns
        -------
        Z: (M) Tensor
            The sampled value of Z_{s_i} component l_i.

        """
        K = self.covariance_mat.list
        mu = self.mean_vec.list

        # Sample M independent N(0, 1) RVs.
        # TODO: Determine if this is better than doing Cholesky ourselves.
        lower_chol = psd_safe_cholesky(K, jitter=1e-3)
        distr = MultivariateNormal(
                loc=mu,
                scale_tril=lower_chol)
        sample = distr.sample()

        return GeneralizedVector.from_list(sample.float(), self.n_points, self.n_out)

    def _eibv_part_2(self, S_y_inds, L_y, lower, upper=None, noise_std=None):
        h = 2 # In case we want to generalize later (see Proposition 2).

        id_h = torch.eye(h, dtype=torch.float32)
        full_h = torch.full((h, h), 1.0, dtype=torch.float32)

        # Extract the covariance reduction at every point. This is pointwise,
        # i.e. ignore correlations between different spatial locations.
        pw_cov_reduction = torch.diagonal(
                self.compute_cov_reduction(S_y_inds, L_y, noise_std).isotopic,
                dim1=0, dim2=1).T

        # Build the concatenated covariance matrix and mean vector
        # WARNING: At each point, we want to multiply the dimension of the
        # response by h. This means that we have to be careful to expand along
        # the response dimension (the last one) and not along the batch
        # dimension (the spatial one).
        pw_covariance_cat = (
                kronecker(id_h, self.pointwise_cov)
                + kronecker(full_h - id_h, pw_cov_reduction))

        one_vector = torch.full((h, 1), 1.0)
        mean_cat = torch.cat(h * [self.mean_vec.isotopic], dim=1)

        # Now concatenate the thresholds.
        lower_cat = torch.cat(h * [lower])
        if upper is not None:
            upper_cat = torch.cat(h * [upper])
        else: upper_cat = None

        part2 = coverage_fct_fixed_location(
                    mean_cat, pw_covariance_cat, lower_cat, upper_cat)
        return part2

    def eibv(self, S_y_inds, L_y, lower, upper=None, noise_std=None):
        """ Computes the expected IBV if we were to make observations at the
        generalized locations (S_y, L_y).
        Since we are on a grid, we do not directly specify the spatial
        locations S_y, but the corresponding grid indices S_y_inds instead.

        Parameters
        ----------
        S_y_inds: (M) Tensor
            Indices (in the grid) of the spatial locations of the measurements.
        L_y: (M) Tensor
            Response indices of the measurements.
        noise_std: float
            Noise standard deviation. Uniform across all measurments.

        """
        part1 = coverage_fct_fixed_location(
                    self.mean_vec.isotopic, self.pointwise_cov, lower, upper)
        part2 = self._eibv_part_2(S_y_inds, L_y, lower, upper, noise_std)

        return torch.sum(part1) - torch.sum(part2)
