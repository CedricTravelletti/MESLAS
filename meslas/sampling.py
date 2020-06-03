""" Sample from multivariate GRF.

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

    def variance(self, S, L):
        """ Compute the variances at generalized location (S, L).

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
        # Reshape to isotopic form. Begin by adding a dimension for the
        # response indices.
        mu_cond_iso = mu_cond_list.reshape((n_pts, self.n_out))

        if compute_post_var: 
            # Reshape to isotopic form by adding dimensions for the response
            # indices.
            var_cond_iso = var_cond_list.reshape(
                    (n_pts, self.n_out))
            return mu_cond_list, mu_cond_iso, var_cond_list, var_cond_iso

        elif compute_post_cov: 
            # Reshape to isotopic form by adding dimensions for the response
            # indices.
            K_cond_iso = K_cond_list.reshape(
                    (n_pts, self.n_out, n_pts, self.n_out)).transpose(1, 2)
            return mu_cond_list, mu_cond_iso, K_cond_list, K_cond_iso

        return mu_cond_list, mu_cond_iso

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
        variance_reduction_iso = variance_reduction_list.reshape((n_pts, self.n_out))
        return variance_reduction_iso
