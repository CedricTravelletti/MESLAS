""" Random fields for inverse problems. This allows conditioning the field on
with respect to linear operators.

This is not fundamentally different from usual conditioning, and could be
straightforwardly implemented if Pytorch allowd for tensors of variable
dimensions.

Indeed, if a measurement operator only involves field values and a few set of
points, then the natural implementation would be a sparse one, computing only
the covariances involved in the measurement.
But this makes matmuls awkward to handle since the number of dimensions in each
line might vary.

Thats why we take another philosophy: a dense one, namely, we always comute all
covariances between all elements of the design.

Thus, we simply take the implementation of a DiscreteGRF, but each time one can
specify locations and measurement indices, we replace the location vectors by a
vector specifying to compute for ALL locations in the grid.

"""
import torch
from meslas.random_fields import DiscreteGRF


class InverseDiscreteGRF(DiscreteGRF):
    def get_full_measurement_inds(self):
        """ Create the mesurement vectors that specifiy to compute things for
        the WHOLE grid.

        Returns
        -------
        S_inds: (self.grid.n_points * self.n_out) Tensor
            Vector containing ALL grid indices (and duplicated for isotopic
            measurement).
        L: (self.grid.n_points * self.n_out) Tensor
            Same, but for response indices.

        """
        # Create the generalized measurement vector corresponding to prediction
        # on the whole grid.
        S_inds, L = self.grid.get_isotopic_generalized_location_inds(
                self.grid.points, self.n_out)
        return S_inds, L

    def inverse_update(self, G, y, noise_std=None,
            mean_vec=None, covariance_mat=None):
        """ Observe some linear operator data and update the field. This will compute the new
        values of the mean vector and covariance matrix.

        Parameters
        ----------
        G: (M, self.grid.n_points, self.n_out) Tensor
            Measurement matrix.
            Response indices of the measurements.
        y: (M) Tensor
            Measured values.
        noise_std: (M) Tensor
            Noise standard deviation for each datapoints, defaults to zero.
        mean_vec: GeneralizedVector, optional
            Can specify a mean vector to start from if do not want to use the
            current mean.
        covariance_mat: GeneralizedMatrix, optional
            Can specify another covariance matrix if do not want to use the
            current one.

        """
        # If not provided, use the current model to update.
        if mean_vec is None: mean_vec = self.mean_vec
        if covariance_mat is None: covariance_mat = self.covariance_mat

        # We need y to be a single dimensional vector.
        y = y.reshape(-1)

        # We always take the full coavariance matrix and mean vector.
        # TODO: This is currently only compatible for n_dims = 1
        if not self.n_out == 1:
            raise NotImplementedError("Currently only implemented for univariate.")
        mean_vec = mean_vec.list
        covariance_mat = covariance_mat.list

        K_pred_y = covariance_mat @ G.T
        K_yy = G @ K_pred_y
        mu_y = G @ mean_vec

        # Create the noise matrix.
        if noise_std is None: noise_std = torch.zeros(self.n_out)
        noise = torch.diag(noise_std**2)

        weights = K_pred_y @ torch.inverse(K_yy + noise)

        # Directly update the one dimensional list of values for the mean
        # vector.
        self.mean_vec.set_vals(mean_vec + weights @ (y - mu_y))
            
        self.covariance_mat.set_vals(covariance_mat - weights @ K_pred_y.t())

        return self.mean_vec, self.covariance_mat

    def sample_foxy(self, G, y, noise_std):
        # TODO: Inefficiency here, we re-discretize the prior two times.
        # For sampling and for conditioning from scratch.
        uncond_sample = self.sample(from_prior=True)

        prior_mean_vec, prior_covariance_mat = self.discretize_prior(
                self.prior_grf, self.grid)

        # Realized data.
        y_real = G @ uncond_sample.list

        krigged_real, _  = self.inverse_update(G, y_real, noise_std,
            mean_vec=prior_mean_vec, covariance_mat=prior_covariance_mat)

        cond_mean, _ = self.inverse_update(G, y, noise_std,
                mean_vec=prior_mean_vec, covariance_mat=prior_covariance_mat)

        return cond_mean.list + uncond_sample.list - krigged_real.list

    def inverse_conditional_cov(self, G, noise_std=None, covariance_mat=None):
        """ Observe some linear operator data and compute the conditional
        covariance.
        Note that it doesnt depend on the value of the observed data.

        Parameters
        ----------
        G: (M, self.grid.n_points, self.n_out) Tensor
            Measurement matrix.
            Response indices of the measurements.
        noise_std: (M) Tensor
            Noise standard deviation for each datapoints, defaults to zero.
        covariance_mat: GeneralizedMatrix, optional
            Can specify another covariance matrix if do not want to use the
            current one.
        
        Returns
        -------
        cov_mat: (self.grid.n_points, self.grid.n_points) Tensor

        """
        # If not provided, use the current model to update.
        if covariance_mat is None: covariance_mat = self.covariance_mat

        # We always take the full coavariance matrix and mean vector.
        # TODO: This is currently only compatible for n_dims = 1
        if not self.n_out == 1:
            raise NotImplementedError("Currently only implemented for univariate.")
        covariance_mat = covariance_mat.list

        K_pred_y = covariance_mat @ G.T
        K_yy = G @ K_pred_y

        # Create the noise matrix.
        if noise_std is None: noise_std = torch.zeros(self.n_out)
        noise = torch.diag(noise_std**2)

        weights = K_pred_y @ torch.inverse(K_yy + noise)
            
        cond_covariance_mat = covariance_mat - weights @ K_pred_y.t()

        return cond_covariance_mat
