""" Sample from multivariate GRF.

"""
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from meslas.grid import Grid, get_isotopic_generalized_location


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
        distr = MultivariateNormal(
                loc=mu,
                covariance_matrix=K)
        sample = distr.sample()

        #sample = mu + chol @ v 

        return sample

    def sample_grid(self, grid):
        """ Sample the GRF (all components) on a grid.

        Parameters
        ----------
        grid: Grid

        Returns
        -------
        sample: (n1, ..., n_d, ,p) Tensor
            The sampled field on the grid. Here p is the number of output
            components and n1, ..., nd are the number of cells along each axis.

        """
        S_iso, L_iso = get_isotopic_generalized_location(
                grid.coordinate_vector, self.n_out)

        sample = self.sample(S_iso, L_iso)

        # Separate indices.
        sample = sample.reshape((self.n_out, grid.size**grid.dim)).t()
        # Put back in grid form.
        sample = sample.reshape((*grid.shape, self.n_out))

        return sample

    def krig(self, S, L, S_y, L_y, y, noise_std=0.0, compute_post_cov=False):
        """ Predict field at some points, based on some measured data at other
        points.
    
        Parameters
        ----------
    
        Returns
        -------
        m
        K
    
        """
        mu_pred = self.mean(S, L)
        mu_y = self.mean(S_y, L_y)
        K_pred_y = self.covariance.K(S, S_y, L, L_y)
        K_yy = self.covariance.K(S_y, S_y, L_y, L_y)

        noise = noise_std**2 * torch.eye(y.shape[0])

        weights = K_pred_y @ torch.inverse(K_yy + noise)
        mu_cond = mu_pred + weights @ (y - mu_y)
        if compute_post_cov:
            K = self.covariance.K(S, S, L, L)
            K_cond = K - weights @ K_pred_y.t()
            return mu_cond, K_cond

        return mu_cond

    def krig_grid(self, grid, S_y, L_y, y, noise_std=0.0, compute_post_cov=False):
        """ Predict field at some points, based on some measured data at other
        points.
    
        Parameters
        ----------
    
        Returns
        -------
        m
        K
    
        """
        # Generate prediction locations corrresponding to the full grid.
        S, L = get_isotopic_generalized_location(
                grid.coordinate_vector, self.n_out)

        if compute_post_cov:
            mu_cond, K_cond = self.krig(S, L, S_y, L_y, y, noise_std=noise_std,
                compute_post_cov=compute_post_cov)
        else: mu_cond = self.krig(S, L, S_y, L_y, y, noise_std=noise_std,
                compute_post_cov=compute_post_cov)

        # Separate indices.
        # Save unreshaped means in its list forms.
        krig_mean_1d = mu_cond
        mu_cond = mu_cond.reshape((self.n_out, grid.size**grid.dim)).t()
        # Put back in grid form.
        mu_cond = mu_cond.reshape((*grid.shape, self.n_out))

        if compute_post_cov: return mu_cond, krig_mean_1d, K_cond

        return mu_cond
