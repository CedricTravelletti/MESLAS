""" Code for multidimensional sampling.
We will be considering multivariate random fields Z=(Z^1, ..., Z^p).
The term *response index* denotes the index of the component of the field we
ate considering.

We will sometime use the word measurement point to denote a
(location, response index) pair.

We will be using the notation conventions from the papers.

    x's will denote a location
    j's will denote reponse a index

Uppercase for concatenated quantities, i.e. a big X is a vector of x's.

First dimension of tensors represent the different samples/locations (batch
dimension).
Other dimensions are for the "dimensions" of the repsonse (or input domain).

THIS IS FOR HETEROTOPIC SAMPLING (most general form).

# TODO: Inmplement convenience methods for full sampling (all indices).

Conventions
-----------
Spatial locations will be denoted by s, capital letters for bunches.
Response indices denoted by l.
Couple of (locations, response indices) denoted by x.

"""
import numpy as np
import torch
from torch.distributions import multivariate_normal


def matern32(H, lmbda, sigma):
    """ Given a matrix of euclidean distances between pairs, compute the
    corresponding Matern 3/2 covariance matrix.

    Note that in the multivariate case, we usually set sigma to 1 and define
    the variances in the cross-covariance function.

    Parameters
    ----------
    H: (M, N) Tensor
    lmbda: Tensor
        Lengthscale parameter.
    sigma: Tensor
        Standard deviation.

    Returns
    -------
    K: (M, N) Tensor

    """
    sqrt3 = torch.sqrt(torch.Tensor([3]))
    K = sigma**2 * (1 + sqrt3/lmbda * H) * torch.exp(- sqrt3/lmbda * H)
    return K

def uniform_mixing_crosscov(L1, L2, gamma0, sigmas):
    """ Given two vectors of response indices,
    comutes the uniform mixing cross covariance (only response index part).

    Parameters
    ----------
    L1: (M) integer Tensor
    L2: (N) integer Tensor
    gamma0: Tensor
        Coupling parameter.
    sigmas: (p) Tensor
        Vector on individual standar deviation for each to the p components.

    Returns
    -------
    gamma: (M, N) Tensor

    """
    # Turn to matrices of size (M, N).
    L1mat, L2mat = torch.meshgrid(L1, L2)
    # Coupling part.
    # Have to extract the float value from gamma0 to use fill.
    gamma_mat = (torch.Tensor(L1mat.shape).fill_(gamma0.item())
            + (1- gamma0) * (L1mat == L2mat))

    # Notice the GENIUS of Pytorch: If we want A_ij to contain sigma[Aij]
    # we just do simga[A] and get the whole matrix, with the same shape as A.
    # This is beautiful.
    sigma_mat1, sigma_mat2 = sigmas[L1mat], sigmas[L2mat]

    return sigma_mat1 * sigma_mat2 * gamma_mat


class Covariance():
    """
    Covariance(factor_stationary_cov)

    Covariance module

    Parameters
    ----------
    factor_stationary_cov: function(H, L1, L2)
        Covariance function. Only allow covariances that factor into a
        stationary spatial part that only depends on the euclidean distance
        matrix H and a purely response index component. L1 and L2 are the
        index matrice.

    """
    def __init__(self, factor_stationary_cov):
        self.factor_stationary_cov = factor_stationary_cov

    def K(self, S1, S2, L1, L2):
        """ Same as above, but for vector of measurements.

        Parameters
        ----------
        S1: (M, d) Tensor
            Spatial location vector. Note if d=1, should still have two
            dimensions.
        S2: (N, d) Tensor
            Spatial location vector.
        L1: (M) Tensor
            Response indices vector.
        L2: (N) Tensor
            Response indices vector.
    
        Returns
        -------
        K: (M, N) Tensor
            Covariane matrix between the two sets of measurements.
    
        """
        # Distance matrix.
        H = torch.cdist(S1, S2, p=2)
    
        return self.factor_stationary_cov(H, L1, L2)