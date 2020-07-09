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
import torch
from torch.distributions import multivariate_normal


class StationaryCovariance():
    """ Stationary covariance function.


    Parameters
    ----------
    factor_stationary_cov: function(H, L1, L2)
        Covariance function. Only allow covariances that factor into a
        stationary spatial part that only depends on the euclidean distance
        matrix H and a purely response index component. L1 and L2 are the
        index matrix.
    n_out: int
        Number of output dimensions.

    """
    def __init__(self, factor_stationary_cov):
        self.factor_stationary_cov = factor_stationary_cov
        self.n_out = n_out

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
        H = torch.cdist(S1, S2, p=2,
                compute_mode='donot_use_mm_for_euclid_dist')
    
        return self.factor_stationary_cov(H, L1, L2)

class FactorStationaryCovariance(StationaryCovariance):
    """ Convenience class for specifiying a factor model.

    Parameters
    ----------
    spatial_cov: function(H)
        Spatial covariance function.
    cross_cov: function(L1, L2)
        Cross-covariance function.
    n_out: int
        Number of output dimensions.

    """
    def __init__(self, spatial_cov, cross_cov, n_out):
        self.factor_stationary_cov = lambda H, L1, L2: spatial_cov(H) * cross_cov(L1, L2)
        self.n_out = n_out

    def __repr__(self):
        out_string = ("Factor Covariance Module:\n"
                "------------------\n"
                "Spatial part: \n"
                "\t cross-corellation parameter gamma0:\n"
                "\t individual variances sigma0s:\n")
        return out_string

class FactorCovariance():
    """ General (not necessarily stationary) separable covariance function.

    Parameters
    ----------
    spatial_cov: function(H)
        Spatial covariance function.
    cross_cov: function(L1, L2)
        Cross-covariance function.
    n_out: int
        Number of output dimensions.

    """
    def __init__(self, spatial_cov, cross_cov, n_out):
        self.spatial_cov = spatial_cov
        self.cross_cov = cross_cov
        self.n_out = n_out

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
        H = torch.cdist(S1, S2, p=2,
                compute_mode='donot_use_mm_for_euclid_dist')
    
        return self.spatial_cov(H) * self.cross_cov(S1, L1, S2, L2)
