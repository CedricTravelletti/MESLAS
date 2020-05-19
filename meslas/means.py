""" Mean functions for GRFs.

"""
import torch


class ConstantMean():
    """ Constant mean function.

    Parameters
    ----------
    means: (p) array-like.
        Constant mean of each of the p-components.

    """
    def __init__(self, means):
        # Convert to tensor if not already one.
        self.means = torch.tensor(means)
        self.dim = self.means.shape[0]

    def __call__(self, S, L):
        """
        Parameters
        ----------
        S: (M, d) Tensor
            List of spatial locations.
        L: (M) Tensor
            List of response indices.
    
        Returns
        -------
        mu: (M) Tensor
            The mean of Z_{s_i} component l_i.

        """
        return self.means[L]

class LinearMean():
    """ Linear trend mean function.
    The mean at location x will be given by
    beta0 + beta1 x

    Parameters
    ----------
    betas0: (p) array-like.
        Constant mean mean term for of each of the p-components.
    betas1: (p, d) array-like.
        Linear trend matrix for each of the p-components.

    """
    def __init__(self, beta0s, beta1s):
        # Convert to tensor if not already one.
        self.beta0s = torch.Tensor(beta0s)
        self.beta1s = torch.Tensor(beta1s)
        self.dim = self.beta0s.shape[0]

    def __call__(self, S, L):
        """
        Parameters
        ----------
        S: (M, d) Tensor
            List of spatial locations.
        L: (M) Tensor
            List of response indices.
    
        Returns
        -------
        mu: (M) Tensor
            The mean of Z_{s_i} component l_i.

        """
        # The code is a bit convoluted since we need to perform dot products
        # for each row.
        return self.beta0s[L] + (self.beta1s[L, :] * S).sum(1)
