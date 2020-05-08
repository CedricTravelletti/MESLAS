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
        self.means = torch.Tensor(means)
        self.dim = means.shape[0]

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
