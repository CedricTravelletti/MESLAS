""" Mean functions for GRFs.

"""


class ConstantMean():
    """ Constant mean function.

    Parameters
    ----------
    means: (p) Tensor
        Constant mean of each of the p-components.

    """
    def __init__(self, means):
        self.means = means
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
