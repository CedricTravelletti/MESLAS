""" Basic covariance functions.

"""
import torch


def Matern32(lmbda, sigma=1.0):
    """ Create a Matern32 covariance function.

    Note that in the multivariate case, we usually set sigma to 1 and define
    the variances in the cross-covariance function.

    Parameters
    ----------
    lmbda: Tensor
        Lengthscale parameter.
    sigma: Tensor
        Standard deviation.

    Returns
    -------
    function(H)
        Matern32 covariance function.Take matrix of euclidean distances as
        input.

    """
    if not torch.is_tensor(lmbda): lmbda = torch.tensor(lmbda)
    if not torch.is_tensor(sigma): sigma = torch.tensor(sigma)

    return lambda H: _matern32(H, lmbda, sigma)

def _matern32(H, lmbda, sigma):
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
