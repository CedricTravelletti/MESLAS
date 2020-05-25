""" Saving ideas from the first reflections about implementation.

"""


def k(x1, x2, j1, j2):
    """ Computes (cross)-covariance between a set of measures.

    TODO: We only consider stationary, so maybe get rid of x1, x2 and just
    consider h.

    Parameters
    ----------
    x1: Tensor
        Locations of first measurement.
    j1: Tensor
        Response index of first measurement.
    x2: Tensor
        Locations of second measurement.
    j2: Tensor
        Response index of second measurement.

    Returns
    -------
    Scalar Tensor
        Cross-covariance between Z^j1(x1) and Z^j2(x2).

    """


def K_isotopic(X1, X2, J_full):
    """ Compute the covariance matrices between all components of X1 and X2.
    Saves time since distance between each point are only computed once (would
    need to compute once for each response index otherwise.

    """

def mu(X, J):
    """ computes mean vector for given set of locations/response indices.

    """
    return torch.zeros(X.shape[0])

def sample(X, J, seed=0):
    """ Sample a multivariate random field at measurement points (X,J).

    Returns
    -------
    Tensor
        Shape (X.shape[0]), sampled values.

    """
    mu = mu(X, J)
    cov_mat = K(X, X, J, J)
    distribution = multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=cov_mat)
    return distribution.sample()

def sample_isotopic(X):
    """ Sample all response components at given locations.

    """
