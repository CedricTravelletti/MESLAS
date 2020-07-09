""" Cross-covariance models.

"""
import torch
torch.set_default_dtype(torch.float32)


class UniformMixing():
    """ Create a uniform mixing cross-covariance. 

    Parameters
    ----------
    gamma0: Tensor
        Coupling parameter.
    sigmas: (p) Tensor
        Vector of individual standard deviation for each to the p components.

    Returns
    -------
    function(L1, L2)

    """
    def __init__(self, gamma0, sigmas):
        # Convert to tensor if not.
        if not torch.is_tensor(gamma0): gamma0 = torch.tensor(gamma0).float()
        if not torch.is_tensor(sigmas): sigmas = torch.tensor(sigmas).float()
        self.gamma0 = gamma0
        self.sigmas = sigmas

    def __call__(self, S1, L1, S2, L2):
        return _uniform_mixing_crosscov(L1, L2, self.gamma0, self.sigmas)

    def __repr__(self):
        out_string = ("Uniform mixing covariance:\n"
                "\t cross-corellation parameter gamma0: {}\n"
                "\t individual variances sigma0s: {}\n").format(self.gamma0, self.sigmas)
        return out_string

def _uniform_mixing_crosscov(L1, L2, gamma0, sigmas):
    """ Given two vectors of response indices,
    comutes the uniform mixing cross covariance (only response index part).


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

def _linear_trend_crosscov(S1, L1, S2, L2, gamma0, sigma0s, betas, s0):
    """ Same as above, but for non-stationary variance models.

    The model is the following:
        \sigma_i(s) = \sigma0_i + \vec{\beta}_i \cdot (s - s0)

    The genial simplicity of the implementation is this: instead of having an
    array containing the variances of each response, have an array of functions
    giving the variances of each function at some place given as argument to
    the function.
    The beauty of having functions as first-class citizens in Python has rarely
    shone so brightly.

    Parameters
    ----------
    S1: (M, d) Tensor
        Spatial location vector. Note if d=1, should still have two
        dimensions.
    L1: (M) Tensor
        Response indices vector.
    S2: (N, d) Tensor
        Spatial location vector.
    L2: (N) Tensor
        Response indices vector.
    gamma0: Tensor
        Coupling parameter.
    sigma0s: (p) Tensor
        Vector of individual standard deviation (constant part) for each to the p components.
    betas: (p, n_dim) Tensor
        List of vectors of linear trends for each of the p components.
        The final model is sigma_i(s) = sigma0s[i] + betas[i, :] * (s - s0).
    s0: (n_dim) Tensor
        Reference point in the domain for linear trend.

    Returns
    -------
    gamma: (M, N) Tensor

    """
    # Turn to matrices of size (M, N).
    L1mat, L2mat = torch.meshgrid(L1, L2)

    # Same for the spatiality. Matrices of size (M, N, n_dim).
    S1mat, _ = torch.meshgrid(S1.reshape(-1), S2[:, 0])
    _, S2mat = torch.meshgrid(S1[:, 0], S2.reshape(-1))

    S1mat = S1mat.reshape(S1.shape[0], S1.shape[1], S2.shape[0]).transpose(1,2)
    S2mat = S2mat.reshape(S1.shape[0], S2.shape[0], S2.shape[1])

    # Coupling part.
    # Have to extract the float value from gamma0 to use fill.
    gamma_mat = (torch.Tensor(L1mat.shape).fill_(gamma0.item())
            + (1- gamma0) * (L1mat == L2mat))

    # Notice the GENIUS of Pytorch: If we want A_ij to contain sigma[Aij]
    # we just do simga[A] and get the whole matrix, with the same shape as A.
    # This is beautiful.
    sigma0_mat1, sigma0_mat2 = sigma0s[L1mat], sigma0s[L2mat]

    # Fetch the spatial coefficient vectors.
    beta_mat1, beta_mat2 = betas[L1mat], betas[L2mat]

    # Perform dot product.
    # Might want to switch to tensordot, but I think the notation is a bit
    # convoluted.
    dot_mat1 = (beta_mat1 * (S1mat - s0)).sum(2)
    dot_mat2 = (beta_mat2 * (S2mat - s0)).sum(2)

    return (sigma0_mat1 + dot_mat1) * (sigma0_mat2 + dot_mat2) * gamma_mat


# TODO: Exclude possibility of negatice variances on the diagonal.
class LinearTrendCrossCov():
    """ Create a linear trend cross-covariance.

    Parameters
    ----------
    gamma0: Tensor
        Coupling parameter.
    sigma0s: (p) Tensor
        Vector of individual standard deviation (constant part) for each to the p components.
    betas: (p, n_dim) Tensor
        List of vectors of linear trends for each of the p components.
        The final model is sigma_i(s) = sigma0s[i] + betas[i, :] * s.

    Returns
    -------
    function(L1, L2)

    """
    def __init__(self, gamma0, sigma0s, betas, s0):
        # Convert to tensor if not.
        if not torch.is_tensor(gamma0): gamma0 = torch.tensor(gamma0).float()
        if not torch.is_tensor(sigma0s): sigma0s = torch.tensor(sigma0s).float()
        if not torch.is_tensor(betas): betas = torch.tensor(betas).float()
        if not torch.is_tensor(s0): s0 = torch.tensor(s0).float()
        self.gamma0 = gamma0
        self.sigma0s = sigma0s
        self.betas = betas
        self.s0 = s0

    def __call__(self, S1, L1, S2, L2):
        return _linear_trend_crosscov(S1, L1, S2, L2,
                self.gamma0, self.sigma0s, self.betas, self.s0)

    def __repr__(self):
        out_string = ("Linear trend covariance:\n"
                "\t cross-corellation parameter: gamma0 {}\n"
                "\t individual variances, constant term:  sigma0s {}\n"
                "\t individual variances, trend term: betas {}\n").format(
                "\t individual variances, reference point: s0 {}\n").format(
                        self.gamma0, self.sigma0s, self.betas, self.s0)
        return out_string

class ParabolicTrendCrossCov():
    """ Create a parabolic trend cross-covariance.

    Parameters
    ----------
    gamma0: Tensor
        Coupling parameter.
    sigma0s: (p) Tensor
        Vector of individual standard deviation (constant part) for each to the p components.
    betas: (p, n_dim, n_dim) Tensor
        Parabolic trend defined by a quadratic form for each component.
        The final model is
            sigma_i(s) = sigma0s[i] + (s - s0)**2 * betas[i, :, :] * (s - s0)**2.

    Returns
    -------
    function(L1, L2)

    """
    def __init__(self, gamma0, sigma0s, betas, s0):
        # Convert to tensor if not.
        if not torch.is_tensor(gamma0): gamma0 = torch.tensor(gamma0).float()
        if not torch.is_tensor(sigma0s): sigma0s = torch.tensor(sigma0s).float()
        if not torch.is_tensor(betas): betas = torch.tensor(betas).float()
        if not torch.is_tensor(s0): s0 = torch.tensor(s0).float()
        self.gamma0 = gamma0
        self.sigma0s = sigma0s
        self.betas = betas
        self.s0 = s0

    def __call__(self, S1, L1, S2, L2):
        return _parabolic_trend_crosscov(S1, L1, S2, L2,
                self.gamma0, self.sigma0s, self.betas, self.s0)

    def __repr__(self):
        out_string = ("Parabolic trend covariance:\n"
                "\t cross-corellation parameter: gamma0 {}\n"
                "\t individual variances, constant term:  sigma0s {}\n"
                "\t individual variances, quadratic form term: betas {}\n").format(
                "\t individual variances, reference point: s0 {}\n").format(
                        self.gamma0, self.sigma0s, self.betas, self.s0)
        return out_string


def _parabolic_trend_crosscov(S1, L1, S2, L2, gamma0, sigma0s, betas, s0):
    """ Same as above, but for non-stationary variance models.

    The model is the following:
        \sigma_i(s) = \sigma0_i + \beta_i * (s - s0)**2

    The genial simplicity of the implementation is this: instead of having an
    array containing the variances of each response, have an array of functions
    giving the variances of each function at some place given as argument to
    the function.
    The beauty of having functions as first-class citizens in Python has rarely
    shone so brightly.

    Parameters
    ----------
    S1: (M, d) Tensor
        Spatial location vector. Note if d=1, should still have two
        dimensions.
    L1: (M) Tensor
        Response indices vector.
    S2: (N, d) Tensor
        Spatial location vector.
    L2: (N) Tensor
        Response indices vector.
    gamma0: Tensor
        Coupling parameter.
    sigma0s: (p) Tensor
        Vector of individual standard deviation (constant part) for each to the p components.
    betas: (p, n_dim, n_dim) Tensor
        Parabolic trend defined by a quadratic form for each component.
        The final model is
            sigma_i(s) = sigma0s[i] + (s - s0)**2 * betas[i, :, :] * (s - s0)**2.
    s0: (n_dim) Tensor
        Reference point in the domain for linear trend.

    Returns
    -------
    gamma: (M, N) Tensor

    """
    # Turn to matrices of size (M, N).
    L1mat, L2mat = torch.meshgrid(L1, L2)

    # Same for the spatiality. Matrices of size (M, N, n_dim).
    S1mat, _ = torch.meshgrid(S1.reshape(-1), S2[:, 0])
    _, S2mat = torch.meshgrid(S1[:, 0], S2.reshape(-1))

    S1mat = S1mat.reshape(S1.shape[0], S1.shape[1], S2.shape[0]).transpose(1,2)
    S2mat = S2mat.reshape(S1.shape[0], S2.shape[0], S2.shape[1])

    # Coupling part.
    # Have to extract the float value from gamma0 to use fill.
    gamma_mat = (torch.Tensor(L1mat.shape).fill_(gamma0.item())
            + (1- gamma0) * (L1mat == L2mat))

    # Notice the GENIUS of Pytorch: If we want A_ij to contain sigma[Aij]
    # we just do simga[A] and get the whole matrix, with the same shape as A.
    # This is beautiful.
    sigma0_mat1, sigma0_mat2 = sigma0s[L1mat], sigma0s[L2mat]

    # Fetch the spatial coefficient vectors.
    beta_mat1, beta_mat2 = betas[L1mat], betas[L2mat]

    # Compute the quadratic form.
    diff1 = S1mat - s0
    diff2 = S2mat - s0
    dot_mat1 = torch.einsum("ija,ijab,ijb->ij", diff1, beta_mat1, diff1)
    dot_mat2 = torch.einsum("ija,ijab,ijb->ij", diff2, beta_mat2, diff2)

    return (sigma0_mat1 + dot_mat1) * (sigma0_mat2 + dot_mat2) * gamma_mat
