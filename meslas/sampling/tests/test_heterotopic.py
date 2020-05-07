import torch
from meslas.sampling.heterotopic import matern32, uniform_mixing_crosscov, Covariance


# Array of locations.
S1 = torch.Tensor([[0, 0], [0, 1], [0, 2], [3, 0]]).float()
S2 = torch.Tensor([[0, 0], [3, 0], [5, 4]]).float()

# Corresponding response indices.
L1 = torch.Tensor([0, 0, 0, 1]).long()
L2 = torch.Tensor([0, 3, 0]).long()

# Spatial parameters.
lmbda = torch.Tensor([1.0])
sigma = torch.Tensor([1.0]) # Let cross covariance handle the variances.

# Cross covariance parameters.
gamma0 = torch.Tensor([0.6])
sigmas = torch.sqrt(torch.Tensor([0.25, 0.3, 0.4, 0.5]))

def my_matern32(H):
    return matern32(H, lmbda, sigma)
def my_crosscov(L1, L2):
    return uniform_mixing_crosscov(L1, L2, gamma0, sigmas)

def my_factor_cov(H, L1, L2):
    return my_matern32(H) * my_crosscov(L1, L2)

my_covariance = Covariance(my_factor_cov)

print(my_covariance.K(S1, S2, L1, L2))
