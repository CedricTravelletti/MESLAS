""" Test script for meslas.means

"""
import torch
from meslas.means import ConstantMean, LinearMean


# Array of locations.
S1 = torch.Tensor([[0, 0], [0, 1], [0, 2], [3, 0], [2,2]]).float()
S2 = torch.Tensor([[0, 0], [3, 0], [5, 4]]).float()

# Corresponding response indices.
L1 = torch.Tensor([0, 0, 0, 1, 1]).long()
L2 = torch.Tensor([0, 3, 0]).long()

# Constant mean of each component.
means = torch.tensor([1.0, -2.0, 4.0, 33.0])

my_mean = ConstantMean(means)
print(my_mean(S1, L1))

# Test the linear trend mean function.
beta0s = means
beta1s = torch.tensor([
        [1.0, 1.0],
        [2.0,2.0],
        [0.0,0.0],
        [-1.0,0.0]])

my_linear_mean = LinearMean(beta0s, beta1s)
print(my_linear_mean(S1, L1))
