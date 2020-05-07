""" Test script for meslas.means

"""
import torch
from meslas.means import ConstantMean


# Array of locations.
S1 = torch.Tensor([[0, 0], [0, 1], [0, 2], [3, 0]]).float()
S2 = torch.Tensor([[0, 0], [3, 0], [5, 4]]).float()

# Corresponding response indices.
L1 = torch.Tensor([0, 0, 0, 1]).long()
L2 = torch.Tensor([0, 3, 0]).long()

# Constant mean of each component.
means = torch.Tensor([1.0, -2.0, 4.0, 33.0])

my_mean = ConstantMean(means)
print(my_mean(S1, L1))

