""" Test script for meslas.geometry.regular_grids

"""
import torch
from meslas.geometry.regular_grids import square_grid

print(square_grid(10, 4))
print(square_grid(10, 4).shape)
