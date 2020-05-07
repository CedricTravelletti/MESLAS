""" Test script for meslas.grid

"""
import torch
from meslas.grid import square_grid

print(square_grid(4, 10))
print(square_grid(4, 10).shape)
