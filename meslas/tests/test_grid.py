""" Test script for meslas.grid

"""
import torch
from meslas.grid import square_grid

print(square_grid(10, 4))
print(square_grid(10, 4).shape)
