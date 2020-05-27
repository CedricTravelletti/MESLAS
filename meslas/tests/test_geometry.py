""" Test script for meslas.geometry submodule

"""
import torch
from meslas.geometry.regular_grids import create_square_grid, TriangularGrid

print(create_square_grid(10, 4))
print(create_square_grid(10, 4).shape)

my_grid = TriangularGrid(3)
