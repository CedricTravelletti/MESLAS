""" Test script for meslas.geometry submodule

"""
import torch
from meslas.geometry.grid import create_square_grid, TriangularGrid
import matplotlib.pyplot as plt

print(create_square_grid(10, 4))
print(create_square_grid(10, 4).shape)

my_grid = TriangularGrid(40)

neighbors_inds = my_grid.get_neighbors(100)

plt.scatter(my_grid.points[:, 0], my_grid.points[:, 1], c='k', s=1)
plt.scatter(my_grid.points[neighbors_inds, 0], my_grid.points[neighbors_inds, 1], c='b', s=1)
plt.scatter(my_grid.points[100, 0], my_grid.points[100, 1], c='r', s=1)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
