""" Test the Sensor class.
"""
import numpy as np
import torch
from meslas.means import ConstantMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid, SquareGrid
from meslas.sampling import GRF
from meslas.excursion import coverage_fct_fixed_location
from meslas.sensor import Sensor


# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.0, sigmas=[np.sqrt(1.0), np.sqrt(1.5)])

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function
mean = ConstantMean([0.0, 0.0])

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create an equilateral triangular grid in 2 dims.
# Number of respones.
my_grid = TriangularGrid(40)
print(my_grid.n_points)

# Initialize a sensor.
my_sensor = Sensor(my_grid, myGRF)

# Observe some data.
S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0],
        [0.6, 0.5]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0])
y = torch.tensor(11*[-6]).float()

my_sensor.add_data(S_y, L_y, y)

# Move to the middle of the image.
location = [0.5, 0.5]
my_sensor.set_location(location)

# Compute the excursion probabilities of the neighbors of the midpoints.
lower = torch.tensor([-1.0, -1.0]).float()
neighbors_excu_proba = my_sensor.compute_neighbors_exursion_prob(lower)
print(neighbors_excu_proba)
