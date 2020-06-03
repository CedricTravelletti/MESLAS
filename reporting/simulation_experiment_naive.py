""" Test the Sensor class.
"""
import numpy as np
import torch
from meslas.means import ConstantMean, LinearMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid, SquareGrid
from meslas.random_fields import GRF
from meslas.excursion import coverage_fct_fixed_location
from meslas.sensor import Sensor


# Dimension of the response.
n_out = 2

# Spatial Covariance.
lmbda = 0.5
matern_cov = Matern32(lmbda=lmbda, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.2, sigmas=[2.25, 2.25])
covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Specify mean function, here it is a linear trend that decreases with the
# horizontal coordinate.
beta0s = np.array([7.8, 24.0])
beta1s = np.array([
        [0, -7.0],
        [0, -5.0]])
mean = LinearMean(beta0s, beta1s)

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create a regular square grid in 2 dims.
# Number of repsones.
dim = 2
my_grid = TriangularGrid(80)
print("Working on equilateral triangular grid with {} nodes.".format(my_grid.points.shape[0]))

# Sample all components at all locations to build a simulated ground truth.
ground_truth = myGRF.sample_isotopic(my_grid.points)

# Plot.
from meslas.plotting_physical import plot_grid_values, plot_grid_probas
plot_grid_values(my_grid, ground_truth)
np.save("./ground_truth.npy", ground_truth.numpy())

# Initialize a sensor.
my_sensor = Sensor(my_grid, myGRF)

# Start on the middle of the map, at the southern border.
my_sensor.set_location([0.5, 0.0])

# Move north statically and collect data.
for i in np.linspace(0.0, 1, 11):
    my_sensor.set_location([i, 0.5])
    # Measure isotopically at current location.
    my_sensor.add_data(
            torch.cat([my_sensor.location,my_sensor.location], dim=0),
            torch.Tensor([0, 1]).long(),
            # Have to reshape to column vector.
            ground_truth[my_sensor.current_node_ind])

# Now that we have collected everything, estimate the excursion probability
# at every point over the design.
lower = torch.tensor([2.5, 20.0]).float()
excursion_probas = my_sensor.compute_exursion_prob(my_grid.points, lower)

plot_grid_probas(my_grid, excursion_probas,
        my_grid.points[my_sensor.visited_node_inds])

"""
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

"""
