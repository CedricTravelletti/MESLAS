import numpy as np
import torch
from meslas.means import LinearMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid, SquareGrid, get_isotopic_generalized_location_inds
from meslas.random_fields import GRF, DiscreteGRF
from meslas.excursion import coverage_fct_fixed_location
from meslas.plotting import plot_grid_values, plot_grid_probas
from meslas.sensor import DiscreteSensor


# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.1, sigma=1.0)

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

covariance = FactorCovariance(matern_cov, cross_cov, n_out=n_out)

# Create the GRF.
myGRF = GRF(mean, covariance)

# Create an equilateral triangular grid in 2 dims.
# Number of respones.
my_grid = TriangularGrid(70)
my_square_grid = SquareGrid(50, 2)

my_discrete_grf = DiscreteGRF.from_model(myGRF, my_grid)

# Observe some data.
S_y = torch.tensor([[0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
        [0.2, 0.4], [0.2, 0.5], [0.2, 0.6],
        [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0],
        [0.6, 0.5]])
L_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0])
y = torch.tensor(11*[-6]).float()

# Since we are working with a discrete GRF, we can only observa data at grid
# nodes. Hence get the ones corresponding to the measured data.
S_y_inds = my_grid.get_closest(S_y)

my_sensor = DiscreteSensor(my_grid, my_discrete_grf)

# Excursion threshold.
lower = torch.tensor([2.5, 20.0]).float()

# Sequentially reveal the data and plot current excursion probability.
excursions_list = []
visited_points_list = []
for i in range(S_y.shape[0]):
    my_sensor.update_design(S_y_inds[:i], L_y[:i], y[:i], noise_std=0.05)
    excursions = my_sensor.compute_exursion_prob(lower)
    excursions_list.append(excursions)
    visited_points = my_sensor.S_y_tot
    visited_points_list.append(visited_points)
    
    plot_grid_probas(my_grid, excursions, visited_points,
            output_filename="gif{}.png".format(i))
