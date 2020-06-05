""" Walkthrough for meeting June 05 2020.
"""
import numpy as np
import torch
from meslas.means import LinearMean
from meslas.covariance.covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import TriangularGrid
from meslas.random_fields import GRF, DiscreteGRF
from meslas.excursion import coverage_fct_fixed_location
from meslas.plotting import plot_grid_values

# ------------------------------------------------------
# END IMPORT STATEMENTS.
# ------------------------------------------------------
print("Begining Walkthrough.")



# ------------------------------------------------------
# DEFINITION OF THE MODEL
# ------------------------------------------------------
# Dimension of the response.
n_out = 2

# Spatial Covariance.
matern_cov = Matern32(lmbda=0.5, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.2, sigmas=[2.25, 2.25])
covariance = FactorCovariance(
        spatial_cov=matern_cov,
        cross_cov=cross_cov,
        n_out=n_out)

# Specify mean function, here it is a linear trend that decreases with the
# horizontal coordinate.
beta0s = np.array([5.8, 24.0])
beta1s = np.array([
        [0, -4.0],
        [0, -3.8]])
mean = LinearMean(beta0s, beta1s)

# Create the GRF.
myGRF = GRF(mean, covariance)
# ------------------------------------------------------






# ------------------------------------------------------
# DISCRETIZE EVERYTHING
# ------------------------------------------------------

# Create a regular square grid in 2 dims.
my_grid = TriangularGrid(25)
print("Working on an equilateral triangular grid with {} nodes.".format(my_grid.n_points))

# Discretize the GRF on a grid and be done with it.
# From now on we only consider locatoins on the grid.
my_discrete_grf = DiscreteGRF.from_model(myGRF, my_grid)

# ------------------------------------------------------











# ------------------------------------------------------
# Sample and plot
# ------------------------------------------------------
# Sample all components at all locations.
sample = my_discrete_grf.sample()

# Plot.
plot_grid_values(my_grid, sample)

# ------------------------------------------------------







# ------------------------------------------------------
# (CO) KRIGING
# ------------------------------------------------------


# ---------------------------------------------------------------
# From now on, we will consider the drawn sample as ground truth.
# ---------------------------------------------------------------
ground_truth = sample

# ------------------------------------------------------
# We will collect 10 datapoints by moving northwards along the middle of the
# domain and compute the (co)kriging predictor.
#
# Note that the generic kriging setup is to specify the spatial location S_y
# of the observations and to provide a list L_y specifying which response was
# observed.
#
# Here, we will observe ALL components at each spatial location (isotopic
# kriging).
# We thus use a convenience function to generate the observation vector
# corresponding to observing ALL components.
#
# Also, since we work on a grid, we use node indices instead of coordinates for
# the spatial locations.
# ------------------------------------------------------







# Measure some data on the middle of the grid.
S_y = torch.tensor([[0.1, 0.5], [0.2, 0.5], [0.3, 0.5],
        [0.4, 0.5], [0.5, 0.5], [0.6, 0.5],
        [0.7, 0.5], [0.8, 0.5], [0.9, 0.5]])

# Get the generalized location corresponding to measuring both repsonses at
# those spatial locations.
S_y_inds, L_y = my_grid.get_isotopic_generalized_location_inds(S_y, p=2)

# Get the data at those locations
y = ground_truth[S_y_inds, L_y]

# Compute kriging predictor.
cond_mean, cond_cov = my_discrete_grf.update(S_y_inds, L_y, y, noise_std=0.05)

# Plot kriging predictor.
plot_grid_values(my_grid, my_discrete_grf.mean_vec,
        my_grid.points[S_y_inds], L_y )








# ------------------------------------------------------
# NOW EXCURSIONS AND RELATED STUFF.
# ------------------------------------------------------
# Excursion threshold.
lower = torch.tensor([2.3, 22.0]).float()

# Get the real excursion set and plot it.
excursion_ground_truth = (sample.isotopic > lower).float()
plot_grid_values(my_grid, excursion_ground_truth, cmap="proba")

# Plot the real excursion set.
plot_grid_values(my_grid, excursion_ground_truth.sum(dim=1), cmap="proba")











# --------------------------------------------------------
# Now consider a new potential point and compute the EIBV.
# --------------------------------------------------------

# Say we are considering point nr 100 in the grid.
S_y_new = my_grid.points[100, :][None, :]

# Get the generalized location corresponding to measuring both repsonses at
# this spatial locations.
S_y_new_inds, L_y_new = my_grid.get_isotopic_generalized_location_inds(S_y_new, p=2)

# Compute the EIBV criterion for this potential observation location.
eibv = my_discrete_grf.eibv(S_y_new_inds, L_y_new, lower, noise_std=0.05)
print("EIBV at point {}: {}".format(S_y_new, eibv))










# Loop.
eibvs = []
for i in range(my_grid.points.shape[0]):
    print(i)
    # Consider some possible measurement loation.
    S_y_new = my_grid.points[i, :][None, :]
    
    # Get the generalized location corresponding to measuring both repsonses at
    # this spatial locations.
    S_y_new_inds, L_y_new = my_grid.get_isotopic_generalized_location_inds(S_y_new, p=2)
    
    # Compute the EIBV criterion for this potential observation location.
    eibv = my_discrete_grf.eibv(S_y_new_inds, L_y_new, lower, noise_std=0.05)
    print("EIBV for this cell {}".format(eibv.item()))
    eibvs.append(eibv)

np.save("./eibvs.npy", np.array(eibvs))
