""" Tests the mvnorm implementation.

We want to make sure that the cdf indeed works as expected.

"""
from mvnorm import multivariate_normal_cdf

# In the end, we want to make sure that the batch dimensions are indeed treated
# as expected. To this end, create a 3-dim Gaussian, and compute excursion on
# two points.
mean_vec = torch.tensor(
        [[1, 2, 3],
        [4, 5, 6]]).double()
cov = torch.tensor([
        [[2, 3, 3]
         [ 2, 5, 6],
         [3, 6, 10]],
        [[2, 3, 3]
         [ 2, 9, 6],
         [3, 6, 10]]
        ]).double()
lower = torch.tensor([1, 2, 1]).double()
lower = torch.tensor([4, 2, 6]).double()

cdf = multivariate_normal_cdf(
            lower=None, upper=None,
            loc=mean_vec, covariance_matrix=cov_mat)
