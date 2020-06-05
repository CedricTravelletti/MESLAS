""" Encapsulation of the reshaping between isotopic and non-isotopic form.

"""
import torch


class GeneralizedVector():
    """ A generalized vector is used to implement the types of vectors that
    appear when working with multivariate random fields. Given *n_points*
    points in space, the value of a *n_out* dimensional random process at those
    points may be represented as a (n_points, n_out) vector. But for some
    operations (like sampling) it makes sense to expand this as a one
    dimensional vector of lenght (n_points, n_out). This is what this class is
    used for.

    The two dimensional form is called the *isotopic* form, by analogy with the
    process of measuring all components of a random field, whereas the
    one-dimensional form is called the list form.

    The underlying data structure is always a one-dimensional list of values.
    The class is indexable as GeneralizedVector[...], and the indexing will be
    performed on the isotopic from. What will be returned is a subset of the
    isotopic tensor.

    """
    def __init__(self, vals, n_points, n_out):
        self.vals = vals
        self.n_points = n_points
        self.n_out = n_out

    def __getitem__(self, inds):
        return self.isotopic.__getitem__(inds)

    def set_vals(self, vals):
        self.vals = vals

    @classmethod
    def from_list(cls, vals, n_points, n_out):
        return cls(vals, n_points, n_out)

    @classmethod
    def from_isotopic(cls, vals_iso):
        return cls(vals_iso.reshape(-1), vals_iso.shape[0], vals_iso.shape[1])

    @property
    def shape(self):
        return (self.n_points, self.n_out)

    @property
    def list(self):
        """ Returns the 1D list form of the generalized vector.

        Returns
        -------
        vals_list: (self.n_points * self.n_out) Tensor

        """
        return self.vals

    @property
    def isotopic(self):
        """ Returns the 2D isotopic form of the generalized vector.

        Returns
        -------
        vals_isotopic: (self.n_points, self.n_out) Tensor

        """
        return self.vals.reshape((self.n_points, self.n_out))

class GeneralizedMatrix():
    """ Same as GeneralizedVector, but for matrices.
    """
    def __init__(self, vals, n_points1, n_out1, n_points2, n_out2):
        self.vals = vals
        self.n_points1 = n_points1
        self.n_points2 = n_points2
        self.n_out1 = n_out2
        self.n_out2 = n_out2

    def __getitem__(self, inds):
        return self.isotopic.__getitem__(inds)

    def set_vals(self, vals):
        self.vals = vals

    @classmethod
    def from_list(cls, vals, n_points1, n_out1, n_points2, n_out2):
        return cls(vals, n_points1, n_out1, n_points2, n_out2)

    @classmethod
    def from_isotopic(cls, vals_iso):
        vals_list = vals_iso.transpose(1,2).reshape(
                (vals_iso.shape[0]*vals_iso.shape[2],vals_iso.shape[1]*vals_iso.shape[3]))
        return cls(vals_list,
                vals_iso.shape[0], vals_iso.shape[2],
                vals_iso.shape[1], vals_iso.shape[3])

    @property
    def shape(self):
        return (self.n_points1, self.n_points2, self.n_out1, self.n_out2)

    @property
    def list(self):
        """ Returns the 1D list form of the generalized vector.

        Returns
        -------
        vals_list: (self.n_points * self.n_out) Tensor

        """
        return self.vals

    @property
    def isotopic(self):
        """ Returns the 2D isotopic form of the generalized vector.

        Returns
        -------
        vals_isotopic: (self.n_points, self.n_out) Tensor

        """
        return self.vals.reshape(
                (self.n_points1, self.n_out1, self.n_points2, self.n_out2)).transpose(1, 2)
