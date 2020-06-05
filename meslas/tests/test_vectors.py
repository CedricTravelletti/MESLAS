""" Test the generalized vector module.

"""
import torch
from meslas.vectors import GeneralizedVector, GeneralizedMatrix

vec_iso = torch.tensor([
        [1,2,3],
        [4,5,6]])
vec_list = torch.tensor([1,2,3,4,5,6])

vec = GeneralizedVector.from_list(vec_list)
vec2 = GeneralizedVector.from_isotopic(vec_iso)
