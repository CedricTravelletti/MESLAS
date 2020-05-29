.. role:: hidden
   :class: hidden-section

Philosophy of the MESLAS package
================================
The MESLAS package (**M**\ulti-variate **E**\xcursion **S**\et **L**\earning by **A**\daptive **S**\ampling) is a toolbox for simulation and prediction of mulitivariate Gaussian random fields.

The setup of the package is the following: :math:`\gp` is a :math:`\no`-dimensional random
field on a domain :math:`\nd`-dimensional
domain :math:`D`.

Our philosophy is to always specify spatial location and response indices
together. That is, one should always specify \textbf{where} and \textbf{what}.

Spatial locations are denoted by :math:`s` and response indices by :math:`\ell`. We will
use boldface (or, in the code, alternatively uppercase or plurals) to denote
vectors of such objects.

A generalized sampling location is thus entirely defined by specifying two vectors

.. math::
   :label: important
   :nowrap:

   \begin{align}
    \bm{s} &= \left(s_1, ..., s_n\right)\in D^n\\
    \bm{\ell} &= \left(\ell_1, ..., \ell_n\right) \in \lbrace 1, ..., \no
    \rbrace^n
   \end{align}

We will refer to :math:`n` as the \textit{dimension} of the generalized sampling
location and usually just talk of location, using the word \textit{spatial
location} when we want to specifically refer to points in :math:`D`. Also, we will
use boldface :math:`x` as a shortcut to refer to the couple $\left(\bm{s},
\bm{\ell}\right)$ of spatial location vector and response index vector.
The shortcut notation :math:`\gp[\bm{x}]` thus refers to the
vector

.. math::
   \gp[\bm{x}]:=\left(\gp[s_1]^{\ell_1}, ..., \gp[s_n]^{\ell_n}\right) \in \mathbb{R}^n.
