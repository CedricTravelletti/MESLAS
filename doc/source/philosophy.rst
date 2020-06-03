.. role:: hidden
   :class: hidden-section

Philosophy of the MESLAS package
================================
The MESLAS package (**M**\ulti-variate **E**\xcursion **S**\et **L**\earning by **A**\daptive **S**\ampling) is a toolbox for simulation and prediction of mulitivariate Gaussian random fields.

The setup of the package is the following: :math:`Z` is a :math:`p`-dimensional random
field on a domain :math:`d`-dimensional
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
    \boldsymbol{s} &= \left(s_1, ..., s_n\right)\in D^n\\
    \boldsymbol{\ell} &= \left(\ell_1, ..., \ell_n\right) \in \lbrace 1, ..., p
    \rbrace^n
   \end{align}

We will refer to :math:`n` as the \textit{dimension} of the generalized sampling
location and usually just talk of location, using the word \textit{spatial
location} when we want to specifically refer to points in :math:`D`. Also, we will
use boldface :math:`x` as a shortcut to refer to the couple :math:`\left(\boldsymbol{s},
\boldsymbol{\ell}\right)` of spatial location vector and response index vector.
The shortcut notation :math:`Z_{\boldsymbol{x}}` thus refers to the
vector

.. math::
   Z_{\boldsymbol{x}}:=\left(Z_{s_1}^{\ell_1}, ..., Z_{s_n}^{\ell_n}\right) \in \mathbb{R}^n.


Generalized Vectors
-------------------
Say we have a :math:`p`-dimensional random field :math:`Z`, which we observe at
:math:`n` different spatial locations :math:`s_1,...,s_n`. Then the most
natural way to organize the observation in a single object is to bundle them
into a :math:`n\times p` dimensional vector

.. math::
   \begin{pmatrix}
     Z^1_{s_1} & \dots & Z^p_{s_1}\\
     \vdots & & \vdots\\
     Z^1_{s_n} & \dots & Z^p_{s_n}
   \end{pmatrix}.

We will refer to such a (generalized) vector as a vector in **isotopic** form.
This form might seem the most natural one, but it is not always the most
appropriate. For example, when sampling such a field :math:`Z` at the different
locations :math:`s_1,...,s_n`, correlations between all loations and all
response indices have to be considered. It is thus easier to organise the
responses in a one dimensional vector

.. math::
   \begin{pmatrix}
     Z^1_{s_1}& \dots & Z^p_{s_1}
     & \dots & Z^1_{s_n} & \dots & Z^p_{s_n}
   \end{pmatrix}

which can then be multiplied by the appropriate covariance matrix. We call this
the **list** form.

One should note the that the way in which the reordering is performed does not
matter, as long as it is consistent throughout. We thus won't ever mention it
again.

In the same fashion, say we want to consider the correlations between a
:math:`p` dimensional field :math:`Z` at locations :math:`s_1,...,s_n` and a
:math:`q` dimensional field :math:`Y` at locations :math:`t_1,...,t_m`.

Then, for sampling purposes, the most natural way to arrange the covariance
information is in **list** form

.. math::
   \begin{pmatrix}
     Cov(Z^1_{s_1}, Y^1_{t_1}) & \dots & Cov(Z^1_{s_1}, Y^q_{t_1}) & \dots &
     Cov(Z^1_{s_1}, Y^1_{t_m}) & \dots & Cov(Z^1_{s_1}, Y^q_{t_m})\\
     \vdots & & & & & & \vdots \\
     Cov(Z^p_{s_1}, Y^1_{t_1}) & \dots & Cov(Z^p_{s_1}, Y^q_{t_1}) & \dots &
     Cov(Z^p_{s_1}, Y^1_{t_m}) & \dots & Cov(Z^p_{s_1}, Y^q_{t_m})\\
     \vdots & & & & & & \vdots \\
     Cov(Z^1_{s_n}, Y^1_{t_1}) & \dots & Cov(Z^1_{s_n}, Y^q_{t_1}) & \dots &
     Cov(Z^1_{s_n}, Y^1_{t_m}) & \dots & Cov(Z^1_{s_n}, Y^q_{t_m})\\
     \vdots & & & & & & \vdots \\
     Cov(Z^p_{s_n}, Y^1_{t_1}) & \dots & Cov(Z^p_{s_n}, Y^q_{t_1}) & \dots &
     Cov(Z^p_{s_n}, Y^1_{t_m}) & \dots & Cov(Z^p_{s_n}, Y^q_{t_m})
   \end{pmatrix}.

But when considering, for example, pointwise excursion probability, it makes
sense to have covariance matrices between couple of points. I.e., to organise
covariance information into a :math:`n\times m \time p \time q` matrix, such
that at index :math:`i,j` we have a :math:`p\times q` matrix of covariances
between spatial locations :math:`s_i` and :math:`t_j`.

What is Natural?
----------------
The isotopic form is the natural one when we want to emphasize spatiality.
Please note the way this sentence is formulated: spatiality has no prefered
role, we might want to emphasize it, but it is not any more fundamental than
the other *dimensions.*. Indeed, a multidimensional random field correlates
different responses at different locations, so there is no difference between
correlations at the same location :math:`Cov(Z^1_x, Z^2_x)` or at different
locations :math:`Cov(Z_x^1, Z_y^2)`.
