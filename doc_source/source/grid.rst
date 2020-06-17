.. role:: hidden
   :class: hidden-section

Gridding Module
===============
All computations are performed on a discretization of the underlying space.
This module is responsible for handling this discretization.


=========================== ============================================================
Grid Module Functionalities
========================================================================================
coverage_fct                    Compute the excursion probability above a given
threshold, at a given point
compute_excursion_probs         For each cell, compute its excursion probability above the given threshold
vorobev_quantile_inds           Get cells belonging to the Vorob'ev quantile at a given level, for a given threshold
vorobev_expectation_inds        Get cells belonging to the Vorob'ev expectation
expected_excursion_measure      Expected measure of excursion set above given threshold
vorobev_deviation               Compute Vorob'ev deviaiton of a given set at a given threshold
=========================== ============================================================

Module Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: meslas.geometry.grid
   :members:
   :undoc-members:
   :show-inheritance:
