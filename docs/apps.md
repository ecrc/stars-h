Set of predefined applications {#apps}
==============================

STARS-H allows to play with different application in a matrix-free form,
avoiding big dense matrices. Matrix-free form is defined by some physical data,
representing rows, some physical data, representing columns, and function of
interaction of a subset of row physical data with a subset of column physical
data. Simply speaking, matrix elements are generated as required by
computations, and there is no need to store entire dense matrix.

Currently, full installation of STARS-H contains following applications:

- [Spatial statistics problem](@ref app-spatial) with exponential, square
    exponential and matern kernels.
- [Electrostatics problem](@ref app-electrostatics) with 1/r kernel.
- [Electrodynamics problem](@ref app-electrodynamics) with sin(kr)/r and
    cos(kr)/r kernels.
- [Random synthetic tile low-rank](@ref app-randtlr) matrix generator.
- [Cauchy](@ref app-cauchy) matrix generator.
