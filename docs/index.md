What is STARS-H? {#mainpage}
================

STARS-H is a **parallel open-source** software, aimed at measuring performance
of different libraries for hierarchical computations (including itself). Its
abbreviation stands for **Software for Testing Accuracy, Reliability and
Scalability of Hierarchical computations**. Idea behind this testing is to
provide comparison of operations in different hierarchical formats for a given
hardware with a focus on clusters of CPUs and GPUs.

Vision of STARS-H
=================

Main purpose of STARS-H is to serve as a connection between different synthetic
and real problems and block-wise low-rank matrices/tensors. Such matrix/tensor
formats are presented by, but not limited to: tile low-rank (matrix is divided
into equal tiles), HSS, HODLR, H and H^2. Many other hierarchical libraries
already can do it, but they are either not open-source or not parallel at all.
STARS-H is meant to fill in this gap. Also, functionality of STARS-H is limited
to building approximations in different formats and multiplication of matrices
in such formats by dense matrices. This is due to another ECRC project, called
HiCMA, aimed at performing different operations on hierarchical matrices.

Possibilities of STARS-H
========================

This project is WIP, only few things are working right now.

Operations:
1.  Approximation by TLR matrix,
2.  Multiplication of TLR matrix by dense matrix.

Backends:
1.  OpenMP,
2.  MPI,
3.  StarPU (without MPI support).
