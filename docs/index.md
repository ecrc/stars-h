About STARS-H {#mainpage}
=============

What is STARS-H?
----------------

STARS-H is an **open-source** project, aimed at providing test matrices to
measure performance of different libraries for hierarchical computations
(including itself). Its abbreviation stands for **Software for
Testing Accuracy, Reliability and Scalability of Hierarchical computations**.
Idea behind this testing is to provide comparison of operations in different
hierarchical formats for different given hardware with a focus on clusters of
CPUs and GPUs.

Why hierarchical computations?
------------------------------

Matrices, coming from different real or synthetic applications, often can be
approximated by matrices of special block-wise low-rank structure, usually
called as hierarchical matrices. Such matrices use much less memory and enable
fast operations due to less flops required. There are several formats of such
matrices, each one with its own complexity, memory footprint and scalability.
Which one is the best for a given problem on a given hardware? This is the
question STARS-H has to answer.

Vision of STARS-H
-----------------

Role of STARS-H is to generate hierarchical matrices in the framework of HPC to
feed them to our high-performance [HiCMA](https://github.com/ecrc/hicma) or
other hierarchical libraries, becoming an H-matrix market within HPC framework.
Hierarchical matrix formats are: tile low-rank (matrix is divided into equal
tiles), HSS, HODLR, H and H^2. Many other hierarchical libraries
already can do it, but they are either not **open-source** or not intended for
HPC. STARS-H is meant to fill in this gap. Also, functionality of STARS-H is
limited to building approximations in different formats and multiplication of
matrices in such formats by dense matrices. This is due to [HiCMA](
https://github.com/ecrc/hicma) project, aimed at performing different
operations on hierarchical matrices.

Features of STARS-H
-------------------

This project is WIP, only few things are working right now.

Applications in matrix-free form:
1. Cauchy matrix,
2. Electrostatics (1/r),
3. Electrodynamics (sin(kr)/r and cos(kr)/r),
4. Random synthetic TLR matrix,
5. Spatial statistics (exponential, square exponential and matern kernels).

Operations:
1. Approximation by TLR matrix,
2. Multiplication of TLR matrix by dense matrix.

Backends:
1. MPI+OpenMP (both pure and hybrid),
2. StarPU (with and without MPI).

Low-rank engines:
1. Ordinary SVD,
2. Rank-revealing QR,
3. Randomized SVD.

Additional:
1. CG method for symmetric positive-definite (SPD) systems.

Current research
----------------

1. Matrix-free forms for other problems (e.g. like in
    [SMART](http://smart.math.purdue.edu)),
2. Approximation by hierarchical (H) and hierarchical with nested basises (H2)
    matrices in the HPC framework,
3. Adaptive cross approximation as a low-rank engine for dense tiles/blocks,
4. Run everywhere backend-wise: from pure OpenMP and MPI to task-based HPC
    libraries,
5. Run everywhere hardware-wise: clusters of CPU and/or GPU.
