What is STARS-H?
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

Features of STARS-H
===================

This project is WIP, with current features limited to:

The only supported format is Tile Low Rank (TLR):
1.  Approximation
2.  Multiplication of TLR matrix by dense matrix

Backends:
1.  OpenMP
2.  MPI
3.  StarPU (shared-memory support only)

Applications:
1.  Synthetic TLR problems 
2.  Spatial statistics (i.e., exponential, square exponential, and 
    Matern kernels)

TODO List
=========

1.  Add support for more matrix kernels and applications 
2.  Extend support to hardware accelerators (i.e, GPUs)
3.  Provide full StarPU support (GPUs and distributed-memory systems)
4.  Implement additional formats: HODLR/H/HSS/H^2

Installation
============

Installation requires **CMake** of version 3.2.3 at least. To build STARS-H,
follow these instructions:

1.  Get STARS-H from git repository

        git clone git@github.com:ecrc/stars-h

    or

        git clone https://github.com/ecrc/stars-h

2.  Go into STARS-H folder

        cd stars-h

3.  Get submodules

        git submodule update --init

4.  Create build directory and go there

        mkdir build && cd build

5.  Use CMake to get all the dependencies

        cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install/

6.  Build STARS-H

        make -j

7.  Run tests (optional)

        make test

8.  Build local documentation (optional)

        make docs

9.  Install STARS-H

        make install

10. Add line

        export PKG_CONFIG_PATH=/path/to/install/lib/pkgconfig:$PKG_CONFIG_PATH

    to your bashrc file.

Now you can use pkg-config executable to collect compiler and linker flags for
STARS-H.

Examples
========

The directory `examples` contains two subfolders: `problem` and `approximation`.
The sources in `problem` show how to generate problems (e.g., spatial statistics, 
minimal or dense) and how to create STARSH\_problem instance, required for every 
step of STARS-H. The examples in `approximation` are based on problem generations 
and have additional steps on approximation of corresponding matrices.

*Important notice: approximation does not require dense matrix to be stored
anywhere, only required matrix elements are computed on the fly.*
