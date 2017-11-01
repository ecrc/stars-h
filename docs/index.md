@mainpage About STARS-H

### What is STARS-H? ###

STARS-H is a **high performance parallel open-source** software standing for
**Software for Testing Accuracy, Reliability and Scalability of Hierarchical 
computations**. Its core idea is to provide a hierarchical matrix 
market in order to benchmark performance of various libraries for hierarchical 
matrix compressions and computations (including itself). STARS-H
intends to provide a standard for assessing accuracy and performance
of hierarchical matrix libraries on a given hardware architecture environment.
STARS-H currently supports only the tile low-rank (TLR) data format for approximation
on shared and distributed-memory systems, using MPI, OpenMP and task-based programming
models.

### Vision of STARS-H ###

The vision of STARS-H is to design, implement and provide a community code for
hierarchical matrix generator with support of various data formats for approximation, 
including, but limited to, TLR, HSS, HODLR, H and H^2. STARS-H aspires to be 
for the low-rank approximation community what UF Sparse Matrix Collection is 
for the sparse linear algebra community, by generating hierarchical matrices 
coming from a variety of synthetic and real-world applications. Furthermore, 
extracting the performance of the underlying hardware resources (i.e., x86 and GPUs) 
is in the DNA of STARS-H, since the approximation phase can be time-consuming 
on large-scale scientific applications.

### Current Features of STARS-H ###

This project is WIP, with current features limited to:

The only supported data format is Tile Low Rank (TLR):
1.  TLR Approximation
2.  Multiplication of TLR matrix by dense matrix

Programming models:
1.  OpenMP
2.  MPI
3.  Task-based using StarPU (shared-memory x86 support only)

Applications:
1.  Synthetic TLR problems 
2.  Spatial statistics (i.e., exponential, square exponential, and 
    Matern kernels)
3. There is more here... Alex?
