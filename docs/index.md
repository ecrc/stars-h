@mainpage STARS-H documentation

What is STARS-H?
====================

STARS-H is a software aimed at measuring performance of different libraries
for hierarchical computations. Abbreviation stands for @__S
oftware for Testing of Accuracy, Reliability and Scalability of Hierarchical computations.

Limitations.
============

This project is WIP, only few things are working right now. Only tile low-rank
format is supported as of now. You can approximate your matrix by TLR matrix
and multiply it by dense. Approximation is parallel due to OpenMP and StarPU.
Multiplication is only sequential as of now.

Vision of **STARS-H**
=====================

In a few years, it is intended to implement different block hierarchical
formats like H, HSS, HODLR, H2. But this is not the only goal! The goal is to
be able to measure performance with other software for supported formats like
Hlib pro.

Installation
============

Installation requires CMake, StarPU and OpenMP-supporting compiler.
