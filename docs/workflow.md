@page workflow

Workflow of STARS-H
===================

STARS-H is meant to build hierarchical approximations of different matrices. It
supports both dense and matrix-free matrices as input. Here we show how to use
STARS-H in a matrix-free manner:

1.  Generate `physical` data, corresponding to rows of input matrix. This can
    be coordinates of particles, triangles of a trinagle mesh or something
    similar.
2.  Generate `physical` data, corresponding to columns of input matrix. It can
    be the same data, as `physical` data for rows. Note, that in this case
    matrix can be symmetric, but this is not guaranteed (i.e. for collocation
    method for integral equations).
3.  Define STARSH\_problem instance, glueing `physical` data with corresponding
    kernel. This is all, what you need to generate any element or a submatrix
    of input matrix. So, STARSH\_problem stores input matrix in a matrix-free
    form.
4.  Build clusterization of `physical` row data and `physical` column data.
    Right now only plain clusterization is supported (one, which leads to TLR
    format). This is stored in STARSH\_cluster instance.
5.  Organize block-wise low-rank format by clusterizations of rows and columns.
    This is stored in STARSH\_blrf instance.
6.  Compute approximation using function starsh_blrm_approximate(). Resulting
    block-wise low-rank matrix is stored in STARSH\_blrm instance.

For better understanding please take a look at @ref examples.
