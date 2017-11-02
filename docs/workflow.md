Workflow of STARS-H {#workflow}
===================

STARS-H builds hierarchical approximations of different matrices. It
supports both dense and matrix-free matrices as input. Here, we demonstrate 
how to use STARS-H in a matrix-free manner:

1.  Generate `physical` data, corresponding to rows of input matrix. This can
    be coordinates of particles, triangles of a triangle mesh or something
    similar.
2.  Generate `physical` data, corresponding to columns of input matrix. It can
    be the same data, as `physical` data for rows. Note, that in this case
    matrix can be symmetric, but this is not guaranteed (i.e., for collocation
    method for integral equations).
3.  Define @ref STARSH_problem instance, glueing `physical` data with
    corresponding kernel. This is all, what you need to generate any element or
    a submatrix of input matrix. So, @ref STARSH_problem stores input matrix
    in a matrix-free form. Operations on @ref STARSH_problem structure can be
    found in module @ref problem.
4.  Build clusterization of `physical` row data and `physical` column data.
    Right now only plain clusterization is supported (one, which leads to TLR
    format). This is stored in @ref STARSH\_cluster instance (module @ref
    cluster).
5.  Organize block-wise low-rank format by clusterizations of rows and columns.
    This is stored in @ref STARSH\_blrf instance (module @ref blrf).
6.  Init internal STARS-H parameters by a call to @ref starsh_init() function.
7.  Compute approximation using function @ref starsh_blrm_approximate().
    Resulting block-wise low-rank matrix is stored in @ref STARSH_blrm
    instance (module @ref blrm).

For further information, please take a look at @ref examples.
