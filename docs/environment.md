Environment variables {#environment}
=====================

Currently, STARS-H uses only 3 environment variables. More information about
these variables can be accessed in documentation of @ref starsh_init()
function. For improved readability, we also give some explanation here:

    STARSH_BACKEND

Select programming model (backend), possible values are: `SEQUENTIAL`,
`OPENMP`, `MPI`, `MPI_OPENMP`, `STARPU`, `MPI_STARPU`. Be careful, do not
choose programming models, disabled during compilation.

    STARSH_LRENGINE

Select low-rank approxination technique (low-rank engine), possible values are:
`SVD`, `RRQR` and `RSVD`.

    STARSH_OVERSAMPLE

Oversampling size for rank-revealing QR (RRQR) and randomized SVD (RSVD).
Default value is `10`.
