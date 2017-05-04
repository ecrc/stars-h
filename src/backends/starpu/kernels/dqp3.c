#include "common.h"
#include "starsh.h"

void starsh_kernel_dqp3_starpu(void *buffer[], void *cl_arg)
//! STARPU kernel for RRQR on a tile.
{
    STARSH_blrf *F;
    int maxrank;
    int oversample;
    double tol;
    starpu_codelet_unpack_args(cl_arg, &F, &maxrank, &oversample, &tol);
    STARSH_problem *P = F->problem;
    STARSH_kernel kernel = P->kernel;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    size_t bi = *(size_t *)STARPU_VARIABLE_GET_PTR(buffer[0]);
    int *rank = (int *)STARPU_VARIABLE_GET_PTR(buffer[1]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[3]);
    int i = F->block_far[2*bi];
    int j = F->block_far[2*bi+1];
    size_t nrows = RC->size[i];
    size_t ncols = CC->size[j];
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    double *work = D+nrows*ncols;
    size_t lwork = STARPU_VECTOR_GET_NX(buffer[4])-nrows*ncols;
    int *iwork = (int *)STARPU_VECTOR_GET_PTR(buffer[5]);
    kernel(nrows, ncols, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
            RD, CD, D);
    starsh_kernel_dqp3(nrows, ncols, D, U, V, rank,
            maxrank, oversample, tol, work, lwork, iwork);
}

