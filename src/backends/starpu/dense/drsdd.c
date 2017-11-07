/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/dense/drsdd.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu.h"

void starsh_dense_dlrrsdd_starpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    STARSH_blrf *F;
    int maxrank;
    int oversample;
    double tol;
    starpu_codelet_unpack_args(cl_arg, &F, &maxrank, &oversample, &tol);
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    STARSH_int bi = *(STARSH_int *)STARPU_VARIABLE_GET_PTR(buffer[0]);
    int *rank = (int *)STARPU_VARIABLE_GET_PTR(buffer[1]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[3]);
    STARSH_int i = F->block_far[2*bi];
    STARSH_int j = F->block_far[2*bi+1];
    STARSH_int nrows = RC->size[i];
    STARSH_int ncols = CC->size[j];
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    double *work = D+nrows*ncols;
    int lwork = STARPU_VECTOR_GET_NX(buffer[4])-nrows*ncols;
    int *iwork = (int *)STARPU_VECTOR_GET_PTR(buffer[5]);
    kernel(nrows, ncols, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
            RD, CD, D, nrows);
    starsh_dense_dlrrsdd(nrows, ncols, D, nrows, U, nrows, V, ncols, rank,
            maxrank, oversample, tol, work, lwork, iwork);
}
