/*! @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/dense/kernel.c
 * @version 0.3.1
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu-cuda.h"

void starsh_dense_kernel_starpu_cuda_cpu(void *buffer[], void *cl_arg)
//! STARPU kernel for matrix kernel.
{
    STARSH_blrf *F;
    STARSH_int i, j;
    starpu_codelet_unpack_args(cl_arg, &F, &i, &j);
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    STARSH_int nrows = RC->size[i];
    STARSH_int ncols = CC->size[j];
    double *D = (double *)STARPU_MATRIX_GET_PTR(buffer[0]);
    kernel(nrows, ncols, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
            RD, CD, D, nrows);
}

