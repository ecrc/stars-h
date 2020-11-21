/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/dense/kernel.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu-kblas.h"
#include <omp.h>

void starsh_dense_kernel_starpu_kblas2_cpu(void *buffers[], void *cl_arg)
//! STARPU kernel for matrix kernel.
{
    //double time0 = omp_get_wtime();
    STARSH_blrf *F;
    int batch_size;
    starpu_codelet_unpack_args(cl_arg, &F, &batch_size);
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    STARSH_int *ind = (STARSH_int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    // This works only for equal square tiles
    STARSH_int N = RC->size[0];
    STARSH_int stride = N*N;
    int pool_size = starpu_combined_worker_get_size();
    int pool_rank = starpu_combined_worker_get_rank();
    for(STARSH_int ibatch = pool_rank; ibatch < batch_size;
            ibatch += pool_size)
    {
        int i = ind[ibatch*2];
        int j = ind[ibatch*2+1];
        kernel(N, N, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
                RD, CD, D + ibatch*stride, N);
    }
}

