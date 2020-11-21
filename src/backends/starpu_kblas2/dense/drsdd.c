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
#include "starsh-starpu-kblas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <kblas.h>
#include "batch_rand.h"
#include <omp.h>

void starsh_dense_dlrrsdd_starpu_kblas2_gpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int batch_size;
    int nb;
    int maxrank;
    int oversample;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    double **work;
    int lwork;
    int **iwork;
    double tol;
    starpu_codelet_unpack_args(cl_arg, &batch_size, &nb, &maxrank, &oversample,
            &tol, &cublas_handles, &kblas_handles, &kblas_states);
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffer[0]);
    double *W = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[3]);
    double *S = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int mn = maxrank+oversample;
    if(mn > nb)
        mn = nb;
    int id = starpu_worker_get_id();
    kblasHandle_t khandle = kblas_handles[id];
    cublasHandle_t cuhandle = cublas_handles[id];
    kblasRandState_t state = kblas_states[id];
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Create copy of first mn columns of D, since kblas_rsvd spoils it
    cublasDcopy(cuhandle, batch_size*nb*nb, D, 1, W, 1);
    // Run randomized SVD, get left singular vectors and singular values
    kblasDrsvd_batch_strided(khandle, nb, nb, mn, W, nb, nb*nb, S, mn, state,
            batch_size);
    double one = 1.0;
    double zero = 0.0;
    for(int bi = 0; bi < batch_size; ++bi)
        cublasDcopy(cuhandle, nb*maxrank, W+bi*nb*nb, 1, U+bi*maxrank*nb, 1);
    kblasDgemm_batch_strided(khandle, KBLAS_Trans, KBLAS_NoTrans, nb, maxrank,
            nb, one, D, nb, nb*nb, U, nb, nb*maxrank, zero, V, nb,
            maxrank*nb, batch_size);
}

void starsh_dense_dlrrsdd_starpu_kblas2_getrank(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int batch_size;
    int nb;
    int maxrank;
    int oversample;
    double tol;
    starpu_codelet_unpack_args(cl_arg, &batch_size, &nb, &maxrank, &oversample,
            &tol);
    double *tmp_U = (double *)STARPU_VECTOR_GET_PTR(buffer[0]);
    double *tmp_V = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *tmp_S = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[3]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[5]);
    int mn = maxrank+oversample;
    if(mn > nb)
        mn = nb;
    int pool_size = starpu_combined_worker_get_size();
    int pool_rank = starpu_combined_worker_get_rank();
    size_t stride = maxrank * nb;
    for(STARSH_int ibatch = pool_rank; ibatch < batch_size;
            ibatch += pool_size)
    {
        int local_rank = starsh_dense_dsvfr(mn, tmp_S+ibatch*mn, tol);
        if(local_rank >= nb/2 || local_rank > maxrank)
            rank[ibatch] = -1;
        else
        {
            double *local_U = U + ibatch*stride;
            double *local_V = V + ibatch*stride;
            double *local_tmp_U = tmp_U + ibatch*stride;
            double *local_tmp_V = tmp_V + ibatch*stride;
            cblas_dcopy(local_rank*nb, local_tmp_U, 1, local_U, 1);
            cblas_dcopy(local_rank*nb, local_tmp_V, 1, local_V, 1);
            rank[ibatch] = local_rank;
        }
    }
}

