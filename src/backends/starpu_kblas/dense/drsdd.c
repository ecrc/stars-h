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

void starsh_dense_dlrrsdd_starpu_kblas_cpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int batch_size;
    int nb;
    int maxrank;
    int oversample;
    double tol;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    double **work;
    int lwork;
    int **iwork;
    starpu_codelet_unpack_args(cl_arg, &batch_size, &nb, &maxrank, &oversample,
            &tol, &cublas_handles, &kblas_handles, &kblas_states, &work,
            &lwork, &iwork);
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffer[0]);
    double *Dcopy = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[3]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int id = starpu_worker_get_id();
    int pool_size = starpu_combined_worker_get_size();
    int pool_rank = starpu_combined_worker_get_rank();
    for(STARSH_int bi = pool_rank; bi < batch_size; bi += pool_size)
    {
        starsh_dense_dlrrsdd(nb, nb, D + bi*nb*nb, nb, U + bi*maxrank*nb, nb,
                V + bi*maxrank*nb, nb, rank+bi, maxrank, oversample, tol,
                work[id], lwork, iwork[id]);
    }
}

void starsh_dense_dlrrsdd_starpu_kblas_gpu(void *buffer[], void *cl_arg)
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
            &tol, &cublas_handles, &kblas_handles, &kblas_states, &work,
            &lwork, &iwork);
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffer[0]);
    double *Dcopy = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[3]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int mn = maxrank+oversample;
    if(mn > nb)
        mn = nb;
    int id = starpu_worker_get_id();
    kblasHandle_t khandle = kblas_handles[id];
    cublasHandle_t cuhandle = cublas_handles[id];
    kblasRandState_t state = kblas_states[id];
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Create copy of D, since kblas_rsvd spoils it
    cublasDcopy(cuhandle, batch_size*nb*nb, D, 1, Dcopy, 1);
    // Run randomized SVD, get left singular vectors and singular values
    //*
    kblasDrsvd_batch_strided(khandle, nb, nb, mn, D, nb, nb*nb, U, mn, state,
            batch_size);
    cudaMemcpyAsync(work[id], U, mn*batch_size*sizeof(double),
            cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for(int bi = 0; bi < batch_size; ++bi)
    {
        int local_rank = starsh_dense_dsvfr(mn, work[id] + bi*mn, tol);
        if(local_rank >= nb/2 || local_rank > maxrank)
        {
            iwork[id][bi] = -1;
            //printf("RANK=-1\n");
        }
        else
        {
            double one = 1.0;
            double zero = 0.0;
            cublasDgemm(cuhandle, CUBLAS_OP_T, CUBLAS_OP_N, nb, local_rank, nb,
                    &one, Dcopy + bi*nb*nb, nb, D + bi*nb*nb, nb, &zero,
                    V + bi*maxrank*nb, nb);
            cublasDcopy(cuhandle, nb*local_rank, D + bi*nb*nb, 1,
                    U + bi*maxrank*nb, 1);
            iwork[id][bi] = local_rank;
            //printf("RANK=%d\n", local_rank);
        }
    }
    cudaMemcpyAsync(rank, iwork[id], batch_size*sizeof(int),
            cudaMemcpyHostToDevice, stream);
    //*/
}

