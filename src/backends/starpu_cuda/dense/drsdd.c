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
#include "starsh-starpu-cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

void starsh_dense_dlrrsdd_starpu_cuda_cpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int maxrank;
    int oversample;
    double tol;
    cublasHandle_t *cublas_handles;
    cusolverDnHandle_t *cusolver_handles;
    curandGenerator_t *curand_handles;
    int **devinfo;
    double *singular_values;
    starpu_codelet_unpack_args(cl_arg, &maxrank, &oversample, &tol,
            &cublas_handles, &cusolver_handles, &curand_handles, &devinfo,
            &singular_values);
    //printf("CODELET: %p, %p, %p\n", cublas_handles, cusolver_handles,
    //        singular_values);
    double *D = (double *)STARPU_MATRIX_GET_PTR(buffer[0]);
    int nrows = STARPU_MATRIX_GET_NX(buffer[0]);
    int ncols = STARPU_MATRIX_GET_NY(buffer[0]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[3]);
    double *work = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int lwork = STARPU_VECTOR_GET_NX(buffer[4]);
    int *iwork = (int *)STARPU_VECTOR_GET_PTR(buffer[5]);
    starsh_dense_dlrrsdd(nrows, ncols, D, nrows, U, nrows, V, ncols, rank,
            maxrank, oversample, tol, work, lwork, iwork);
}

void starsh_dense_dlrrsdd_starpu_cuda_gpu(void *buffer[], void *cl_arg)
//! STARPU kernel for 1-way randomized SVD on a tile.
{
    int maxrank;
    int oversample;
    double tol;
    cublasHandle_t *cublas_handles;
    cusolverDnHandle_t *cusolver_handles;
    curandGenerator_t *curand_handles;
    int **devinfo;
    double *singular_values;
    starpu_codelet_unpack_args(cl_arg, &maxrank, &oversample, &tol,
            &cublas_handles, &cusolver_handles, &curand_handles, &devinfo,
            &singular_values);
    double *D = (double *)STARPU_MATRIX_GET_PTR(buffer[0]);
    int nrows = STARPU_MATRIX_GET_NX(buffer[0]);
    int ncols = STARPU_MATRIX_GET_NY(buffer[0]);
    double *U = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *V = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    int *rank = (int *)STARPU_VECTOR_GET_PTR(buffer[3]);
    double *work = (double *)STARPU_VECTOR_GET_PTR(buffer[4]);
    int lwork = STARPU_VECTOR_GET_NX(buffer[4]);
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    if(mn2 > mn)
        mn2 = mn;
    int id = starpu_worker_get_id();
    cusolverDnHandle_t cusolverhandle = cusolver_handles[id];
    cublasHandle_t cuhandle = cublas_handles[id];
    curandGenerator_t curandhandle = curand_handles[id];
    int *mydevinfo  = devinfo[id];
    double *host_S = singular_values+id*(maxrank+oversample);
    double *device_X = work; // ncols-by-mn2-by random matrix
    double *device_Q = device_X+ncols*mn2; // nrows-by-mn2 matrix
    double *device_tau = device_Q+nrows*mn2; // mn2 elements
    double *device_S = device_tau;
    double *device_U = device_tau+mn2; // ncols-by-mn2 matrix
    double *device_V = device_U+ncols*mn2; // mn2-by-mn2 matrix
    double *device_rwork = device_V+mn2*mn2;
    double *device_work = device_rwork+mn2;
    lwork -= (2*ncols+nrows+2*mn2+1)*mn2;
    //printf("lwork=%d\n", lwork);
    double one = 1.0;
    double zero = 0.0;
    cusolverStatus_t status;
    cublasStatus_t status2;
    curandGenerateNormalDouble(curandhandle, device_X, ncols*mn2, zero, one);
    status2 = cublasDgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_N, nrows, mn2, ncols,
            &one, D, nrows, device_X, ncols, &zero, device_Q, nrows);
    if(status2)
    {
        printf("STATUS GEMM=%d\n", status2);
    }
    cudaMemcpy(host_S, device_Q, sizeof(*device_Q), cudaMemcpyDeviceToHost);
    status = cusolverDnDgeqrf(cusolverhandle, nrows, mn2, device_Q, nrows,
            device_tau, device_work, lwork, mydevinfo);
    if(status)
    {
        printf("STATUS GEQRF=%d\n", status);
    }
    status = cusolverDnDorgqr(cusolverhandle, nrows, mn2, mn2, device_Q, nrows,
            device_tau, device_work, lwork, mydevinfo);
    if(status)
    {
        printf("STATUS ORGQR=%d\n", status);
    }
    cublasDgemm(cuhandle, CUBLAS_OP_T, CUBLAS_OP_N, ncols, mn2, nrows, &one, D,
            nrows, device_Q, nrows, &zero, device_X, ncols);
    status = cusolverDnDgesvd(cusolverhandle, 'S', 'S', ncols, mn2, device_X,
            ncols, device_S, device_U, ncols, device_V, mn2, device_work,
            lwork, device_rwork, mydevinfo);
    if(status)
    {
        printf("STATUS GESVD=%d\n", status);
    }
    cudaMemcpy(host_S, device_S, mn2*sizeof(*host_S), cudaMemcpyDeviceToHost);
    //printf("SV:");
    //for(int i = 0; i < 5; i++)
    //    printf(" %f", host_S[i]);
    //printf("\n");
    // Get rank, corresponding to given error tolerance
    int local_rank = starsh_dense_dsvfr(mn2, host_S, tol);
    if(local_rank < mn/2 && local_rank <= maxrank)
    {
        // Compute right factor of low-rank approximation, using given left
        // singular vectors
        cublasDgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T, nrows, local_rank,
                mn2, &one, device_Q, nrows, device_V, mn2, &zero, U, nrows);
        cublasDcopy(cuhandle, ncols*local_rank, device_U, 1, V, 1);
        for(int i = 0; i < local_rank; i++)
        {
            cublasDscal(cuhandle, ncols, &host_S[i], V+i*ncols, 1);
        }
    }
    else
        local_rank = -1;
    cudaError_t err;
    // Write new rank back into device memory
    err = cudaMemcpy(rank, &local_rank, sizeof(local_rank),
            cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        printf("ERROR IN CUDAMEMCPY\n");
}

