/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/dense/dgemm.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu.h"

void starsh_dense_dgemm_starpu(void *buffer[], void *cl_arg)
//! STARPU kernel for matrix kernel.
{
    int TN1, TN2;
    int m, n, k, lda, ldb, ldc;
    double alpha, beta;
    starpu_codelet_unpack_args(cl_arg, &TN1, &TN2, &m, &n, &k, &alpha, &lda,
            &ldb, &beta, &ldc);
    double *A = (double *)STARPU_VECTOR_GET_PTR(buffer[0]);
    double *B = (double *)STARPU_VECTOR_GET_PTR(buffer[1]);
    double *C = (double *)STARPU_VECTOR_GET_PTR(buffer[2]);
    //printf("%d %d %d %f %d %d %f %d\n", m, n, k, alpha, lda, ldb, beta, ldc);
    cblas_dgemm(CblasColMajor, TN1, TN2, m, n, k, alpha, A, lda, B, ldb, beta,
            C, ldc);
}

