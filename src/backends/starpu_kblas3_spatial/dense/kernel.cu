/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/dense/kernel.cu
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "starpu.h"
extern "C"
{
#include <cuda_runtime.h>
#include "common.h"
#include "starsh.h"
#include "starsh-starpu-kblas.h"
#include "starsh-spatial.h"
#include <omp.h>

static __global__ void local_gpu_kernel_for_spatial(STARSH_ssdata *data,
        STARSH_int *block_far, int N, double *D, int ldD, int stride)
//! Exponential kernel for 2-dimensional spatial statistics problem on GPU
{
    int tile_i = N * block_far[2*blockIdx.x];
    int tile_j = N * block_far[2*blockIdx.x + 1];
    //printf("blockidx=%d\n", blockIdx.x);
    // Read parameters
    double beta = -data->beta;
    double noise = data->noise;
    double sigma = data->sigma;
    // Get coordinates
    STARSH_int count = data->particles.count;
    double *x, *y, *z;
    x = data->particles.point;
    y = x + count;
    //z = y + count;
    double *buffer = D + (size_t)stride*blockIdx.x;
    // Fill column-major matrix
    for(int j = threadIdx.x; j < N; j += blockDim.x)
    {
        int index_j = tile_j + j;
        double x_j = x[index_j];
        double y_j = y[index_j];
        //double z_j = z[index_j];
        for(int i = threadIdx.y; i < N; i += blockDim.y)
        {
            int index_i = tile_i + i;
            double dx = x[index_i] - x_j;
            double dy = y[index_i] - y_j;
            //double dz = z[index_i] - z_j;
            //double dist = norm3d(dx, dy, dz) / beta;
            double dist = sqrt(dx*dx + dy*dy) / beta;
            if(dist == 0)
                buffer[j*(size_t)ldD+i] = sigma + noise;
            else
                buffer[j*(size_t)ldD+i] = sigma * exp(dist);
            //printf("A(%d,%d,%d)=%f\n", index_i, index_j, j*ldD+i, buffer[j*ldD+i]);
        }
    }
}

void starsh_dense_kernel_starpu_kblas3_gpu(void *buffers[], void *cl_arg)
//! STARPU kernel for matrix kernel.
{
    //double time0 = omp_get_wtime();
    STARSH_ssdata **data_gpu;
    int batch_size;
    int N;
    int id = starpu_worker_get_id();
    starpu_codelet_unpack_args(cl_arg, &data_gpu, &N, &batch_size);
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    STARSH_int *ind = (STARSH_int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    dim3 threads(16, 16);
    cudaStream_t stream = starpu_cuda_get_local_stream();
    local_gpu_kernel_for_spatial<<<batch_size, threads, 0, stream>>>(data_gpu[id],
            ind, N, D, N, N*N);
}

} // extern "C"
