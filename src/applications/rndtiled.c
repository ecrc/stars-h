#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mkl.h>
#include "starsh.h"
#include "starsh-rndtiled.h"

static void starsh_rndtiled_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result)
{
    STARSH_rndtiled *data = row_data;
    int n = data->n;
    int nblocks = data->nblocks;
    int block_size = data->block_size;
    double *U = data->U;
    double *S = data->S;
    double *rndS = data->rndS;
    double *buffer = result;
    for(int i = 0; i < nrows; i++)
    {
        int ibrow = i/block_size;
        for(int j = 0; j < ncols; j++)
        {
            int jbcol = j/block_size;
            double res = 0;
            for(int k = 0; k < block_size; k++)
                res += U[i+k*n]*U[j+k*n]*
                    (S[k]+rndS[ibrow*block_size+jbcol*n+k]);
            buffer[j*nrows+i] = res;
        }
    }
}

int starsh_rndtiled_gen(STARSH_rndtiled **data, STARSH_kernel *kernel,
        int nblocks, int block_size, double decay, double noise)
//! Generate random tiled matrix by a special rule
/*! @param[out] data: Address to pointer to `STARSH_rndtiled` object.
 * @param[out] kernel: Interaction kernel.
 * @param[in] nblocks: Number of tiles in one dimension.
 * @param[in] block_size: Size of tile in one dimension.
 * @param[in] decay: Decay of singular values.
 * @param[in] noise: Level of noise in singular values of tiles.
 * */
{
    int n = nblocks*block_size;
    int iseed[4] = {0, 0, 0, 1};
    double *U, *S, *rndS, *tau, *work;
    int lwork = block_size;
    STARSH_MALLOC(U, n*block_size);
    STARSH_MALLOC(S, block_size);
    STARSH_MALLOC(rndS, n*nblocks);
    STARSH_MALLOC(tau, block_size);
    STARSH_MALLOC(work, lwork);
    LAPACKE_dlarnv_work(3, iseed, n*block_size, U);
    for(int i = 0; i < nblocks; i++)
    {
        LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, block_size, block_size, U, n,
                tau, work, lwork);
        LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, block_size, block_size,
                block_size, U, n, tau, work, lwork);
    }
    free(work);
    free(tau);
    LAPACKE_dlarnv_work(2, iseed, n*nblocks, rndS);
    S[0] = 1.;
    for(int i = 1; i < block_size; i++)
        S[i] = S[i-1]*decay;
    STARSH_MALLOC(*data, 1);
    (*data)->n = n;
    (*data)->nblocks = nblocks;
    (*data)->block_size = block_size;
    (*data)->U = U;
    (*data)->S = S;
    (*data)->rndS = rndS;
    *kernel = starsh_rndtiled_kernel;
    return 0;
}

int starsh_rndtiled_free(STARSH_rndtiled *data)
//! Free data
{
    free(data->U);
    free(data->S);
    free(data->rndS);
    free(data);
}
