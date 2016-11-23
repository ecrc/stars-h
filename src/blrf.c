#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <string.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"
#include "lapacke.h"


STARS_BLRF *STARS_BLRF_init(STARS_Problem *problem, char symm,
        STARS_Cluster *row_cluster, STARS_Cluster *col_cluster,
        size_t nblocks_far, size_t *block_far, size_t nblocks_near,
        size_t *block_near, STARS_BLRF_Type type)
// Initialization of structure STARS_BLRF
// Parameters:
//   problem: pointer to a structure, holding all the information about problem
//   symm: 'S' if problem and division into blocks are both symmetric, 'N'
//     otherwise.
//   row_cluster: clusterization of rows into block rows.
//   col_cluster: clusterization of columns into block columns.
//   nblocks_far: number of admissible far-field blocks.
//   block_far: array of pairs of admissible far-filed block rows and block
//     columns. block_far[2*i] is an index of block row and block_far[2*i+1]
//     is an index of block column.
//   nblocks_near: number of admissible far-field blocks.
//   block_near: array of pairs of admissible near-filed block rows and block
//     columns. block_near[2*i] is an index of block row and block_near[2*i+1]
//     is an index of block column.
//   type: type of block low-rank format. Tiled with STARS_BLRF_Tiled or
//     hierarchical with STARS_BLRF_H or STARS_BLRF_HOLDR.
{
    size_t i, j, bi;
    size_t *size;
    STARS_BLRF *blrf = malloc(sizeof(*blrf));
    blrf->problem = problem;
    blrf->symm = symm;
    blrf->nblocks_far = nblocks_far;
    blrf->block_far = block_far;
    blrf->block_near = block_near;
    blrf->nblocks_near = nblocks_near;
    blrf->row_cluster = row_cluster;
    int nbrows = blrf->nbrows = row_cluster->nblocks;
    // Set far-field block columns for each block row in compressed format
    blrf->brow_far_start = malloc((nbrows+1)*sizeof(*blrf->brow_far_start));
    blrf->brow_far = malloc(nblocks_far*sizeof(*blrf->brow_far));
    size = malloc(nbrows*sizeof(*size));
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_far; bi++)
        size[block_far[2*bi]]++;
    blrf->brow_far_start[0] = 0;
    for(i = 0; i < nbrows; i++)
        blrf->brow_far_start[i+1] = blrf->brow_far_start[i]+size[i];
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_far; bi++)
    {
        i = block_far[2*bi];
        j = blrf->brow_far_start[i]+size[i];
        blrf->brow_far[j] = bi;//block_far[2*bi+1];
        size[i]++;
    }
    free(size);
    // Set near-field block columns for each block row in compressed format
    blrf->brow_near_start = malloc((nbrows+1)*sizeof(*blrf->brow_near_start));
    blrf->brow_near = malloc(nblocks_near*sizeof(*blrf->brow_near));
    size = malloc(nbrows*sizeof(*size));
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_near; bi++)
        size[block_near[2*bi]]++;
    blrf->brow_near_start[0] = 0;
    for(i = 0; i < nbrows; i++)
        blrf->brow_near_start[i+1] = blrf->brow_near_start[i]+size[i];
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_near; bi++)
    {
        i = block_near[2*bi];
        j = blrf->brow_near_start[i]+size[i];
        blrf->brow_near[j] = bi;//block_near[2*bi+1];
        size[i]++;
    }
    free(size);
    if(symm == 'N')
    {
        blrf->col_cluster = col_cluster;
        int nbcols = blrf->nbcols = col_cluster->nblocks;
        // Set far-field block rows for each block column in compressed format
        blrf->bcol_far_start = malloc((nbcols+1)*
                sizeof(*blrf->bcol_far_start));
        blrf->bcol_far = malloc(nblocks_far*sizeof(*blrf->bcol_far));
        size = malloc(nbcols*sizeof(*size));
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
            size[block_far[2*bi]]++;
        blrf->bcol_far_start[0] = 0;
        for(i = 0; i < nbcols; i++)
            blrf->bcol_far_start[i+1] = blrf->bcol_far_start[i]+size[i];
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            i = block_far[2*bi];
            j = blrf->bcol_far_start[i]+size[i];
            blrf->bcol_far[j] = bi;//block_far[2*bi+1];
            size[i]++;
        }
        free(size);
        // Set near-field block rows for each block column in compressed format
        blrf->bcol_near_start = malloc((nbcols+1)*
                sizeof(*blrf->bcol_near_start));
        blrf->bcol_near = malloc(nblocks_near*sizeof(*blrf->bcol_near));
        size = malloc(nbcols*sizeof(*size));
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
            size[block_near[2*bi]]++;
        blrf->bcol_near_start[0] = 0;
        for(i = 0; i < nbcols; i++)
            blrf->bcol_near_start[i+1] = blrf->bcol_near_start[i]+size[i];
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = block_near[2*bi];
            j = blrf->bcol_near_start[i]+size[i];
            blrf->bcol_near[j] = bi;//block_near[2*bi+1];
            size[i]++;
        }
        free(size);
    }
    else
    {
        blrf->col_cluster = row_cluster;
        blrf->nbcols = row_cluster->nblocks;
        // Set far-field block rows for each block column in compressed format
        blrf->bcol_far_start = blrf->brow_far_start;
        blrf->bcol_far = blrf->brow_far;
        // Set near-field block rows for each block column in compressed format
        blrf->bcol_near_start = blrf->brow_near_start;
        blrf->bcol_near = blrf->brow_near;
    }
    blrf->type = type;
    return blrf;
}

void STARS_BLRF_free(STARS_BLRF *blrf)
// Free memory, used by block low rank format (partitioning of array into
// blocks)
{
    if(blrf == NULL)
    {
        fprintf(stderr, "STARS_BLRF instance is NOT initialized\n");
        return;
    }
    free(blrf->brow_far_start);
    free(blrf->brow_far);
    free(blrf->brow_near_start);
    free(blrf->brow_near);
    if(blrf->symm == 'N')
    {
        free(blrf->bcol_far_start);
        free(blrf->bcol_far);
        free(blrf->bcol_near_start);
        free(blrf->bcol_near);
    }
    free(blrf);
}

void STARS_BLRF_info(STARS_BLRF *blrf)
// Print short info on block partitioning
{
    if(blrf == NULL)
    {
        fprintf(stderr, "STARS_BLRF instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRF at %p, '%c' symmetric, %zu block rows, %zu "
            "block columns, %zu far-field blocks, %zu near-field blocks>\n",
            blrf, blrf->symm, blrf->nbrows, blrf->nbcols, blrf->nblocks_far,
            blrf->nblocks_near);
}

void STARS_BLRF_print(STARS_BLRF *blrf)
// Print full info on block partitioning
{
    size_t i, j;
    if(blrf == NULL)
    {
        printf("STARS_BLRF instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRF at %p, '%c' symmetric, %zu block rows, %zu "
            "block columns, %zu far-field blocks, %zu near-field blocks>\n",
            blrf, blrf->symm, blrf->nbrows, blrf->nbcols, blrf->nblocks_far,
            blrf->nblocks_near);
    // Printing info about far-field blocks
    for(i = 0; i < blrf->nbrows; i++)
    {
        if(blrf->brow_far_start[i+1] > blrf->brow_far_start[i])
            printf("Admissible far-field block columns for block row "
                    "%zu: %zu", i, blrf->brow_far[blrf->brow_far_start[i]]);
        for(j = blrf->brow_far_start[i]+1; j < blrf->brow_far_start[i+1]; j++)
        {
            printf(" %zu", blrf->brow_far[j]);
        }
        if(blrf->brow_far_start[i+1] > blrf->brow_far_start[i])
            printf("\n");
    }
    // Printing info about near-field blocks
    for(i = 0; i < blrf->nbrows; i++)
    {
        if(blrf->brow_near_start[i+1] > blrf->brow_near_start[i])
            printf("Admissible near-field block columns for block row "
                    "%zu: %zu", i, blrf->brow_near[blrf->brow_near_start[i]]);
        for(j = blrf->brow_near_start[i]+1; j < blrf->brow_near_start[i+1];
                j++)
        {
            printf(" %zu", blrf->brow_near[j]);
        }
        if(blrf->brow_near_start[i+1] > blrf->brow_near_start[i])
            printf("\n");
    }
}

STARS_BLRF *STARS_BLRF_init_tiled(STARS_Problem *problem, STARS_Cluster
        *row_cluster, STARS_Cluster *col_cluster, char symm)
// Create plain division into tiles/blocks using plain cluster trees for rows
// and columns without actual pivoting
{
    if(symm == 'S' && problem->symm == 'N')
    {
        fprintf(stderr, "Since problem is NOT symmetric, can not proceed with "
                "symmetric flag on in STARS_BLRF_plain\n");
        exit(1);
    }
    if(symm == 'S' && row_cluster != col_cluster)
    {
        fprintf(stderr, "Since problem is symmetric, clusters should be the "
                "same (both pointers should be equal)\n");
        exit(1);
    }
    size_t nbrows = row_cluster->nblocks, nbcols = col_cluster->nblocks;
    size_t i, j, k = 0, nblocks_far, *block_far;
    if(symm == 'N')
    {
        nblocks_far = nbrows*nbcols;
        block_far = malloc(2*nblocks_far*sizeof(*block_far));
        for(i = 0; i < nbrows; i++)
            for(j = 0; j < nbcols; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                k++;
            }
    }
    else
    {
        nblocks_far = nbrows*(nbrows+1)/2;
        block_far = malloc(2*nblocks_far*sizeof(*block_far));
        for(i = 0; i < nbrows; i++)
            for(j = 0; j <= i; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                k++;
            }
    }
    return STARS_BLRF_init(problem, symm, row_cluster, col_cluster,
            nblocks_far, block_far, 0, NULL, STARS_BLRF_Tiled);
}

void STARS_BLRF_getblock(STARS_BLRF *blrf, size_t i, size_t j, size_t *shape,
        void **D)
// PLEASE CLEAN MEMORY POINTER *D AFTER USE
{
    STARS_Problem *problem = blrf->problem;
    if(problem->ndim != 2)
    {
        fprintf(stderr, "Non-scalar kernels are not supported in STARS_BLRF_"
                "getblock\n");
        exit(1);
    }
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    size_t nrows = row_cluster->size[i], ncols = col_cluster->size[j];
    shape[0] = nrows;
    shape[1] = ncols;
    *D = malloc(problem->entry_size*nrows*ncols);
    (problem->kernel)(nrows, ncols, row_cluster->pivot+row_cluster->start[i],
            col_cluster->pivot+col_cluster->start[j], problem->row_data,
            problem->col_data, *D);
}

