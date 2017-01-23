#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"
#include "lapacke.h"
#include "misc.h"


int starsh_blrf_new(STARSH_blrf **F, STARSH_problem *P, char symm,
        STARSH_cluster *R, STARSH_cluster *C, size_t nblocks_far, int *block_far,
        size_t nblocks_near, int *block_near, STARSH_blrf_type type)
// Initialization of structure STARSH_blrf
// Parameters:
//   problem: pointer to a structure, holding all the information about problem
//   symm: 'S' if problem and division into blocks are both symmetric, 'N'
//     otherwise.
//   R: clusterization of rows into block rows.
//   C: clusterization of columns into block columns.
//   nblocks_far: number of admissible far-field blocks.
//   block_far: array of pairs of admissible far-filed block rows and block
//     columns. block_far[2*i] is an index of block row and block_far[2*i+1]
//     is an index of block column.
//   nblocks_near: number of admissible far-field blocks.
//   block_near: array of pairs of admissible near-filed block rows and block
//     columns. block_near[2*i] is an index of block row and block_near[2*i+1]
//     is an index of block column.
//   type: type of block low-rank format. Tiled with starsh_blrf_Tiled or
//     hierarchical with starsh_blrf_H or starsh_blrf_HOLDR.
{
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARSH_ERROR("invalid value of `symm`");
        return 1;
    }
    if(R == NULL)
    {
        STARSH_ERROR("invalid value of `R`");
        return 1;
    }
    if(C == NULL)
    {
        STARSH_ERROR("invalid value of `C`");
        return 1;
    }
    if(nblocks_far > 0 && block_far == NULL)
    {
        STARSH_ERROR("invalid value of `block_far`");
        return 1;
    }
    if(nblocks_near > 0 && block_near == NULL)
    {
        STARSH_ERROR("invalid value of `block_near`");
        return 1;
    }
    int i, *size;
    size_t bi, bj;
    STARSH_MALLOC(*F, 1);
    STARSH_blrf *F2 = *F;
    F2->problem = P;
    F2->symm = symm;
    F2->block_far = NULL;
    if(nblocks_far > 0)
        F2->block_far = block_far;
    F2->block_near = NULL;
    if(nblocks_near > 0)
        F2->block_near = block_near;
    F2->nblocks_far = nblocks_far;
    F2->nblocks_near = nblocks_near;
    F2->row_cluster = R;
    int nbrows = F2->nbrows = R->nblocks;
    // Set far-field block columns for each block row in compressed format
    F2->brow_far_start = NULL;
    F2->brow_far = NULL;
    if(nblocks_far > 0)
    {
        STARSH_MALLOC(F2->brow_far_start, nbrows+1);
        STARSH_MALLOC(F2->brow_far, nblocks_far);
        STARSH_MALLOC(size, nbrows);
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
            size[block_far[2*bi]]++;
        F2->brow_far_start[0] = 0;
        for(i = 0; i < nbrows; i++)
            F2->brow_far_start[i+1] = F2->brow_far_start[i]+size[i];
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            i = block_far[2*bi];
            bj = F2->brow_far_start[i]+size[i];
            F2->brow_far[bj] = bi;//block_far[2*bi+1];
            size[i]++;
        }
        free(size);
    }
    // Set near-field block columns for each block row in compressed format
    F2->brow_near_start = NULL;
    F2->brow_near = NULL;
    if(nblocks_near > 0)
    {
        STARSH_MALLOC(F2->brow_near_start, nbrows+1);
        STARSH_MALLOC(F2->brow_near,nblocks_near);
        STARSH_MALLOC(size, nbrows);
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
            size[block_near[2*bi]]++;
        F2->brow_near_start[0] = 0;
        for(i = 0; i < nbrows; i++)
            F2->brow_near_start[i+1] = F2->brow_near_start[i]+size[i];
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = block_near[2*bi];
            bj = F2->brow_near_start[i]+size[i];
            F2->brow_near[bj] = bi;//block_near[2*bi+1];
            size[i]++;
        }
        free(size);
    }
    if(symm == 'N')
    {
        F2->col_cluster = C;
        int nbcols = F2->nbcols = C->nblocks;
        // Set far-field block rows for each block column in compressed format
        F2->bcol_far_start = NULL;
        F2->bcol_far = NULL;
        if(nblocks_far > 0)
        {
            STARSH_MALLOC(F2->bcol_far_start, nbcols+1);
            STARSH_MALLOC(F2->bcol_far,nblocks_far);
            STARSH_MALLOC(size, nbcols);
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_far; bi++)
                size[block_far[2*bi]]++;
            F2->bcol_far_start[0] = 0;
            for(i = 0; i < nbcols; i++)
                F2->bcol_far_start[i+1] = F2->bcol_far_start[i]+size[i];
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                i = block_far[2*bi];
                bj = F2->bcol_far_start[i]+size[i];
                F2->bcol_far[bj] = bi;//block_far[2*bi+1];
                size[i]++;
            }
            free(size);
        }
        // Set near-field block rows for each block column in compressed format
        F2->bcol_near_start = NULL;
        F2->bcol_near = NULL;
        if(nblocks_near > 0)
        {
            STARSH_MALLOC(F2->bcol_near_start, nbcols+1);
            STARSH_MALLOC(F2->bcol_near, nblocks_near);
            STARSH_MALLOC(size, nbcols);
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_near; bi++)
                size[block_near[2*bi]]++;
            F2->bcol_near_start[0] = 0;
            for(i = 0; i < nbcols; i++)
                F2->bcol_near_start[i+1] = F2->bcol_near_start[i]+size[i];
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_near; bi++)
            {
                i = block_near[2*bi];
                bj = F2->bcol_near_start[i]+size[i];
                F2->bcol_near[bj] = bi;//block_near[2*bi+1];
                size[i]++;
            }
            free(size);
        }
    }
    else
    {
        F2->col_cluster = R;
        F2->nbcols = R->nblocks;
        // Set far-field block rows for each block column in compressed format
        F2->bcol_far_start = F2->brow_far_start;
        F2->bcol_far = F2->brow_far;
        // Set near-field block rows for each block column in compressed format
        F2->bcol_near_start = F2->brow_near_start;
        F2->bcol_near = F2->brow_near;
    }
    F2->type = type;
    return 0;
}

int starsh_blrf_free(STARSH_blrf *F)
// Free memory, used by block low rank format (partitioning of array into
// blocks)
{
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    if(F->nblocks_far > 0)
    {
        free(F->block_far);
        free(F->brow_far_start);
        free(F->brow_far);
    }
    if(F->nblocks_near > 0)
    {
        free(F->block_near);
        free(F->brow_near_start);
        free(F->brow_near);
    }
    if(F->symm == 'N')
    {
        if(F->nblocks_far > 0)
        {
            free(F->bcol_far_start);
            free(F->bcol_far);
        }
        if(F->nblocks_near > 0)
        {
            free(F->bcol_near_start);
            free(F->bcol_near);
        }
    }
    free(F);
    return 0;
}

void starsh_blrf_swap(STARSH_blrf *F, STARSH_blrf *F2)
{
    STARSH_blrf tmp = *F;
    *F = *F2;
    *F2 = tmp;
}

void starsh_blrf_swap2(STARSH_blrf *F, STARSH_blrf *F2)
// Swap content about admissible blocks of two BLR formats (fields `problem`,
// `symm`, `row_cluster` and `col_cluster` should be equal for input `F` and
// `F2`). Useful when inplace modification of one of them is required due to
// new information (more accurate lists of far-field and near-filed blocks).
{
    void *tmp_ptr;
    size_t tmp_size;

    tmp_size = F->nblocks_far;
    F->nblocks_far = F2->nblocks_far;
    F2->nblocks_far = tmp_size;

    tmp_size = F->nblocks_near;
    F->nblocks_near = F2->nblocks_near;
    F2->nblocks_near = tmp_size;

    tmp_ptr = F->block_far;
    F->block_far = F2->block_far;
    F2->block_far = tmp_ptr;

    tmp_ptr = F->block_near;
    F->block_near = F2->block_near;
    F2->block_near = tmp_ptr;

    tmp_ptr = F->brow_far_start;
    F->brow_far_start = F2->brow_far_start;
    F2->brow_far_start = tmp_ptr;

    tmp_ptr = F->brow_far;
    F->brow_far = F2->brow_far;
    F2->brow_far = tmp_ptr;

    tmp_ptr = F->brow_near_start;
    F->brow_near_start = F2->brow_near_start;
    F2->brow_near_start = tmp_ptr;

    tmp_ptr = F->brow_near;
    F->brow_near = F2->brow_near;
    F2->brow_near = tmp_ptr;

    tmp_ptr = F->bcol_far_start;
    F->bcol_far_start = F2->bcol_far_start;
    F2->bcol_far_start = tmp_ptr;

    tmp_ptr = F->bcol_far;
    F->bcol_far = F2->bcol_far;
    F2->bcol_far = tmp_ptr;

    tmp_ptr = F->bcol_near_start;
    F->bcol_near_start = F2->bcol_near_start;
    F2->bcol_near_start = tmp_ptr;

    tmp_ptr = F->bcol_near;
    F->bcol_near = F2->bcol_near;
    F2->bcol_near = tmp_ptr;
}

int starsh_blrf_info(STARSH_blrf *F)
// Print short info on block partitioning
{
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    printf("<STARSH_blrf at %p, '%c' symmetric, %d block rows, %d "
            "block columns, %zu far-field blocks, %zu near-field blocks>\n",
            F, F->symm, F->nbrows, F->nbcols, F->nblocks_far, F->nblocks_near);
    return 0;
}

int starsh_blrf_print(STARSH_blrf *F)
// Print full info on block partitioning
{
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    int i;
    size_t j;
    printf("<STARSH_blrf at %p, '%c' symmetric, %d block rows, %d "
            "block columns, %zu far-field blocks, %zu near-field blocks>\n",
            F, F->symm, F->nbrows, F->nbcols, F->nblocks_far, F->nblocks_near);
    // Printing info about far-field blocks
    if(F->nblocks_far > 0)
    {
        for(i = 0; i < F->nbrows; i++)
        {
            if(F->brow_far_start[i+1] > F->brow_far_start[i])
                printf("Admissible far-field blocks for block row "
                        "%d: %zu", i, F->brow_far[F->brow_far_start[i]]);
            for(j = F->brow_far_start[i]+1; j < F->brow_far_start[i+1]; j++)
            {
                printf(" %zu", F->brow_far[j]);
            }
            if(F->brow_far_start[i+1] > F->brow_far_start[i])
                printf("\n");
        }
    }
    // Printing info about near-field blocks
    if(F->nblocks_near > 0)
    {
        for(i = 0; i < F->nbrows; i++)
        {
            if(F->brow_near_start[i+1] > F->brow_near_start[i])
                printf("Admissible near-field blocks for block row "
                        "%d: %zu", i, F->brow_near[F->brow_near_start[i]]);
            for(j = F->brow_near_start[i]+1; j < F->brow_near_start[i+1]; j++)
            {
                printf(" %zu", F->brow_near[j]);
            }
            if(F->brow_near_start[i+1] > F->brow_near_start[i])
                printf("\n");
        }
    }
    return 0;
}

int starsh_blrf_new_tiled(STARSH_blrf **F, STARSH_problem *P, STARSH_cluster *R,
        STARSH_cluster *C, char symm)
// Create plain division into tiles/blocks using plain cluster trees for rows
// and columns without actual pivoting
{
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    if(R == NULL)
    {
        STARSH_ERROR("invalid value of `R`");
        return 1;
    }
    if(C == NULL)
    {
        STARSH_ERROR("invalid value of `C`");
        return 1;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARSH_ERROR("invalid value of `symm`");
        return 1;
    }
    if(symm == 'S' && P->symm == 'N')
    {
        STARSH_ERROR("invalid value of `symm`");
        return 1;
    }
    if(symm == 'S' && R != C)
    {
        STARSH_ERROR("`R` and `C` should be equal");
        return 1;
    }
    int nbrows = R->nblocks, nbcols = C->nblocks;
    int i, j, *block_far;
    size_t k = 0, nblocks_far;
    if(symm == 'N')
    {
        nblocks_far = (size_t)nbrows*(size_t)nbcols;
        STARSH_MALLOC(block_far, 2*nblocks_far);
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
        nblocks_far = (size_t)nbrows*(size_t)(nbrows+1)/2;
        STARSH_MALLOC(block_far, 2*nblocks_far);
        for(i = 0; i < nbrows; i++)
            for(j = 0; j <= i; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                k++;
            }
    }
    return starsh_blrf_new(F, P, symm, R, C, nblocks_far, block_far, 0, NULL,
            STARSH_PLAIN);
}

int starsh_blrf_get_block(STARSH_blrf *F, int i, int j, int *shape, void **D)
// PLEASE CLEAN MEMORY POINTER *D AFTER USE
{
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    if(i < 0 || i >= F->nbrows)
    {
        STARSH_ERROR("invalid value of `i`");
        return 1;
    }
    if(j < 0 || j >= F->nbcols)
    {
        STARSH_ERROR("invalid value of `j`");
        return 1;
    }
    if(shape == NULL)
    {
        STARSH_ERROR("invalid value of `shape`");
        return 1;
    }
    if(D == NULL)
    {
        STARSH_ERROR("invalid value of `D`");
        return 1;
    }
    STARSH_problem *P = F->problem;
    if(P->ndim != 2)
    {
        STARSH_ERROR("only scalar kernels are supported");
        return 1;
    }
    STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
    int nrows = R->size[i], ncols = C->size[j], info;
    shape[0] = nrows;
    shape[1] = ncols;
    STARSH_MALLOC(*D, P->entry_size*(size_t)nrows*(size_t)ncols);
    info = P->kernel(nrows, ncols, R->pivot+R->start[i], C->pivot+C->start[j],
            P->row_data, P->col_data, *D);
    return info;
}

