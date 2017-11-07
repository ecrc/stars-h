/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/blrf.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-mpi.h"

int starsh_blrf_new(STARSH_blrf **format, STARSH_problem *problem, char symm,
        STARSH_cluster *row_cluster, STARSH_cluster *col_cluster,
        STARSH_int nbrows, STARSH_int nbcols, STARSH_int nblocks_far,
        STARSH_int *block_far, STARSH_int *brow_far_start,
        STARSH_int *brow_far, STARSH_int *bcol_far_start,
        STARSH_int *bcol_far, STARSH_int nblocks_far_local,
        STARSH_int *block_far_local, STARSH_int nblocks_near,
        STARSH_int *block_near, STARSH_int *brow_near_start,
        STARSH_int *brow_near, STARSH_int *bcol_near_start,
        STARSH_int *bcol_near, STARSH_int nblocks_near_local,
        STARSH_int *block_near_local, enum STARSH_BLRF_TYPE type)
//! Init @ref STARSH_blrf object.
/*! This function simply allocates memory and fills fields of @ref STARSH_blrf
 * object. Look at @ref STARSH_blrf to get more info about meaning of each
 * field.
 *
 * @param[out] format: Pointer to @ref STARSH_blrf object.
 * @param[in] problem: pointer to @ref STARSH_problem object.
 * @param[in] symm: 'S' if format is symmetric and 'N' otherwise.
 * @param[in] row_cluster, col_cluster: pointers to @ref STARSH_cluster
 *      objects, corresponding to clusterization of rows and columns.
 * @param[in] nbrows, nbcols: Number of clusters of rows and columns.
 * @param[in] nblocks_far: Number of admissible far-field blocks.
 * @param[in] block_far: Coordinates of admissible far-field blocks.
 * @param[in] brow_far_start, brow_far: List of all admissible far-field
 *      blocks, stored in CSR format.
 * @param[in] bcol_far_start, bcol_far: List of all admissible far-field
 *      blocks, stored in CSC format.
 * @param[in] nblocks_far_local, block_far_local: List of local admissible
 *      far-field blocks, stored on current MPI node.
 * @param[in] nblocks_near: Number of admissible near-field blocks.
 * @param[in] block_near: Coordinates of admissible near-field blocks.
 * @param[in] brow_near_start, brow_near: List of all admissible near-field
 *      blocks, stored in CSR format.
 * @param[in] bcol_near_start, bcol_near: List of all admissible near-field
 *      blocks, stored in CSC format.
 * @param[in] nblocks_near_local, block_near_local: List of local admissible
 *      near-field blocks, stored on current MPI node.
 * @param[in] type: Type of format. Look at @ref STARSH_BLRF_TYPE for more
 *      info.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_blrf_new_from_coo(), starsh_blrf_free().
 * @ingroup blrf
 * */
{
    STARSH_blrf *F;
    STARSH_MALLOC(F, 1);
    *format = F;
    F->problem = problem;
    F->symm = symm;
    F->row_cluster = row_cluster;
    F->col_cluster = col_cluster;
    F->nbrows = nbrows;
    F->nbcols = nbcols;
    F->nblocks_far = nblocks_far;
    F->block_far = block_far;
    F->brow_far_start = brow_far_start;
    F->brow_far = brow_far;
    F->bcol_far_start = bcol_far_start;
    F->bcol_far = bcol_far;
    F->nblocks_far_local = nblocks_far_local;
    F->block_far_local = block_far_local;
    F->nblocks_near = nblocks_near;
    F->block_near = block_near;
    F->brow_near_start = brow_near_start;
    F->brow_near = brow_near;
    F->bcol_near_start = bcol_near_start;
    F->bcol_near = bcol_near;
    F->nblocks_near_local = nblocks_near_local;
    F->block_near_local = block_near_local;
    return STARSH_SUCCESS;
}

int starsh_blrf_new_from_coo(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster,
        STARSH_int nblocks_far, STARSH_int *block_far, STARSH_int nblocks_near,
        STARSH_int *block_near, enum STARSH_BLRF_TYPE type)
//! Init @ref STARSH_blrf object by lists of admissible blocks.
/*! Allocate memory and fill ifields of @ref STARSH_blrf object. Uses lists
 * of admissible far-field and near-field blocks in COO sparse format to copy
 * it into CSR and CSC formats. Look at @ref STARSH_blrf to get more info about
 * meaning of each field.
 *
 * @param[out] format: Address of pointer to @ref STARSH_blrf object.
 * @param[in] problem: pointer to @ref STARSH_problem object.
 * @param[in] symm: 'S' if format is symmetric and 'N' otherwise.
 * @param[in] row_cluster, col_cluster: pointers to @ref STARSH_cluster
 *      objects, corresponding to clusterization of rows and columns.
 * @param[in] nblocks_far: Number of admissible far-field blocks.
 * @param[in] block_far: Coordinates of admissible far-field blocks.
 * @param[in] nblocks_near: Number of admissible near-field blocks.
 * @param[in] block_near: Coordinates of admissible near-field blocks.
 * @param[in] type: Type of format.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_blrf_new(), starsh_blrf_free().
 * @ingroup blrf
 * */
{
    if(format == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return STARSH_WRONG_PARAMETER;
    }
    if(problem == NULL)
    {
        STARSH_ERROR("Invalid value of `problem`");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARSH_ERROR("Invalid value of `symm`");
        return STARSH_WRONG_PARAMETER;
    }
    if(row_cluster == NULL)
    {
        STARSH_ERROR("Invalid value of `row_cluster`");
        return STARSH_WRONG_PARAMETER;
    }
    if(col_cluster == NULL)
    {
        STARSH_ERROR("Invalid value of `col_cluster`");
        return STARSH_WRONG_PARAMETER;
    }
    if(nblocks_far > 0 && block_far == NULL)
    {
        STARSH_ERROR("Invalid value of `block_far`");
        return STARSH_WRONG_PARAMETER;
    }
    if(nblocks_near > 0 && block_near == NULL)
    {
        STARSH_ERROR("Invalid value of `block_near`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_int i, *size;
    STARSH_int bi, bj;
    STARSH_blrf *F;
    STARSH_MALLOC(F, 1);
    *format = F;
    F->problem = problem;
    F->symm = symm;
    F->block_far = NULL;
    if(nblocks_far > 0)
        F->block_far = block_far;
    F->block_near = NULL;
    if(nblocks_near > 0)
        F->block_near = block_near;
    F->nblocks_far = nblocks_far;
    F->nblocks_near = nblocks_near;
    F->row_cluster = row_cluster;
    STARSH_int nbrows = F->nbrows = row_cluster->nblocks;
    // Set far-field block columns for each block row in compressed format
    F->brow_far_start = NULL;
    F->brow_far = NULL;
    if(nblocks_far > 0)
    {
        STARSH_MALLOC(F->brow_far_start, nbrows+1);
        STARSH_MALLOC(F->brow_far, nblocks_far);
        STARSH_MALLOC(size, nbrows);
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
            size[block_far[2*bi]]++;
        F->brow_far_start[0] = 0;
        for(i = 0; i < nbrows; i++)
            F->brow_far_start[i+1] = F->brow_far_start[i]+size[i];
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            i = block_far[2*bi];
            bj = F->brow_far_start[i]+size[i];
            F->brow_far[bj] = bi;
            size[i]++;
        }
        free(size);
    }
    // Set near-field block columns for each block row in compressed format
    F->brow_near_start = NULL;
    F->brow_near = NULL;
    if(nblocks_near > 0)
    {
        STARSH_MALLOC(F->brow_near_start, nbrows+1);
        STARSH_MALLOC(F->brow_near,nblocks_near);
        STARSH_MALLOC(size, nbrows);
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
            size[block_near[2*bi]]++;
        F->brow_near_start[0] = 0;
        for(i = 0; i < nbrows; i++)
            F->brow_near_start[i+1] = F->brow_near_start[i]+size[i];
        for(i = 0; i < nbrows; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = block_near[2*bi];
            bj = F->brow_near_start[i]+size[i];
            F->brow_near[bj] = bi;
            size[i]++;
        }
        free(size);
    }
    if(symm == 'N')
    {
        F->col_cluster = col_cluster;
        STARSH_int nbcols = F->nbcols = col_cluster->nblocks;
        // Set far-field block rows for each block column in compressed format
        F->bcol_far_start = NULL;
        F->bcol_far = NULL;
        if(nblocks_far > 0)
        {
            STARSH_MALLOC(F->bcol_far_start, nbcols+1);
            STARSH_MALLOC(F->bcol_far,nblocks_far);
            STARSH_MALLOC(size, nbcols);
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_far; bi++)
                size[block_far[2*bi+1]]++;
            F->bcol_far_start[0] = 0;
            for(i = 0; i < nbcols; i++)
                F->bcol_far_start[i+1] = F->bcol_far_start[i]+size[i];
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                i = block_far[2*bi+1];
                bj = F->bcol_far_start[i]+size[i];
                F->bcol_far[bj] = bi;
                size[i]++;
            }
            free(size);
        }
        // Set near-field block rows for each block column in compressed format
        F->bcol_near_start = NULL;
        F->bcol_near = NULL;
        if(nblocks_near > 0)
        {
            STARSH_MALLOC(F->bcol_near_start, nbcols+1);
            STARSH_MALLOC(F->bcol_near, nblocks_near);
            STARSH_MALLOC(size, nbcols);
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_near; bi++)
                size[block_near[2*bi+1]]++;
            F->bcol_near_start[0] = 0;
            for(i = 0; i < nbcols; i++)
                F->bcol_near_start[i+1] = F->bcol_near_start[i]+size[i];
            for(i = 0; i < nbcols; i++)
                size[i] = 0;
            for(bi = 0; bi < nblocks_near; bi++)
            {
                i = block_near[2*bi+1];
                bj = F->bcol_near_start[i]+size[i];
                F->bcol_near[bj] = bi;
                size[i]++;
            }
            free(size);
        }
    }
    else
    {
        F->col_cluster = row_cluster;
        F->nbcols = row_cluster->nblocks;
        // Set far-field block rows for each block column in compressed format
        F->bcol_far_start = F->brow_far_start;
        F->bcol_far = F->brow_far;
        // Set near-field block rows for each block column in compressed format
        F->bcol_near_start = F->brow_near_start;
        F->bcol_near = F->brow_near;
    }
    F->type = type;
    return STARSH_SUCCESS;
}

int starsh_blrf_new_tlr(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster)
//! TLR partitioning of problem with given plain clusters.
/*! Uses non-hierarchical clusterization of rows and columns to generate plain
 * division of problem into admissible blocks.
 *
 * @param[out] format: Address of pointer to @ref STARSH_blrf object.
 * @param[in] problem: Pointer to @ref STARSH_problem object.
 * @param[in] symm: 'S' if format is symmetric and 'N' otherwise.
 * @param[in] row_cluster, col_cluster: pointers to @ref STARSH_cluster
 *      objects, corresponding to clusterization of rows and columns.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_blrf_new(), starsh_blrf_new_from_coo().
 * @ingroup blrf
 * */
{
    if(format == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return STARSH_WRONG_PARAMETER;
    }
    if(problem == NULL)
    {
        STARSH_ERROR("Invalid value of `problem`");
        return STARSH_WRONG_PARAMETER;
    }
    if(row_cluster == NULL)
    {
        STARSH_ERROR("Invalid value of `row_cluster`");
        return STARSH_WRONG_PARAMETER;
    }
    if(col_cluster == NULL)
    {
        STARSH_ERROR("Invalid value of `col_cluster`");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARSH_ERROR("Invalid value of `symm`");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm == 'S' && problem->symm == 'N')
    {
        STARSH_ERROR("Invalid value of `symm`");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm == 'S' && row_cluster != col_cluster)
    {
        STARSH_ERROR("`row_cluster` and `col_cluster` should be equal");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_int nbrows = row_cluster->nblocks, nbcols = col_cluster->nblocks;
    STARSH_int i, j, *block_far;
    STARSH_int k = 0, nblocks_far;
    if(symm == 'N')
    {
        nblocks_far = nbrows*nbcols;
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
        nblocks_far = nbrows*(nbrows+1)/2;
        STARSH_MALLOC(block_far, 2*nblocks_far);
        for(i = 0; i < nbrows; i++)
            for(j = 0; j <= i; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                k++;
            }
    }
    return starsh_blrf_new_from_coo(format, problem, symm, row_cluster,
            col_cluster, nblocks_far, block_far, 0, NULL, STARSH_TLR);
}

void starsh_blrf_free(STARSH_blrf *format)
//! Free @ref STARSH_blrf object.
//! @ingroup blrf
{
    if(format == NULL)
        return;
    if(format->nblocks_far > 0)
    {
        free(format->block_far);
        free(format->brow_far_start);
        free(format->brow_far);
    }
    if(format->nblocks_near > 0)
    {
        free(format->block_near);
        free(format->brow_near_start);
        free(format->brow_near);
    }
    if(format->symm == 'N')
    {
        if(format->nblocks_far > 0)
        {
            free(format->bcol_far_start);
            free(format->bcol_far);
        }
        if(format->nblocks_near > 0)
        {
            free(format->bcol_near_start);
            free(format->bcol_near);
        }
    }
    free(format);
}

void starsh_blrf_info(STARSH_blrf *format)
//! Print short info about @ref STARSH_blrf object.
//! @ingroup blrf
{
    if(format == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return;
    }
    STARSH_blrf *F = format;
    printf("<STARSH_blrf at %p, '%c' symmetric, %d block rows, %d "
            "block columns, %zu far-field blocks, %zu near-field blocks>\n",
            F, F->symm, F->nbrows, F->nbcols, F->nblocks_far, F->nblocks_near);
}

void starsh_blrf_print(STARSH_blrf *format)
//! Print full info about @ref STARSH_blrf object.
//! @ingroup blrf
{
    if(format == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return;
    }
    STARSH_int i;
    STARSH_int j;
    STARSH_blrf *F = format;
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
}

int starsh_blrf_get_block(STARSH_blrf *format, STARSH_int i, STARSH_int j,
        int *shape, void **D)
//! Returns dense block on intersection of given block row and column.
/*! Allocate memory and compute kernel function for a given submatrix on
 * intersection of `i`-th row cluster with `j`-th column cluster. User have to
 * free memory after usage.
 *
 * @param[in] format: Pointer to @ref STARSH_blrf object.
 * @param[in] i, j: Indexes of row and column clusters.
 * @param[out] shape: Shape of output submatrix.
 * @param[out] D: Submatrix on intersection of `i`-th row cluster and `j`-th
 *      column cluster.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup blrf
 * */
{
    if(format == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return STARSH_WRONG_PARAMETER;
    }
    if(i < 0 || i >= format->nbrows)
    {
        STARSH_ERROR("Invalid value of `i`");
        return STARSH_WRONG_PARAMETER;
    }
    if(j < 0 || j >= format->nbcols)
    {
        STARSH_ERROR("Invalid value of `j`");
        return STARSH_WRONG_PARAMETER;
    }
    if(shape == NULL)
    {
        STARSH_ERROR("Invalid value of `shape`");
        return STARSH_WRONG_PARAMETER;
    }
    if(D == NULL)
    {
        STARSH_ERROR("Invalid value of `D`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_problem *P = format->problem;
    if(P->ndim != 2)
    {
        STARSH_ERROR("Only scalar kernels are supported");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_cluster *R = format->row_cluster, *C = format->col_cluster;
    STARSH_int nrows = R->size[i], ncols = C->size[j], info;
    shape[0] = nrows;
    shape[1] = ncols;
    STARSH_MALLOC(*D, P->entry_size*(size_t)nrows*(size_t)ncols);
    P->kernel(nrows, ncols, R->pivot+R->start[i], C->pivot+C->start[j],
            P->row_data, P->col_data, *D, nrows);
    return info;
}

#ifdef MPI
int starsh_blrf_new_from_coo_mpi(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster,
        STARSH_int nblocks_far, STARSH_int *block_far,
        STARSH_int nblocks_far_local, STARSH_int *block_far_local,
        STARSH_int nblocks_near, STARSH_int *block_near,
        STARSH_int nblocks_near_local, STARSH_int *block_near_local,
        enum STARSH_BLRF_TYPE type)
//! Create new @ref STARSH_blrf object on MPI node.
/*! Allocate memory and fill fields of @ref STARSH_blrf object on a
 * distributed-memory system (using MPI). Uses lists of admissible far-field
 * and near-field blocks in COO sparse format to copy it into CSR and CSC
 * formats. Look at @ref STARSH_blrf to get more info about meaning of each
 * field.
 *
 * @param[out] format: Address of pointer to @ref STARSH_blrf object.
 * @param[in] problem: pointer to @ref STARSH_problem object.
 * @param[in] symm: 'S' if format is symmetric and 'N' otherwise.
 * @param[in] row_cluster, col_cluster: pointers to @ref STARSH_cluster
 *      objects, corresponding to clusterization of rows and columns.
 * @param[in] nblocks_far: Number of admissible far-field blocks.
 * @param[in] block_far: Coordinates of admissible far-field blocks.
 * @param[in] nblocks_far_local: Number of admissible far-field blocks, stored
 *      locally on MPI node.
 * @param[in] block_far_local: List of admissible far-field blocks, stored
 *      locally on MPI node.
 * @param[in] nblocks_near: Number of admissible near-field blocks.
 * @param[in] block_near: Coordinates of admissible near-field blocks.
 * @param[in] nblocks_near_local: Number of admissible near-field blocks,
 *      stored locally on MPI node.
 * @param[in] block_near_local: List of admissible near-field blocks, stored
 *      locally on MPI node.
 * @param[in] type: Type of format.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_blrf_new_from_coo(), starsh_blrf_new_tlr_mpi().
 * @ingroup blrf
 * */
{
    int info;
    info = starsh_blrf_new_from_coo(format, problem, symm, row_cluster,
            col_cluster, nblocks_far, block_far, nblocks_near, block_near,
            type);
    (*format)->nblocks_far_local = nblocks_far_local;
    (*format)->block_far_local = block_far_local;
    (*format)->nblocks_near_local = nblocks_near_local;
    (*format)->block_near_local = block_near_local;
    return info;
}

int starsh_blrf_new_tlr_mpi(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster)
//! TLR partitioning on MPI nodes with 2D block cycling distribution.
/*! Uses non-hierarchical clusterization of rows and columns to generate plain
 * division of problem into admissible far-field and near-field blocks, placed
 * over MPI nodes by 2D block cycling distribution.
 *
 * @param[out] format: Address of pointer to @ref STARSH_blrf object.
 * @param[in] problem: Pointer to @ref STARSH_problem object.
 * @param[in] symm: 'S' if format is symmetric and 'N' otherwise.
 * @param[in] row_cluster, col_cluster: pointers to @ref STARSH_cluster
 *      objects, corresponding to clusterization of rows and columns.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_blrf_new_tlr().
 * @ingroup blrf
 * */
{
    if(format == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return STARSH_WRONG_PARAMETER;
    }
    if(problem == NULL)
    {
        STARSH_ERROR("Invalid value of `problem`");
        return STARSH_WRONG_PARAMETER;
    }
    if(row_cluster == NULL)
    {
        STARSH_ERROR("Invalid value of `row_cluster`");
        return STARSH_WRONG_PARAMETER;
    }
    if(col_cluster == NULL)
    {
        STARSH_ERROR("Invalid value of `col_cluster`");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARSH_ERROR("Invalid value of `symm`");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm == 'S' && problem->symm == 'N')
    {
        STARSH_ERROR("Invalid value of `symm`");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm == 'S' && row_cluster != col_cluster)
    {
        STARSH_ERROR("`row_cluster` and `col_cluster` should be equal");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_int nbrows = row_cluster->nblocks, nbcols = col_cluster->nblocks;
    STARSH_int i, j, *block_far;
    STARSH_int k = 0, nblocks_far, nblocks_far_local, li = 0;
    STARSH_int *block_far_local;
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int grid_nx = sqrt(mpi_size), grid_ny = grid_nx, grid_x, grid_y;
    if(grid_nx*grid_ny != mpi_size)
        STARSH_ERROR("MPI SIZE MUST BE SQUARE OF INTEGER!");
    grid_ny = mpi_size / grid_nx;
    grid_x = mpi_rank % grid_nx;
    grid_y = mpi_rank / grid_nx;
    //STARSH_WARNING("MPI GRID=%d x %d, MPI RANK=%d, MPI COORD=(%d, %d)", grid_nx, grid_ny, mpi_rank, grid_x, grid_y);
    if(symm == 'N')
    {
        nblocks_far = nbrows*nbcols;
        STARSH_MALLOC(block_far, 2*nblocks_far);
        //nblocks_far_local = (nblocks_far+mpi_size-1-mpi_rank)/mpi_size;
        nblocks_far_local = ((nbrows+grid_nx-1-grid_x)/grid_nx)*
                ((nbcols+grid_ny-1-grid_y)/grid_ny);
        STARSH_MALLOC(block_far_local, nblocks_far_local);
        for(i = 0; i < nbrows; i++)
            for(j = 0; j < nbcols; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                if(i % grid_nx == grid_x && j % grid_ny == grid_y)
                //if(k % mpi_size == mpi_rank)
                {
                    block_far_local[li] = k;
                    li++;
                }
                k++;
            }
        if(li != nblocks_far_local)
            STARSH_ERROR("WRONG COUNT FOR LOCAL BLOCKS");
    }
    else
    {
        nblocks_far = nbrows*(nbrows+1)/2;
        STARSH_MALLOC(block_far, 2*nblocks_far);
        nblocks_far_local = (nbrows+grid_nx-1-grid_x)/grid_nx-1;
        nblocks_far_local = nblocks_far_local*(nblocks_far_local+1)/2;
        if(grid_x >= grid_y)
        {
            nblocks_far_local += nbrows/grid_nx;
            if(nbrows % grid_nx > grid_x && nbcols % grid_ny > grid_y)
                nblocks_far_local += 1;
        }
        //nblocks_far_local = (nblocks_far+mpi_size-1-mpi_rank)/mpi_size;
        //STARSH_WARNING("(%d, %d): LOCAL BLOCKS=%zu", grid_x, grid_y, nblocks_far_local);
        STARSH_MALLOC(block_far_local, nblocks_far_local);
        for(i = 0; i < nbrows; i++)
            for(j = 0; j <= i; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                if(i % grid_nx == grid_x && j % grid_ny == grid_y)
                //if(k % mpi_size == mpi_rank)
                {
                    block_far_local[li] = k;
                    li++;
                    if(li > nblocks_far_local)
                    {
                        STARSH_ERROR("WRONG COUNG FOR LOCAL BLOCKS 2");
                    }
                }
                k++;
            }
        if(li != nblocks_far_local)
            STARSH_ERROR("WRONG COUNT FOR LOCAL BLOCKS");
    }
    return starsh_blrf_new_from_coo_mpi(format, problem, symm, row_cluster,
            col_cluster, nblocks_far, block_far, nblocks_far_local,
            block_far_local, 0, NULL, 0, NULL, STARSH_TLR);
}
#endif // MPI
