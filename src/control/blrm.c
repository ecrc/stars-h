/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/blrm.c
 * @version 0.1.1
 * @author Aleksandr Mikhalev
 * @date 2018-11-06
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-mpi.h"

int starsh_blrm_new(STARSH_blrm **matrix, STARSH_blrf *format, int *far_rank,
        Array **far_U, Array **far_V, int onfly, Array **near_D, void *alloc_U,
        void *alloc_V, void *alloc_D, char alloc_type)
//! Init @ref STARSH_blrm object.
/*! 
 * @param[out] matrix: Address of pointer to @ref STARSH_blrm object.
 * @param[in] format: Pointer to @ref STARSH_blrf object.
 * @param[in] far_rank: Array of ranks of far-field blocks.
 * @param[in] far_U: Array of low-rank factors `U`.
 * @param[in] far_V: Array of low-rank factors `V`.
 * @param[in] onfly: Whether not to store dense blocks.
 * @param[in] near_D: Array of dense near-field blocks.
 * @param[in] alloc_U: Pointer to big buffer for all `far_U`.
 * @param[in] alloc_V: Pointer to big buffer for all `far_V`.
 * @param[in] alloc_D: Pointer to big buffer for all `near_D`.
 * @param[in] alloc_type: Type of memory allocation. `1` if big buffers
 *     are used.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup blrm
 * */
{
    if(matrix == NULL)
    {
        STARSH_ERROR("Invalid value of `matrix`");
        return STARSH_WRONG_PARAMETER;
    }
    if(format == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_blrf *F = format;
    if(far_rank == NULL && F->nblocks_far > 0)
    {
        STARSH_ERROR("Invalid value of `far_rank`");
        return STARSH_WRONG_PARAMETER;
    }
    if(far_U == NULL && F->nblocks_far > 0)
    {
        STARSH_ERROR("Invalid value of `far_U`");
        return STARSH_WRONG_PARAMETER;
    }
    if(far_V == NULL && F->nblocks_far > 0)
    {
        STARSH_ERROR("Invalid value of `far_V`");
        return STARSH_WRONG_PARAMETER;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARSH_ERROR("Invalid value of `onfly`");
        return STARSH_WRONG_PARAMETER;
    }
    if(near_D == NULL && F->nblocks_near > 0 && onfly == 0)
    {
        STARSH_ERROR("Invalid value of `near_D`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_type != '1' && alloc_type != '2')
    {
        STARSH_ERROR("Invalid value of `alloc_type`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_U == NULL && F->nblocks_far > 0 && alloc_type == '1')
    {
        STARSH_ERROR("Invalid value of `alloc_U`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_V == NULL && F->nblocks_far > 0 && alloc_type == '1')
    {
        STARSH_ERROR("Invalid value of `alloc_V`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_D == NULL && F->nblocks_near > 0 && alloc_type == '1' &&
            onfly == 0)
    {
        STARSH_ERROR("Invalid value of `alloc_D`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_blrm *M;
    STARSH_MALLOC(M, 1);
    *matrix = M;
    M->format = F;
    M->far_rank = far_rank;
    M->far_U = far_U;
    M->far_V = far_V;
    M->onfly = onfly;
    M->near_D = near_D;
    M->alloc_U = alloc_U;
    M->alloc_V = alloc_V;
    M->alloc_D = alloc_D;
    M->alloc_type = alloc_type;
    STARSH_int bi, data_size = 0, size = 0;
    size += sizeof(*M);
    size += F->nblocks_far*(sizeof(*far_rank)+sizeof(*far_U)+sizeof(*far_V));
    for(bi = 0; bi < F->nblocks_far; bi++)
    {
        size += far_U[bi]->nbytes+far_V[bi]->nbytes;
        data_size += far_U[bi]->data_nbytes+far_V[bi]->data_nbytes;
    }
    if(onfly == 0)
    {
        size += F->nblocks_near*sizeof(*near_D);
        for(bi = 0; bi < F->nblocks_near; bi++)
        {
            size += near_D[bi]->nbytes;
            data_size += near_D[bi]->data_nbytes;
        }
    }
    M->nbytes = size;
    M->data_nbytes = data_size;
    return STARSH_SUCCESS;
}

void starsh_blrm_free(STARSH_blrm *matrix)
//! Free memory of a non-nested block low-rank matrix.
//! @ingroup blrm
{
    STARSH_blrm *M = matrix;
    if(M == NULL)
        return;
    STARSH_blrf *F = M->format;
    STARSH_int bi;
    int info;
    if(F->nblocks_far > 0)
    {
        if(M->alloc_type == '1')
        {
            free(M->alloc_U);
            free(M->alloc_V);
            for(bi = 0; bi < F->nblocks_far; bi++)
            {
                M->far_U[bi]->data = NULL;
                array_free(M->far_U[bi]);
                M->far_V[bi]->data = NULL;
                array_free(M->far_V[bi]);
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_far; bi++)
            {
                array_free(M->far_U[bi]);
                array_free(M->far_V[bi]);
            }
        }
        free(M->far_rank);
        free(M->far_U);
        free(M->far_V);
    }
    if(F->nblocks_near > 0 && M->onfly == 0)
    {
        if(M->alloc_type == '1')
        {
            free(M->alloc_D);
            for(bi = 0; bi < F->nblocks_near; bi++)
            {
                M->near_D[bi]->data = NULL;
                array_free(M->near_D[bi]);
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_near; bi++)
            {
                array_free(M->near_D[bi]);
            }
        }
        free(M->near_D);
    }
    free(M);
}

void starsh_blrm_info(STARSH_blrm *matrix)
//! Print short info on non-nested block low-rank matrix.
//! @ingroup blrm
{
    STARSH_blrm *M = matrix;
    if(M == NULL)
        return;
    printf("<STARSH_blrm at %p, %d onfly, allocation type '%c', %f MB memory "
            "footprint>\n", M, M->onfly, M->alloc_type, M->nbytes/1024./1024.);
}

int starsh_blrm_get_block(STARSH_blrm *matrix, STARSH_int i, STARSH_int j,
        int *shape, int *rank, void **U, void **V, void **D)
//! Get shape, rank and low-rank factors or dense representation of a block.
/*! If block is admissible and low-rank, then its low-rank factors are returned
 * and `D` is NULL (since dense block is not stored). If block is admissible
 * and not low-rank, then its dense version is returned and `U` and `V` are
 * NULL. If block is NOT admissible, then it is computed and returned as dense.
 *
 * @param[in] matrix: Pointer to @ref STARSH_blrm object.
 * @param[in] i: Index of block row.
 * @param[in] j: Index of block column.
 * @param[out] shape: Shape of corresponding block.
 * @param[out] rank: Rank of corresponding block.
 * @param[out] U: Low-rank factor U.
 * @param[out] V: Low-rank factor V.
 * @param[out] D: Dense block.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup blrm
 * */
{
    STARSH_blrm *M = matrix;
    if(M == NULL)
    {
        STARSH_ERROR("Invalid value of `matrix`");
        return STARSH_WRONG_PARAMETER;
    }
    if(shape == NULL)
    {
        STARSH_ERROR("Invalid value of `shape`");
        return STARSH_WRONG_PARAMETER;
    }
    if(rank == NULL)
    {
        STARSH_ERROR("Invalid value of `rank`");
        return STARSH_WRONG_PARAMETER;
    }
    if(U == NULL)
    {
        STARSH_ERROR("Invalid value of `U`");
        return STARSH_WRONG_PARAMETER;
    }
    if(V == NULL)
    {
        STARSH_ERROR("Invalid value of `V`");
        return STARSH_WRONG_PARAMETER;
    }
    if(D == NULL)
    {
        STARSH_ERROR("Invalid value of `D`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_blrf *F = M->format;
    if(i < 0 || i >= F->nbrows)
    {
        STARSH_ERROR("Invalid value of `i`");
        return STARSH_WRONG_PARAMETER;
    }
    if(j < 0 || j >= F->nbcols)
    {
        STARSH_ERROR("Invalid value of `j`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_problem *P = F->problem;
    if(P->ndim != 2)
    {
        STARSH_ERROR("Only scalar kernels are supported");
        return STARSH_WRONG_PARAMETER;
    }
    int onfly = M->onfly;
    STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
    STARSH_int nrows = R->size[i], ncols = C->size[j], info = 0;
    shape[0] = nrows;
    shape[1] = ncols;
    *rank = nrows < ncols ? nrows : ncols;
    *U = NULL;
    *V = NULL;
    *D = NULL;
    STARSH_int bi = -1, k;
    if(F->nblocks_far > 0)
    {
        k = F->brow_far_start[i];
        while(k < F->brow_far_start[i+1])
        {
            if(F->block_far[2*F->brow_far[k]+1] == j)
            {
                bi = k;
                break;
            }
            k++;
        }
        if(bi != -1)
        {
            *rank = M->far_rank[bi];
            *U = M->far_U[bi]->data;
            *V = M->far_V[bi]->data;
            return info;
        }
    }
    if(F->nblocks_near > 0)
    {
        k = F->brow_near_start[i];
        while(k < F->brow_near_start[i+1])
        {
            if(F->block_near[2*F->brow_near[k]+1] == j)
            {
                bi = k;
                break;
            }
            k++;
        }
        if(bi != -1)
        {
            if(onfly == 0)
                *D = M->near_D[bi]->data;
            else
                info = starsh_blrf_get_block(F, i, j, shape, D);
            return info;
        }
    }
    STARSH_WARNING("Required block (%d, %d) is not admissible!\n", i, j);
    info = starsh_blrf_get_block(F, i, j, shape, D);
    return info;
}

#ifdef MPI
int starsh_blrm_new_mpi(STARSH_blrm **matrix, STARSH_blrf *format,
        int *far_rank, Array **far_U, Array **far_V, int onfly, Array **near_D,
        void *alloc_U, void *alloc_V, void *alloc_D, char alloc_type)
//! Init @ref STARSH_blrm object.
/*! 
 * @param[out] matrix: Address of pointer to @ref STARSH_blrm object.
 * @param[in] format: Pointer to @ref STARSH_blrf object.
 * @param[in] far_rank: Array of ranks of far-field blocks.
 * @param[in] far_U: Array of low-rank factors `U`.
 * @param[in] far_V: Array of low-rank factors `V`.
 * @param[in] onfly: Whether not to store dense blocks.
 * @param[in] near_D: Array of dense near-field blocks.
 * @param[in] alloc_U: Pointer to big buffer for all `far_U`.
 * @param[in] alloc_V: Pointer to big buffer for all `far_V`.
 * @param[in] alloc_D: Pointer to big buffer for all `near_D`.
 * @param[in] alloc_type: Type of memory allocation. `1` if big buffers
 *     are used.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup blrm
 * */
{
    if(matrix == NULL)
    {
        STARSH_ERROR("Invalid value of `matrix`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_blrf *F = format;
    if(F == NULL)
    {
        STARSH_ERROR("Invalid value of `format`");
        return STARSH_WRONG_PARAMETER;
    }
    if(far_rank == NULL && F->nblocks_far_local > 0)
    {
        STARSH_ERROR("Invalid value of `far_rank`");
        return STARSH_WRONG_PARAMETER;
    }
    if(far_U == NULL && F->nblocks_far_local > 0)
    {
        STARSH_ERROR("Invalid value of `far_U`");
        return STARSH_WRONG_PARAMETER;
    }
    if(far_V == NULL && F->nblocks_far_local > 0)
    {
        STARSH_ERROR("Invalid value of `far_V`");
        return STARSH_WRONG_PARAMETER;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARSH_ERROR("Invalid value of `onfly`");
        return STARSH_WRONG_PARAMETER;
    }
    if(near_D == NULL && F->nblocks_near_local > 0 && onfly == 0)
    {
        STARSH_ERROR("Invalid value of `near_D`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_type != '1' && alloc_type != '2')
    {
        STARSH_ERROR("Invalid value of `alloc_type`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_U == NULL && F->nblocks_far_local > 0 && alloc_type == '1')
    {
        STARSH_ERROR("Invalid value of `alloc_U`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_V == NULL && F->nblocks_far_local > 0 && alloc_type == '1')
    {
        STARSH_ERROR("Invalid value of `alloc_V`");
        return STARSH_WRONG_PARAMETER;
    }
    if(alloc_D == NULL && F->nblocks_near_local > 0 && alloc_type == '1' &&
            onfly == 0)
    {
        STARSH_ERROR("Invalid value of `alloc_D`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_blrm *M;
    STARSH_MALLOC(M, 1);
    *matrix = M;
    M->format = F;
    M->far_rank = far_rank;
    M->far_U = far_U;
    M->far_V = far_V;
    M->onfly = onfly;
    M->near_D = near_D;
    M->alloc_U = alloc_U;
    M->alloc_V = alloc_V;
    M->alloc_D = alloc_D;
    M->alloc_type = alloc_type;
    STARSH_int lbi, bi;
    size_t data_size = 0, size = 0;
    size += sizeof(*M);
    size += F->nblocks_far_local
        *(sizeof(*far_rank)+sizeof(*far_U)+sizeof(*far_V));
    for(lbi = 0; lbi < F->nblocks_far_local; lbi++)
    {
        size += far_U[lbi]->nbytes+far_V[lbi]->nbytes;
        data_size += far_U[lbi]->data_nbytes+far_V[lbi]->data_nbytes;
    }
    if(onfly == 0)
    {
        size += F->nblocks_near_local*sizeof(*near_D);
        for(lbi = 0; lbi < F->nblocks_near_local; lbi++)
        {
            size += near_D[lbi]->nbytes;
            data_size += near_D[lbi]->data_nbytes;
        }
    }
    M->nbytes = 0;
    M->data_nbytes = 0;
    MPI_Allreduce(&size, &(M->nbytes), 1, my_MPI_SIZE_T, MPI_SUM,
            MPI_COMM_WORLD);
    MPI_Allreduce(&data_size, &(M->data_nbytes), 1, my_MPI_SIZE_T, MPI_SUM,
            MPI_COMM_WORLD);
    return STARSH_SUCCESS;
}

void starsh_blrm_free_mpi(STARSH_blrm *matrix)
//! Free memory of a non-nested block low-rank matrix.
//! @ingroup blrm
{
    STARSH_blrm *M = matrix;
    if(M == NULL)
        return;
    STARSH_blrf *F = M->format;
    STARSH_int lbi;
    int info;
    if(F->nblocks_far_local > 0)
    {
        if(M->alloc_type == '1')
        {
            free(M->alloc_U);
            free(M->alloc_V);
            for(lbi = 0; lbi < F->nblocks_far_local; lbi++)
            {
                M->far_U[lbi]->data = NULL;
                array_free(M->far_U[lbi]);
                M->far_V[lbi]->data = NULL;
                array_free(M->far_V[lbi]);
            }
        }
        else// M->alloc_type == '2'
        {
            for(lbi = 0; lbi < F->nblocks_far_local; lbi++)
            {
                array_free(M->far_U[lbi]);
                array_free(M->far_V[lbi]);
            }
        }
        free(M->far_rank);
        free(M->far_U);
        free(M->far_V);
    }
    if(F->nblocks_near_local > 0 && M->onfly == 0)
    {
        if(M->alloc_type == '1')
        {
            free(M->alloc_D);
            for(lbi = 0; lbi < F->nblocks_near_local; lbi++)
            {
                M->near_D[lbi]->data = NULL;
                array_free(M->near_D[lbi]);
            }
        }
        else// M->alloc_type == '2'
        {
            for(lbi = 0; lbi < F->nblocks_near_local; lbi++)
            {
                array_free(M->near_D[lbi]);
            }
        }
        free(M->near_D);
    }
    free(M);
}

void starsh_blrm_info_mpi(STARSH_blrm *matrix)
//! Print short info on non-nested block low-rank matrix.
//! @ingroup blrm
{
    STARSH_blrm *M = matrix;
    if(M == NULL)
        return;
    printf("<STARSH_blrm at %p, %d onfly, allocation type '%c', %f MB memory "
            "footprint>\n", M, M->onfly, M->alloc_type, M->nbytes/1024./1024.);
    return;
}
#endif // MPI
