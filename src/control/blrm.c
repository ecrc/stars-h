/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/blrm.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-13
 * */

#include "common.h"
#include "starsh.h"

int starsh_blrm_new(STARSH_blrm **M, STARSH_blrf *F, int *far_rank,
        Array **far_U, Array **far_V, int onfly,
        Array **near_D, void *alloc_U, void *alloc_V,
        void *alloc_D, char alloc_type)
//! Init procedure for a non-nested block low-rank matrix.
/*! @ingroup blrm
 * @param[out] M: Address of pointer to `STARSH_blrm` object.
 * @param[in] F: Block low-rank format.
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
 * @return Error code.
 * */
{
    if(M == NULL)
    {
        STARSH_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    if(far_rank == NULL && F->nblocks_far > 0)
    {
        STARSH_ERROR("invalid value of `far_rank`");
        return 1;
    }
    if(far_U == NULL && F->nblocks_far > 0)
    {
        STARSH_ERROR("invalid value of `far_U`");
        return 1;
    }
    if(far_V == NULL && F->nblocks_far > 0)
    {
        STARSH_ERROR("invalid value of `far_V`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARSH_ERROR("invalid value of `onfly`");
        return 1;
    }
    if(near_D == NULL && F->nblocks_near > 0 && onfly == 0)
    {
        STARSH_ERROR("invalid value of `near_D`");
        return 1;
    }
    if(alloc_type != '1' && alloc_type != '2')
    {
        STARSH_ERROR("invalid value of `alloc_type`");
        return 1;
    }
    if(alloc_U == NULL && F->nblocks_far > 0 && alloc_type == '1')
    {
        STARSH_ERROR("invalid value of `alloc_U`");
        return 1;
    }
    if(alloc_V == NULL && F->nblocks_far > 0 && alloc_type == '1')
    {
        STARSH_ERROR("invalid value of `alloc_V`");
        return 1;
    }
    if(alloc_D == NULL && F->nblocks_near > 0 && alloc_type == '1' &&
            onfly == 0)
    {
        STARSH_ERROR("invalid value of `alloc_D`");
        return 1;
    }
    STARSH_MALLOC(*M, 1);
    STARSH_blrm *M2 = *M;
    M2->format = F;
    M2->far_rank = far_rank;
    M2->far_U = far_U;
    M2->far_V = far_V;
    M2->onfly = onfly;
    M2->near_D = near_D;
    M2->alloc_U = alloc_U;
    M2->alloc_V = alloc_V;
    M2->alloc_D = alloc_D;
    M2->alloc_type = alloc_type;
    size_t bi, data_size = 0, size = 0;
    size += sizeof(*M2);
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
    M2->nbytes = size;
    M2->data_nbytes = data_size;
    return 0;
}

int starsh_blrm_free(STARSH_blrm *M)
//! Free memory of a non-nested block low-rank matrix.
//! @ingroup blrm
{
    if(M == NULL)
    {
        STARSH_ERROR("invalid value of `M`");
        return 1;
    }
    STARSH_blrf *F = M->format;
    size_t bi;
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
                info = array_free(M->far_U[bi]);
                if(info != 0)
                    return info;
                M->far_V[bi]->data = NULL;
                info = array_free(M->far_V[bi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_far; bi++)
            {
                info = array_free(M->far_U[bi]);
                if(info != 0)
                    return info;
                info = array_free(M->far_V[bi]);
                if(info != 0)
                    return info;
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
                info = array_free(M->near_D[bi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_near; bi++)
            {
                info = array_free(M->near_D[bi]);
                if(info != 0)
                    return info;
            }
        }
        free(M->near_D);
    }
    free(M);
    return 0;
}

int starsh_blrm_info(STARSH_blrm *M)
//! Print short info on non-nested block low-rank matrix.
//! @ingroup blrm
{
    if(M == NULL)
    {
        STARSH_ERROR("invalid value of `M`");
        return 1;
    }
    printf("<STARSH_blrm at %p, %d onfly, allocation type '%c', %f MB memory "
            "footprint>\n", M, M->onfly, M->alloc_type, M->nbytes/1024./1024.);
    return 0;
}

int starsh_blrm_get_block(STARSH_blrm *M, int i, int j, int *shape, int *rank,
        void **U, void **V, void **D)
//! Get shape, rank and low-rank factors or dense representation of a block.
//! @ingroup blrm
{
    if(M == NULL)
    {
        STARSH_ERROR("invalid value of `M`");
        return 1;
    }
    if(shape == NULL)
    {
        STARSH_ERROR("invalid value of `shape`");
        return 1;
    }
    if(rank == NULL)
    {
        STARSH_ERROR("invalid value of `rank`");
        return 1;
    }
    if(U == NULL)
    {
        STARSH_ERROR("invalid value of `U`");
        return 1;
    }
    if(V == NULL)
    {
        STARSH_ERROR("invalid value of `V`");
        return 1;
    }
    if(D == NULL)
    {
        STARSH_ERROR("invalid value of `D`");
        return 1;
    }
    STARSH_blrf *F = M->format;
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
    STARSH_problem *P = F->problem;
    if(P->ndim != 2)
    {
        STARSH_ERROR("only scalar kernels are supported");
        return 1;
    }
    int onfly = M->onfly;
    STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
    int nrows = R->size[i], ncols = C->size[j], info = 0;
    shape[0] = nrows;
    shape[1] = ncols;
    *rank = nrows < ncols ? nrows : ncols;
    *U = NULL;
    *V = NULL;
    *D = NULL;
    size_t bi = -1, k;
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

int starsh_blrm_approximate(STARSH_blrm **M, STARSH_blrf *F, int maxrank,
        int oversample, double tol, int onfly, const char *scheme)
//! Main call to get approximation in non-nested block low-rank format.
/*! @ingroup blrm
 * @param[out] M: Address of pointer to `STARSH_blrm` object.
 * @param[in,out] F: Format of a matrix. May be changed on exit.
 * @param[in] maxrank: Maximum rank.
 * @param[in] oversample: Oversampling parameter for randomized SVD. Use `10`
 *     as default value.
 * @param[in] tol: Relative error tolerance.
 * @param[in] onfly: Whether not to store dense near-field blocks.
 * @param[in] scheme: Scheme to use for low=-rank approximations. Possible
 *     values are: `sdd`, `qp3`, `rsdd`, `rsdd2`, `omp_sdd`, `omp_qp3`,
 *     `omp_rsdd`, `omp_rsdd2`, `starpu_sdd`, `starpu_qp3`, `starpu_rsdd`,
 *     `starpu_rsdd2`, 
 * @return Error code.
 * */
{
    if(strcmp(scheme, "sdd") == 0)
        starsh_blrm__dsdd(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "rsdd") == 0)
        starsh_blrm__drsdd(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "rsdd2") == 0)
        starsh_blrm__drsdd2(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "qp3") == 0)
        starsh_blrm__dqp3(M, F, maxrank, oversample, tol, onfly);
#ifdef OPENMP
    else if(strcmp(scheme, "omp_sdd") == 0)
        starsh_blrm__dsdd_omp(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "omp_rsdd") == 0)
        starsh_blrm__drsdd_omp(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "omp_rsdd2") == 0)
        starsh_blrm__drsdd2_omp(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "omp_qp3") == 0)
        starsh_blrm__dqp3_omp(M, F, maxrank, oversample, tol, onfly);
#endif
#ifdef STARPU
    else if(strcmp(scheme, "starpu_sdd") == 0)
        starsh_blrm__dsdd_starpu(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "starpu_rsdd") == 0)
        starsh_blrm__drsdd_starpu(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "starpu_rsdd2") == 0)
        starsh_blrm__drsdd2_starpu(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "starpu_qp3") == 0)
        starsh_blrm__dqp3_starpu(M, F, maxrank, oversample, tol, onfly);
#endif
#ifdef MPI
    else if(strcmp(scheme, "mpi_sdd") == 0)
        starsh_blrm__dsdd_mpi(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "mpi_rsdd") == 0)
        starsh_blrm__drsdd_mpi(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "mpi_qp3") == 0)
        starsh_blrm__dqp3_mpi(M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "mpi_na") == 0)
        starsh_blrm__dna_mpi(M, F, maxrank, oversample, tol, onfly);
#endif
    else
    {
        STARSH_ERROR("wrong scheme (possible: sdd, rsdd, qp3, starpu_sdd, "
                "starpu_rsdd, starpu_rsdd2, starpu_qp3, omp_sdd, omp_rsdd, "
                "omp_rsdd2, omp_qp3, mpi_sdd, mpi_rsdd, mpi_qp3)");
        return 1;
    }
    return 0;
}

#ifdef MPI
int starsh_blrm_new_mpi(STARSH_blrm **M, STARSH_blrf *F, int *far_rank,
        Array **far_U, Array **far_V, int onfly,
        Array **near_D, void *alloc_U, void *alloc_V,
        void *alloc_D, char alloc_type)
//! Init procedure for a non-nested block low-rank matrix with MPI.
/*! @ingroup blrm
 * @param[out] M: Address of pointer to `STARSH_blrm` object.
 * @param[in] F: Block low-rank format.
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
 * @return Error code.
 * */
{
    if(M == NULL)
    {
        STARSH_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARSH_ERROR("invalid value of `F`");
        return 1;
    }
    if(far_rank == NULL && F->nblocks_far_local > 0)
    {
        STARSH_ERROR("invalid value of `far_rank`");
        return 1;
    }
    if(far_U == NULL && F->nblocks_far_local > 0)
    {
        STARSH_ERROR("invalid value of `far_U`");
        return 1;
    }
    if(far_V == NULL && F->nblocks_far_local > 0)
    {
        STARSH_ERROR("invalid value of `far_V`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARSH_ERROR("invalid value of `onfly`");
        return 1;
    }
    if(near_D == NULL && F->nblocks_near_local > 0 && onfly == 0)
    {
        STARSH_ERROR("invalid value of `near_D`");
        return 1;
    }
    if(alloc_type != '1' && alloc_type != '2')
    {
        STARSH_ERROR("invalid value of `alloc_type`");
        return 1;
    }
    if(alloc_U == NULL && F->nblocks_far_local > 0 && alloc_type == '1')
    {
        STARSH_ERROR("invalid value of `alloc_U`");
        return 1;
    }
    if(alloc_V == NULL && F->nblocks_far_local > 0 && alloc_type == '1')
    {
        STARSH_ERROR("invalid value of `alloc_V`");
        return 1;
    }
    if(alloc_D == NULL && F->nblocks_near_local > 0 && alloc_type == '1' &&
            onfly == 0)
    {
        STARSH_ERROR("invalid value of `alloc_D`");
        return 1;
    }
    STARSH_MALLOC(*M, 1);
    STARSH_blrm *M2 = *M;
    M2->format = F;
    M2->far_rank = far_rank;
    M2->far_U = far_U;
    M2->far_V = far_V;
    M2->onfly = onfly;
    M2->near_D = near_D;
    M2->alloc_U = alloc_U;
    M2->alloc_V = alloc_V;
    M2->alloc_D = alloc_D;
    M2->alloc_type = alloc_type;
    size_t lbi, bi, data_size = 0, size = 0;
    size += sizeof(*M2);
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
    M2->nbytes = 0;
    M2->data_nbytes = 0;
    MPI_Allreduce(&size, &(M2->nbytes), 1, my_MPI_SIZE_T, MPI_SUM,
            MPI_COMM_WORLD);
    MPI_Allreduce(&data_size, &(M2->data_nbytes), 1, my_MPI_SIZE_T, MPI_SUM,
            MPI_COMM_WORLD);
    return 0;
}

int starsh_blrm_free_mpi(STARSH_blrm *M)
//! Free memory of a non-nested block low-rank matrix.
//! @ingroup blrm
{
    if(M == NULL)
    {
        STARSH_ERROR("invalid value of `M`");
        return 1;
    }
    STARSH_blrf *F = M->format;
    size_t lbi;
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
                info = array_free(M->far_U[lbi]);
                if(info != 0)
                    return info;
                M->far_V[lbi]->data = NULL;
                info = array_free(M->far_V[lbi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(lbi = 0; lbi < F->nblocks_far_local; lbi++)
            {
                info = array_free(M->far_U[lbi]);
                if(info != 0)
                    return info;
                info = array_free(M->far_V[lbi]);
                if(info != 0)
                    return info;
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
                info = array_free(M->near_D[lbi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(lbi = 0; lbi < F->nblocks_near_local; lbi++)
            {
                info = array_free(M->near_D[lbi]);
                if(info != 0)
                    return info;
            }
        }
        free(M->near_D);
    }
    free(M);
    return 0;
}

int starsh_blrm_info_mpi(STARSH_blrm *M)
//! Print short info on non-nested block low-rank matrix.
//! @ingroup blrm
{
    if(M == NULL)
    {
        STARSH_ERROR("invalid value of `M`");
        return 1;
    }
    printf("<STARSH_blrm at %p, %d onfly, allocation type '%c', %f MB memory "
            "footprint>\n", M, M->onfly, M->alloc_type, M->nbytes/1024./1024.);
    return 0;
}
#endif // MPI
