#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include "stars.h"
//#include "cblas.h"
//#include "lapacke.h"
#include "misc.h"


int STARS_BLRM_new(STARS_BLRM **M, STARS_BLRF *F, int *far_rank,
        Array **far_U, Array **far_V, int onfly,
        Array **near_D, void *alloc_U, void *alloc_V,
        void *alloc_D, char alloc_type)
// Init procedure for a non-nested block low-rank matrix
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARS_ERROR("invalid value of `F`");
        return 1;
    }
    if(far_rank == NULL && F->nblocks_far > 0)
    {
        STARS_ERROR("invalid value of `far_rank`");
        return 1;
    }
    if(far_U == NULL && F->nblocks_far > 0)
    {
        STARS_ERROR("invalid value of `far_U`");
        return 1;
    }
    if(far_V == NULL && F->nblocks_far > 0)
    {
        STARS_ERROR("invalid value of `far_V`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARS_ERROR("invalid value of `onfly`");
        return 1;
    }
    if(near_D == NULL && F->nblocks_near > 0 && onfly == 0)
    {
        STARS_ERROR("invalid value of `near_D`");
        return 1;
    }
    if(alloc_type != '1' && alloc_type != '2')
    {
        STARS_ERROR("invalid value of `alloc_type`");
        return 1;
    }
    if(alloc_U == NULL && alloc_type == '1')
    {
        STARS_ERROR("invalid value of `alloc_U`");
        return 1;
    }
    if(alloc_V == NULL && alloc_type == '1')
    {
        STARS_ERROR("invalid value of `alloc_V`");
        return 1;
    }
    if(alloc_D == NULL && alloc_type == '1' && onfly == 0)
    {
        STARS_ERROR("invalid value of `alloc_D`");
        return 1;
    }
    STARS_MALLOC(*M, 1);
    STARS_BLRM *M2 = *M;
    M2->blrf = F;
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

int STARS_BLRM_free(STARS_BLRM *M)
// Free memory of a non-nested block low-rank matrix
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    STARS_BLRF *F = M->blrf;
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
                info = Array_free(M->far_U[bi]);
                if(info != 0)
                    return info;
                M->far_V[bi]->data = NULL;
                info = Array_free(M->far_V[bi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_far; bi++)
            {
                info = Array_free(M->far_U[bi]);
                if(info != 0)
                    return info;
                info = Array_free(M->far_V[bi]);
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
                info = Array_free(M->near_D[bi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_near; bi++)
            {
                info = Array_free(M->near_D[bi]);
                if(info != 0)
                    return info;
            }
        }
        free(M->near_D);
    }
    free(M);
    return 0;
}

int STARS_BLRM_info(STARS_BLRM *M)
// Print short info on non-nested block low-rank matrix
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    printf("<STARS_BLRM at %p, %d onfly, allocation type '%c', %f MB memory "
            "footprint>\n", M, M->onfly, M->alloc_type, M->nbytes/1024./1024.);
    return 0;
}

int STARS_BLRM_get_block(STARS_BLRM *M, int i, int j, int *shape, int *rank,
        void **U, void **V, void **D)
// Returns shape of block, its rank and low-rank factors or dense
// representation of a block
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(shape == NULL)
    {
        STARS_ERROR("invalid value of `shape`");
        return 1;
    }
    if(rank == NULL)
    {
        STARS_ERROR("invalid value of `rank`");
        return 1;
    }
    if(U == NULL)
    {
        STARS_ERROR("invalid value of `U`");
        return 1;
    }
    if(V == NULL)
    {
        STARS_ERROR("invalid value of `V`");
        return 1;
    }
    if(D == NULL)
    {
        STARS_ERROR("invalid value of `D`");
        return 1;
    }
    STARS_BLRF *F = M->blrf;
    if(i < 0 || i >= F->nbrows)
    {
        STARS_ERROR("invalid value of `i`");
        return 1;
    }
    if(j < 0 || j >= F->nbcols)
    {
        STARS_ERROR("invalid value of `j`");
        return 1;
    }
    STARS_Problem *P = F->problem;
    if(P->ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
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
            *D = M->near_D[bi]->data;
            if(*D == NULL)
                info = STARS_BLRF_get_block(F, i, j, shape, D);
            return info;
        }
    }
    STARS_WARNING("Required block (%d, %d) is not admissible!\n", i, j);
    info = STARS_BLRF_get_block(F, i, j, shape, D);
    return info;
}

