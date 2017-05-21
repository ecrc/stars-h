/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/cluster.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-05-21
 * */

#include "common.h"
#include "starsh.h"

int starsh_cluster_new(STARSH_cluster **C, void *data, int ndata, int *pivot,
        int nblocks, int nlevels, int *level, int *start, int *size,
        int *parent, int *child_start, int *child,
        enum STARSH_CLUSTER_TYPE type)
//! Init for STARSH_cluster instance.
/*! @ingroup cluster
 * @param[out] C: Address of pointer to `STARSH_cluster` object.
 * @param[in] data: Pointer structure, holding physical data.
 * @param[in] ndata: Number of discrete elements, corresponding to
 *     physical data.
 * @param[in] pivot: Pivoting of clusterization. After applying this
 *     pivoting, rows (or columns), corresponding to one block are
 *     one after another.
 * @param[in] nblocks: Number of blocks/block rows/block
 *     columns/subclusters.
 * @param[in] nlevels: Number of levels of hierarchy.
 * @param[in] level: Indexes of subclusters for each level of hierarchy.
 *     Subclusters with indexes from `level[i]` to `level[i+1]-1` inclusively
 *     belong to `i`-th level of hierarchy.
 * @param[in] start: Start point of of indexes of discrete elements of each
 *     block/subcluster in array `pivot`.
 * @param[in] size: Size of each block/subcluster/block row/block column.
 * @param[in] parent: Array of parents.
 * @param[in] child_start: Array of start points in array `child` of each
 *    subcluster.
 * @param[in] child: aAray of children of each subcluster.
 * @param[in] type: Type of clustrization. `STARSH_ClusterTiled` for tiled and
 *     `STARS_ClusterHierarchical` for hierarchical.
 * */
{
    STARSH_MALLOC(*C, 1);
    STARSH_cluster *C2 = *C;
    C2->data = data;
    C2->ndata = ndata;
    C2->pivot = pivot;
    C2->nblocks = nblocks;
    C2->nlevels = nlevels;
    C2->level = level;
    C2->start = start;
    C2->size = size;
    C2->parent = parent;
    C2->child_start = child_start;
    C2->child = child;
    C2->type = type;
    return 0;
}

int starsh_cluster_free(STARSH_cluster *C)
//! Free fields and structure of the clusterization.
//! @ingroup cluster
{
    if(C == NULL)
    {
        STARSH_ERROR("invalid value of `C`");
        return 1;
    }
    free(C->pivot);
    if(C->level != NULL)
        free(C->level);
    free(C->start);
    free(C->size);
    if(C->parent != NULL)
        free(C->parent);
    if(C->child_start != NULL)
        free(C->child_start);
    if(C->child != NULL)
        free(C->child);
    free(C);
    return 0;
}

int starsh_cluster_info(STARSH_cluster *C)
//! Print some info about clusterization.
//! @ingroup cluster
{
    if(C == NULL)
    {
        STARSH_ERROR("invalid value of `C`");
        return 1;
    }
    printf("<STARS_Cluster at %p, ", C);
    if(C->type == STARSH_PLAIN)
        printf("tiled, ");
    else
        printf("hierarchical, ");
    printf("%d blocks>\n", C->nblocks);
    return 0;
}

int starsh_cluster_new_tiled(STARSH_cluster **C, void *data, int ndata,
        int block_size)
//! Plain division of data into blocks of discrete elements.
//! @ingroup cluster
/*! Non-pivoted non-hierarchical clusterization. */
{
    if(C == NULL)
    {
        STARSH_ERROR("invalid value of `C`");
        return 1;
    }
    if(ndata < 0)
    {
        STARSH_ERROR("invalid value of `ndata`");
        return 1;
    }
    if(block_size < 0)
    {
        STARSH_ERROR("invalid value of `block_sizez");
        return 1;
    }
    int i = 0, j, k = 0, nblocks = (ndata-1)/block_size+1;
    int *start, *size, *pivot;
    STARSH_MALLOC(start, nblocks);
    STARSH_MALLOC(size, nblocks);
    while(i < ndata)
    {
        j = i+block_size;
        if(j > ndata)
            j = ndata;
        start[k] = i;
        size[k] = j-i;
        i = j;
        k++;
    }
    STARSH_MALLOC(pivot, ndata);
    for(i = 0; i < ndata; i++)
        pivot[i] = i;
    return starsh_cluster_new(C, data, ndata, pivot, nblocks, 0, NULL, start,
            size, NULL, NULL, NULL, STARSH_PLAIN);
}

