/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/cluster.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"

int starsh_cluster_new(STARSH_cluster **cluster, void *data, STARSH_int ndata,
        STARSH_int *pivot, STARSH_int nblocks, STARSH_int nlevels,
        STARSH_int *level, STARSH_int *start, STARSH_int *size,
        STARSH_int *parent, STARSH_int *child_start, STARSH_int *child,
        enum STARSH_CLUSTER_TYPE type)
//! Init @ref STARSH_cluster object.
/*! This function simply alocates memory and fills fields of @ref
 * STARSH_cluster object. Look at @ref STARSH_cluster to get more info about
 * meaning of each field.
 *
 * @param[out] cluster: Address of pointer to @ref STARSH_cluster object.
 * @param[in] data: Pointer structure, holding physical data.
 * @param[in] ndata: Number of discrete elements, corresponding to
 *      physical data.
 * @param[in] pivot: Pivoting of clusterization. After applying this
 *      pivoting, rows (or columns), corresponding to one block are
 *      one after another.
 * @param[in] nblocks: Number of blocks / block rows / block columns /
 *      subclusters.
 * @param[in] nlevels: Number of levels of hierarchy.
 * @param[in] level: Indexes of subclusters for each level of hierarchy.
 *      Subclusters with indexes from `level[i]` to `level[i+1]-1` inclusively
 *      belong to `i`-th level of hierarchy.
 * @param[in] start: Start point of of indexes of discrete elements of each
 *      block / subcluster in array `pivot`.
 * @param[in] size: Size of each block / subcluster / block row / block column.
 * @param[in] parent: Array of parents.
 * @param[in] child_start: Array of start points in array `child` of each
 *      subcluster.
 * @param[in] child: Array of children of each subcluster.
 * @param[in] type: Type of clusterization.  Look at @ref STARSH_CLUSTER_TYPE
 *      for more info.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup cluster
 * */
{
    STARSH_cluster *C;
    STARSH_MALLOC(C, 1);
    *cluster = C;
    C->data = data;
    C->ndata = ndata;
    C->pivot = pivot;
    C->nblocks = nblocks;
    C->nlevels = nlevels;
    C->level = level;
    C->start = start;
    C->size = size;
    C->parent = parent;
    C->child_start = child_start;
    C->child = child;
    C->type = type;
    return STARSH_SUCCESS;
}

void starsh_cluster_free(STARSH_cluster *cluster)
//! Free @ref STARSH_cluster object.
//! @ingroup cluster
{
    STARSH_cluster *C = cluster;
    if(C == NULL)
        return;
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
}

void starsh_cluster_info(STARSH_cluster *cluster)
//! Print some info about @ref STARSH_cluster object.
//! @ingroup cluster
{
    STARSH_cluster *C = cluster;
    if(C == NULL)
        return;
    printf("<STARS_Cluster at %p, ", C);
    if(C->type == STARSH_PLAIN)
        printf("tiled, ");
    else
        printf("hierarchical, ");
    printf("%d blocks>\n", C->nblocks);
}

int starsh_cluster_new_plain(STARSH_cluster **cluster, void *data,
        STARSH_int ndata, STARSH_int block_size)
//! Plain division of data into blocks of discrete elements.
/*! Non-pivoted non-hierarchical clusterization.
 *
 * @param[out] cluster: Address of pointer to @ref STARSH_cluster object.
 * @param[in] data: Pointer to structure, holding physical data.
 * @param[in] ndata: Number of discrete elements in physical data.
 * @param[in] block_size: How many discrete elements should be combined into
 *      single block.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_cluster_new().
 * @ingroup cluster
 * */
{
    if(cluster == NULL)
    {
        STARSH_ERROR("Invalid value of `cluster`");
        return 1;
    }
    STARSH_int i = 0, j, k = 0;
    STARSH_int nblocks = (ndata-1)/block_size+1;
    STARSH_int *start, *size, *pivot;
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
    return starsh_cluster_new(cluster, data, ndata, pivot, nblocks, 0, NULL,
            start, size, NULL, NULL, NULL, STARSH_PLAIN);
}

