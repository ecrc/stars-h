#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stars.h"


STARS_Cluster *STARS_Cluster_init(void *data, size_t ndata, size_t *pivot,
        size_t nblocks, size_t nlevels, size_t *level, size_t *start,
        size_t *size, ssize_t *parent, size_t *child_start, ssize_t *child,
        STARS_ClusterType type)
// Init for STARS_Cluster instance
// Parameters:
//   data: pointer structure, holding to physical data.
//   ndata: number of discrete elements (particles or mesh elements),
//     corresponding to physical data.
//   pivot: pivoting of clusterization. After applying this pivoting, rows (or
//     columns), corresponding to one block are placed in a row.
//   nblocks: number of blocks/block rows/block columns/subclusters.
//   nlevels: number of levels of hierarchy.
//   level: array of size nlevels+1, indexes of blocks from level[i] to
//     level[i+1]-1 inclusively belong to i-th level of hierarchy.
//   start: start point of of indexes of discrete elements of each
//     block/subcluster in array pivot.
//   size: size of each block/subcluster/block row/block column.
//   parent: array of parents, size is nblocks.
//   child_start: array of start points in array child of each subcluster.
//   child: array of children of each subcluster.
//   type: type of cluster. Tiled with STARS_ClusterTiled or hierarchical with
//     STARS_ClusterHierarchical.
{
    STARS_Cluster *cluster = malloc(sizeof(*cluster));
    cluster->data = data;
    cluster->ndata = ndata;
    cluster->pivot = pivot;
    cluster->nblocks = nblocks;
    cluster->nlevels = nlevels;
    cluster->level = level;
    cluster->start = start;
    cluster->size = size;
    cluster->parent = parent;
    cluster->child_start = child_start;
    cluster->child = child;
    cluster->type = type;
    return cluster;
}

void STARS_Cluster_free(STARS_Cluster *cluster)
// Free data buffers, consumed by clusterization information.
{
    if(cluster == NULL)
    {
        fprintf(stderr, "STARS_Cluster instance is NOT initialized\n");
        return;
    }
    free(cluster->pivot);
    if(cluster->level != NULL)
        free(cluster->level);
    free(cluster->start);
    free(cluster->size);
    if(cluster->parent != NULL)
        free(cluster->parent);
    if(cluster->child_start != NULL)
        free(cluster->child_start);
    if(cluster->child != NULL)
        free(cluster->child);
    free(cluster);
}

void STARS_Cluster_info(STARS_Cluster *cluster)
// Print some info about clusterization
{
    printf("<STARS_Cluster at %p, ", cluster);
    if(cluster->type == STARS_ClusterTiled)
        printf("tiled, ");
    else
        printf("hierarchical, ");
    printf("%zu blocks>\n", cluster->nblocks);
}

STARS_Cluster *STARS_Cluster_init_tiled(void *data, size_t ndata,
        size_t block_size)
// Plain (non-hierarchical) division of data into blocks of discrete elements.
{
    size_t i = 0, j, k = 0, nblocks = (ndata-1)/block_size+1;
    size_t *start = malloc(nblocks*sizeof(*start));
    size_t *size = malloc(nblocks*sizeof(*size));
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
    size_t *pivot = malloc(ndata*sizeof(*pivot));
    for(i = 0; i < ndata; i++)
        pivot[i] = i;
    return STARS_Cluster_init(data, ndata, pivot, nblocks, 0, NULL, start,
            size, NULL, NULL, NULL, STARS_ClusterTiled);
}

