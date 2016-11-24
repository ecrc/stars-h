#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stars.h"
#include "misc.h"

int STARS_Cluster_new(STARS_Cluster **C, void *data, int ndata, int *pivot,
        int nblocks, int nlevels, int *level, int *start, int *size,
        int *parent, int *child_start, int *child, STARS_ClusterType type)
// Init for STARS_Cluster instance
// Parameters:
//   data: pointer structure, holding to physical data.
//   ndata: number of discrete elements (particles or mesh elements),
//     corresponding to physical data.
//   pivot: pivoting of Cization. After applying this pivoting, rows (or
//     columns), corresponding to one block are placed in a row.
//   nblocks: number of blocks/block rows/block columns/subCs.
//   nlevels: number of levels of hierarchy.
//   level: array of size nlevels+1, indexes of blocks from level[i] to
//     level[i+1]-1 inclusively belong to i-th level of hierarchy.
//   start: start point of of indexes of discrete elements of each
//     block/subC in array pivot.
//   size: size of each block/subC/block row/block column.
//   parent: array of parents, size is nblocks.
//   child_start: array of start points in array child of each subC.
//   child: array of children of each subC.
//   type: type of C. Tiled with STARS_ClusterTiled or hierarchical with
//     STARS_ClusterHierarchical.
{
    *C = malloc(sizeof(**C));
    STARS_Cluster *C2 = *C;
    if(C2 == NULL)
    {
        STARS_error("STARS_Cluster_new", "malloc() failed");
        return 1;
    }
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

int STARS_Cluster_free(STARS_Cluster *C)
// Free data buffers, consumed by Cization information.
{
    if(C == NULL)
    {
        STARS_error("STARS_Cluster_free", "invalid value of `C`");
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

int STARS_Cluster_info(STARS_Cluster *C)
// Print some info about Cization
{
    if(C == NULL)
    {
        STARS_error("STARS_Cluster_free", "invalid value of `C`");
        return 1;
    }
    printf("<STARS_Cluster at %p, ", C);
    if(C->type == STARS_ClusterTiled)
        printf("tiled, ");
    else
        printf("hierarchical, ");
    printf("%d blocks>\n", C->nblocks);
    return 0;
}

int STARS_Cluster_new_tiled(STARS_Cluster **C, void *data, int ndata,
        int block_size)
// Plain (non-hierarchical) division of data into blocks of discrete elements.
{
    if(C == NULL)
    {
        STARS_error("STARS_Cluster_new_tiled", "invalid value of `C`");
        return 1;
    }
    if(ndata < 0)
    {
        STARS_error("STARS_Cluster_new_tiled", "invalid value of `ndata`");
        return 1;
    }
    if(block_size < 0)
    {
        STARS_error("STARS_Cluster_new_tiled", "invalid value of "
                "`block_sizez");
        return 1;
    }
    int i = 0, j, k = 0, nblocks = (ndata-1)/block_size+1;
    int *start = malloc(nblocks*sizeof(*start));
    if(start == NULL)
    {
        STARS_error("STARS_Cluster_new_tiled", "malloc() failed");
        return 1;
    }
    int *size = malloc(nblocks*sizeof(*size));
    if(size == NULL)
    {
        STARS_error("STARS_Cluster_new_tiled", "malloc() failed");
        return 1;
    }
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
    int *pivot = malloc(ndata*sizeof(*pivot));
    if(pivot == NULL)
    {
        STARS_error("STARS_Cluster_new_tiled", "malloc() failed");
    }
    for(i = 0; i < ndata; i++)
        pivot[i] = i;
    return STARS_Cluster_new(C, data, ndata, pivot, nblocks, 0, NULL, start,
            size, NULL, NULL, NULL, STARS_ClusterTiled);
}

