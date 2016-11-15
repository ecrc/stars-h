#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stars.h"


STARS_Cluster *STARS_Cluster_init(void *data, int ndata, int *pivot,
        int nblocks, int *start, int *size, STARS_ClusterType type)
// Init for STARS_Cluster instance
// Parameters:
//   data: pointer structure, holding to physical data, stores pointer to a
//     given data without any copy.
//   ndata: number of discrete elements (particles or mesh elements),
//     corresponding to physical data
//   pivot: pivoting of clusterization. After applying this pivoting, rows (or
//     columns), corresponding to one block are placed in a row. Stores pointer
//     to a given array without copying anything.
//   nblocks: number of blocks/block rows/block columns/subclusters
//   start: start point of of indexes of discrete elements of each
//     block/subcluster in array pivot. No copying is done.
//   size: size of each block/subcluster/block row/block column. No copying is
//     done.
{
    STARS_Cluster *cluster = malloc(sizeof(*cluster));
    cluster->data = data;
    cluster->ndata = ndata;
    cluster->pivot = pivot;
    cluster->nblocks = nblocks;
    cluster->start = start;
    cluster->size = size;
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
    free(cluster->start);
    free(cluster->size);
    free(cluster);
}

void STARS_Cluster_info(STARS_Cluster *cluster)
// Print some info about clusterization
{
    printf("<STARS_Cluster at %p, %d type, %d blocks>\n", cluster,
            cluster->type, cluster->nblocks);
}

STARS_Cluster *STARS_Cluster_init_tiled(void *data, int ndata, int block_size)
// Plain (non-hierarchical) division of data into blocks of discrete elements.
{
    int i = 0, j, k = 0;
    int nblocks = (ndata-1)/block_size+1;
    int *start = malloc(nblocks*sizeof(int));
    int *size = malloc(nblocks*sizeof(int));
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
    int *pivot = malloc(ndata*sizeof(int));
    for(i = 0; i < ndata; i++)
        pivot[i] = i;
    return STARS_Cluster_init(data, ndata, pivot, nblocks, start, size,
            STARS_ClusterTiled);
}

