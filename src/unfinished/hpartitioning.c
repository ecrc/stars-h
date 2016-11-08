#include <stdlib.h>
#include <stdio.h>
#include "hpartitioning.h"

STARS_hcluster *STARS_hcluster_init(uint64_t count, void *data,
        uint64_t block_size, aux_get_func aux_get, aux_free_func aux_free,
        divide_cluster_func divide_cluster)
{
    uint64_t num_clusters_step = 2*count/block_size;
    STARS_hcluster *hcl = (STARS_hcluster *)malloc(sizeof(STARS_hcluster));
    hcl->count = count;
    hcl->data = data;
    hcl->index = (uint64_t *)malloc(count*sizeof(uint64_t));
    for(uint64_t i = 0; i < count; i++)
        hcl->index[i] = i;
    hcl->num_clusters = 1;
    hcl->num_alloc_clusters = num_clusters_step;
    hcl->parent = (uint64_t *)malloc(num_clusters_step*sizeof(uint64_t));
    hcl->child = (uint64_t *)malloc(num_clusters_step*sizeof(uint64_t));
    hcl->child_start = (uint64_t *)malloc(num_clusters_step*sizeof(uint64_t));
    hcl->child_size = (uint64_t *)malloc(num_clusters_step*sizeof(uint64_t));
    hcl->aux_get = aux_get;
    hcl->aux_free = aux_free;
    hcl->aux = malloc(num_clusters_step*sizeof(void *));
    hcl->aux[0] = aux_get(data, count, hcl->index);
    hcl->divide_cluster = divide_cluster;
    return hcl;
}

int STARS_hcluster_free(STARS_hcluster *hcl)
{
    free(hcl->index);
    free(hcl->parent);
    free(hcl->child);
    free(hcl->child_start);
    free(hcl->child_size);
    for(uint64_t i = 0; i < hcl->num_clusters; i++)
        hcl->aux_free(aux[i]);
    free(hcl->aux);
    free(hcl);
}
