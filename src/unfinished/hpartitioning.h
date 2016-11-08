#include <inttypes.h>

typedef void *(*aux_get_func)(void *data, int count, int *index);
typedef void (*aux_free_func)(void *aux);
typedef uint64_t *(*divide_cluster_func)(void *data, int count, int *index);

typedef struct STARS_hcluster
{
    uint64_t count;
    void *data;
    uint64_t *index;
    uint64_t block_size;
    uint64_t num_clusters;
    uint64_t num_alloc_clusters;
    uint64_t *parent;
    uint64_t *child;
    uint64_t *child_start;
    uint64_t *child_size;
    aux_get_func aux_get;
    aux_free_func aux_free;
    void **aux;
    divide_cluster_func divide_cluster;
} STARS_hcluster;

STARS_hcluster *STARS_hcluster_init(uint64_t count, void *data,
        uint64_t block_size, aux_get_func aux_get, aux_free_func aux_free,
        divide_cluster_func divide_cluster);
int STARS_hcluster_free(STARS_hcluster *hcl);

typedef struct STARS_subcluster
{
    STARS_hcluster hcluster;
    uint64_t count;
    uint64_t *index;
    void *aux;
} STARS_subcluster;
