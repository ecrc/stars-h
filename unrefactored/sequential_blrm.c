#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include "stars.h"
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
//#include "cblas.h"
//#include "lapacke.h"
#include "misc.h"

int starsh_blrm_heatmap(STARSH_blrm *M, char *filename)
{
    STARSH_blrf *F = M->format;
    STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
    int *rank_map = malloc((size_t)F->nbrows*(size_t)F->nbcols*
            sizeof(*rank_map));
    size_t bi;
    for(bi = 0; bi < F->nblocks_far; bi++)
    {
        size_t i = F->block_far[2*bi];
        size_t j = F->block_far[2*bi+1];
        rank_map[i*F->nbcols+j] = M->far_rank[bi];
        if(i != j && F->symm == 'S')
            rank_map[j*F->nbcols+i] = M->far_rank[bi];
    }
    for(bi = 0; bi < F->nblocks_near; bi++)
    {
        size_t i = F->block_near[2*bi];
        size_t j = F->block_near[2*bi+1];
        int nrowsi = R->size[i];
        int ncolsj = C->size[j];
        int rank = nrowsi < ncolsj ? nrowsi : ncolsj;
        rank_map[i*F->nbcols+j] = rank;
        if(i != j && F->symm == 'S')
            rank_map[j*F->nbcols+i] = rank;
    }
    FILE *fd = fopen(filename, "w");
    fprintf(fd, "%d %d\n", F->nbrows, F->nbcols);
    for(size_t i = 0; i < F->nbrows; i++)
    {
        for(size_t j = 0; j < F->nbcols; j++)
            fprintf(fd, " %d", rank_map[i*F->nbcols+j]);
        fprintf(fd, "\n");
    }
    fclose(fd);
    return 0;
}
