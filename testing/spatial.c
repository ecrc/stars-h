#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stars.h"
#include "stars-spatial.h"

int main(int argc, char **argv)
    // Example of how to use STARS library for spatial statistics.
    // For more information on STARS structures look inside of header files.
{
    if(argc < 9)
    {
        printf("%d\n", argc);
        printf("spatial.out row_blocks col_blocks block_size maxrank "
                "tol beta KADIR heatmap-filename\n");
        exit(0);
    }
    int row_blocks = atoi(argv[1]), col_blocks = atoi(argv[2]);
    int block_size = atoi(argv[3]), maxrank = atoi(argv[4]);
    double tol = atof(argv[5]), beta = atof(argv[6]);
    int KADIR = atoi(argv[7]);
    char *heatmap_fname = argv[8];
    printf("rb=%d, cb=%d, bs=%d, mr=%d, tol=%e, beta=%f, KADIR=%d\n",
            row_blocks, col_blocks, block_size, maxrank, tol, beta, KADIR);
    STARS_Problem *problem;
    STARS_BLR *format;
    STARS_BLRmatrix *matrix = NULL;
    while(matrix == NULL)
    {
        format = STARS_gen_ss_blrformat(row_blocks, col_blocks, block_size, beta);
        // Problem is generated inside STARS_gen_ss_blrformat
        matrix = STARS_blr__compress_algebraic_svd(format, maxrank, tol, KADIR);
        if(matrix == NULL)
        {
            free(format->problem->row_data);
            free(format->problem);
            STARS_BLR_free(format);
        }
    }
    //printf("Measuring error!\n");
    //STARS_BLRmatrix_error(matrix);
    //STARS_BLRmatrix_info(matrix);
    //STARS_BLRmatrix_printKADIR(matrix);
    STARS_BLRmatrix_heatmap(matrix, heatmap_fname);
    STARS_BLRmatrix_free(matrix);
    //STARS_BLR_info(format);
    int nbrows = format->nbrows, shape[2], i;
    char order = 'C';
    double *buffer = NULL;
    FILE *fd;
    Array *U, *S, *V, *block;
    STARS_BLR_getblock(format, nbrows/2, nbrows/2, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'F', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_getblock(format, 5*nbrows/8, 3*nbrows/8, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'F', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_getblock(format, 3*nbrows/4, nbrows/4, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'F', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_getblock(format, nbrows-1, 0, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'F', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_free(format);
    return 0;
}
