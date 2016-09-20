typedef struct
{
    int count;
    double *point;
    double beta;
} STARS_ssdata;

typedef struct
{
    int count, bcount, bsize;
    double *U, *S, *V;
} STARS_synthdata;

block_func *STARS_get_kernel(char *name);
void zsort(int n, double *points);
void block_exp_kernel(int rows, int cols, int *row, int *col,
        STARS_ssdata *row_data, STARS_ssdata *col_data, double *result);
void block_synth_kernel(int rows, int cols, int *row, int *col,
        STARS_synthdata *row_data, STARS_synthdata *col_data, double *result);

