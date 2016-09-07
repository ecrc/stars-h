typedef struct
{
    int count;
    double *point;
    double beta;
} STARS_ssdata;

block_func *STARS_get_kernel(char *name);
void zsort(int n, double *points);
void block_exp_kernel(int rows, int cols, int *row, int *col,
        STARS_ssdata *row_data, STARS_ssdata *col_data, double *result);
