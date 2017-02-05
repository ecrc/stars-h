typedef struct STARS_esdata
{
    size_t count;
    double *point;
} STARS_esdata;

int STARS_esdata_block_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result);
STARS_esdata *STARS_gen_esdata(int row_blocks, int col_blocks,
        int block_size);
void STARS_esdata_free(STARS_esdata *data);
