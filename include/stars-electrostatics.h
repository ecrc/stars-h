typedef struct STARS_esdata
{
    size_t count;
    double *point;
} STARS_esdata;

int STARS_esdata_block_kernel(size_t nrows, size_t ncols, size_t *irow,
        size_t *icol, void *row_data, void *col_data, void *result);
STARS_esdata *STARS_gen_esdata(size_t row_blocks, size_t col_blocks,
        size_t block_size);
void STARS_esdata_free(STARS_esdata *data);
