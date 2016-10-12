Array *block_es_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data);
void *STARS_gen_esdata(int count);
STARS_Problem *STARS_gen_esproblem(int count);
STARS_BLR *STARS_gen_es_blrformat(int block_size, int block_count);
