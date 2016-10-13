Array *block_es_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data);
STARS_Problem *STARS_gen_esproblem(int row_blocks, int col_blocks,
        int block_size);
STARS_BLR *STARS_gen_es_blrformat(int row_blocks, int col_blocks,
        int block_size);
