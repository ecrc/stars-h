typedef void block_func(int, int, int *, int *, void *, void *, void *);

typedef struct
{
    char symm; // 'N' if nonsymmetric problem, 'S' if symmetric
    char dtype; // data type of problem ('s', 'd', 'c', 'z')
    int rows, cols; // number of rows and columns of a matrix
    int *row_order, *col_order; // permutation of rows and columns
    void *row_data, *col_data; // objects behind rows and columns
    int row_bcount, col_bcount; // number of block rows and block columns
    int *row_bstart, *col_bstart; // start point of block rows and block
        // columns (in correspondance to row_order and col_order)
    int *row_bsize, *col_bsize; // size of each block row and block column
    int block_count, *block_index, *block_rank; // total number of blocks,
        // each block index (pair of block coordinates) and rank
    void **U, **V, **A; // buffers to save low rank approximation of a block
        // in UV format or, if block is not low rank, block itself as a
        // submatrix A. Right now it is done as a simple vector of pointers to
        // corresponding data. In future, it is planned to change it.
    block_func *kernel; // block kernel function, generating matrix
} STARS_blr;

STARS_blr *unified_compress_symm(int block_count, int block_size, void *data,
        block_func kernel, double tol);

void compress_blr(STARS_blr *A, double tol);
