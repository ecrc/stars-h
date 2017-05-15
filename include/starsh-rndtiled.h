#ifndef _RNDTILED_H_
#define _RNDTILED_H_

typedef struct starsh_rndtiled STARSH_rndtiled;

struct starsh_rndtiled
//! Structure for generating synthetic BLR matrices.
{
    int n;
    //!< Number of rows of synthetic matrix.
    char dtype;
    //!< Precision of elements of a matrix.
    int nblocks;
    //! < Number of tiles in one dimension.
    int block_size;
    //!< Size of each tile.
    double *U;
    //!< Pointer to `n`-by-`block_size` matrix-generator.
    double *S;
    //!< Array of singular values, which is common for all tiles.
    double *rndS;
    //!< Array of noise in singular values for each tile.
    double add_diag;
    //!< Value to add to each diagonal element (for positive definiteness).
};

enum STARSH_RNDTILED_PARAM
{
    STARSH_RNDTILED_NB = 1,
    STARSH_RNDTILED_DECAY = 2,
    STARSH_RNDTILED_DIAG = 3
};

int starsh_rndtiled_new(STARSH_rndtiled **data, int n, char dtype,
        int block_size, double decay, double add_diag);
int starsh_rndtiled_new_va(STARSH_rndtiled **data, int n, char dtype,
        va_list args);
int starsh_rndtiled_new_el(STARSH_rndtiled **data, int n, char dtype, ...);
int starsh_rndtiled_get_kernel(STARSH_kernel *kernel, STARSH_rndtiled *data,
        int type);
int starsh_rndtiled_free(STARSH_rndtiled *data);

#endif // _RNDTILED_H_
