#ifndef _RNDTILED_H_
#define _RNDTILED_H_

typedef struct starsh_rndtiled STARSH_rndtiled;

struct starsh_rndtiled
//! Structure for generating synthetic BLR matrices.
{
    int n;
    //!< Number of rows of synthetic matrix.
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
};

int starsh_rndtiled_new(STARSH_rndtiled **data, int n, char dtype,
        int block_size, double decay);
int starsh_rndtiled_new_va(STARSH_rndtiled **data, int n, char dtype,
        va_list args);
int starsh_rndtiled_new_el(STARSH_rndtiled **data, int n, char dtype, ...);

int starsh_rndtiled_get_kernel(STARSH_kernel *kernel, const char *type,
        char dtype);

int starsh_rndtiled_free(STARSH_rndtiled *data);

#endif // _RNDTILED_H_
