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

int starsh_rndtiled_gen(STARSH_rndtiled **data, STARSH_kernel *kernel,
        int nblocks, int block_size, double decay, double noise);

int starsh_rndtiled_free(STARSH_rndtiled *data);

#endif // _RNDTILED_H_
