#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>

static void tiled_compress_algebaric_svd_kernel(void *buffers[], void *args)
{

}

int STARS_BLRM_tiled_compress_algebraic_svd_starpu(STARS_BLRM **M,
        STARS_BLRF *F, int fixrank, double tol, int onfly)
// Static scheduling of work on different MPI nodes (for simplicity) plus
// dynamic scheduling of work on a single MPI node by means of StarPU
{
    struct starpu_codelet codelet = {
        .cpu_funcs = {tiled_compress_algebraic_svd_kernel},
        .nbuffers = 3,
        .modes = {STARPU_RW, STARPU_W, STARPU_W}
    };
    return 0;
}


// Testing main function
int main(int argc, char **argv)
{
    int size = 10000;
    int nblocks = 20;
    STARS_BLRM_tiled_compress_algebraic_svd_starpu(size, nblocks, 0, 1e-3, 0);
    return 0;
}
