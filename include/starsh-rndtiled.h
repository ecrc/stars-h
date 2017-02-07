#ifndef _RNDTILED_H_
#define _RNDTILED_H_

typedef struct starsh_rndtiled STARSH_rndtiled;

struct starsh_rndtiled
{
    int n;
    int nblocks;
    int block_size;
    double *U;
    double *S;
    double *rndS;
};

int starsh_rndtiled_gen(STARSH_rndtiled **data, STARSH_kernel *kernel,
        int nblocks, int block_size, double decay, double noise);

int starsh_rndtiled_free(STARSH_rndtiled *data);

#endif // _RNDTILED_H_
