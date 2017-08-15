/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/init.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-13
 * */

#include "common.h"
#include "starsh.h"

struct starsh_params starsh_params;
const struct starsh_params starsh_params_default = {};

// Array of backends (string, enum constant and function pointer)
struct
{
    const char *string;
    enum STARSH_BACKEND backend;
} const backend[] =
{
    {"SEQUENTIAL", STARSH_BACKEND_SEQUENTIAL},
    {"OMP", STARSH_BACKEND_OMP},
    {"MPI", STARSH_BACKEND_MPI},
    {"MPIOMP", STARSH_BACKEND_MPIOMP},
    {"STARPU", STARSH_BACKEND_STARPU},
    {"STARPUMPI", STARSH_BACKEND_STARPUMPI}
};

const int backend_num = sizeof(backend)/sizeof(backend[0]);
const int backend_default = 3;

// Array of low-rank engines (string, enum constant and function pointer)
struct
{
    const char *string;
    enum STARSH_LRENGINE lrengine;
} const lrengine[] =
{
    {"SVD", STARSH_LRENGINE_SVD},
    {"DCSVD", STARSH_LRENGINE_DCSVD},
    {"RRQR", STARSH_LRENGINE_RRQR},
    {"RSVD", STARSH_LRENGINE_RSVD},
    {"CROSS", STARSH_LRENGINE_CROSS},
};

const int lrengine_num = sizeof(lrengine)/sizeof(lrengine[0]);
const int lrengine_default = 3;

STARSH_blrm_approximate *(approximate_funcs[6][5]) =
{
    // Funcs for SEQUENTIAL
    {starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3, starsh_blrm__drsdd, starsh_blrm__drsdd},
    // Funcs for OMP
    {starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3, starsh_blrm__drsdd, starsh_blrm__drsdd},
    // Funcs for MPI
    {starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3, starsh_blrm__drsdd, starsh_blrm__drsdd},
    // Funcs for MPIOMP
    {starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3, starsh_blrm__drsdd, starsh_blrm__drsdd},
    // Funcs for STARPU
    {starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3, starsh_blrm__drsdd, starsh_blrm__drsdd},
    // Funcs for STARPUMPI
    {starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3, starsh_blrm__drsdd, starsh_blrm__drsdd}
};

int starsh_init()
{
    const char *str_backend = "STARSH_BACKEND";
    const char *str_lrengine = "STARSH_LRENGINE";
    int info = 0;
    info |= starsh_set_backend(getenv(str_backend));
    info |= starsh_set_lrengine(getenv(str_lrengine));
    starsh_params.oversample = 10;
    return info;
}

int starsh_set_backend(const char *string)
{
    int i, selected = -1;
    if(string == NULL)
    {
        selected = backend_default;
    }
    else
    {
        for(i = 0; i < backend_num; i++)
        {
            if(!strcmp(string, backend[i].string))
            {
                selected = i;
                break;
            }
        }
    }
    if(selected == -1)
    {
        fprintf(stderr, "environment variable \"STARSH_BACKEND=%s\" "
                "is not supported\n", string);
        return 1;
    }
    fprintf(stderr, "selected backend %s (FAKE message)\n",
            backend[selected].string);
    return 0;
}

int starsh_set_lrengine(const char *string)
{
    int i, selected = -1;
    if(string == NULL)
    {
        selected = lrengine_default;
    }
    else
    {
        for(i = 0; i < lrengine_num; i++)
        {
            if(!strcmp(string, lrengine[i].string))
            {
                selected = i;
                break;
            }
        }
    }
    if(selected == -1)
    {
        fprintf(stderr, "environment variable \"STARSH_LRENGINE=%s\" "
                "is not supported\n", string);
        return 1;
    }
    fprintf(stderr, "selected engine %s (FAKE message)\n",
            lrengine[selected].string);
    return 0;
}
