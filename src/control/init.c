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

// Array of backends (string, enum constant and function pointer)
struct
{
    const char *string;
    enum STARSH_BACKEND backend;
} static backend[] =
{
    {"SEQUENTIAL", STARSH_BACKEND_SEQUENTIAL},
    {"OMP", STARSH_BACKEND_OMP},
    {"MPI", STARSH_BACKEND_MPI},
    {"MPIOMP", STARSH_BACKEND_MPIOMP},
    {"STARPU", STARSH_BACKEND_STARPU},
    {"STARPUMPI", STARSH_BACKEND_STARPUMPI}
};

static int backend_num = 6;
static int backend_default = 3;

// Array of low-rank engines (string, enum constant and function pointer)
struct
{
    const char *string;
    enum STARSH_LRENGINE lrengine;
    
} static lrengine[] =
{
    {"SVD", STARSH_LRENGINE_SVD},
    {"DCSVD", STARSH_LRENGINE_DCSVD},
    {"RRQR", STARSH_LRENGINE_RRQR},
    {"RSVD", STARSH_LRENGINE_RSVD},
    {"CROSS", STARSH_LRENGINE_CROSS},
};

static int lrengine_num = 5;
static int lrengine_default = 3;

int starsh_init()
{
    const char *STARSH_BACKEND = "STARSH_BACKEND";
    const char *STARSH_LRENGINE = "STARSH_LRENGINE";
    int info = 0;
    info |= starsh_set_backend(getenv(STARSH_BACKEND));
    info |= starsh_set_lrengine(getenv(STARSH_LRENGINE));
    starsh_params.rsvd_oversample = 10;
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
