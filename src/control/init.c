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

#include "starsh.h"
#include "common.h"
#include "control/init.h"

int starsh_init()
//! Initialize backend and low-rank engine to be used
/*! Reads environment variables and sets up backend (etc. MPI) and low-rank
 * engine (etc. SVD).
 * Environment variables
 * ---------------------
 *  STARSH_BACKEND: SEQUENTIAL, MPI (pure MPI), OPENMP (pure OpenMP) or
 *  MPI_OPENMP (hybrid MPI with OpenMP)
 *
 *  STARSH_LRENGINE: DCSVD (divide-and-conquer SVD), RRQR (LAPACK *geqp3) or
 *  RSVD(randomized SVD)
 * */
{
    const char *str_backend = "STARSH_BACKEND";
    const char *str_lrengine = "STARSH_LRENGINE";
    starsh_params = starsh_params_default;
    int info = 0, i;
    info |= starsh_set_backend(getenv(str_backend));
    info |= starsh_set_lrengine(getenv(str_lrengine));
    return info;
}

int starsh_set_backend(const char *string)
//! Set backend (MPI or OpenMP or other scheduler) for computations
/*! @param[in] string: Name of backend to use. Possible values:
 * "SEQUENTIAL", "OPENMP", "MPI", "MPI_OPENMP"
 * */
{
    int i, selected = -1;
    if(string == NULL)
    {
        selected = BACKEND_DEFAULT;
    }
    else
    {
        for(i = 0; i < BACKEND_NUM; i++)
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
        selected = BACKEND_DEFAULT;
        fprintf(stderr, "Environment variable STARSH_BACKEND=%s is invalid, "
                "using default backend (%s)\n",
                string, backend[selected].string);
    }
    else if(backend[selected].backend == STARSH_BACKEND_NOTSUPPORTED)
    {
        selected = BACKEND_DEFAULT;
        fprintf(stderr, "Backend %s (environment variable STARSH_BACKEND) "
                "is not supported, using default backend (%s)\n",
                string, backend[selected].string);
    }
    starsh_params.backend = backend[selected].backend;
    fprintf(stderr, "Selected backend is %s\n", backend[selected].string);
    starsh_blrm_approximate = dlr[starsh_params.backend][
        starsh_params.lrengine];
    return 0;
}

int starsh_set_lrengine(const char *string)
//! Set low-rank engine (SVD, Randomized SVD or Cross) for computations
/*! @param[in] string: Name of low-rank engine to use. Possible values:
 * "SVD", "DCSVD", "RRQR", "RSVD", "CROSS"
 * */
{
    int i, selected = -1;
    if(string == NULL)
    {
        selected = LRENGINE_DEFAULT;
    }
    else
    {
        for(i = 0; i < LRENGINE_NUM; i++)
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
        selected = LRENGINE_DEFAULT;
        fprintf(stderr, "Environment variable STARSH_LRENGINE=%s is invalid, "
                "using default low-rank engine (%s)\n",
                string, lrengine[selected].string);
    }
    starsh_params.lrengine = lrengine[selected].lrengine;
    fprintf(stderr, "Selected low-rank engine is %s\n",
            lrengine[selected].string);
    starsh_blrm_approximate = dlr[starsh_params.backend][
        starsh_params.lrengine];
    return 0;
}
