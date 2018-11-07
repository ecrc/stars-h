/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/init.c
 * @version 0.1.1
 * @author Aleksandr Mikhalev
 * @date 2018-11-06
 * */

#include "starsh.h"
#include "starsh-mpi.h"
#include "starsh-starpu.h"
#include "starsh-mpi-starpu.h"
#include "common.h"
#include "control/init.h"

// Approximation routine, chosen by starsh_init()
STARSH_blrm_approximate *starsh_blrm_approximate = NULL;

int starsh_init()
//! Initialize backend and low-rank engine to be used.
/*! Read environment variables and sets up backend (etc. MPI) and low-rank
 * engine (etc. SVD).
 *
 * Environment variables
 * ---------------------
 *  STARSH_BACKEND: SEQUENTIAL, MPI (pure MPI), OPENMP (pure OpenMP) or
 *  MPI_OPENMP (hybrid MPI with OpenMP).
 *
 *  STARSH_LRENGINE: SVD (divide-and-conquer SVD), RRQR (LAPACK *geqp3) or
 *  RSVD (randomized SVD).
 *
 *  STARSH_OVERSAMPLE: Number of oversampling vectors for randomized SVD and
 *  RRQR.
 *
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_set_backend(), starsh_set_lrengine().
 * */
{
    const char *str_backend = "STARSH_BACKEND";
    const char *str_lrengine = "STARSH_LRENGINE";
    const char *str_oversample = "STARSH_OVERSAMPLE";
    //starsh_params = starsh_params_default;
    int info = 0, i;
    // Set backend by STARSH_BACKEND
    info = starsh_set_backend(getenv(str_backend));
    // If attempt to use user-defined value fails, then use default one
    if(info != STARSH_SUCCESS)
        starsh_set_backend(NULL);
    // Set low-rank engine by STARSH_LRENGINE
    info = starsh_set_lrengine(getenv(str_lrengine));
    // If attempt to use user-defined value fails, then use default one
    if(info != STARSH_SUCCESS)
        starsh_set_lrengine(NULL);
    // Set oversampling size by STARSH_OVERSAMPLE
    info = starsh_set_oversample(getenv(str_oversample));
    // If attempt to use user-defined value fails, then use default one
    if(info != STARSH_SUCCESS)
        starsh_set_oversample(NULL);
    return STARSH_SUCCESS;
}

int starsh_set_backend(const char *string)
//! Set backend (MPI or OpenMP or other scheduler) for computations.
/*! @param[in] string: Environment variable and value, encoded in a string.
 *      Example: "STARSH_BACKEND=SEQUENTIAL".
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_init(), starsh_set_lrengine().
 * */
{
    int i, selected = -1;
    if(string == NULL)
    {
        selected = starsh_params_default.backend;
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
        fprintf(stderr, "Environment variable STARSH_BACKEND=%s is invalid\n",
                string);
        return STARSH_WRONG_PARAMETER;
    }
    if(backend[selected].backend == STARSH_BACKEND_NOTSUPPORTED)
    {
        fprintf(stderr, "Specified backend %s is not supported\n", string);
        return STARSH_WRONG_PARAMETER;
    }
    starsh_params.backend = backend[selected].backend;
    fprintf(stderr, "Selected backend is %s\n", backend[selected].string);
    starsh_blrm_approximate = dlr[starsh_params.backend][
        starsh_params.lrengine];
    return STARSH_SUCCESS;
}

int starsh_set_lrengine(const char *string)
//! Set low-rank engine (SVD, Randomized SVD or Cross) for computations.
/*! @param[in] string: Environment variable and value, encoded in a string.
 *      Example: "STARSH_LRENGINE=SVD".
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_init(), starsh_set_backends().
 * */
{
    int i, selected = -1;
    if(string == NULL)
    {
        selected = starsh_params_default.lrengine;
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
        fprintf(stderr, "Environment variable STARSH_LRENGINE=%s is invalid\n",
                string);
        return STARSH_WRONG_PARAMETER;
    }
    starsh_params.lrengine = lrengine[selected].lrengine;
    fprintf(stderr, "Selected low-rank engine is %s\n",
            lrengine[selected].string);
    starsh_blrm_approximate = dlr[starsh_params.backend][
        starsh_params.lrengine];
    return STARSH_SUCCESS;
}

int starsh_set_oversample(const char *string)
//! Set oversampling size for randomized SVD and RRQR.
/*! @param[in] string: Environment variable and value, encoded in a string.
 *      Example: "STARSH_OVERSAMPLE=10".
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_init().
 * */
{
    int value;
    if(string == NULL)
    {
        value = starsh_params_default.oversample;
    }
    else
    {
        value = atoi(string);
    }
    if(value <= 0)
    {
        fprintf(stderr, "Environment variable STARSH_OVERSAMPLE=%s is "
                "invalid\n", string);
        return STARSH_WRONG_PARAMETER;
    }
    fprintf(stderr, "Selected oversample size %d\n", value);
    starsh_params.oversample = value;
    return STARSH_SUCCESS;
}
