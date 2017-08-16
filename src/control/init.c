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

//! Parameters of STARS-H
struct starsh_params starsh_params;

//! Set number of backends and default one
#define BACKEND_NUM 6
#define BACKEND_DEFAULT STARSH_BACKEND_SEQUENTIAL
#ifdef OPENMP
    #undef BACKEND_DEFAULT
    #define BACKEND_DEFAULT STARSH_BACKEND_OPENMP
#endif
#ifdef MPI
    #undef BACKEND_DEFAULT
    #define BACKEND_DEFAULT STARSH_BACKEND_MPI
#endif
#if defined(OPENMP) && defined(MPI)
    #undef BACKEND_DEFAULT
    #define BACKEND_DEFAULT STARSH_BACKEND_MPI_OPENMP
#endif

//! Array of backends, each presented by string and enum value
struct
{
    const char *string;
    enum STARSH_BACKEND backend;
} const backend[BACKEND_NUM] =
{
    {"SEQUENTIAL", STARSH_BACKEND_SEQUENTIAL},
#ifdef OPENMP
    {"OPENMP", STARSH_BACKEND_OPENMP},
#else
    {"OPENMP", STARSH_BACKEND_NOTSUPPORTED},
#endif
#ifdef MPI
    {"MPI", STARSH_BACKEND_MPI},
#else
    {"MPI", STARSH_BACKEND_NOTSUPPORTED},
#endif
#if defined(OPENMP) && defined(MPI)
    {"MPI_OPENMP", STARSH_BACKEND_MPI_OPENMP},
#else
    {"MPI_OPENMP", STARSH_BACKEND_NOTSUPPORTED},
#endif
#ifdef STARPU
    {"STARPU", STARSH_BACKEND_STARPU},
#else
    {"STARPU", STARSH_BACKEND_NOTSUPPORTED},
#endif
#if defined(STARPU) && defined(MPI)
    {"MPI_STARPU", STARSH_BACKEND_MPI_STARPU},
#else
    {"MPI_STARPU", STARSH_BACKEND_NOTSUPPORTED},
#endif
};

//! Set number of low-rank engines and default one
#define LRENGINE_NUM 5
#define LRENGINE_DEFAULT STARSH_LRENGINE_RSVD
//! Array of low-rank engines, presented by string and enum value
struct
{
    const char *string;
    enum STARSH_LRENGINE lrengine;
} const lrengine[LRENGINE_NUM] =
{
    {"SVD", STARSH_LRENGINE_SVD},
    {"DCSVD", STARSH_LRENGINE_DCSVD},
    {"RRQR", STARSH_LRENGINE_RRQR},
    {"RSVD", STARSH_LRENGINE_RSVD},
    {"CROSS", STARSH_LRENGINE_CROSS},
};

//! Array of approximation functions for NOTSUPPORTED backend
static STARSH_blrm_approximate *(dlr_none[LRENGINE_NUM]) = {};

//! Array of approximation functions for SEQUENTIAL backend
static STARSH_blrm_approximate *(dlr_seq[LRENGINE_NUM]) =
{
    starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3,
    starsh_blrm__drsdd, starsh_blrm__drsdd
};

//! Array of approximation functions for OMP backend
static STARSH_blrm_approximate *(dlr_omp[LRENGINE_NUM]) =
{
    starsh_blrm__dsdd_omp, starsh_blrm__dsdd_omp, starsh_blrm__dqp3_omp,
    starsh_blrm__drsdd_omp, starsh_blrm__drsdd_omp
};

//! Array of approximation functions for MPI and MPI_OPENMP backends
static STARSH_blrm_approximate *(dlr_mpi[LRENGINE_NUM]) =
{
    starsh_blrm__dsdd_mpi, starsh_blrm__dsdd_mpi, starsh_blrm__dqp3_mpi,
    starsh_blrm__drsdd_mpi, starsh_blrm__drsdd_mpi
};

//! Array of approximation functions, depending on backend
static STARSH_blrm_approximate *(*dlr[BACKEND_NUM]) =
{
    dlr_seq, dlr_omp, dlr_mpi, dlr_mpi, dlr_none, dlr_none
};

int starsh_init()
//! Initialize backend and low-rank engine to be used
{
    const char *str_backend = "STARSH_BACKEND";
    const char *str_lrengine = "STARSH_LRENGINE";
    const static struct starsh_params starsh_params_default =
    {
        BACKEND_DEFAULT, LRENGINE_DEFAULT, 10
    };
    //starsh_params.oversample = 10;
    //starsh_params.backend = STARSH_BACKEND_OPENMP;
    //starsh_params.lrengine = STARSH_LRENGINE_RSVD;
    starsh_params = starsh_params_default;
    int info = 0, i;
    info |= starsh_set_backend(getenv(str_backend));
    info |= starsh_set_lrengine(getenv(str_lrengine));
    return info;
}

int starsh_set_backend(const char *string)
//! Set backend (MPI or OpenMP or other scheduler) for computations
/*! @param[in] string: name of backend to use
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
/*! @param[in] string: name of low-rank engine to use
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
