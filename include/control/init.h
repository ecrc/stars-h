/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/init.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-13
 * */

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

//! Parameters of STARS-H
struct starsh_params starsh_params =
{
    STARSH_BACKEND_NOTSELECTED, STARSH_LRENGINE_NOTSELECTED, -1
};

const static struct starsh_params starsh_params_default =
{
    BACKEND_DEFAULT, LRENGINE_DEFAULT, 10
};

//! Array of approximation functions for NOTSUPPORTED backend
static STARSH_blrm_approximate *(dlr_none[LRENGINE_NUM]) = {};

//! Array of approximation functions for SEQUENTIAL backend
static STARSH_blrm_approximate *(dlr_seq[LRENGINE_NUM]) =
{
    starsh_blrm__dsdd, starsh_blrm__dsdd, starsh_blrm__dqp3,
    starsh_blrm__drsdd, starsh_blrm__drsdd
};

//! Array of approximation functions for OPENMP backend
static STARSH_blrm_approximate *(dlr_omp[LRENGINE_NUM]) =
{
    #ifdef OPENMP
    starsh_blrm__dsdd_omp, starsh_blrm__dsdd_omp, starsh_blrm__dqp3_omp,
    starsh_blrm__drsdd_omp, starsh_blrm__drsdd_omp
    #endif
};

//! Array of approximation functions for MPI and MPI_OPENMP backends
static STARSH_blrm_approximate *(dlr_mpi[LRENGINE_NUM]) =
{
    #ifdef MPI
    starsh_blrm__dsdd_mpi, starsh_blrm__dsdd_mpi, starsh_blrm__dqp3_mpi,
    starsh_blrm__drsdd_mpi, starsh_blrm__drsdd_mpi
    #endif
};

//! Array of approximation functions for STARPU backend
static STARSH_blrm_approximate *(dlr_starpu[LRENGINE_NUM]) =
{
    #ifdef STARPU
    starsh_blrm__dsdd_starpu, starsh_blrm__dsdd_starpu,
    starsh_blrm__dqp3_starpu, starsh_blrm__drsdd_starpu,
    starsh_blrm__drsdd_starpu
    #endif
};

//! Array of approximation functions for MPI_STARPU backend
static STARSH_blrm_approximate *(dlr_starpu_mpi[LRENGINE_NUM]) =
{
    #if defined(STARPU) && defined(MPI)
    starsh_blrm__dsdd_mpi_starpu, starsh_blrm__dsdd_mpi_starpu,
    starsh_blrm__dqp3_mpi_starpu, starsh_blrm__drsdd_mpi_starpu,
    starsh_blrm__drsdd_mpi_starpu
    #endif
};

//! Array of approximation functions, depending on backend
static STARSH_blrm_approximate *(*dlr[BACKEND_NUM]) =
{
    dlr_seq, dlr_omp, dlr_mpi, dlr_mpi, dlr_starpu, dlr_starpu_mpi
};

