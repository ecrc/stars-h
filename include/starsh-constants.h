/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-constants.h
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-22
 * */

#ifndef __STARSH_CONSTANTS_H__
#define __STARSH_CONSTANTS_H__

//! Enum for backend types
enum STARSH_BACKEND
{
    STARSH_BACKEND_NOTSELECTED = -2,
    //!< Backend has not been yet selected
    STARSH_BACKEND_NOTSUPPORTED = -1,
    //!< Error, backend is not supported
    STARSH_BACKEND_SEQUENTIAL = 0,
    //!< Sequential
    STARSH_BACKEND_OPENMP = 1,
    //!< OpenMP
    STARSH_BACKEND_MPI = 2,
    //!< MPI
    STARSH_BACKEND_MPI_OPENMP = 3,
    //!< Hybrid MPI + OpenMP
    STARSH_BACKEND_STARPU = 4,
    //!< StarPU (without MPI)
    STARSH_BACKEND_MPI_STARPU = 5
    //!< StarPU (with MPI)
};

//! Enum for low-rank engine (approximation technique)
enum STARSH_LRENGINE
{
    STARSH_LRENGINE_NOTSELECTED = -2,
    //!< Engine has not been yet selected
    STARSH_LRENGINE_NOTSUPPORTED = -1,
    //!< Error, engine is not supported
    STARSH_LRENGINE_SVD = 0,
    //!< Usual Singular Values Decomposition
    STARSH_LRENGINE_DCSVD = 1,
    //!< Divide-and-Conquer SVD
    STARSH_LRENGINE_RRQR = 2,
    //!< Rank-Revealing QR
    STARSH_LRENGINE_RSVD = 3,
    //!< Randomized SVD
    STARSH_LRENGINE_CROSS = 4,
    //!< Cross approximation
};

//! Enum for error codes
enum STARSH_ERRNO
{
    STARSH_SUCCESS = 0,
    //!< No error
    STARSH_MALLOC_ERROR = 1,
    //!< Error of malloc(), maybe there were no free memory
    STARSH_FILE_NOT_EXIST = 2,
    //!< Such file does not exist
    STARSH_FILE_WRONG_INPUT = 3,
    //!< Wrong format of data inside ASCII file
    STARSH_WRONG_PARAMETER = 4,
    //!< Wrong value of one of parameters
    STARSH_FPRINTF_ERROR = 5,
    //!< Error during fprintf()
    STARSH_FWRITE_ERROR = 6,
    //!< Error during fwrite()
    STARSH_UNKNOWN_ERROR = -1
    //!< Error, not listed in enum STARSH_ERRNO
};

//! Enum for problem types
enum STARSH_PROBLEM_TYPE
{
    STARSH_MINIMAL = 1,
    //!< Minimal working example
    STARSH_RNDTILED = 2,
    //!< Random tile-low rank matrix
    STARSH_SPATIAL = 3
    //!< Geospatial statistics
};

//! Enum type to show actual block low-rank format
enum STARSH_BLRF_TYPE
{
    STARSH_TILED = 1,
    //!< Tile low-rank format
    //STARSH_H = 2,
    ////!< H format
    //STARSH_HODLR = 3
    ////!< HODLR format
};

//! Enum type to show type of clusterization
enum STARSH_CLUSTER_TYPE
{
    STARSH_PLAIN = 1,
    //!< No hierarchy in clusterization
    //STARSH_HIERARCHICAL = 2
    ////!< Hierarchical clusterization
};

//! Enum type to show file format (ASCII or binary)
enum STARSH_FILE_TYPE
{
    STARSH_ASCII = 1,
    //!< ASCII file
    STARSH_BINARY = 2
    //!< Binary file
};

//! Enum type to select placement of particles for STARSH_particles
enum STARSH_PARTICLES_PLACEMENT
{
    STARSH_PARTICLES_RAND = 1,
    //!< Uniform random distribution in [0,1] range
    //STARSH_PARTICLES_RANDN = 2,
    ////!< Normal random distribution with mean=0 and variance=1
    STARSH_PARTICLES_UNIFORM = 3,
    //!< Uniform in [0,1] grid
    STARSH_PARTICLES_RANDGRID = 4,
    //!< Grid, based on uniform in [0,1] distribution of coordinates
    //STARSH_PARTICLES_RANDNGRID = 5,
    ////!< Grid, based on normal (0,1) ditribution of coordinates
    STARSH_PARTICLES_QUASIUNIFORM1 = 6,
    //!< Uniform in [0,1] grid, but each particle is slightly shifted
    STARSH_PARTICLES_QUASIUNIFORM2 = 7,
    //!< Uniform in [0,1] grid, but each grid coordinate is slightly shifted
    STARSH_PARTICLES_OBSOLETE1 = -1,
    //!< Old version of STARSH_PARTICLES_QUASIUNIFORM1 (keep for compatibility)
    STARSH_PARTICLES_OBSOLETE2 = -2,
    //!< Old version of STARSH_PARTICLES_QUASIUNIFORM2 (keep for compatibility)
};

#endif // __STARSH_CONSTANTS_H__
