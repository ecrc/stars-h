/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-constants.h
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-13
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
    MALLOC_ERROR = 1,
    STARSH_UNKNOWN_ERROR = -1,
    STARSH_WRONG_PARAMETER
};

//! Enum for problem types
enum STARSH_PROBLEM_TYPE
{
    STARSH_MINIMAL = 1,
    STARSH_RNDTILED = 2,
    STARSH_SPATIAL = 3
};

//! Enum type to show actual block low-rank format
enum STARSH_BLRF_TYPE
{
    STARSH_TILED = 1,
    STARSH_H = 2,
    STARSH_HODLR = 3
};

//! Enum type to show type of clusterization
enum STARSH_CLUSTER_TYPE
{
    STARSH_PLAIN = 1,
    STARSH_HIERARCHICAL = 2
};

#endif // __STARSH_CONSTANTS_H__
