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
    STARSH_BACKEND_SEQUENTIAL = 1,
    //!< Sequential
    STARSH_BACKEND_OMP = 2,
    //!< OpenMP
    STARSH_BACKEND_MPI = 3,
    //!< MPI
    STARSH_BACKEND_MPIOMP = 4,
    //!< Hybrid MPI + OpenMP
    STARSH_BACKEND_STARPU = 5,
    //!< StarPU (without MPI)
    STARSH_BACKEND_STARPUMPI = 6,
    //!< StarPU (with MPI)
};

//! Enum for low-rank engine (approximation technique)
enum STARSH_LRENGINE
{
    STARSH_LRENGINE_SVD = 1,
    //!< Usual Singular Values Decomposition
    STARSH_LRENGINE_DCSVD = 2,
    //!< Divide-and-Conquer SVD
    STARSH_LRENGINE_RRQR = 3,
    //!< Rank-Revealing QR
    STARSH_LRENGINE_RSVD = 4,
    //!< Randomized SVD
    STARSH_LRENGINE_CROSS = 5,
    //!< Cross approximation
};

//! Enum for error codes
enum STARSH_ERRNO
{
    STARSH_SUCCESS = 0,
    MALLOC_ERROR = 1,
    STARSH_UNKNOWN_ERROR = -1
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
