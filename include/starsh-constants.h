/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * @file starsh-constants.h
 * @version 1.0.0.2
 * @author Aleksandr Mikhalev
 * @date 16 May 2017
 * */

#ifndef __STARSH_CONSTANTS_H__
#define __STARSH_CONSTANTS_H__

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
