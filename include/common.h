/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/common.h
 *
 * @cond
 * This command in pair with endcond will prevent file from being documented.
 *
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <limits.h>
#include <stdint.h>

#ifdef MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#ifdef OPENMP
    #include <omp.h>
#endif

#ifdef MPI
    #include <mpi.h>
    #ifndef SIZE_MAX
        #error "SIZE_MAX not defined"
    #endif
    #if SIZE_MAX == UCHAR_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
    #elif SIZE_MAX == USHRT_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
    #elif SIZE_MAX == UINT_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED
    #elif SIZE_MAX == ULONG_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
    #elif SIZE_MAX == ULLONG_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
    #else
       #error "No MPI data type fits size_t"
    #endif
#endif

#ifdef STARPU
    #include <starpu.h>
#endif

#ifdef GSL
    #include <gsl/gsl_sf.h>
#endif

#define STARSH_ERROR(format, ...)\
{\
    fprintf(stderr, "STARSH ERROR: %s(): ", __func__);\
    fprintf(stderr, format, ##__VA_ARGS__);\
    fprintf(stderr, "\n");\
}

#ifdef SHOW_WARNINGS
    #define STARSH_WARNING(format, ...)\
    {\
        fprintf(stderr, "STARSH WARNING: %s(): ", __func__);\
        fprintf(stderr, format, ##__VA_ARGS__);\
        fprintf(stderr, "\n");\
    }
#else
    #define STARSH_WARNING(...)
#endif

#define STARSH_MALLOC(var, expr_nitems)\
{\
    var = malloc(sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("line %d: malloc() failed", __LINE__);\
        return STARSH_MALLOC_ERROR;\
    }\
}

#define STARSH_REALLOC(var, expr_nitems)\
{\
    var = realloc(var, sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        return STARSH_MALLOC_ERROR;\
    }\
}

#define STARSH_PMALLOC(var, expr_nitems, var_info)\
{\
    var = malloc(sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        var_info = STARSH_MALLOC_ERROR;\
    }\
}

#define STARSH_PREALLOC(var, expr_nitems, var_info)\
{\
    var = realloc(var, sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        var_info = STARSH_MALLOC_ERROR;\
    }\
}

int cmp_size_t(const void *a, const void *b);

#endif // __COMMON_H__

//! @endcond
