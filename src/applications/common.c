/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/common.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-05-21
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-minimal.h"
#include "starsh-rndtiled.h"

int starsh_application(void **data, STARSH_kernel *kernel, int n, char dtype,
        int problem_type, int kernel_type, ...)
//! Generates data and matrix kernel for one of predefined applications
/*!
 * @ingroup applications
 * @param[out] data: Address of pointer for structure, holding all data
 * @param[out] kernel: matrix kernel
 * @param[in] n: desired size of corresponding matrix
 * @param[in] dtype: precision of each element of a matrix ('d' for double)
 * @param[in] problem_type: possible values are of
 *                          type enum STARSH_PROBLEM_TYPE
 * @param[in] kernel_type: possible values are of corresponding
 *                          type enum STARSH_*_*, where first star corresponds
 *                          to problem name and seconds start corresponds to
 *                          name of kernel
 * 
 * All other parameters go by pairs: integer, indicating what kind of parameter
 * is following after it, with the value of parameter. This list ends with
 * integer 0.
 */
{
    va_list args;
    va_start(args, kernel_type);
    int info = 0;
    switch(problem_type)
    {
        case STARSH_MINIMAL:
            info = starsh_mindata_new_va((STARSH_mindata **)data, n, dtype,
                    args);
            if(info != 0)
                return info;
            info = starsh_mindata_get_kernel(kernel, *data, kernel_type);
            break;
        case STARSH_RNDTILED:
            info = starsh_rndtiled_new_va((STARSH_rndtiled **)data, n, dtype,
                    args);
            if(info != 0)
                return info;
            info = starsh_rndtiled_get_kernel(kernel, *data, kernel_type);
            break;
        case STARSH_SPATIAL:
            info = starsh_ssdata_new_va((STARSH_ssdata **)data, n, dtype,
                    args);
            if(info != 0)
                return info;
            info = starsh_ssdata_get_kernel(kernel, *data, kernel_type);
            break;
        default:
            STARSH_ERROR("Wrong value of problem_type");
            return 1;
    }
    va_end(args);
    return info;
}
