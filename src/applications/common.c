/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/common.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-electrostatics.h"
#include "starsh-electrodynamics.h"
#include "starsh-minimal.h"
#include "starsh-randtlr.h"
#include "starsh-cauchy.h"

int starsh_application(void **data, STARSH_kernel **kernel, STARSH_int count,
        char dtype, int problem_type, int kernel_type, ...)
//! Generates data and matrix kernel for one of predefined applications.
/*! All parameters in `...` go by pairs: integer, indicating what kind of
 * parameter is following after it, with the value of parameter. This list ends
 * with integer 0. Parameter `dtype` is ignored as of now, but it is here to
 * support different precisions.
 *
 * @param[out] data: Address of pointer for structure, holding all data.
 * @param[out] kernel: @ref STARSH_kernel function.
 * @param[in] count: Desired size of corresponding matrix.
 * @param[in] dtype: Precision of each element of a matrix ('d' for double).
 * @param[in] problem_type: Type of problem.
 * @param[in] kernel_type: Type of kernel, depends on problem.
 * @sa starsh_ssdata_generate(), starsh_ssdata_get_kernel(),
 *      starsh_randtlr_generate(), starsh_randtlr_get_kernel(),
 *      starsh_esdata_generate(), starsh_esdata_get_kernel(),
 *      starsh_eddata_generate(), starsh_eddata_get_kernel(),
 *      starsh_cauchy_new(), starsh_cauchy_get_kernel(),
 *      starsh_mindata_new(), starsh_mindata_get_kernel().
 * @ingroup applications
 */
{
    va_list args;
    va_start(args, kernel_type);
    int info = STARSH_SUCCESS;
    switch(problem_type)
    {
        case STARSH_MINIMAL:
            // Minimal example simply does not support any parameters. So, if
            // there are any, throw error.
            if(va_arg(args, int) != 0)
                return STARSH_WRONG_PARAMETER;
            info = starsh_mindata_new((STARSH_mindata **)data, count, dtype);
            if(info != STARSH_SUCCESS)
                return info;
            info = starsh_mindata_get_kernel(kernel, *data, kernel_type);
            break;
        case STARSH_RANDTLR:
            // Since random TLR matrix supports only double precision, dtype
            // is ignored
            info = starsh_randtlr_generate_va((STARSH_randtlr **)data, count,
                    args);
            if(info != STARSH_SUCCESS)
                return info;
            info = starsh_randtlr_get_kernel(kernel, *data, kernel_type);
            break;
        case STARSH_SPATIAL:
            // Since Spatial statistics supports only double precision, dtype
            // is ignored
            info = starsh_ssdata_generate_va((STARSH_ssdata **)data, count,
                    args);
            if(info != STARSH_SUCCESS)
                return info;
            info = starsh_ssdata_get_kernel(kernel, *data, kernel_type);
            if(info != STARSH_SUCCESS)
                return info;
            break;
        case STARSH_ELECTROSTATICS:
            // Since electrostatics supports only double precision, dtype
            // is ignored
            info = starsh_esdata_generate_va((STARSH_esdata **)data, count,
                    args);
            if(info != STARSH_SUCCESS)
                return info;
            info = starsh_esdata_get_kernel(kernel, *data, kernel_type);
            if(info != STARSH_SUCCESS)
                return info;
            break;
        case STARSH_ELECTRODYNAMICS:
            // Since electrodynamics supports only double precision, dtype
            // is ignored
            info = starsh_eddata_generate_va((STARSH_eddata **)data, count,
                    args);
            if(info != STARSH_SUCCESS)
                return info;
            info = starsh_eddata_get_kernel(kernel, *data, kernel_type);
            if(info != STARSH_SUCCESS)
                return info;
            break;
        case STARSH_CAUCHY:
            info = starsh_cauchy_new_va((STARSH_cauchy **)data, count, args);
            if(info != STARSH_SUCCESS)
                return info;
            info = starsh_cauchy_get_kernel(kernel, *data, kernel_type);
            if(info != STARSH_SUCCESS)
                return info;
            break;
        default:
            STARSH_ERROR("Wrong value of problem_type");
            return STARSH_WRONG_PARAMETER;
    }
    va_end(args);
    return info;
}
