/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/spatial.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-22
 */

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"

int starsh_ssdata_new(STARSH_ssdata **data, int n, char dtype, int ndim,
        double beta, double nu, double noise,
        enum STARSH_PARTICLES_PLACEMENT place, double sigma)
//! Generate spatial statistics data.
/*! @ingroup applications
 * @param[out] data: Address of pointer to `STARSH_ssdata` object.
 * @param[in] n: Size of matrix.
 * @param[in] dtype: Precision ('s', 'd', 'c' or 'z').
 * @param[in] ndim: Dimensionality of spatial statisics problem.
 * @param[in] beta: Correlation length.
 * @param[in] nu: Smoothing parameter for Matern kernel.
 * @param[in] noise: Value to add to diagonal elements.
 * @param[in] place: Placement strategy for spatial points.
 * @param[in] sigma: Variance.
 * @return Error code.
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("invalid value of `data`");
        return STARSH_WRONG_PARAMETER;
    }
    if(n <= 0)
    {
        STARSH_ERROR("invalid value of `n`");
        return STARSH_WRONG_PARAMETER;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("invalid value of `beta`");
        return STARSH_WRONG_PARAMETER;
    }
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return STARSH_WRONG_PARAMETER;
    }
    if(noise < 0)
    {
        STARSH_ERROR("invalid value of `noise`");
        return STARSH_WRONG_PARAMETER;
    }
    if(sigma < 0)
    {
        STARSH_ERROR("invalid value of `sigma`");
        return STARSH_WRONG_PARAMETER;
    }
    int info;
    STARSH_particles *particles;
    info = starsh_particles_generate(&particles, n, ndim, place);
    if(info != STARSH_SUCCESS)
    {
        fprintf(stderr, "INFO=%d\n", info);
        return info;
    }
    STARSH_MALLOC(*data, 1);
    (*data)->particles = *particles;
    free(particles);
    (*data)->dtype = dtype;
    (*data)->beta = beta;
    (*data)->nu = nu;
    (*data)->noise = noise;
    (*data)->sigma = sigma;
    return STARSH_SUCCESS;
}

int starsh_ssdata_new_va(STARSH_ssdata **data, int n, char dtype,
        va_list args)
//! Generate spatial statistics data with va_list.
//! For more info look at starsh_ssdata_new().
//! @ingroup applications
{
    int arg_type;
    // Set default values
    int ndim = 2;
    double beta = 0.1;
    double nu = 0.5;
    double noise = 0;
    int place = STARSH_PARTICLES_UNIFORM;
    double sigma = 1.0;
    int info;
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return STARSH_WRONG_PARAMETER;
    }
    while((arg_type = va_arg(args, int)) != 0)
    {
        switch(arg_type)
        {
            case STARSH_SPATIAL_NDIM:
                ndim = va_arg(args, int);
                break;
            case STARSH_SPATIAL_BETA:
                beta = va_arg(args, double);
                break;
            case STARSH_SPATIAL_NU:
                nu = va_arg(args, double);
                break;
            case STARSH_SPATIAL_NOISE:
                noise = va_arg(args, double);
                break;
            case STARSH_SPATIAL_PLACE:
                place = va_arg(args, int);
                break;
            case STARSH_SPATIAL_SIGMA:
                sigma = va_arg(args, double);
                break;
            default:
                STARSH_ERROR("Wrong parameter type");
                return STARSH_WRONG_PARAMETER;
        }
    }
    info = starsh_ssdata_new(data, n, dtype, ndim, beta, nu, noise, place,
            sigma);
    return info;
}

int starsh_ssdata_new_el(STARSH_ssdata **data, int n, char dtype, ...)
//! Generate spatial statistics data with ellipsis.
//! For more info look at starsh_ssdata_new().
//! @ingroup applications
{
    va_list args;
    va_start(args, dtype);
    int info = starsh_ssdata_new_va(data, n, dtype, args);
    va_end(args);
    return info;
}

void starsh_ssdata_free(STARSH_ssdata *data)
//! Free data.
//! @ingroup applications
{
    starsh_particles_free(&data->particles);
}

static int starsh_ssdata_get_kernel_1d(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for 1-dimensional spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return STARSH_WRONG_PARAMETER;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_1d;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_1d_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqrexp_kernel_1d;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqrexp_kernel_1d_simd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_1d;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_1d_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_1d;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_1d_simd;
            break;
#else
        case STARSH_SPATIAL_MATERN:
        case STARSH_SPATIAL_MATERN_SIMD:
        case STARSH_SPATIAL_MATERN2:
        case STARSH_SPATIAL_MATERN2_SIMD:
            STARSH_ERROR("Matern kernel requires GSL library, which was "
                    "not found");
            return STARSH_WRONG_PARAMETER;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;;
    }
    return STARSH_SUCCESS;
}

static int starsh_ssdata_get_kernel_2d(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for 2-dimensional spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return STARSH_WRONG_PARAMETER;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_2d;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_2d_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqrexp_kernel_2d;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqrexp_kernel_2d_simd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_2d;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_2d_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_2d;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_2d_simd;
            break;
#else
        case STARSH_SPATIAL_MATERN:
        case STARSH_SPATIAL_MATERN_SIMD:
        case STARSH_SPATIAL_MATERN2:
        case STARSH_SPATIAL_MATERN2_SIMD:
            STARSH_ERROR("Matern kernel requires GSL library, which was "
                    "not found");
            return STARSH_WRONG_PARAMETER;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_ssdata_get_kernel_3d(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for 3-dimensional spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return STARSH_WRONG_PARAMETER;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_3d;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_3d_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqrexp_kernel_3d;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqrexp_kernel_3d_simd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_3d;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_3d_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_3d;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_3d_simd;
            break;
#else
        case STARSH_SPATIAL_MATERN:
        case STARSH_SPATIAL_MATERN_SIMD:
        case STARSH_SPATIAL_MATERN2:
        case STARSH_SPATIAL_MATERN2_SIMD:
            STARSH_ERROR("Matern kernel requires GSL library, which was "
                    "not found");
            return STARSH_WRONG_PARAMETER;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_ssdata_get_kernel_4d(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for 4-dimensional spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return STARSH_WRONG_PARAMETER;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_4d;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_4d_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqrexp_kernel_4d;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqrexp_kernel_4d_simd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_4d;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_4d_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_4d;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_4d_simd;
            break;
#else
        case STARSH_SPATIAL_MATERN:
        case STARSH_SPATIAL_MATERN_SIMD:
        case STARSH_SPATIAL_MATERN2:
        case STARSH_SPATIAL_MATERN2_SIMD:
            STARSH_ERROR("Matern kernel requires GSL library, which was "
                    "not found");
            return STARSH_WRONG_PARAMETER;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_ssdata_get_kernel_nd(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for n-dimensional spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return STARSH_WRONG_PARAMETER;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_nd;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_nd_simd;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqrexp_kernel_nd_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqrexp_kernel_nd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_nd;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_nd_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_nd;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_nd_simd;
            break;
#else
        case STARSH_SPATIAL_MATERN:
        case STARSH_SPATIAL_MATERN_SIMD:
        case STARSH_SPATIAL_MATERN2:
        case STARSH_SPATIAL_MATERN2_SIMD:
            STARSH_ERROR("Matern kernel requires GSL library, which was "
                    "not found");
            return STARSH_WRONG_PARAMETER;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

int starsh_ssdata_get_kernel(STARSH_kernel *kernel, STARSH_ssdata *data,
        int type)
//! Get corresponding kernel for spatial statistics problem.
//! @ingroup applications
{
    switch(data->particles.ndim)
    {
        case 1:
            return starsh_ssdata_get_kernel_1d(kernel, type, data->dtype);
        case 2:
            return starsh_ssdata_get_kernel_2d(kernel, type, data->dtype);
        case 3:
            return starsh_ssdata_get_kernel_3d(kernel, type, data->dtype);
        case 4:
            return starsh_ssdata_get_kernel_4d(kernel, type, data->dtype);
        default:
            return starsh_ssdata_get_kernel_nd(kernel, type, data->dtype);
    }
}
