/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/spatial.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 */

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"

int starsh_ssdata_new(STARSH_ssdata **data, STARSH_int count, int ndim)
//! Allocate memory for @ref STARSH_ssdata object.
/*! This functions only allocates memory for particles without setting
 * coordinates to any value.
 * Do not forget to sort `data->particles` by starsh_particles_zsort_inplace()
 * to take advantage of low-rank submatrices.
 *
 * @param[out] data: Address of pointer to @ref STARSH_ssdata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_ssdata_init(), starsh_ssdata_free(),
 *      starsh_ssdata_generate(), starsh_ssdata_read_from_file(),
 *      starsh_particles_zsort_inplace(), STARSH_particles.
 * @ingroup app-spatial
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("Invalid value of `data`");
        return STARSH_WRONG_PARAMETER;
    }
    if(ndim <= 0)
    {
        STARSH_ERROR("Invalid value of `ndim`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_ssdata *tmp;
    STARSH_MALLOC(tmp, 1);
    tmp->particles.count = count;
    tmp->particles.ndim = ndim;
    STARSH_MALLOC(tmp->particles.point, count*ndim);
    return STARSH_SUCCESS;
}

int starsh_ssdata_init(STARSH_ssdata **data, STARSH_int count, int ndim,
        double *point, double beta, double nu, double noise, double sigma)
//! Initialize @ref STARSH_ssdata object by given data.
/*! Array `point` should be stored in a special way: `x_1 x_2 ... x_count y_1
 * y_2 ... y_count z_1 z_2 ...`.
 * This function does not allocate memory for coordinates and uses provided
 * pointer `point`. Do not free memory of `point` until you finish using
 * returned @ref STARSH_ssdata object.
 * Do not forget to sort `data->particles` by starsh_particles_zsort_inplace()
 * to take advantage of low-rank submatrices.
 *
 * @param[out] data: Address of pointer to @ref STARSH_ssdata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @param[in] point: Pointer to array of coordinates of particles.
 * @param[in] beta: Correlation length.
 * @param[in] nu: Smoothing parameter for Mat&eacute;rn kernel.
 * @param[in] noise: Value to add to diagonal elements.
 * @param[in] sigma: Variance.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_ssdata_new(), starsh_ssdata_free(),
 *      starsh_ssdata_generate(), starsh_ssdata_read_from_file(),
 *      starsh_particles_zsort_inplace().
 * @ingroup app-spatial
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("Invalid value of `data`");
        return STARSH_WRONG_PARAMETER;
    }
    if(ndim <= 0)
    {
        STARSH_ERROR("Invalid value of `ndim`");
        return STARSH_WRONG_PARAMETER;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("Invalid value of `beta`");
        return STARSH_WRONG_PARAMETER;
    }
    if(nu < 0)
    {
        STARSH_ERROR("Invalid value of `nu`");
        return STARSH_WRONG_PARAMETER;
    }
    if(noise < 0)
    {
        STARSH_ERROR("Invalid value of `noise`");
        return STARSH_WRONG_PARAMETER;
    }
    if(sigma < 0)
    {
        STARSH_ERROR("Invalid value of `sigma`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_ssdata *tmp;
    STARSH_MALLOC(tmp, 1);
    tmp->particles.count = count;
    tmp->particles.ndim = ndim;
    tmp->particles.point = point;
    tmp->beta = beta;
    tmp->nu = nu;
    tmp->noise = noise;
    tmp->sigma = sigma;
    return STARSH_SUCCESS;
}

int starsh_ssdata_generate(STARSH_ssdata **data, STARSH_int count, int ndim,
        double beta, double nu, double noise,
        enum STARSH_PARTICLES_PLACEMENT place, double sigma)
//! Generate @ref STARSH_ssdata object by given distribution.
/*! @param[out] data: Address of pointer to @ref STARSH_ssdata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @param[in] beta: Correlation length.
 * @param[in] nu: Smoothing parameter for Mat&eacute;rn kernel.
 * @param[in] noise: Value to add to diagonal elements.
 * @param[in] place: Placement strategy for spatial points.
 * @param[in] sigma: Variance.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_ssdata_generate_va(), starsh_ssdata_generate_el().
 * @ingroup app-spatial
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("Invalid value of `data`");
        return STARSH_WRONG_PARAMETER;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("Invalid value of `beta`");
        return STARSH_WRONG_PARAMETER;
    }
    if(nu < 0)
    {
        STARSH_ERROR("Invalid value of `nu`");
        return STARSH_WRONG_PARAMETER;
    }
    if(noise < 0)
    {
        STARSH_ERROR("Invalid value of `noise`");
        return STARSH_WRONG_PARAMETER;
    }
    if(sigma < 0)
    {
        STARSH_ERROR("Invalid value of `sigma`");
        return STARSH_WRONG_PARAMETER;
    }
    int info;
    STARSH_particles *particles;
    info = starsh_particles_generate(&particles, count, ndim, place);
    if(info != STARSH_SUCCESS)
    {
        fprintf(stderr, "INFO=%d\n", info);
        return info;
    }
    STARSH_MALLOC(*data, 1);
    (*data)->particles = *particles;
    free(particles);
    (*data)->beta = beta;
    (*data)->nu = nu;
    (*data)->noise = noise;
    (*data)->sigma = sigma;
    return STARSH_SUCCESS;
}

int starsh_ssdata_generate_va(STARSH_ssdata **data, STARSH_int count,
        va_list args)
//! Generate @ref STARSH_ssdata object with incomplete set of parameters.
/*! Parse possibly incomplete set of parameters for starsh_ssdata_generate().
 * If argument is not in the `args`, then its default value is used:
 *
 * Argument | Default value | Type
 * ---------|---------------|--------
 * `ndim`   | 2             | int
 * `beta`   | 0.1           | double
 * `nu`     | 0.5           | double
 * `noise`  | 0.0           | double
 * `place`  | @ref STARSH_PARTICLES_UNIFORM | @ref STARSH_PARTICLES_PLACEMENT
 * `sigma`  | 1.0           | double
 *
 * List of arguments `args` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_SPATIAL_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_ssdata object.
 * @param[in] count: Number of particles.
 * @param[in] args: Arguments, packed into va_args.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      void generate(size_t count, ...)
 *      {
 *          STARSH_ssdata *data;
 *          va_list args;
 *          va_start(args, count);
 *          starsh_ssdata_generate_va(&data, count, args);
 *          va_end(args);
 *      }
 * @endcode
 * @sa starsh_ssdata_generate(), starsh_ssdata_generate_el().
 * @ingroup app-spatial
 * */
{
    int arg_type;
    // Set default values
    int ndim = 2;
    double beta = 0.1;
    double nu = 0.5;
    double noise = 0;
    enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
    double sigma = 1.0;
    int info;
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
                place = va_arg(args, enum STARSH_PARTICLES_PLACEMENT);
                break;
            case STARSH_SPATIAL_SIGMA:
                sigma = va_arg(args, double);
                break;
            default:
                STARSH_ERROR("Wrong parameter type");
                return STARSH_WRONG_PARAMETER;
        }
    }
    info = starsh_ssdata_generate(data, count, ndim, beta, nu, noise, place,
            sigma);
    return info;
}

int starsh_ssdata_generate_el(STARSH_ssdata **data, STARSH_int count, ...)
//! Generate @ref STARSH_ssdata object with incomplete set of parameters.
/*! Parse possibly incomplete set of parameters for starsh_ssdata_generate().
 * If argument is not in the `...`, then its default value is used:
 *
 * Argument | Default value | Type
 * ---------|---------------|--------
 * `ndim`   | 2             | int
 * `beta`   | 0.1           | double
 * `nu`     | 0.5           | double
 * `noise`  | 0.0           | double
 * `place`  | @ref STARSH_PARTICLES_UNIFORM | @ref STARSH_PARTICLES_PLACEMENT
 * `sigma`  | 1.0           | double
 *
 * List of arguments in `...` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_SPATIAL_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_ssdata object.
 * @param[in] count: Number of particles.
 * @param[in] ...: Variable amount of arguments.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      starsh_ssdata_generate_el(&data, count,
 *          STARSH_SPATIAL_PLACE, STARSH_PARTICLES_RAND,
 *          STARSH_SPATIAL_NOISE, 0.1,
 *          STARSH_SPATIAL_SIGMA, 0.9,
 *          0).
 * @endcode
 * @arg @code{.c}
 *      starsh_ssdata_generate_el(&data, count,
 *          STARSH_SPATIAL_NDIM, 3,
 *          STARSH_SPATIAL_BETA, 0.2,
 *          STARSH_SPATIAL_NU, 1.5,
 *          STARSH_SPATIAL_NOISE, 0.1,
 *          STARSH_SPATIAL_PLACE, STARSH_PARTICLES_RAND,
 *          STARSH_SPATIAL_SIGMA, 0.9,
 *          0).
 * @endcode
 * @sa starsh_ssdata_generate(), starsh_ssdata_generate_va().
 * @ingroup app-spatial
 * */
{
    va_list args;
    va_start(args, count);
    int info = starsh_ssdata_generate_va(data, count, args);
    va_end(args);
    return info;
}

void starsh_ssdata_free(STARSH_ssdata *data)
//! Free memory of @ref STARSH_ssdata object.
/*! @sa starsh_ssdata_new(), starsh_ssdata_init(), starsh_ssdata_generate().
 * @ingroup app-spatial
 * */
{
    starsh_particles_free(&data->particles);
}

static int starsh_ssdata_get_kernel_1d(STARSH_kernel **kernel,
        enum STARSH_SPATIAL_KERNEL type)
// Get kernel for 1-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
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
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_ssdata_get_kernel_2d(STARSH_kernel **kernel,
        enum STARSH_SPATIAL_KERNEL type)
// Get kernel for 2-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
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

static int starsh_ssdata_get_kernel_3d(STARSH_kernel **kernel,
        enum STARSH_SPATIAL_KERNEL type)
// Get kernel for 3-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
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

static int starsh_ssdata_get_kernel_4d(STARSH_kernel **kernel,
        enum STARSH_SPATIAL_KERNEL type)
// Get kernel for 4-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
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

static int starsh_ssdata_get_kernel_nd(STARSH_kernel **kernel,
        enum STARSH_SPATIAL_KERNEL type)
// Get corresponding kernel for n-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
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

int starsh_ssdata_get_kernel(STARSH_kernel **kernel, STARSH_ssdata *data,
        enum STARSH_SPATIAL_KERNEL type)
//! Get kernel for spatial statistics problem.
/*! Kernel can be selected with this call or manually. To select kernel
 * manually look into @ref app-spatial-kernels.
 *
 * @param[out] kernel: Address of pointer to @ref STARSH_kernel function.
 * @param[in] data: Pointer to @ref STARSH_ssdata object.
 * @param[in] type: Type of kernel. For more info look at @ref
 *      STARSH_SPATIAL_KERNEL.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_ssdata_block_exp_kernel_nd(),
 *      starsh_ssdata_block_exp_kernel_nd_simd(),
 *      starsh_ssdata_block_sqrexp_kernel_nd(),
 *      starsh_ssdata_block_sqrexp_kernel_nd_simd(),
 *      starsh_ssdata_block_matern_kernel_nd(),
 *      starsh_ssdata_block_matern_kernel_nd_simd(),
 *      starsh_ssdata_block_matern2_kernel_nd(),
 *      starsh_ssdata_block_matern2_kernel_nd_simd().
 * @ingroup app-spatial
 * */
{
    switch(data->particles.ndim)
    {
        case 1:
            return starsh_ssdata_get_kernel_1d(kernel, type);
        case 2:
            return starsh_ssdata_get_kernel_2d(kernel, type);
        case 3:
            return starsh_ssdata_get_kernel_3d(kernel, type);
        case 4:
            return starsh_ssdata_get_kernel_4d(kernel, type);
        default:
            return starsh_ssdata_get_kernel_nd(kernel, type);
    }
}
