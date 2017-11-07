/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/electrodynamics.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 */

#include "common.h"
#include "starsh.h"
#include "starsh-electrodynamics.h"

int starsh_eddata_new(STARSH_eddata **data, STARSH_int count, int ndim)
//! Allocate memory for @ref STARSH_eddata object.
/*! This functions only allocates memory for particles without setting
 * coordinates to any value.
 * Do not forget to sort `data->particles` by starsh_particles_zsort_inplace()
 * to take advantage of low-rank submatrices.
 *
 * @param[out] data: Address of pointer to @ref STARSH_ssdata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_eddata_init(), starsh_eddata_free(),
 *      starsh_eddata_generate(), starsh_eddata_read_from_file(),
 *      starsh_particles_zsort_inplace(), STARSH_particles.
 * @ingroup app-electrodynamics
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
    STARSH_eddata *tmp;
    STARSH_MALLOC(tmp, 1);
    tmp->particles.count = count;
    tmp->particles.ndim = ndim;
    STARSH_MALLOC(tmp->particles.point, count*ndim);
    return STARSH_SUCCESS;
}

int starsh_eddata_init(STARSH_eddata **data, STARSH_int count, int ndim,
        double *point, double k, double diag)
//! Initialize @ref STARSH_eddata object by given data.
/*! Array `point` should be stored in a special way: `x_1 x_2 ... x_count y_1
 * y_2 ... y_count z_1 z_2 ...`.
 * This function does not allocate memory for coordinates and uses provided
 * pointer `point`. Do not free memory of `point` until you finish using
 * returned @ref STARSH_eddata object.
 * Do not forget to sort `data->particles` by starsh_particles_zsort_inplace()
 * to take advantage of low-rank submatrices.
 *
 * @param[out] data: Address of pointer to @ref STARSH_eddata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @param[in] point: Pointer to array of coordinates of particles.
 * @param[in] k: Wave number.
 * @param[in] diag: Value of diagonal elements.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_eddata_new(), starsh_eddata_free(),
 *      starsh_eddata_generate(), starsh_eddata_read_from_file(),
 *      starsh_particles_zsort_inplace().
 * @ingroup app-electrodynamics
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
    STARSH_eddata *tmp;
    STARSH_MALLOC(tmp, 1);
    tmp->particles.count = count;
    tmp->particles.ndim = ndim;
    tmp->particles.point = point;
    tmp->k = k;
    tmp->diag = diag;
    return STARSH_SUCCESS;
}

int starsh_eddata_generate(STARSH_eddata **data, STARSH_int count, int ndim,
        double k, double diag, enum STARSH_PARTICLES_PLACEMENT place)
//! Generate @ref STARSH_ssdata object by given distribution.
/*! @param[out] data: Address of pointer to @ref STARSH_eddata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @param[in] k: Wave number.
 * @param[in] diag: Value of diagonal elements.
 * @param[in] place: Placement strategy for spatial points.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_eddata_generate_va(), starsh_eddata_generate_el().
 * @ingroup app-electrodynamics
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("Invalid value of `data`");
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
    (*data)->k = k;
    (*data)->diag = diag;
    return STARSH_SUCCESS;
}

int starsh_eddata_generate_va(STARSH_eddata **data, STARSH_int count,
        va_list args)
//! Generate @ref STARSH_eddata object with incomplete set of parameters.
/*! Parses possibly incomplete set of parameters for starsh_eddata_generate().
 * If argument is not in the `args`, then its default value is used:
 *
 * Argument | Default value | Type
 * ---------|---------------|--------
 * `ndim`   | 2             | int
 * `k`      | 1.0           | double
 * `diag`   | 1.0           | double
 * `place`  | @ref STARSH_PARTICLES_UNIFORM | @ref STARSH_PARTICLES_PLACEMENT
 *
 * List of arguments `args` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_ELECTRODYNAMICS_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_ssdata object.
 * @param[in] count: Number of particles.
 * @param[in] args: Arguments, packed into va_args.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      void generate(STARSH_int count, ...)
 *      {
 *          STARSH_eddata *data;
 *          va_list args;
 *          va_start(args, count);
 *          starsh_eddata_generate_va(&data, count, args);
 *          va_end(args);
 *      }
 * @endcode
 * @sa starsh_eddata_generate(), starsh_eddata_generate_el().
 * @ingroup app-electrodynamics
 * */
{
    int arg_type;
    // Set default values
    int ndim = 2;
    double k = 1.0;
    double diag = 1.0;
    enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
    int info;
    while((arg_type = va_arg(args, int)) != 0)
    {
        switch(arg_type)
        {
            case STARSH_ELECTRODYNAMICS_NDIM:
                ndim = va_arg(args, int);
                break;
            case STARSH_ELECTRODYNAMICS_K:
                k = va_arg(args, double);
                break;
            case STARSH_ELECTRODYNAMICS_DIAG:
                diag = va_arg(args, double);
                break;
            case STARSH_ELECTRODYNAMICS_PLACE:
                place = va_arg(args, enum STARSH_PARTICLES_PLACEMENT);
                break;
            default:
                STARSH_ERROR("Wrong parameter type");
                return STARSH_WRONG_PARAMETER;
        }
    }
    info = starsh_eddata_generate(data, count, ndim, k, diag, place);
    return info;
}

int starsh_eddata_generate_el(STARSH_eddata **data, STARSH_int count, ...)
//! Generate @ref STARSH_eddata object with incomplete set of parameters.
/*! Parses possibly incomplete set of parameters for starsh_eddata_generate().
 * If argument is not in the `...`, then its default value is used:
 *
 * Argument | Default value | Type
 * ---------|---------------|--------
 * `ndim`   | 2             | int
 * `k`      | 1.0           | double
 * `diag`   | 1.0           | double
 * `place`  | @ref STARSH_PARTICLES_UNIFORM | @ref STARSH_PARTICLES_PLACEMENT
 *
 * List of arguments in `...` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_ELECTRODYNAMICS_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_eddata object.
 * @param[in] count: Number of particles.
 * @param[in] ...: Variable amount of arguments.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      starsh_eddata_generate_el(&data, count,
 *          STARSH_ELECTRODYNAMICS_PLACE, STARSH_PARTICLES_RAND,
 *          STARSH_ELECTRODYNAMICS_K, 0.1,
 *          STARSH_ELECTRODYNAMICS_DIAG, 0.9,
 *          0).
 * @endcode
 * @arg @code{.c}
 *      starsh_eddata_generate_el(&data, count,
 *          STARSH_ELECTRODYNAMICS_NDIM, 3,
 *          STARSH_ELECTRODYNAMICS_K, 1.5,
 *          STARSH_ELECTRODYNAMICS_PLACE, STARSH_PARTICLES_RAND,
 *          0).
 * @endcode
 * @sa starsh_eddata_generate(), starsh_eddata_generate_va().
 * @ingroup app-electrodynamics
 * */
{
    va_list args;
    va_start(args, count);
    int info = starsh_eddata_generate_va(data, count, args);
    va_end(args);
    return info;
}

void starsh_eddata_free(STARSH_eddata *data)
//! Free memory of @ref STARSH_eddata object.
/*! @sa starsh_eddata_new(), starsh_eddata_init(), starsh_eddata_generate().
 * @ingroup app-electrodynamics
 * */
{
    starsh_particles_free(&data->particles);
}

static int starsh_eddata_get_kernel_1d(STARSH_kernel **kernel,
        enum STARSH_ELECTRODYNAMICS_KERNEL type)
// Get kernel for 1-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTRODYNAMICS_SIN:
            *kernel = starsh_eddata_block_sin_kernel_1d;
            break;
        case STARSH_ELECTRODYNAMICS_SIN_SIMD:
            *kernel = starsh_eddata_block_sin_kernel_1d_simd;
            break;
        case STARSH_ELECTRODYNAMICS_COS:
            *kernel = starsh_eddata_block_cos_kernel_1d;
            break;
        case STARSH_ELECTRODYNAMICS_COS_SIMD:
            *kernel = starsh_eddata_block_cos_kernel_1d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_eddata_get_kernel_2d(STARSH_kernel **kernel,
        enum STARSH_ELECTRODYNAMICS_KERNEL type)
// Get kernel for 2-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTRODYNAMICS_SIN:
            *kernel = starsh_eddata_block_sin_kernel_2d;
            break;
        case STARSH_ELECTRODYNAMICS_SIN_SIMD:
            *kernel = starsh_eddata_block_sin_kernel_2d_simd;
            break;
        case STARSH_ELECTRODYNAMICS_COS:
            *kernel = starsh_eddata_block_cos_kernel_2d;
            break;
        case STARSH_ELECTRODYNAMICS_COS_SIMD:
            *kernel = starsh_eddata_block_cos_kernel_2d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_eddata_get_kernel_3d(STARSH_kernel **kernel,
        enum STARSH_ELECTRODYNAMICS_KERNEL type)
// Get kernel for 3-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTRODYNAMICS_SIN:
            *kernel = starsh_eddata_block_sin_kernel_3d;
            break;
        case STARSH_ELECTRODYNAMICS_SIN_SIMD:
            *kernel = starsh_eddata_block_sin_kernel_3d_simd;
            break;
        case STARSH_ELECTRODYNAMICS_COS:
            *kernel = starsh_eddata_block_cos_kernel_3d;
            break;
        case STARSH_ELECTRODYNAMICS_COS_SIMD:
            *kernel = starsh_eddata_block_cos_kernel_3d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_eddata_get_kernel_4d(STARSH_kernel **kernel,
        enum STARSH_ELECTRODYNAMICS_KERNEL type)
// Get kernel for 4-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTRODYNAMICS_SIN:
            *kernel = starsh_eddata_block_sin_kernel_4d;
            break;
        case STARSH_ELECTRODYNAMICS_SIN_SIMD:
            *kernel = starsh_eddata_block_sin_kernel_4d_simd;
            break;
        case STARSH_ELECTRODYNAMICS_COS:
            *kernel = starsh_eddata_block_cos_kernel_4d;
            break;
        case STARSH_ELECTRODYNAMICS_COS_SIMD:
            *kernel = starsh_eddata_block_cos_kernel_4d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_eddata_get_kernel_nd(STARSH_kernel **kernel,
        enum STARSH_ELECTRODYNAMICS_KERNEL type)
// Get kernel for n-dimensional spatial statistics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTRODYNAMICS_SIN:
            *kernel = starsh_eddata_block_sin_kernel_nd;
            break;
        case STARSH_ELECTRODYNAMICS_SIN_SIMD:
            *kernel = starsh_eddata_block_sin_kernel_nd_simd;
            break;
        case STARSH_ELECTRODYNAMICS_COS:
            *kernel = starsh_eddata_block_cos_kernel_nd;
            break;
        case STARSH_ELECTRODYNAMICS_COS_SIMD:
            *kernel = starsh_eddata_block_cos_kernel_nd_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

int starsh_eddata_get_kernel(STARSH_kernel **kernel, STARSH_eddata *data,
        enum STARSH_ELECTRODYNAMICS_KERNEL type)
//! Get kernel for electrodynamics problem.
/*! Kernel can be selected with this call or manually. To select kernel
 * manually look into @ref app-electrodynamics-kernels.
 *
 * @param[out] kernel: Address of pointer to @ref STARSH_kernel function.
 * @param[in] data: Pointer to @ref STARSH_eddata object.
 * @param[in] type: Type of kernel. For more info look at @ref
 *      STARSH_ELECTRODYNAMICS_KERNEL.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_eddata_block_sin_kernel_nd(),
 *      starsh_eddata_block_sin_kernel_nd_simd(),
 *      starsh_eddata_block_cos_kernel_nd(),
 *      starsh_eddata_block_cos_kernel_nd_simd().
 * @ingroup app-electrodynamics
 * */
{
    switch(data->particles.ndim)
    {
        case 1:
            return starsh_eddata_get_kernel_1d(kernel, type);
        case 2:
            return starsh_eddata_get_kernel_2d(kernel, type);
        case 3:
            return starsh_eddata_get_kernel_3d(kernel, type);
        case 4:
            return starsh_eddata_get_kernel_4d(kernel, type);
        default:
            return starsh_eddata_get_kernel_nd(kernel, type);
    }
}
