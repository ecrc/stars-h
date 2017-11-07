/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/electrostatics.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 */

#include "common.h"
#include "starsh.h"
#include "starsh-electrostatics.h"

int starsh_esdata_new(STARSH_esdata **data, STARSH_int count, int ndim)
//! Allocate memory for @ref STARSH_esdata object.
/*! This functions only allocates memory for particles without setting
 * coordinates to any value.
 * Do not forget to sort `data->particles` by starsh_particles_zsort_inplace()
 * to take advantage of low-rank submatrices.
 *
 * @param[out] data: Address of pointer to @ref STARSH_esdata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_esdata_init(), starsh_esdata_free(),
 *      starsh_esdata_generate(), starsh_esdata_read_from_file(),
 *      starsh_particles_zsort_inplace(), STARSH_particles.
 * @ingroup app-electrostatics
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
    STARSH_esdata *tmp;
    STARSH_MALLOC(tmp, 1);
    tmp->count = count;
    tmp->ndim = ndim;
    STARSH_MALLOC(tmp->point, count*ndim);
    return STARSH_SUCCESS;
}

int starsh_esdata_init(STARSH_esdata **data, STARSH_int count, int ndim,
        double *point)
//! Initialize @ref STARSH_esdata object by given data.
/*! Array `point` should be stored in a special way: `x_1 x_2 ... x_count y_1
 * y_2 ... y_count z_1 z_2 ...`.
 * This function does not allocate memory for coordinates and uses provided
 * pointer `point`. Do not free memory of `point` until you finish using
 * returned @ref STARSH_esdata object.
 * Do not forget to sort `data->particles` by starsh_particles_zsort_inplace()
 * to take advantage of low-rank submatrices.
 *
 * @param[out] data: Address of pointer to @ref STARSH_esdata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @param[in] point: Pointer to array of coordinates of particles.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_esdata_new(), starsh_esdata_free(),
 *      starsh_esdata_generate(), starsh_esdata_read_from_file(),
 *      starsh_particles_zsort_inplace().
 * @ingroup app-electrostatics
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
    STARSH_esdata *tmp;
    STARSH_MALLOC(tmp, 1);
    tmp->count = count;
    tmp->ndim = ndim;
    tmp->point = point;
    return STARSH_SUCCESS;
}

int starsh_esdata_generate(STARSH_esdata **data, STARSH_int count, int ndim,
        enum STARSH_PARTICLES_PLACEMENT place)
//! Generate @ref STARSH_esdata object by given distribution.
/*! @param[out] data: Address of pointer to @ref STARSH_esdata object.
 * @param[in] count: Number of particles.
 * @param[in] ndim: Dimensionality of space.
 * @param[in] place: Placement strategy for spatial points.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_esdata_generate_va(), starsh_esdata_generate_el().
 * @ingroup app-electrostatics
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("Invalid value of `data`");
        return STARSH_WRONG_PARAMETER;
    }
    int info;
    STARSH_particles **particles = data;
    info = starsh_particles_generate(particles, count, ndim, place);
    if(info != STARSH_SUCCESS)
    {
        fprintf(stderr, "INFO=%d\n", info);
        return info;
    }
    return STARSH_SUCCESS;
}

int starsh_esdata_generate_va(STARSH_esdata **data, STARSH_int count,
        va_list args)
//! Generate @ref STARSH_esdata object with incomplete set of parameters.
/*! Parss possibly incomplete set of parameters for starsh_esdata_generate().
 * If argument is not in the `args`, then its default value is used:
 *
 * Argument | Default value | Type
 * ---------|---------------|--------
 * `ndim`   | 2             | int
 * `place`  | @ref STARSH_PARTICLES_UNIFORM | @ref STARSH_PARTICLES_PLACEMENT
 *
 * List of arguments `args` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_ELECTROSTATICS_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_esdata object.
 * @param[in] count: Number of particles.
 * @param[in] args: Arguments, packed into va_args.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      void generate(size_t count, ...)
 *      {
 *          STARSH_esdata *data;
 *          va_list args;
 *          va_start(args, count);
 *          starsh_esdata_generate_va(&data, count, args);
 *          va_end(args);
 *      }
 * @endcode
 * @sa starsh_esdata_generate(), starsh_esdata_generate_el().
 * @ingroup app-electrostatics
 * */
{
    int arg_type;
    // Set default values
    int ndim = 2;
    enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
    int info;
    while((arg_type = va_arg(args, int)) != 0)
    {
        switch(arg_type)
        {
            case STARSH_ELECTROSTATICS_NDIM:
                ndim = va_arg(args, int);
                break;
            case STARSH_ELECTROSTATICS_PLACE:
                place = va_arg(args, enum STARSH_PARTICLES_PLACEMENT);
                break;
            default:
                STARSH_ERROR("Wrong parameter type");
                return STARSH_WRONG_PARAMETER;
        }
    }
    info = starsh_esdata_generate(data, count, ndim, place);
    return info;
}

int starsh_esdata_generate_el(STARSH_esdata **data, STARSH_int count, ...)
//! Generate @ref STARSH_esdata object with incomplete set of parameters.
/*! Parses possibly incomplete set of parameters for starsh_esdata_generate().
 * If argument is not in the `...`, then its default value is used:
 *
 * Argument | Default value | Type
 * ---------|---------------|--------
 * `ndim`   | 2             | int
 * `place`  | @ref STARSH_PARTICLES_UNIFORM | @ref STARSH_PARTICLES_PLACEMENT
 *
 * List of arguments in `...` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_ELECTROSTATICS_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_esdata object.
 * @param[in] count: Number of particles.
 * @param[in] ...: Variable amount of arguments.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      starsh_esdata_generate_el(&data, count,
 *          STARSH_SPATIAL_PLACE, STARSH_PARTICLES_RAND,
 *          0).
 * @endcode
 * @arg @code{.c}
 *      starsh_esdata_generate_el(&data, count,
 *          STARSH_SPATIAL_NDIM, 3,
 *          STARSH_SPATIAL_PLACE, STARSH_PARTICLES_RAND,
 *          0).
 * @endcode
 * @sa starsh_esdata_generate(), starsh_esdata_generate_va().
 * @ingroup app-electrostatics
 * */
{
    va_list args;
    va_start(args, count);
    int info = starsh_esdata_generate_va(data, count, args);
    va_end(args);
    return info;
}

void starsh_esdata_free(STARSH_esdata *data)
//! Free memory of @ref STARSH_esdata object.
/*! @sa starsh_esdata_new(), starsh_esdata_init(), starsh_esdata_generate().
 * @ingroup app-electrostatics
 * */
{
    starsh_particles_free(data);
}

static int starsh_esdata_get_kernel_1d(STARSH_kernel **kernel,
        enum STARSH_ELECTROSTATICS_KERNEL type)
// Get kernel for 1-dimensional electrostatics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_1d;
            break;
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL_SIMD:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_1d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_esdata_get_kernel_2d(STARSH_kernel **kernel,
        enum STARSH_ELECTROSTATICS_KERNEL type)
// Get kernel for 2-dimensional electrostatics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_2d;
            break;
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL_SIMD:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_2d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_esdata_get_kernel_3d(STARSH_kernel **kernel,
        enum STARSH_ELECTROSTATICS_KERNEL type)
// Get kernel for 3-dimensional electrostatics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_3d;
            break;
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL_SIMD:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_3d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_esdata_get_kernel_4d(STARSH_kernel **kernel,
        enum STARSH_ELECTROSTATICS_KERNEL type)
// Get kernel for 4-dimensional electrostatics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_4d;
            break;
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL_SIMD:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_4d_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

static int starsh_esdata_get_kernel_nd(STARSH_kernel **kernel,
        enum STARSH_ELECTROSTATICS_KERNEL type)
// Get kernel for n-dimensional electrostatics problem.
// This function is static not to be visible outside this module.
{
    switch(type)
    {
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_nd;
            break;
        case STARSH_ELECTROSTATICS_COULOMB_POTENTIAL_SIMD:
            *kernel = starsh_esdata_block_coulomb_potential_kernel_nd_simd;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

int starsh_esdata_get_kernel(STARSH_kernel **kernel, STARSH_esdata *data,
        enum STARSH_ELECTROSTATICS_KERNEL type)
//! Get kernel for electrostatics problem.
/*! Kernel can be selected with this call or manually. To select kernel
 * manually look into @ref app-electrostatics-kernels.
 *
 * @param[out] kernel: Address of pointer to @ref STARSH_kernel function.
 * @param[in] data: Pointer to @ref STARSH_esdata object.
 * @param[in] type: Type of kernel. For more info look at @ref
 *      STARSH_ELECTROSTATICS_KERNEL.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_esdata_block_coulomb_potential_kernel_nd(),
 *      starsh_esdata_block_coulomb_potential_kernel_nd_simd().
 * @ingroup app-spatial
 * */
{
    switch(data->ndim)
    {
        case 1:
            return starsh_esdata_get_kernel_1d(kernel, type);
        case 2:
            return starsh_esdata_get_kernel_2d(kernel, type);
        case 3:
            return starsh_esdata_get_kernel_3d(kernel, type);
        case 4:
            return starsh_esdata_get_kernel_4d(kernel, type);
        default:
            return starsh_esdata_get_kernel_nd(kernel, type);
    }
}
