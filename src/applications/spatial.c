/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/spatial.c
 * @version 0.1.1
 * @author Aleksandr Mikhalev
 * @date 2018-11-06
 */

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"

#define PI 3.14159265358979323846264338327950288 

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
    *data = tmp;
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
 * @param[in] sigma: Square of variance.
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
    *data = tmp;
    return STARSH_SUCCESS;
}



int starsh_ssdata_init_parsimonious(STARSH_ssdata **data, STARSH_int count, int ndim,
		double *point, double sigma1, double sigma2, double beta, double nu1,
		double nu2, double corr, double noise)
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
	 * @param[in] sigma: Square of variance.
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
	if(nu1 < 0 || nu2 < 0)
	{
		STARSH_ERROR("Invalid value of `nu`");
		return STARSH_WRONG_PARAMETER;
	}
	if(noise < 0)
	{        STARSH_ERROR("Invalid value of `noise`");
		return STARSH_WRONG_PARAMETER;
	}
	if(sigma1 < 0 || sigma2 < 0)
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
	tmp->nu = nu1;
	tmp->noise = noise;
	tmp->sigma = sigma1;
        tmp->nu2 = nu2;
        tmp->noise = noise;
        tmp->sigma2 = sigma2;
        tmp->corr = corr;
	*data = tmp;
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
	 * @param[in] sigma: Square of variance.
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


int starsh_ssdata_generate_parsimonious(STARSH_ssdata **data, STARSH_int count, int ndim,  double beta, double nu, double noise,  enum STARSH_PARTICLES_PLACEMENT place, double sigma, double sigma2, double nu2, double corr)
	//! Generate @ref STARSH_ssdata object by given distribution.
	/*! @param[out] data: Address of pointer to @ref STARSH_ssdata object.
	 * @param[in] count: Number of particles.
	 * @param[in] ndim: Dimensionality of space.
	 * @param[in] beta: Correlation length.
	 * @param[in] nu: Smoothing parameter for Mat&eacute;rn kernel.
	 * @param[in] noise: Value to add to diagonal elements.
	 * @param[in] place: Placement strategy for spatial points.
	 * @param[in] sigma: Square of variance.
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
        (*data)->nu2 = nu2;
        (*data)->noise2 = noise;
        (*data)->sigma2 = sigma2;
        (*data)->corr = corr;
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
	double nu2 = 0.5;
	double noise = 0;
	enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
	double sigma = 1.0;
	double sigma2 = -1.0;
	double corr = 0.0;
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
			case STARSH_SPATIAL_SIGMA2:
				sigma2 = va_arg(args, double);
				break;
			case STARSH_SPATIAL_NU2:
				nu2 = va_arg(args, double);
				break;
			case STARSH_SPATIAL_CORR:
				corr = va_arg(args, double);
				break;
			default:
				STARSH_ERROR("Wrong parameter type");
				return STARSH_WRONG_PARAMETER;
		}
	}
	if(sigma2 < 0)	
		info = starsh_ssdata_generate(data, count, ndim, beta, nu, noise, place,
				sigma);
	else
		info = starsh_ssdata_generate_parsimonious(data, count, ndim, beta, nu, noise, place, sigma, sigma2, nu2, corr);
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
		case STARSH_SPATIAL_PARSIMONIOUS_SIMD:
		case STARSH_SPATIAL_PARSIMONIOUS2_SIMD:
			STARSH_ERROR("Matern kernel requires GSL library, which was "
					"not found");
			return STARSH_WRONG_PARAMETER;
#endif
		case STARSH_SPATIAL_EXP_GCD:
		case STARSH_SPATIAL_SQREXP_GCD:
		case STARSH_SPATIAL_MATERN_GCD:
		case STARSH_SPATIAL_MATERN2_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS2_GCD:
			STARSH_ERROR("GCD (spherical distance) can be used only for 2D "
					"problem");
			break;
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
		case STARSH_SPATIAL_EXP_GCD:
			*kernel = starsh_ssdata_block_exp_kernel_2d_simd_gcd;
			break;
		case STARSH_SPATIAL_SQREXP_GCD:
			*kernel = starsh_ssdata_block_sqrexp_kernel_2d_simd_gcd;
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
		case STARSH_SPATIAL_PARSIMONIOUS_SIMD:
			{
				printf("STARSH_SPATIAL_PARSIMONIOUS_SIMD\n");
				*kernel = starsh_ssdata_block_parsimonious_kernel_2d_simd;
				break;
			}
		case STARSH_SPATIAL_PARSIMONIOUS2_SIMD:
			*kernel = starsh_ssdata_block_parsimonious2_kernel_2d_simd;
			break;
		case STARSH_SPATIAL_MATERN_GCD:
			*kernel = starsh_ssdata_block_matern_kernel_2d_simd_gcd;
			break;
		case STARSH_SPATIAL_MATERN2_GCD:
			{
				printf("STARSH_SPATIAL_MATERN2_GCD(hi)\n");
				*kernel = starsh_ssdata_block_matern2_kernel_2d_simd_gcd;
				break;
			}
		case STARSH_SPATIAL_PARSIMONIOUS_GCD:
			{
				printf("STARSH_SPATIAL_PARSIMONIOUS_GCD(hi)\n");
				*kernel = starsh_ssdata_block_parsimonious_kernel_2d_simd_gcd;
				break;
			}
		case STARSH_SPATIAL_PARSIMONIOUS2_GCD:
			*kernel = starsh_ssdata_block_parsimonious2_kernel_2d_simd_gcd;
			break;
#else
		case STARSH_SPATIAL_MATERN:
		case STARSH_SPATIAL_MATERN_SIMD:
		case STARSH_SPATIAL_MATERN2:
		case STARSH_SPATIAL_MATERN2_SIMD:
		case STARSH_SPATIAL_PARSIMONIOUS_SIMD:
		case STARSH_SPATIAL_PARSIMONIOUS2_SIMD:
		case STARSH_SPATIAL_MATERN_GCD:
		case STARSH_SPATIAL_MATERN2_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS2_GCD:
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
		case STARSH_SPATIAL_PARSIMONIOUS_SIMD:
		case STARSH_SPATIAL_PARSIMONIOUS2_SIMD:
			STARSH_ERROR("Matern kernel requires GSL library, which was "
					"not found");
			return STARSH_WRONG_PARAMETER;
#endif
		case STARSH_SPATIAL_EXP_GCD:
		case STARSH_SPATIAL_SQREXP_GCD:
		case STARSH_SPATIAL_MATERN_GCD:
		case STARSH_SPATIAL_MATERN2_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS2_GCD:
			STARSH_ERROR("GCD (spherical distance) can be used only for 2D "
					"problem");
			break;
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
		case STARSH_SPATIAL_PARSIMONIOUS_SIMD:
		case STARSH_SPATIAL_PARSIMONIOUS2_SIMD:
			STARSH_ERROR("Matern kernel requires GSL library, which was "
					"not found");
			return STARSH_WRONG_PARAMETER;
#endif
		case STARSH_SPATIAL_EXP_GCD:
		case STARSH_SPATIAL_SQREXP_GCD:
		case STARSH_SPATIAL_MATERN_GCD:
		case STARSH_SPATIAL_MATERN2_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS2_GCD:
			STARSH_ERROR("GCD (spherical distance) can be used only for 2D "
					"problem");
			break;
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
	printf("%============99999999999999999999999 \n" );
	switch(type)
	{
		case STARSH_SPATIAL_EXP:
			*kernel = starsh_ssdata_block_exp_kernel_nd;
			break;
		case STARSH_SPATIAL_EXP_SIMD:
			*kernel = starsh_ssdata_block_exp_kernel_nd_simd;
			break;
		case STARSH_SPATIAL_SQREXP:
			*kernel = starsh_ssdata_block_sqrexp_kernel_nd;
			break;
		case STARSH_SPATIAL_SQREXP_SIMD:
			*kernel = starsh_ssdata_block_sqrexp_kernel_nd_simd;
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
		case STARSH_SPATIAL_PARSIMONIOUS_SIMD:
		case STARSH_SPATIAL_PARSIMONIOUS2_SIMD:
			STARSH_ERROR("Matern kernel requires GSL library, which was "
					"not found");
			return STARSH_WRONG_PARAMETER;
#endif
		case STARSH_SPATIAL_EXP_GCD:
		case STARSH_SPATIAL_SQREXP_GCD:
		case STARSH_SPATIAL_MATERN_GCD:
		case STARSH_SPATIAL_MATERN2_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS_GCD:
		case STARSH_SPATIAL_PARSIMONIOUS2_GCD:
			STARSH_ERROR("GCD (spherical distance) can be used only for 2D "
					"problem");
			break;
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

// This function converts decimal degrees to radians
static double deg2rad(double deg) {
	return (deg * PI / 180.);
}
//  This function converts radians to decimal degrees
static double rad2deg(double rad) {
	return (rad * 180. / PI);
}

#define earthRadiusKm 6371.0

/**
 * Returns the distance between two points on the Earth.
 * Direct translation from http://en.wikipedia.org/wiki/Haversine_formula
 * @param lat1d Latitude of the first point in degrees
 * @param lon1d Longitude of the first point in degrees
 * @param lat2d Latitude of the second point in degrees
 * @param lon2d Longitude of the second point in degrees
 * @return The distance between the two points in kilometers
 */
static double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
	double lat1r, lon1r, lat2r, lon2r, u, v;
	lat1r = deg2rad(lat1d);
	lon1r = deg2rad(lon1d);
	lat2r = deg2rad(lat2d);
	lon2r = deg2rad(lon2d);
	u = sin((lat2r - lat1r)/2);
	v = sin((lon2r - lon1r)/2);
	return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

void starsh_ssdata_block_exp_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Exponential kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 e^{-\frac{r_{ij}}{\beta}} + \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \delta \f$ is the delta function
	 * \f[
	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points, measured by arc on sphere, and variance \f$ \sigma \f$, correlation
	 * length \f$ \beta \f$ and
	 * noise \f$ \mu \f$ come from \p row_data (\ref STARSH_ssdata object). No
	 * memory is allocated in this function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_exp_kernel_1d_simd(),
	 *      starsh_ssdata_block_exp_kernel_2d_simd(),
	 *      starsh_ssdata_block_exp_kernel_3d_simd(),
	 *      starsh_ssdata_block_exp_kernel_4d_simd(),
	 *      starsh_ssdata_block_exp_kernel_nd_simd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	//int ndim = 2;
	double beta = -data1->beta;
	double noise = data1->noise;
	double sigma = data1->sigma;
	// Get coordinates
	size_t count1 = data1->particles.count;
	size_t count2 = data2->particles.count;
	double *x1[2], *x2[2];
	//printf("%===============%f(4), \n", sigma);
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;
	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols; j++)
	{
		for(i = 0; i < nrows; i++)
		{
			dist = distanceEarth(x1[0][irow[i]], x1[1][irow[i]],
					x2[0][icol[j]], x2[1][icol[j]]);
			dist = dist/beta;
			if(dist == 0)
				buffer[j*(size_t)ld+i] = sigma+noise;
			else
				buffer[j*(size_t)ld+i] = sigma*exp(dist);
		}
	}
}

void starsh_ssdata_block_sqrexp_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Square exponential kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 e^{-\frac{1}{2} \left( \frac{r_{ij}}{\beta}
	 *      \right)^2} + \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \delta \f$ is the delta function
	 * \f[
	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$ and
	 * noise \f$ \mu \f$ come from \p row_data (\ref STARSH_ssdata object). No
	 * memory is allocated in this function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_sqrexp_kernel_1d(),
	 *      starsh_ssdata_block_sqrexp_kernel_2d(),
	 *      starsh_ssdata_block_sqrexp_kernel_3d(),
	 *      starsh_ssdata_block_sqrexp_kernel_4d(),
	 *      starsh_ssdata_block_sqrexp_kernel_nd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	double beta = -2*data1->beta*data1->beta;
	double noise = data1->noise;
	double sigma = data1->sigma;
	// Get coordinates
	STARSH_int count1 = data1->particles.count;
	STARSH_int count2 = data2->particles.count;
	double *x1[2], *x2[2];
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
	//printf("%===============(3)%f, \n", sigma);
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;
	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols; j++)
	{
		for(i = 0; i < nrows; i++)
		{
			dist = distanceEarth(x1[0][irow[i]], x1[1][irow[i]],
					x2[0][icol[j]], x2[1][icol[j]]);
			dist = dist*dist/beta;
			if(dist == 0)
				buffer[j*(size_t)ld+i] = sigma+noise;
			else
				buffer[j*(size_t)ld+i] = sigma*exp(dist);
		}
	}
}

#ifdef GSL

void starsh_ssdata_block_matern_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \sqrt{2 \nu}
	 *      \frac{r_{ij}}{\beta} \right)^{\nu} K_{\nu} \left( \sqrt{2 \nu}
	 *      \frac{r_{ij}}{\beta} \right) + \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
	 * Bessel function of the second kind, \f$ \delta \f$ is the delta function
	 * \f[
	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
	 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
	 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
	 * function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_matern_kernel_1d(),
	 *      starsh_ssdata_block_matern_kernel_2d_simd(),
	 *      starsh_ssdata_block_matern_kernel_3d_simd(),
	 *      starsh_ssdata_block_matern_kernel_4d_simd(),
	 *      starsh_ssdata_block_matern_kernel_nd_simd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	double beta = data1->beta;
	double nu = data1->nu;
	double theta = sqrt(2*nu)/beta;
	double noise = data1->noise;
	double sigma = data1->sigma;
	// Get coordinates
	STARSH_int count1 = data1->particles.count;
	STARSH_int count2 = data2->particles.count;
	double *x1[2], *x2[2];
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
	//printf("%===============(2)%f, \n", sigma);
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;
	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols; j++)
	{
		for(i = 0; i < nrows; i++)
		{
			dist = distanceEarth(x1[0][irow[i]], x1[1][irow[i]],
					x2[0][icol[j]], x2[1][icol[j]]);
			dist = dist*theta;
			if(dist == 0)
				buffer[j*(size_t)ld+i] = sigma+noise;
			else
				buffer[j*(size_t)ld+i] = sigma*pow(2.0, 1.0-nu)/
					gsl_sf_gamma(nu)*pow(dist, nu)*
					gsl_sf_bessel_Knu(nu, dist);
		}
	}
}

void starsh_ssdata_block_matern2_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{r_{ij}}
	 *      {\beta} \right)^{\nu} K_{\nu} \left( \frac{r_{ij}}{\beta} \right) +
	 *      \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
	 * Bessel function of the second kind, \f$ \delta \f$ is the delta function
	 * \f[
	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
	 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
	 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
	 * function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_matern2_kernel_1d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_2d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_3d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_4d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_nd_simd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	double beta = data1->beta;
	double nu = data1->nu;
	double noise = data1->noise;
	double sigma = data1->sigma;
	// Get coordinates
	STARSH_int count1 = data1->particles.count;
	STARSH_int count2 = data2->particles.count;
	double *x1[2], *x2[2];
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
	//printf("%(14)===============(test)%f, \n", sigma);
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;
	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols; j++)
	{
		for(i = 0; i < nrows; i++)
		{
			dist = distanceEarth(x1[0][irow[i]], x1[1][irow[i]],
					x2[0][icol[j]], x2[1][icol[j]]);
			dist = dist/beta;
			if(dist == 0)
				buffer[j*(size_t)ld+i] = sigma+noise;
			else
				buffer[j*(size_t)ld+i] = sigma*pow(2.0, 1.0-nu)/
					gsl_sf_gamma(nu)*pow(dist, nu)*
					gsl_sf_bessel_Knu(nu, dist);
		}
	}
}



void starsh_ssdata_block_parsimonious_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{r_{ij}}
	 *      {\beta} \right)^{\nu} K_{\nu} \left( \frac{r_{ij}}{\beta} \right) +
	 *      \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
	 * Bessel function of the second kind, \f$ \delta \f$ is the delta function
	 * \f[
	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
	 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
	 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
	 * function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_matern2_kernel_1d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_2d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_3d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_4d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_nd_simd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	double beta = data1->beta;
	double nu1 = data1->nu;
	double noise1 = data1->noise;
	double sigma1 = data1->sigma;

	double nu2    = data1->nu2;
	double noise2 = data1->noise2;
	double sigma2 = data1->sigma2;
	double corr  = data1->corr;

	//printf("%(13)===============%f, %f, %f, %f, %f, %f\n", sigma1, sigma2, beta, nu1, nu2, corr);
	// Get coordinates
	STARSH_int count1 = data1->particles.count;
	STARSH_int count2 = data2->particles.count;
	double *x1[2], *x2[2];
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;


	//    double con= sigma*pow(2.0, 1.0-nu)/gsl_sf_gamma(nu);
	double con1 = 0.0, con2 = 0.0, con12 = 0.0, rho = 0.0, nu12 = 0.0;

	con1 = pow(2,(nu1-1)) * tgamma(nu1);
	con1 = 1.0/con1;
	con1 = sigma1 * con1;

	con2 = pow(2, (nu2-1)) * tgamma(nu2);
	con2 = 1.0/con2;
	con2 = sigma2 * con2;

	nu12 = 0.5 * (nu1+ nu2);

	rho = corr * sqrt( (tgamma(nu1 + 1)*tgamma(nu2 + 1)) /
			(tgamma(nu1) * tgamma(nu2)) ) *
		tgamma(nu12) / tgamma(nu12 + 1);


	con12 = pow(2,(nu12-1)) * tgamma(nu12);
	con12 = 1.0/con12;
	con12 = rho * sqrt(sigma1 * sigma2) * con12;



	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols; j++)
	{
		for(i = 0; i < nrows; i++)
		{

			dist = distanceEarth(x1[0][irow[i]], x1[1][irow[i]],
					x2[0][icol[j]], x2[1][icol[j]]);
			dist = dist/beta;
			if( i % 2 ==0)
			{
				if(dist == 0)
				{
					if( i == j)
						buffer[j*(size_t)ld+i] = sigma1+noise1;
					else 
						buffer[j*(size_t)ld+i] = rho * sqrt(sigma1 * sigma2) ;


				}
				else

					if( i == j)
						buffer[j*(size_t)ld+i] = con1 * pow(dist, nu1) * gsl_sf_bessel_Knu(nu1, dist);
					else
						buffer[j*(size_t)ld+i] = con12 * pow(dist, nu12) * gsl_sf_bessel_Knu(nu12, dist);
			}
			else
			{
				if(dist == 0)
				{
					if( i == j)
						buffer[j*(size_t)ld+i] = sigma2+noise2;
					else
						buffer[j*(size_t)ld+i] = rho * sqrt(sigma1 * sigma2) ;

				}
				else
				{
					if( i == j)
						buffer[j*(size_t)ld+i] = con1 * pow(dist, nu1) * gsl_sf_bessel_Knu(nu1, dist);
					else
						buffer[j*(size_t)ld+i] = con2 * pow(dist, nu2) * gsl_sf_bessel_Knu(nu2, dist);
				}

			}	
		}
	}
}


void starsh_ssdata_block_parsimonious_kernel_2d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{r_{ij}}
	 *      {\beta} \right)^{\nu} K_{\nu} \left( \frac{r_{ij}}{\beta} \right) +
	 *      \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
	 * Bessel function of the second kind, \f$ \delta \f$ is the delta function  
	 * \f[
	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
	 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
	 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
	 * function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_matern2_kernel_1d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_2d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_3d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_4d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_nd_simd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	double beta = data1->beta;
	double nu1 = data1->nu;
	double noise1 = data1->noise;
	double sigma1 = data1->sigma;

	double nu2    = data1->nu2;
	double noise2 = data1->noise;
	double sigma2 = data1->sigma2;
	double corr  = data1->corr;

	//printf("(12)======%f %f %f %f %f %f %f %f\n", sigma1, sigma2, beta, nu1, nu2, corr, noise1, noise2);
	//exit(0);
	// Get coordinates
	STARSH_int count1 = data1->particles.count;
	STARSH_int count2 = data2->particles.count;
	double *x1[2], *x2[2];
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;


	//    double con= sigma*pow(2.0, 1.0-nu)/gsl_sf_gamma(nu);
	double con1 = 0.0, con2 = 0.0, con12 = 0.0, rho = 0.0, nu12 = 0.0;

	con1 = pow(2,(nu1-1)) * tgamma(nu1);
	con1 = 1.0/con1;
	con1 = sigma1 * con1;

	con2 = pow(2, (nu2-1)) * tgamma(nu2);
	con2 = 1.0/con2;
	con2 = sigma2 * con2;

	nu12 = 0.5 * (nu1+ nu2);

	rho = corr * sqrt( (tgamma(nu1 + 1)*tgamma(nu2 + 1)) /
			(tgamma(nu1) * tgamma(nu2)) ) *
		tgamma(nu12) / tgamma(nu12 + 1);

	con12 = pow(2,(nu12-1)) * tgamma(nu12);
	con12 = 1.0/con12;
	con12 = rho * sqrt(sigma1 * sigma2) * con12;

	//printf("(LR): %f, %f, %f, %f\n", rho, con1, con2, con12);
	//printf("%f, %f, %f, %f, %f, %f\n", sigma1, sigma2, beta, nu1, nu2, corr);
	//exit(0);

	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols-1; j+=2)
	{
		for(i = 0; i < nrows-1; i+=2)
		{
			//printf("i:%d, j:%d,\n", i, j);
			double dist = 0.0;
			for(k = 0; k < 2; k++)
			{
				tmp = pow(x1[k][irow[i]]-x2[k][icol[j]],2);
				dist += tmp;
			}
			dist = sqrt(dist)/beta;

			if(dist == 0)
			{
				buffer[j*(size_t)ld+i] = sigma1+noise1;
				buffer[j*(size_t)ld+(i+1)] = rho * sqrt(sigma1 * sigma2) ;
				buffer[(j+1)*(size_t)ld+i] = rho * sqrt(sigma1 * sigma2) ;
				buffer[(j+1)*(size_t)ld+(i+1)] = sigma2+noise2;

			}
			else
			{
				buffer[j*(size_t)ld+i] = con1 * pow(dist, nu1) * gsl_sf_bessel_Knu(nu1, dist);//+noise1;
				buffer[j*(size_t)ld+(i+1)] = con12 * pow(dist, nu12) * gsl_sf_bessel_Knu(nu12, dist);
				buffer[(j+1)*(size_t)ld+i] = con12 * pow(dist, nu12) * gsl_sf_bessel_Knu(nu12, dist);
				buffer[(j+1)*(size_t)ld+(i+1)] = con2 * pow(dist, nu2) * gsl_sf_bessel_Knu(nu2, dist);//;+noise2;
			}

		}
		//printf ("===(lr) %d, %d, %f, %f, %f, %f, %f,%f\n" ,  i, j, x1[0][irow[i]], x2[0][icol[j]], x1[1][irow[i]], x2[1][icol[j]],  buffer[j*(size_t)ld+i],dist);
	}
}





void starsh_ssdata_block_parsimonious2_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{r_{ij}}
	 *      {\beta} \right)^{\nu} K_{\nu} \left( \frac{r_{ij}}{\beta} \right) +
	 *      \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
	 * Bessel function of the second kind, \f$ \delta \f$ is the delta function
	 * \f[
	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
	 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
	 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
	 * function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_matern2_kernel_1d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_2d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_3d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_4d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_nd_simd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	double beta = data1->beta;
	double nu1 = data1->nu;
	double noise1 = data1->noise;
	double sigma1 = data1->sigma;

	double nu2    = data1->nu2;
	double noise2 = data1->noise2;
	double sigma2 = data1->sigma2;
	double corr  = data1->corr;


	printf("%(11)===============%f, %f, %f, %f, %f, %f\n", sigma1, sigma2, beta, nu1, nu2, corr);
	//exit(0);
	// Get coordinates
	STARSH_int count1 = data1->particles.count;
	STARSH_int count2 = data2->particles.count;
	double *x1[2], *x2[2];
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;
	//    double con= sigma*pow(2.0, 1.0-nu)/gsl_sf_gamma(nu);
	double con1 = 0.0, con2 = 0.0, con12 = 0.0, rho = 0.0, nu12 = 0.0;

	con1 = pow(2,(nu1-1)) * tgamma(nu1);
	con1 = 1.0/con1;
	con1 = sigma1 * con1;

	con2 = pow(2, (nu2-1)) * tgamma(nu2);
	con2 = 1.0/con2;
	con2 = sigma2 * con2;

	nu12 = 0.5 * (nu1+ nu2);

	rho = corr * sqrt( (tgamma(nu1 + 1)*tgamma(nu2 + 1)) /
			(tgamma(nu1) * tgamma(nu2)) ) *
		tgamma(nu12) / tgamma(nu12 + 1);


	con12 = pow(2,(nu12-1)) * tgamma(nu12);
	con12 = 1.0/con12;
	con12 = rho * sqrt(sigma1 * sigma2) * con12;



	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols; j++)
	{
		for(i = 0; i < nrows; i++)
		{

			dist = distanceEarth(x1[0][irow[i]], x1[1][irow[i]],
					x2[0][icol[j]], x2[1][icol[j]]);
			dist = dist/beta;
			if( i % 2 ==0)
			{
				if(dist == 0)
				{
					if( i == j)
						buffer[j*(size_t)ld+i] = sigma1+noise1;
					else
						buffer[j*(size_t)ld+i] = rho * sqrt(sigma1 * sigma2) ;


				}
				else

					if( i == j)
						buffer[j*(size_t)ld+i] = con1 * pow(dist, nu1) * gsl_sf_bessel_Knu(nu1, dist);
					else
						buffer[j*(size_t)ld+i] = con12 * pow(dist, nu12) * gsl_sf_bessel_Knu(nu12, dist);
			}
			else
			{
				if(dist == 0)

				{
					if( i == j)
						buffer[j*(size_t)ld+i] = sigma2+noise2;
					else
						buffer[j*(size_t)ld+i] = rho * sqrt(sigma1 * sigma2) ;

				}
				else
				{
					if( i == j)
						buffer[j*(size_t)ld+i] = con1 * pow(dist, nu1) * gsl_sf_bessel_Knu(nu1, dist);
					else
						buffer[j*(size_t)ld+i] = con2 * pow(dist, nu2) * gsl_sf_bessel_Knu(nu2, dist);
				}

			}
		}
	}
}


void starsh_ssdata_block_parsimonious2_kernel_2d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld)
	//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
	/*! Fills matrix \f$ A \f$ with values
	 * \f[
	 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{r_{ij}}
	 *      {\beta} \right)^{\nu} K_{\nu} \left( \frac{r_{ij}}{\beta} \right) +
	 *      \mu \delta(r_{ij}),
	 * \f]
	 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
	 * Bessel function of the second kind, \f$ \delta \f$ is the delta function
	 * \f[

	 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
	 *      \end{array} \right.,
	 * \f]
	 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
	 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
	 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
	 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
	 * function!
	 *
	 * Uses SIMD instructions.
	 *
	 * @param[in] nrows: Number of rows of \f$ A \f$.
	 * @param[in] ncols: Number of columns of \f$ A \f$.
	 * @param[in] irow: Array of row indexes.
	 * @param[in] icol: Array of column indexes.
	 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
	 * @param[out] result: Pointer to memory of \f$ A \f$.
	 * @param[in] ld: Leading dimension of `result`.
	 * @sa starsh_ssdata_block_matern2_kernel_1d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_2d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_3d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_4d_simd(),
	 *      starsh_ssdata_block_matern2_kernel_nd_simd().
	 * @ingroup app-spatial-kernels
	 * */
{
	int i, j, k;
	STARSH_ssdata *data1 = row_data;
	STARSH_ssdata *data2 = col_data;
	double tmp, dist;
	// Read parameters
	double beta = data1->beta;
	double nu1 = data1->nu;
	double noise1 = data1->noise;
	double sigma1 = data1->sigma;

	double nu2    = data1->nu2;
	double noise2 = data1->noise;
	double sigma2 = data1->sigma2;
	double corr  = data1->corr;

//	printf("(12)======%f %f %f %f %f %f %f %f\n", sigma1, sigma2, beta, nu1, nu2, corr, noise1, noise2);
//	exit(0);
	// Get coordinates
	STARSH_int count1 = data1->particles.count;
	STARSH_int count2 = data2->particles.count;
	double *x1[2], *x2[2];
	x1[0] = data1->particles.point;
	x2[0] = data2->particles.point;
#pragma omp simd
	for(i = 1; i < 2; i++)
	{
		x1[i] = x1[0]+i*count1;
		x2[i] = x2[0]+i*count2;
	}
	double *x1_cur, *x2_cur;
	double *buffer = result;


	//    double con= sigma*pow(2.0, 1.0-nu)/gsl_sf_gamma(nu);
	double con1 = 0.0, con2 = 0.0, con12 = 0.0, rho = 0.0, nu12 = 0.0;

	con1 = pow(2,(nu1-1)) * tgamma(nu1);
	con1 = 1.0/con1;
	con1 = sigma1 * con1;

	con2 = pow(2, (nu2-1)) * tgamma(nu2);
	con2 = 1.0/con2;
	con2 = sigma2 * con2;

	nu12 = 0.5 * (nu1+ nu2);

	rho = corr * sqrt( (tgamma(nu1 + 1)*tgamma(nu2 + 1)) /
			(tgamma(nu1) * tgamma(nu2)) ) *
		tgamma(nu12) / tgamma(nu12 + 1);

	con12 = pow(2,(nu12-1)) * tgamma(nu12);
	con12 = 1.0/con12;
	con12 = rho * sqrt(sigma1 * sigma2) * con12;



//printf("%f %f %f %f %f %f %f\n", con1, con2, nu1, nu2, nu12, rho, con12);
//exit(0);
	// Fill column-major matrix
#pragma omp simd
	for(j = 0; j < ncols/2; j++)
	{
		for(i = 0; i < nrows/2; i++)
		{
			double dist = 0.0;
			for(k = 0; k < 2; k++)
			{
				tmp = pow(x1[k][irow[i]]-x2[k][icol[j]],2);
				dist += tmp;
			}
			dist = sqrt(dist)/beta;

			if(dist == 0)
				buffer[j*(size_t)ld+i] = sigma1;//+1e-4;

			else
{
				buffer[j*(size_t)ld+i] = con1 * pow(dist, nu1) * gsl_sf_bessel_Knu(nu1, dist);//+noise1;

			//printf("\n\n======%f %f %f %f %f (%f, %f), (%f, %f)\n", con1, pow(dist, nu1), nu1, dist, gsl_sf_bessel_Knu(nu1, dist),  x1[0][irow[i]], x2[0][icol[j]], x1[1][irow[i]], x2[1][icol[j]]  );
		//	printf("(1)%6.4e, %f, %f\n", buffer[j*(size_t)ld+i], dist, (con1 * pow(dist, nu1) * gsl_sf_bessel_Knu(nu1, dist)));

//exit(0);
}
		}
	}

	//************************************************
	for(j = ncols/2; j < ncols; j++)
	{
		for(i = nrows/2; i < nrows; i++)
		{
			double dist = 0.0;
			for(k = 0; k < 2; k++)
			{
				tmp = pow(x1[k][irow[i]]-x2[k][icol[j]],2);
				dist += tmp;
			}
			dist = sqrt(dist)/beta;

			if(dist == 0)
				buffer[j*(size_t)ld+i] = sigma2;//+1e-4;

			else
				buffer[j*(size_t)ld+i] = con2 * pow(dist, nu2) * gsl_sf_bessel_Knu(nu2, dist);//+noise1;

		}
		//printf("(2)%f ", buffer[j*(size_t)ld+i]);
	}
	//***************************************************
	for(j = ncols/2; j < ncols; j++)
	{
		for(i = 0; i < nrows/2; i++)
		{
			double dist = 0.0;
			for(k = 0; k < 2; k++)
			{
				tmp = pow(x1[k][irow[i]]-x2[k][icol[j]],2);
				dist += tmp;
			}
			dist = sqrt(dist)/beta;

			if(dist == 0)
				buffer[j*(size_t)ld+i] = rho * sqrt(sigma1 * sigma2);

			else
				buffer[j*(size_t)ld+i] = con12 * pow(dist, nu12) * gsl_sf_bessel_Knu(nu12, dist);//+noise1;

		}
		//printf("(3)%f ", buffer[j*(size_t)ld+i]);
	}
	//***************************************************
	for(j = 0; j < ncols/2; j++)
	{
		for(i = nrows/2; i < nrows; i++)
		{
			double dist = 0.0;
			for(k = 0; k < 2; k++)
			{
				tmp = pow(x1[k][irow[i]]-x2[k][icol[j]],2);
				dist += tmp;
			}
			dist = sqrt(dist)/beta;

			if(dist == 0)
				buffer[j*(size_t)ld+i] = rho * sqrt(sigma1 * sigma2);
			else
				buffer[j*(size_t)ld+i] = con12 * pow(dist, nu12) * gsl_sf_bessel_Knu(nu12, dist);//+noise1;

		}
		//printf("(4)%f ", buffer[j*(size_t)ld+i]);
	}


}
#endif // GSL

