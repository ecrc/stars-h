/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/randtlr.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-randtlr.h"

void starsh_randtlr_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld)
//! The only kernel for @ref STARSH_randtlr object.
/*! @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (@ref STARSH_randtlr object).
 * @param[in] col_data: Pointer to physical data (@ref STARSH_randtlr object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * @ingroup app-randtlr
 * */
{
    STARSH_randtlr *data = row_data;
    STARSH_int count = data->count;
    STARSH_int nblocks = data->nblocks;
    STARSH_int block_size = data->block_size;
    double diag = data->diag;
    double *U = data->U;
    double *S = data->S;
    double *buffer = result;
    for(STARSH_int i = 0; i < nrows; i++)
    {
        STARSH_int ii = irow[i];
        STARSH_int ibrow = ii/block_size;
        for(STARSH_int j = 0; j < ncols; j++)
        {
            STARSH_int jj = icol[j];
            STARSH_int jbcol = jj/block_size;
            double res = 0;
            for(STARSH_int k = 0; k < block_size; k++)
                res += U[ii+k*count]*U[jj+k*count]*S[k];
            if(ii == jj)
                buffer[j*(size_t)ld+i] = res+diag;
            else
                buffer[j*(size_t)ld+i] = res;
        }
    }
}

int starsh_randtlr_generate(STARSH_randtlr **data, STARSH_int count,
        STARSH_int block_size, double decay, double diag)
//! Generate @ref STARSH_randtlr object for random TLR matrix.
/*! @param[out] data: Address of pointer to @ref STARSH_randtlr object.
 * @param[in] count: Size of matrix.
 * @param[in] block_size: Size of each tile.
 * @param[in] decay: Decay of singular values, first value is 1.0.
 * @param[in] diag: Value to add to each diagonal element.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup app-randtlr
 * */
{
    if(count % block_size != 0)
    {
        STARSH_ERROR("`count` mod `block_size` must be equal to 0");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_int nblocks = count/block_size;
    int iseed[4] = {0, 0, 0, 1};
    double *U, *S, *tau, *work;
    int lwork = block_size;
    STARSH_MALLOC(U, count*block_size);
    STARSH_MALLOC(S, block_size);
    STARSH_MALLOC(tau, block_size);
    STARSH_MALLOC(work, lwork);
    LAPACKE_dlarnv_work(3, iseed, count*block_size, U);
    for(STARSH_int i = 0; i < nblocks; i++)
    {
        LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, block_size, block_size,
                U+i*block_size, count, tau, work, lwork);
        LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, block_size, block_size,
                block_size, U+i*block_size, count, tau, work, lwork);
    }
    free(work);
    free(tau);
    S[0] = 1.;
    for(STARSH_int i = 1; i < block_size; i++)
        S[i] = S[i-1]*decay;
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->nblocks = nblocks;
    (*data)->block_size = block_size;
    (*data)->U = U;
    (*data)->S = S;
    (*data)->diag = diag;
    return STARSH_SUCCESS;
}

int starsh_randtlr_generate_va(STARSH_randtlr **data, STARSH_int count,
        va_list args)
//! Generate @ref STARSH_randtlr object with incomplete set of parameters.
/*! Parse possibly incomplete set of parameters for starsh_randtlr_generate().
 * If argument is not in the `args`, then its default value is used:
 *
 * Argument     | Default value | Type
 * -------------|---------------|--------
 * `block_size` | `count`       | STARSH_int
 * `decay`      | 0.0           | double
 * `diag`       | 0.0           | double
 *
 * List of arguments `args` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_RANDTLR_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_randtlr object.
 * @param[in] count: Size of matrix.
 * @param[in] args: Arguments, packed into va_args.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      void generate(STARSH_int count, ...)
 *      {
 *          STARSH_randtlr *data;
 *          va_list args;
 *          va_start(args, count);
 *          starsh_randtlr_generate_va(&data, count, args);
 *          va_end(args);
 *      }
 * @endcode
 * @sa starsh_randtlr_generate(), starsh_randtlr_generate_el().
 * @ingroup app-randtlr
 * */
{
    STARSH_int nb = count;
    double decay = 0;
    double diag = 0;
    int arg_type;
    while((arg_type = va_arg(args, int)) != 0)
    {
        switch(arg_type)
        {
            case STARSH_RANDTLR_NB:
                nb = va_arg(args, STARSH_int);
                break;
            case STARSH_RANDTLR_DECAY:
                decay = va_arg(args, double);
                break;
            case STARSH_RANDTLR_DIAG:
                diag = va_arg(args, double);
                break;
            default:
                STARSH_ERROR("Wrong parameter type");
                return STARSH_WRONG_PARAMETER;
        }
    }
    return starsh_randtlr_generate(data, count, nb, decay, diag);
}

int starsh_randtlr_generate_el(STARSH_randtlr **data, STARSH_int count, ...)
//! Generate @ref STARSH_randtlr object with incomplete set of parameters.
/*! Parses possibly incomplete set of parameters for starsh_randtlr_generate().
 * If argument is not in the `...`, then its default value is used:
 *
 * Argument     | Default value | Type
 * -------------|---------------|--------
 * `block_size` | `count`       | STARSH_int
 * `decay`      | 0.0           | double
 * `diag`       | 0.0           | double
 *
 * List of arguments `...` should look as pairs (Arg.constant, Value) with 0
 * as a last argument. For correspondance of arguments and arg.constants take a
 * look at @ref STARSH_RANDTLR_PARAM.
 *
 * @param[out] data: Address of pointer to @ref STARSH_randtlr object.
 * @param[in] count: Size of matrix.
 * @param[in] ...: Variable amount of arguments.
 * @return Error code @ref STARSH_ERRNO.
 *
 * @par Examples
 * @arg @code{.c}
 *      starsh_randtlr_generate_el(&data, count,
 *          STARSH_SPATIAL_NB, 100,
 *          STARSH_SPATIAL_DECAY, 0.9,
 *          0).
 * @endcode
 * @arg @code{.c}
 *      starsh_ssdata_generate_el(&data, count,
 *          STARSH_SPATIAL_NB, 300,
 *          STARSH_SPATIAL_DECAY, 0.2,
 *          STARSH_SPATIAL_DIAG, 10.0,
 *          0).
 * @endcode
 * @sa starsh_randtlr_generate(), starsh_randtlr_generate_va().
 * @ingroup app-randtlr
 * */
{
    va_list args;
    va_start(args, count);
    int info = starsh_randtlr_generate_va(data, count, args);
    va_end(args);
    return info;
}

int starsh_randtlr_get_kernel(STARSH_kernel **kernel, STARSH_randtlr *data,
        enum STARSH_RANDTLR_KERNEL type)
//! Get kernel for spatial statistics problem.
/*! Kernel can be selected with this call or manually. Currently, there is only
 * one kernel for @ref STARSH_randtlr problem.
 *
 * @param[out] kernel: Address of pointer to @ref STARSH_kernel function.
 * @param[in] data: Pointer to @ref STARSH_randtlr object.
 * @param[in] type: Type of kernel. For more info look at @ref
 *      STARSH_RANDTLR_KERNEL.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_randtlr_block_kernel().
 * @ingroup app-randtlr
 * */
{
    switch(type)
    {
        case STARSH_RANDTLR_KERNEL1:
            *kernel = starsh_randtlr_block_kernel;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

void starsh_randtlr_free(STARSH_randtlr *data)
//! Free memory of @ref STARSH_randtlr object.
/*! @sa starsh_randtlr_generate().
 * @ingroup app-randtlr
 * */
{
    free(data->U);
    free(data->S);
    free(data);
}
