/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/cauchy.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-cauchy.h"

void starsh_cauchy_block_kernel(int nrows, int ncols, STARSH_int *irow, 
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld)
//! The Cauchy kernel for @ref STARSH_cauchy object.
/*! Each off-diagonal element is generated as
 * \f[
 *      A_{ij} = \frac{1}{x_i-y_j},
 * \f]
 * where \f$ x \f$ and \f$ y \f$ are coordinates of points in one-dimensional
 * space. Diagonal elements are simply set as
 * \f[
 *      A_{ii} = d_i,
 * \f]
 * with given values of \f$ d_i \f$.
 *
 * @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (@ref STARSH_cauchy object).
 * @param[in] col_data: Pointer to physical data (@ref STARSH_cauchy object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * @sa starsh_cauchy_get_kernel().
 * @ingroup app-cauchy
 * */
{
    int i, j;
    STARSH_cauchy *data1 = row_data;
    STARSH_cauchy *data2 = col_data;
    STARSH_int count = data1->count;
    double *x = data1->point;
    double *y = data2->point;
    double *diag = x+count;
    double *buffer = result;
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            if(irow[i] == icol[j])
                buffer[j*(size_t)ld+i] = diag[irow[i]];
            else
                buffer[j*(size_t)ld+i] = 1.0/(x[irow[i]]-y[icol[j]]);
        }
}

int starsh_cauchy_init(STARSH_cauchy **data, STARSH_int count, double *point)
//! Create container for Cauchy example.
/*! @param[out] data: Address of pointer to @ref STARSH_cauchy object.
 * @param[in] count: Size of matrix.
 * @param[in] point: Coordinates of points (1D space) and values for diagonals
 *      of Cauchy matrix.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_cauchy_new().
 * @ingroup app-cauchy
 * */
{
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->point = point;
    return STARSH_SUCCESS;
}

int starsh_cauchy_new(STARSH_cauchy **data, STARSH_int count, double *point,
        double *diag)
//! Create container for Cauchy example.
/*! @param[out] data: Address of pointer to @ref STARSH_cauchy object.
 * @param[in] count: Size of matrix.
 * @param[in] point: Coordinates of points (1D space).
 * @param[in] diag: Values for diagonal of Cauchy matrix.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_cauchy_init().
 * @ingroup app-cauchy
 * */
{
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    STARSH_MALLOC((*data)->point, count*2);
    if(point != NULL)
        for(STARSH_int i = 0; i < count ; i++)
            (*data)->point[i] = point[i];
    else
        for(STARSH_int i = 0; i < count ; i++)
            (*data)->point[i] = (double)i;
    if(diag != NULL)
        for(STARSH_int i = 0; i < count ; i++)
            (*data)->point[i+count] = diag[i];
    else
        for(STARSH_int i = 0; i < count ; i++)
            (*data)->point[i+count] = 1.0;
    return STARSH_SUCCESS;
}

int starsh_cauchy_new_va(STARSH_cauchy **data, STARSH_int count, va_list args)
//! Generate @ref STARSH_cauchy object by incomplete set of parameters.
/*!
 * @sa starsh_cauchy_new().
 * @ingroup app-cauchy
 * */
{
    int arg_type;
    double *point = NULL;
    double *diag = NULL;
    while((arg_type = va_arg(args, int)) != 0)
    {
        switch(arg_type)
        {
            case STARSH_CAUCHY_POINT:
                point = va_arg(args, double *);
                break;
            case STARSH_CAUCHY_DIAG:
                diag = va_arg(args, double *);
                break;
            default:
                STARSH_ERROR("Wrong parameter type");
                return STARSH_WRONG_PARAMETER;
        }
    }
    return starsh_cauchy_new(data, count, point, diag);
}

int starsh_cauchy_new_el(STARSH_cauchy **data, STARSH_int count, ...)
//! Generate @ref STARSH_cauchy object by incomplete set of parameters.
/*!
 * @sa starsh_cauchy_new().
 * @ingroup app-cauchy
 * */
{
    va_list args;
    va_start(args, count);
    int info = starsh_cauchy_new_va(data, count, args);
    va_end(args);
    return info;
}

void starsh_cauchy_free(STARSH_cauchy *data)
//! Free data.
//! @ingroup app-cauchy
{
    starsh_particles_free(data);
}

int starsh_cauchy_get_kernel(STARSH_kernel **kernel, STARSH_cauchy *data,
        enum STARSH_CAUCHY_KERNEL type)
//! Get kernel for minimal working example.
/*! Kernel can be selected with this call or manually. Currently, there is only
 * one kernel for @ref STARSH_cauchy problem.
 *
 * @param[out] kernel: Address of @ref STARSH_kernel function.
 * @param[in] data: Pointer to @ref STARSH_cauchy object.
 * @param[in] type: Type of kernel. For more info look at @ref
 *      STARSH_CAUCHY_KERNEL.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_cauchy_block_kernel().
 * @ingroup app-cauchy
 * */
{
    switch(type)
    {
        case STARSH_CAUCHY_KERNEL1:
            *kernel = starsh_cauchy_block_kernel;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

