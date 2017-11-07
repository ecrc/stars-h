/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/minimal.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-minimal.h"

void starsh_mindata_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld)
//! The only kernel for @ref STARSH_mindata object.
/*! @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (@ref STARSH_mindata object).
 * @param[in] col_data: Pointer to physical data (@ref STARSH_mindata object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * @ingroup app-minimal
 * */
{
    int i, j;
    STARSH_mindata *data = row_data;
    STARSH_int n = data->count;
    double *buffer = result;
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            if(irow[i] == icol[j])
                buffer[j*(size_t)ld+i] = n+1;
            else
                buffer[j*(size_t)ld+i] = 1.0;
        }
}

int starsh_mindata_new(STARSH_mindata **data, STARSH_int count, char dtype)
//! Create container for minimal working example.
/*! @param[out] data: Address of pointer to @ref STARSH_mindata object.
 * @param[in] count: Size of matrix.
 * @param[in] dtype: precision ('s', 'd', 'c' or 'z').
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup app-minimal
 * */
{
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->dtype = dtype;
    return STARSH_SUCCESS;
}

void starsh_mindata_free(STARSH_mindata *data)
//! Free data.
//! @ingroup app-minimal
{
    if(data != NULL)
        free(data);
}

int starsh_mindata_get_kernel(STARSH_kernel **kernel, STARSH_mindata *data,
        enum STARSH_MINIMAL_KERNEL type)
//! Get kernel for minimal working example.
/*! Kernel can be selected with this call or manually. Currently, there is only
 * one kernel for @ref STARSH_mindata problem.
 *
 * @param[out] kernel: Address of @ref STARSH_kernel function.
 * @param[in] data: Pointer to @ref STARSH_mindata object.
 * @param[in] type: Type of kernel. For more info look at @ref
 *      STARSH_MINIMAL_KERNEL.
 * @return Error code @ref STARSH_ERRNO.
 * @sa starsh_mindata_block_kernel().
 * @ingroup app-minimal
 * */
{
    switch(type)
    {
        case STARSH_MINIMAL_KERNEL1:
            *kernel = starsh_mindata_block_kernel;
            break;
        default:
            STARSH_ERROR("Wrong type of kernel");
            return STARSH_WRONG_PARAMETER;
    }
    return STARSH_SUCCESS;
}

