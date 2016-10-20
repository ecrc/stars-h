#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include "stars.h"
#include "stars-misc.h"

Array *Array_from_buffer(int ndim, int *shape, char dtype, char order,
        void *buffer)
{
    // Init array from given buffer. Check if all parameters are good and
    // proceed.
    int i, error = 0;
    if(ndim < 0)
    {
        fprintf(stderr, "Number of dimensions can not be less than 0.\n");
        error = 1;
    }
    if(order != 'F' && order != 'C')
    {
        fprintf(stderr, "Order should be one of 'F' or 'C', not '%c'.\n",
                order);
        error = 1;
    }
    if(dtype != 's' && dtype != 'd' && dtype != 'c' && dtype != 'z')
    {
        fprintf(stderr, "Data type should be one of 's', 'd', 'c' or 'z', "
                "not '%c'.\n", dtype);
        error = 1;
    }
    for(i = 0; i < ndim; i++)
        if(shape[i] < 0)
        {
            fprintf(stderr, "Shape[%d] is %d, should be nonnegative.\n",
                    i, shape[i]);
            error = 1;
        }
    if(error)
    {
        exit(1);
    }
    if(ndim == 0)
    {
        Array *array = (Array *)malloc(sizeof(Array));
        array->ndim = ndim;
        array->shape = NULL;
        array->stride = NULL;
        array->order = 'F';
        array->dtype = dtype;
        array->size = 0;
        array->nbytes = 0;
        array->buffer = buffer;
        return array;
    }
    int size = 1, elem_size = 0;
    int *stride = (int *)malloc(ndim*sizeof(int));
    int *newshape = (int *)malloc(ndim*sizeof(int));
    if(order == 'F')
    {
        stride[0] = 1;
        newshape[0] = shape[0];
        for(i = 1; i < ndim; i++)
        {
            newshape[i] = shape[i];
            stride[i] = stride[i-1]*shape[i-1];
        }
        size = stride[ndim-1]*shape[ndim-1];
    }
    else // order == 'C'
    {
        stride[ndim-1] = 1;
        newshape[ndim-1] = shape[ndim-1];
        for(i = ndim-2; i >= 0; i--)
        {
            newshape[i] = shape[i];
            stride[i] = stride[i+1]*shape[i+1];
        }
        size = stride[0]*shape[0];
    }
    if(dtype == 's')
        elem_size = sizeof(float);
    else if(dtype == 'd')
        elem_size = sizeof(double);
    else if(dtype == 'c')
        elem_size = sizeof(float complex);
    else // dtype == 'z'
        elem_size = sizeof(double complex);
    Array *array = (Array *)malloc(sizeof(Array));
    array->ndim = ndim;
    array->shape = newshape;
    array->stride = stride;
    array->order = order;
    array->dtype = dtype;
    array->size = size;
    array->nbytes = size*elem_size;
    array->buffer = buffer;
    return array;
}

Array *Array_new(int ndim, int *shape, char dtype, char order)
{
    // Init with NULL buffer and then allocate it
    Array *array = Array_from_buffer(ndim, shape, dtype, order, NULL);
    array->buffer = malloc(array->nbytes);
    return array;
}

Array *Array_new_like(Array *array)
{
    // Initialization of array with dtype, shape and order as of given array
    // and allocation of memory buffer
    Array *array2 = (Array *)malloc(sizeof(Array));
    array2->ndim = array->ndim;
    array2->size = array->size;
    array2->dtype = array->dtype;
    array2->nbytes = array->nbytes;
    array2->shape = (int *)malloc(array->ndim*sizeof(int));
    array2->stride = (int *)malloc(array->ndim*sizeof(int));
    array2->order = array->order;
    array2->buffer = malloc(array->nbytes);
    memcpy(array2->shape, array->shape, array->ndim*sizeof(int));
    memcpy(array2->stride, array->stride, array->ndim*sizeof(int));
    return array2;
}

Array *Array_copy(Array *array)
{
    // Create copy of array, order and shape keep are the same
    Array *array2 = Array_new_like(array);
    memcpy(array2->buffer, array->buffer, array->nbytes);
    return array2;
}

void Array_free(Array *array)
{
    // Free memory, consumed by array structure and buffer
    if(array == NULL)
    {
        fprintf(stderr, "Can not free Array at NULL.\n");
        exit(1);
    }
    if(array->buffer != NULL)
        free(array->buffer);
    if(array->shape != NULL)
        free(array->shape);
    if(array->stride != NULL)
        free(array->stride);
    free(array);
}

void Array_info(Array *array)
{
    // Print shape, stride, size, data type and ordering of array
    int i;
    printf("<Array at %p of shape (", (char *)array);
    if(array->ndim > 0)
    {
        for(i = 0; i < array->ndim; i++)
            printf("%d,", array->shape[i]);
        printf("\b");
    }
    printf("), stride (");
    if(array->ndim > 0)
    {
        for(i = 0; i < array->ndim; i++)
            printf("%d,", array->stride[i]);
        printf("\b");
    }
    printf("), %d elements, '%c' dtype, '%c' order>\n", array->size,
            array->dtype, array->order);
}

void Array_print(Array *array)
{
    // Simple printing of array, works only for arrays of doubles
    int i, j, offset, row, row_size = array->size/array->shape[0];
    int *index = (int *)malloc(array->ndim*sizeof(int));
    if(array->dtype == 's')
    {
    }
    else if(array->dtype == 'd')
    {
        double *buffer = (double *)array->buffer;
        for(row = 0; row < array->shape[0]; row++)
        {
            index[0] = row;
            for(i = 1; i < array->ndim; i++)
                index[i] = 0;
            for(i = 0; i < row_size; i++)
            {
                offset = 0;
                for(j = 0; j < array->ndim; j++)
                    offset += array->stride[j]*index[j];
                //printf(" %.2lf(%d)", buffer[offset], offset);
                printf(" %.2lf", buffer[offset]);
                index[1] += 1;
                j = 1;
                while(index[j] == array->shape[j])
                {
                    index[j] = 0;
                    j += 1;
                    index[j] += 1;
                }
            }
            printf("\n");
        }
    }
}

void Array_init(Array *array, char *kind)
{
    // Init buffer in a special manner: randn, rand, ones or zeros
    int i;
    if(strcmp(kind, "randn"))
        Array_init_randn(array);
    else if(strcmp(kind, "rand"))
        Array_init_rand(array);
    else if(strcmp(kind, "zeros"))
        Array_init_zeros(array);
    else if(strcmp(kind, "ones"))
        Array_init_ones(array);
    else
    {
        fprintf(stderr, "Parameter kind should be \"randn\", \"rand\", "
                "\"zeros\" or \"ones\", not \"%s\".\n", kind);
        exit(1);
    }
}

void Array_init_randn(Array *array)
{
    // Init buffer of array with random numbers of normal (0,1) distribution
    int i;
    if(array->dtype == 's')
    {
        float *buffer = (float *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn();
    }
    else if(array->dtype == 'd')
    {
        double *buffer = (double *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn();
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = (float complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn()+I*randn();
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn()+I*randn();
    }
}

void Array_init_rand(Array *array)
{
    // Init buffer with random numbers of uniform [0,1] distribution
    int i;
    if(array->dtype == 's')
    {
        float *buffer = (float *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = rand()/RAND_MAX;
    }
    else if(array->dtype == 'd')
    {
        double *buffer = (double *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = rand()/RAND_MAX;
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = (float complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = rand()/RAND_MAX+I*rand()/RAND_MAX;
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = rand()/RAND_MAX+I*rand()/RAND_MAX;
    }
}

void Array_init_zeros(Array *array)
{
    // Set all elements to 0.0
    int i;
    if(array->dtype == 's')
    {
        float *buffer = (float *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
    else if(array->dtype == 'd')
    {
        double *buffer = (double *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = (float complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
}

void Array_init_ones(Array *array)
{
    // Set all elements to 1.0
    int i;
    if(array->dtype == 's')
    {
        float *buffer = (float *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
    else if(array->dtype == 'd')
    {
        double *buffer = (double *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = (float complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
}

void Array_tomatrix(Array *array, char kind)
{
    // Convert N-dimensional array to 2-dimensional array (matrix) by
    // collapsing dimensions. This collapse can be assumed as attempt to look
    // at array as at a matrix with long rows (kind == 'R') or long columns
    // (kind == 'C'). If kind is 'R', dimensions from 1 to the last are
    // collapsed into columns. If kind is 'C', dimensions from 0 to the last
    // minus one are collapsed into rows. Example: array of shape (2,3,4,5)
    // will be collapsed to array of shape (2,60) if kind is 'R' or to array of
    // shape (24,5) if kind is 'C'.
    if(kind != 'R' && kind != 'C')
    {
        fprintf(stderr, "Parameter kind must be equal to 'R' or 'C', not "
                "'%c'.\n", kind);
        exit(1);
    }
    if(kind == 'R')
    {
        array->shape[1] = array->size/array->shape[0];
        if(array->order == 'C')
            array->stride[1] = 1;
    }
    else // kind == 'C'
    {
        array->shape[1] = array->shape[array->ndim-1];
        array->shape[0] = array->size/array->shape[1];
        if(array->order == 'F')
            array->stride[1] = array->stride[array->ndim-1];
        else // array->order == 'C'
        {
            array->stride[0] = array->stride[array->ndim-2];
            array->stride[1] = array->stride[array->ndim-1];
        }
    }
    array->ndim = 2;
}

void Array_trans(Array *array)
{
    // Transposition of array. No real transposition is performed, only changes
    // shape, stride and order.
    int i;
    int *new_shape = (int *)malloc(array->ndim*sizeof(int));
    int *new_stride = (int *)malloc(array->ndim*sizeof(int));
    for(i = 0; i < array->ndim; i++)
    {
        new_shape[i] = array->shape[array->ndim-i-1];
        new_stride[i] = array->stride[array->ndim-i-1];
    }
    free(array->shape);
    free(array->stride);
    array->shape = new_shape;
    array->stride = new_stride;
    array->order = ('F'-array->order)+'C';
}

Array *Array_dot(Array* A, Array *B)
{
    // GEMM for two arrays. Multiplication is performed by last dimension of
    // array A and first dimension of array B. These dimensions, data types and
    // ordering of both arrays should be equal.
    int i, error = 0;
    if(A->dtype != B->dtype)
    {
        fprintf(stderr, "Data type of each array must be the same.\n");
        error = 1;
    }
    if(A->order != B->order)
    {
        fprintf(stderr, "Order of each array must be the same.\n");
        error = 1;
    }
    if(A->shape[A->ndim-1] != B->shape[0])
    {
        fprintf(stderr, "Can not multiply arrays of shapes (");
        if(A->ndim > 0)
        {
            for(i = 0; i < A->ndim; i++)
                fprintf(stderr, "%d,", A->shape[i]);
            fprintf(stderr, "\b");
        }
        fprintf(stderr, ") and (");
        if(B->ndim > 0)
        {
            for(i = 0; i < B->ndim; i++)
                fprintf(stderr, "%d,", B->shape[i]);
            fprintf(stderr, "\b");
        }
        fprintf(stderr, ")\n");
        error = 1;
    }
    if(error)
    {
        exit(1);
    }
    int order;
    if(A->order == 'C')
        order = LAPACK_ROW_MAJOR;
    else // A->order == 'F'
        order = LAPACK_COL_MAJOR;
    int new_ndim = A->ndim+B->ndim-2;
    int m = A->size/A->shape[A->ndim-1];
    int n = B->size/B->shape[0];
    int k = B->shape[0];
    int lda = A->stride[0]*A->stride[A->ndim-1];
    int ldb = B->stride[0]*B->stride[B->ndim-1];
    int *new_shape = (int *)malloc(new_ndim*sizeof(int));
    for(i = 0; i < A->ndim-1; i++)
        new_shape[i] = A->shape[i];
    for(i = 0; i < B->ndim-1; i++)
        new_shape[i+A->ndim-1] = B->shape[i+1];
    Array *C = Array_new(new_ndim, new_shape, A->dtype, A->order);
    int ldc = C->stride[0]*C->stride[C->ndim-1];
    if(C->dtype == 's')
    {
        cblas_sgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, 1., A->buffer,
                lda, B->buffer, ldb, 0., C->buffer, ldc);
    }
    else if(C->dtype == 'd')
    {
        cblas_dgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, 1., A->buffer,
                lda, B->buffer, ldb, 0., C->buffer, ldc);
    }
    else if(C->dtype == 'c')
    {
        float complex one = 1.0, zero = 0.0;
        cblas_cgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, &one,
                A->buffer, lda, B->buffer, ldb, &zero, C->buffer, ldc);
    }
    else // C->dtype == 'z'
    {
        double complex one = 1.0, zero = 0.0;
        cblas_zgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, &one,
                A->buffer, lda, B->buffer, ldb, &zero, C->buffer, ldc);
    }
    return C;
}

void Array_SVD(Array *array, Array **U, Array **S, Array **V)
{
    // Compute SVD of a given 2-dimensional array.
    int error = 0;
    if(array->ndim != 2)
    {
        fprintf(stderr, "Input array must be 2-dimensional to perform SVD.\n");
        error = 1;
    }
    char uv_dtype = array->dtype;
    if(uv_dtype != 's' && uv_dtype != 'd' && uv_dtype != 'c'
            && uv_dtype != 'z')
    {
        fprintf(stderr, "Data type of input array should be one of 's', 'd', "
                "'c' or 'z', not '%c'.\n", uv_dtype);
        error = 1;
    }
    if(error)
    {
        exit(1);
    }
    int order, lda, ldu, ldv;
    int tmp_shape[2];
    int mn = array->shape[0];
    char s_dtype;
    if(uv_dtype == 's' || uv_dtype == 'c')
        s_dtype = 's';
    else // uv_dtype 'd' || uv_dtype == 'z'
        s_dtype = 'd';
    if(mn > array->shape[1])
        mn = array->shape[1];
    if(array->order == 'C')
    {
        order = LAPACK_ROW_MAJOR;
        lda = array->shape[1];
        ldu = mn;
        ldv = lda;
    }
    else // array->order == 'F'
    {
        order = LAPACK_COL_MAJOR;
        lda = array->shape[0];
        ldu = lda;
        ldv = mn;
    }
    tmp_shape[0] = array->shape[0];
    tmp_shape[1] = mn;
    *U = Array_new(2, tmp_shape, uv_dtype, array->order);
    tmp_shape[1] = array->shape[1];
    tmp_shape[0] = mn;
    *V = Array_new(2, tmp_shape, uv_dtype, array->order);
    *S = Array_new(1, tmp_shape, s_dtype, array->order);
    if(uv_dtype == 's')
        LAPACKE_sgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
    else if(uv_dtype == 'd')
        LAPACKE_dgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
    else if(uv_dtype == 'c')
        LAPACKE_cgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
    else // uv_dtype == 'z'
        LAPACKE_zgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
}

void Array_scale(Array *array, char kind, Array *factor)
{
    int error = 0;
    if(kind != 'R' && kind != 'C')
    {
        fprintf(stderr, "Parameter kind should be one of 'R' or 'C'. not "
                "'%c'.\n", kind);
        error = 1;
    }
    if(array->dtype != factor->dtype)
    {
        fprintf(stderr, "Data type of input arrays should be the same.\n");
        error = 1;
    }
    if(factor->ndim != 1)
    {
        fprintf(stderr, "Factor should be 1-dimensional array.\n");
        error = 1;
    }
    if(kind == 'R' && factor->shape[0] != array->shape[0])
    {
        fprintf(stderr, "Input arrays should have same number of rows.\n");
        error = 1;
    }
    if(kind == 'C' && factor->shape[0] != array->shape[array->ndim-1])
    {
        fprintf(stderr, "Input arrays should have same number of columns.\n");
        error = 1;
    }
    if(error)
    {
        exit(1);
    }
    int i, m = array->shape[0], n = array->shape[array->ndim-1];
    int mn = m < n ? m : n;
    if(array->dtype == 's')
        for(i = 0; i < mn; i++)
            cblas_sscal(m, ((float *)factor->buffer)[i], (float *)
                    array->buffer+i*array->stride[1], array->stride[0]);
    else if(array->dtype == 'd')
        for(i = 0; i < mn; i++)
            cblas_dscal(m, ((double *)factor->buffer)[i], (double *)
                    array->buffer+i*array->stride[1], array->stride[0]);
    else if(array->dtype == 'c')
        for(i = 0; i < mn; i++)
            cblas_cscal(m, ((float complex *)factor->buffer)+i,
                    (float complex *)array->buffer+i*array->stride[1],
                    array->stride[0]);
    else // array->dtype == 'z'
        for(i = 0; i < mn; i++)
            cblas_zscal(m, ((double complex *)factor->buffer)+i,
                    (double complex *)array->buffer+i*array->stride[1],
                    array->stride[0]);
}

double Array_diff(Array *array, Array *array2)
{
    int error = 0;
    if(array->dtype != array2->dtype)
    {
        fprintf(stderr, "Data type of input arrays must be the same.\n");
        error = 1;
    }
    if(array->ndim != array2->ndim)
    {
        fprintf(stderr, "Number of dimensions of arrays must be the same.\n");
        error = 1;
    }
    else
    {
        int i;
        for(i = 0; i < array->ndim; i++)
            if(array->shape[i] != array2->shape[i])
            {
                fprintf(stderr, "Shapes of arrays must be equal.\n");
                error = 1;
                break;
            }
    }
    if(error)
    {
        exit(1);
    }
    if(array->dtype == 's')
    {
        float *tmp_buf = (float *)malloc(array->nbytes);
        cblas_scopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_saxpy(array->size, -1.0, array->buffer, 1, tmp_buf, 1);
        float diff = cblas_snrm2(array->size, tmp_buf, 1);
        free(tmp_buf);
        return diff;
    }
    if(array->dtype == 'd')
    {
        double *tmp_buf = (double *)malloc(array->nbytes);
        cblas_dcopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_daxpy(array->size, -1.0, array->buffer, 1, tmp_buf, 1);
        double diff = cblas_dnrm2(array->size, tmp_buf, 1);
        free(tmp_buf);
        return diff;
    }
    if(array->dtype == 'c')
    {
        float complex one = -1;
        float complex *tmp_buf = (float complex *)malloc(array->nbytes);
        cblas_ccopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_caxpy(array->size, &one, array->buffer, 1, tmp_buf, 1);
        float diff = cblas_scnrm2(array->size, tmp_buf, 1);
        free(tmp_buf);
        return diff;
    }
    else// array->dtype == 'z'
    {
        double complex one = -1;
        double complex *tmp_buf = (double complex *)malloc(array->nbytes);
        cblas_zcopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_zaxpy(array->size, &one, array->buffer, 1, tmp_buf, 1);
        double diff = cblas_dznrm2(array->size, tmp_buf, 1);
        free(tmp_buf);
        return diff;
    }
}

double Array_norm(Array *array)
{
    if(array->dtype == 's')
    {
        return cblas_snrm2(array->size, array->buffer, 1);
    }
    if(array->dtype == 'd')
    {
        return cblas_dnrm2(array->size, array->buffer, 1);
    }
    if(array->dtype == 'c')
    {
        return cblas_scnrm2(array->size, array->buffer, 1);
    }
    else// array->dtype == 'z'
    {
        return cblas_dznrm2(array->size, array->buffer, 1);
    }
}

Array *Array_convert(Array *array, char dtype)
{
    int error = 0;
    if(dtype != 's' && dtype != 'd' && dtype != 'c' && dtype != 'z')
    {
        fprintf(stderr, "New data type must be 's', 'd', 'c' or 'z', not '%c'"
                ".\n", dtype);
        error = 1;
    }
    if(array->dtype == dtype)
    {
        fprintf(stderr, "Data type of array must not be equal to new data type"
                ".\n");
        error = 1;
    }
    if(error)
    {
        exit(1);
    }
    int i;
    Array *array2;
    if(dtype == 's')
    {
        float *buffer = (float *)malloc(array->size*sizeof(float));
        if(array->dtype == 's')
        {
            float *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'd')
        {
            double *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'c')
        {
            float complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else// array->dtype == 'z'
        {
            double complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        array2 = Array_from_buffer(array->ndim, array->shape, dtype,
                array->order, buffer);
    }
    if(dtype == 'd')
    {
        double *buffer = (double *)malloc(array->size*sizeof(double));
        if(array->dtype == 's')
        {
            float *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'd')
        {
            double *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'c')
        {
            float complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else// array->dtype == 'z'
        {
            double complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        array2 = Array_from_buffer(array->ndim, array->shape, dtype,
                array->order, buffer);
    }
    if(dtype == 'c')
    {
        float complex *buffer = (float complex *)malloc(array->size*sizeof(
                    float complex));
        if(array->dtype == 's')
        {
            float *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'd')
        {
            double *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'c')
        {
            float complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else// array->dtype == 'z'
        {
            double complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        array2 = Array_from_buffer(array->ndim, array->shape, dtype,
                array->order, buffer);
    }
    else// dtype == 'z'
    {
        double complex *buffer = (double complex *)malloc(array->size*sizeof(
                    double complex));
        if(array->dtype == 's')
        {
            float *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'd')
        {
            double *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else if(array->dtype == 'c')
        {
            float complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        else// array->dtype == 'z'
        {
            double complex *src = array->buffer;
            for(i = 0; i < array->size; i++)
            {
                buffer[i] = src[i];
            }
        }
        array2 = Array_from_buffer(array->ndim, array->shape, dtype,
                array->order, buffer);
    }
    return array2;
}

int SVD_get_rank(Array *S, double tol, char type)
{
    int i, shape[2], size = S->size;
    if(type != 'F' && type != '2')
    {
        fprintf(stderr, "type must be '2' or 'F', not '%c'\n", type);
        exit(1);
    }
    if(S->dtype == 's')
    {
        float stol, tmp, *S2, *Sbuf = (float *)S->buffer;
        if(type == 'F')
        {
            S2 = (float *)malloc(size*sizeof(float));
            i = size-1;
            tmp = Sbuf[i];
            S2[i] = tmp*tmp;
            for(i = size-2; i >= 0; i--)
            {
                tmp = Sbuf[i];
                S2[i] = tmp*tmp+S2[i+1];
            }
            stol = S2[0]*tol*tol;
            i = 1;
            while(S2[i] > stol && i < size)
                i++;
            free(S2);
            return i;
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 1;
            while(Sbuf[i] > stol && i < size)
                i++;
            return i;
        }
    }
    else// S->dtype == 'd'
    {
        double stol, tmp, *S2, *Sbuf = (double *)S->buffer;
        if(type == 'F')
        {
            S2 = (double *)malloc(size*sizeof(double));
            i = size-1;
            tmp = Sbuf[i];
            S2[i] = tmp*tmp;
            for(i = size-2; i >= 0; i--)
            {
                tmp = Sbuf[i];
                S2[i] = tmp*tmp+S2[i+1];
            }
            stol = S2[0]*tol*tol;
            i = 1;
            while(S2[i] > stol && i < size)
                i++;
            free(S2);
            return i;
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 1;
            while(Sbuf[i] > stol && i < size)
                i++;
            return i;
        }
    }
}

void SVD_get_approximation(Array *U, Array *S, Array *V, int rank);
