#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include "stars.h"
#include "stars-misc.h"

Array *Array_from_buffer(size_t ndim, size_t *shape, char dtype, char order,
        void *buffer)
// Init array from given buffer. Check if all parameters are good and
// proceed.
{
    int error = 0;
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
    if(error)
    {
        exit(1);
    }
    size_t dtype_size = 0;
    if(dtype == 's')
        dtype_size = sizeof(float);
    else if(dtype == 'd')
        dtype_size = sizeof(double);
    else if(dtype == 'c')
        dtype_size = sizeof(float complex);
    else // dtype == 'z'
        dtype_size = sizeof(double complex);
    if(ndim == 0)
    {
        Array *array = malloc(sizeof(Array));
        array->ndim = ndim;
        array->shape = NULL;
        array->stride = NULL;
        array->order = 'F';
        array->size = 0;
        array->dtype = dtype;
        array->dtype_size = dtype_size;
        array->nbytes = 0;
        array->buffer = buffer;
        return array;
    }
    size_t size = 1, i;
    ssize_t *stride = malloc(ndim*sizeof(*stride));
    size_t *newshape = malloc(ndim*sizeof(*newshape));
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
        for(i = ndim-1; i > 0; i--)
        {
            newshape[i-1] = shape[i-1];
            stride[i-1] = stride[i]*shape[i];
        }
        size = stride[0]*shape[0];
    }
    Array *array = malloc(sizeof(*array));
    array->ndim = ndim;
    array->shape = newshape;
    array->stride = stride;
    array->order = order;
    array->size = size;
    array->dtype = dtype;
    array->dtype_size = dtype_size;
    array->nbytes = size*dtype_size;
    array->buffer = buffer;
    return array;
}

Array *Array_new(size_t ndim, size_t *shape, char dtype, char order)
// Init with NULL buffer and then allocate it
{
    Array *array = Array_from_buffer(ndim, shape, dtype, order, NULL);
    array->buffer = malloc(array->nbytes);
    return array;
}

Array *Array_new_like(Array *array)
// Initialize new array with exactly the same shape, dtype and so on, but
// with a different memory buffer
{
    Array *array2 = malloc(sizeof(*array));
    array2->ndim = array->ndim;
    array2->shape = malloc(array->ndim*sizeof(*array->shape));
    memcpy(array2->shape, array->shape, array->ndim*sizeof(*array->shape));
    array2->stride = malloc(array->ndim*sizeof(*array->stride));
    memcpy(array2->stride, array->stride, array->ndim*sizeof(*array->shape));
    array2->order = array->order;
    array2->size = array->size;
    array2->dtype = array->dtype;
    array2->dtype_size = array->dtype_size;
    array2->nbytes = array->nbytes;
    array2->buffer = malloc(array->nbytes);
    return array2;
}

Array *Array_copy(Array *array, char order)
// Create copy of array with given data layout or keeping layout if order ==
// 'N'
{
    if(order != 'F' && order != 'C' && order != 'N')
    {
        fprintf(stderr, "Wrong parameter order, should be 'F' or 'C'\n");
        exit(1);
    }
    Array *array2;
    if(order == array->order || order == 'N')
    {
        array2 = Array_new_like(array);
        memcpy(array2->buffer, array->buffer, array->nbytes);
        return array2;
    }
    size_t i, j, ind1 = 0, ind2 = 0;
    ssize_t *coord = malloc(array->ndim*sizeof(*coord));
    for(i = 0; i < array->ndim; i++)
        coord[i] = 0;
    array2 = Array_new(array->ndim, array->shape, array->dtype, order);
    size_t dtype_size = array->dtype_size;
    for(i = 0; i < array->size; i++)
    {
        memcpy(array2->buffer+ind2*dtype_size,
                array->buffer+ind1*dtype_size, dtype_size);
        j = array->ndim-1;
        coord[j] += 1;
        ind1 += array->stride[j];
        ind2 += array2->stride[j];
        while(coord[j] == array->shape[j] && j > 0)
        {
            ind1 -= array->stride[j]*coord[j];
            ind2 -= array2->stride[j]*coord[j];
            coord[j] = 0;
            j -= 1;
            ind1 += array->stride[j];
            ind2 += array2->stride[j];
            coord[j] += 1;
        }
    }
    free(coord);
    return array2;
}

void Array_free(Array *array)
// Free memory, consumed by array structure and buffer
{
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
// Print all the data from Array structure
{
    size_t i;
    printf("<Array at %p of shape (", array);
    if(array->ndim > 0)
    {
        for(i = 0; i < array->ndim; i++)
            printf("%zu,", array->shape[i]);
        printf("\b");
    }
    printf("), stride (");
    if(array->ndim > 0)
    {
        for(i = 0; i < array->ndim; i++)
            printf("%zu,", array->stride[i]);
        printf("\b");
    }
    printf("), '%c' order, %zu elements, '%c' dtype, %zu bytes per element, "
            "%zu total bytes per buffer at %p>\n", array->order, array->size,
            array->dtype, array->dtype_size, array->nbytes, array->buffer);
}

void Array_print(Array *array)
// Print elements of array, different rows of array are printed on different
// rows of output
{
    size_t i, j, row, row_size = array->size/array->shape[0];
    ssize_t offset;
    ssize_t *index = malloc(array->ndim*sizeof(*index));
    if(array->dtype == 's')
    {
        float *buffer = array->buffer;
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
                printf(" %.2f(%zd)", buffer[offset], offset);
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
    else if(array->dtype == 'd')
    {
        double *buffer = array->buffer;
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
                printf(" %.2f(%zd)", buffer[offset], offset);
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
    else if(array->dtype == 'c')
    {
        float complex *buffer = array->buffer;
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
                printf(" %.2f%+.2f(%zd)", crealf(buffer[offset]),
                        cimagf(buffer[offset]), offset);
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
    else// array->dtype == 'z'
    {
        double complex *buffer = array->buffer;
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
                printf(" %.2f%+.2f(%zd)", creal(buffer[offset]),
                        cimag(buffer[offset]), offset);
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
    free(index);
}

void Array_init(Array *array, char *kind)
// Init buffer in a special manner: randn, rand, ones or zeros
{
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
// Init buffer of array with random numbers of normal (0,1) distribution
{
    size_t i;
    if(array->dtype == 's')
    {
        float *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn();
    }
    else if(array->dtype == 'd')
    {
        double *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn();
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn()+I*randn();
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn()+I*randn();
    }
}

void Array_init_rand(Array *array)
// Init buffer with random numbers of uniform [0,1] distribution
{
    size_t i;
    if(array->dtype == 's')
    {
        float *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = (double)rand()/(double)RAND_MAX;
    }
    else if(array->dtype == 'd')
    {
        double *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = (double)rand()/(double)RAND_MAX;
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = (double)rand()/(double)RAND_MAX+
                I*(double)rand()/(double)RAND_MAX;
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = (double)rand()/(double)RAND_MAX+
                I*(double)rand()/(double)RAND_MAX;
    }
}

void Array_init_zeros(Array *array)
// Set all elements to 0.0
{
    size_t i;
    if(array->dtype == 's')
    {
        float *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
    else if(array->dtype == 'd')
    {
        double *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
}

void Array_init_ones(Array *array)
// Set all elements to 1.0
{
    size_t i;
    if(array->dtype == 's')
    {
        float *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
    else if(array->dtype == 'd')
    {
        double *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
    else if(array->dtype == 'c')
    {
        float complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
    else // array->dtype == 'z'
    {
        double complex *buffer = array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
}

void Array_tomatrix(Array *array, char kind)
// Convert N-dimensional array to 2-dimensional array (matrix) by
// collapsing dimensions. This collapse can be assumed as attempt to look
// at array as at a matrix with long rows (kind == 'R') or long columns
// (kind == 'C'). If kind is 'R', dimensions from 1 to the last are
// collapsed into columns. If kind is 'C', dimensions from 0 to the last
// minus one are collapsed into rows. Example: array of shape (2,3,4,5)
// will be collapsed to array of shape (2,60) if kind is 'R' or to array of
// shape (24,5) if kind is 'C'.
{
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
// Transposition of array. No real transposition is performed, only changes
// shape, stride and order.
{
    size_t i;
    size_t *new_shape = malloc(array->ndim*sizeof(*new_shape));
    ssize_t *new_stride = malloc(array->ndim*sizeof(*new_stride));
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
// GEMM for two arrays. Multiplication is performed by last dimension of
// array A and first dimension of array B. These dimensions, data types and
// ordering of both arrays should be equal.
{
    int error = 0;
    size_t i;
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
                fprintf(stderr, "%zu,", A->shape[i]);
            fprintf(stderr, "\b");
        }
        fprintf(stderr, ") and (");
        if(B->ndim > 0)
        {
            for(i = 0; i < B->ndim; i++)
                fprintf(stderr, "%zu,", B->shape[i]);
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
    size_t new_ndim = A->ndim+B->ndim-2;
    int m = A->size/A->shape[A->ndim-1];
    int n = B->size/B->shape[0];
    int k = B->shape[0];
    int lda = A->stride[0]*A->stride[A->ndim-1];
    int ldb = B->stride[0]*B->stride[B->ndim-1];
    size_t *new_shape = malloc(new_ndim*sizeof(*new_shape));
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

int Array_SVD(Array *array, Array **U, Array **S, Array **V)
// Compute SVD of a given 2-dimensional array.
{
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
    size_t tmp_shape[2];
    size_t mn = array->shape[0];
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
        return LAPACKE_sgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
    else if(uv_dtype == 'd')
        return LAPACKE_dgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
    else if(uv_dtype == 'c')
        return LAPACKE_cgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
    else // uv_dtype == 'z'
        return LAPACKE_zgesdd(order, 'S', array->shape[0], array->shape[1],
                array->buffer, lda, (*S)->buffer, (*U)->buffer, ldu,
                (*V)->buffer, ldv);
}

size_t SVD_get_rank(Array *S, double tol, char type)
// Returns rank by given array of singular values, tolerance and type of norm
// ('2' for spectral norm, 'F' for Frobenius norm)
{
    size_t i, size = S->size;
    if(type != 'F' && type != '2')
    {
        fprintf(stderr, "type must be '2' or 'F', not '%c'\n", type);
        exit(1);
    }
    if(S->dtype == 's')
    {
        float stol, tmp, *S2, *Sbuf = S->buffer;
        if(type == 'F')
        {
            S2 = malloc(size*sizeof(*S2));
            i = size-1;
            tmp = Sbuf[i];
            S2[i] = tmp*tmp;
            for(i = size-1; i > 0; i--)
            {
                tmp = Sbuf[i-1];
                S2[i-1] = tmp*tmp+S2[i];
            }
            stol = S2[0]*tol*tol;
            i = 1;
            while(i < size && S2[i] > stol)
                i++;
            free(S2);
            return i;
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 1;
            while(i < size && Sbuf[i] > stol)
                i++;
            return i;
        }
    }
    else// S->dtype == 'd'
    {
        double stol, tmp, *S2, *Sbuf = S->buffer;
        if(type == 'F')
        {
            S2 = malloc(size*sizeof(*S2));
            i = size-1;
            tmp = Sbuf[i];
            S2[i] = tmp*tmp;
            for(i = size-1; i > 0; i--)
            {
                tmp = Sbuf[i-1];
                S2[i-1] = tmp*tmp+S2[i];
            }
            stol = S2[0]*tol*tol;
            i = 1;
            while(i < size && S2[i] > stol)
                i++;
            free(S2);
            return i;
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 1;
            while(i < size && Sbuf[i] > stol)
                i++;
            return i;
        }
    }
}

void Array_scale(Array *array, char kind, Array *factor)
// Apply row or column scaling to array
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
    size_t i, m = array->shape[0], n = array->shape[array->ndim-1];
    size_t mn = m < n ? m : n;
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
// Measure Frobenius error of approximation of array by array2
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
        for(size_t i = 0; i < array->ndim; i++)
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
    double diff = 0;
    void *tmp_buf = malloc(array->nbytes);
    int copied = 0;
    if(array->order != array2->order)
    {
        printf("Input arrays in Array_diff have different data layout "
                "(one is 'C'-order, another is 'F'-order). Creating copy of "
                "2nd array with data layout of 1st array.\n");
        array2 = Array_copy(array2, array->order);
        copied = 1;
    }
    if(array->dtype == 's')
    {
        cblas_scopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_saxpy(array->size, -1.0, array->buffer, 1, tmp_buf, 1);
        diff = cblas_snrm2(array->size, tmp_buf, 1);
    }
    else if(array->dtype == 'd')
    {
        cblas_dcopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_daxpy(array->size, -1.0, array->buffer, 1, tmp_buf, 1);
        diff = cblas_dnrm2(array->size, tmp_buf, 1);
    }
    else if(array->dtype == 'c')
    {
        float complex one = -1;
        cblas_ccopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_caxpy(array->size, &one, array->buffer, 1, tmp_buf, 1);
        diff = cblas_scnrm2(array->size, tmp_buf, 1);
    }
    else// array->dtype == 'z'
    {
        double complex one = -1;
        cblas_zcopy(array->size, array2->buffer, 1, tmp_buf, 1);
        cblas_zaxpy(array->size, &one, array->buffer, 1, tmp_buf, 1);
        diff = cblas_dznrm2(array->size, tmp_buf, 1);
    }
    free(tmp_buf);
    if(copied == 1)
        Array_free(array2);
    return diff;
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
// Copy array and convert data type
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
    size_t i;
    Array *array2;
    if(dtype == 's')
    {
        float *buffer = malloc(array->size*sizeof(float));
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
        double *buffer = malloc(array->size*sizeof(double));
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
        float complex *buffer = malloc(array->size*sizeof(*buffer));
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
        double complex *buffer = malloc(array->size*sizeof(*buffer));
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

int Array_Cholesky(Array *array, char uplo)
// Cholesky factoriation for an array
{
    if(array->ndim != 2)
    {
        fprintf(stderr, "Input array must be 2-dimensional.\n");
        exit(1);
    }
    if(array->shape[0] != array->shape[1])
    {
        fprintf(stderr, "Input array must be square.\n");
        exit(1);
    }
    if(uplo != 'U' && uplo != 'L')
    {
        fprintf(stderr, "Parameter 2 in ARRAY_Cholesky is wrong.\n");
        exit(1);
    }
    int order;
    if(array->order == 'C')
    {
        order = LAPACK_ROW_MAJOR;
    }
    else// array->order =='F'
    {
        order = LAPACK_COL_MAJOR;
    }
    if(array->dtype == 's')
    {
        return LAPACKE_spotrf(order, uplo, array->shape[0], array->buffer,
                array->shape[0]);
    }
    else if(array->dtype == 'd')
    {
        return LAPACKE_dpotrf(order, uplo, array->shape[0], array->buffer,
                array->shape[0]);
    }
    else if(array->dtype == 'c')
    {
        return LAPACKE_cpotrf(order, uplo, array->shape[0], array->buffer,
                array->shape[0]);
    }
    else// array->dtype == 'z'
    {
        return LAPACKE_zpotrf(order, uplo, array->shape[0], array->buffer,
                array->shape[0]);
    }
}
