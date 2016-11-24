#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include "stars.h"
#include "misc.h"

int Array_from_buffer(Array **A, int ndim, int *shape, char dtype,
        char order, void *data)
// Init Array from given buffer. Check if all parameters are good and
// proceed.
{
    if(ndim < 0)
    {
        STARS_error("Array_from_buffer", "illegal value of ndim");
        return 1;
    }
    if(order != 'F' && order != 'C')
    {
        STARS_error("Array_from_buffer", "illegal value of order");
        return 1;
    }
    if(dtype != 's' && dtype != 'd' && dtype != 'c' && dtype != 'z')
    {
        STARS_error("Array_from_buffer", "illegal value of dtype");
        return 1;
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
        *A = malloc(sizeof(**A));
        Array *C = *A;
        C->ndim = ndim;
        C->shape = NULL;
        C->stride = NULL;
        C->order = 'F';
        C->size = 0;
        C->dtype = dtype;
        C->dtype_size = dtype_size;
        C->nbytes = 0;
        C->data = data;
        return 0;
    }
    size_t size = 1;
    int i;
    ssize_t *stride = malloc(ndim*sizeof(*stride));
    int *newshape = malloc(ndim*sizeof(*newshape));
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
    *A = malloc(sizeof(**A));
    Array *C = *A;
    C->ndim = ndim;
    C->shape = newshape;
    C->stride = stride;
    C->order = order;
    C->size = size;
    C->dtype = dtype;
    C->dtype_size = dtype_size;
    C->nbytes = size*dtype_size;
    C->data = data;
    return 0;
}

int Array_new(Array **A, int ndim, int *shape, char dtype, char order)
// Init with NULL buffer and then allocate it
{
    int info = Array_from_buffer(A, ndim, shape, dtype, order, NULL);
    if(info != 0)
        return info;
    (*A)->data = malloc((*A)->nbytes);
    return 0;
}

int Array_new_like(Array **A, Array *B)
// Initialize new A with exactly the same shape, dtype and so on, but
// with a different memory buffer
{
    *A = malloc(sizeof(**A));
    Array *C = *A;
    C->ndim = B->ndim;
    C->shape = malloc(B->ndim*sizeof(*B->shape));
    memcpy(C->shape, B->shape, B->ndim*sizeof(*B->shape));
    C->stride = malloc(B->ndim*sizeof(*B->stride));
    memcpy(C->stride, B->stride, B->ndim*sizeof(*B->stride));
    C->order = B->order;
    C->size = B->size;
    C->dtype = B->dtype;
    C->dtype_size = B->dtype_size;
    C->nbytes = B->nbytes;
    C->data = malloc(B->nbytes);
    return 0;
}

int Array_new_copy(Array **A, Array *B, char order)
// Create copy of A with given data layout or keeping layout if order ==
// 'N'
{
    if(order != 'F' && order != 'C' && order != 'N')
    {
        STARS_error("Array_new_copy", "illegal value of order");
        return 1;
    }
    int info;
    if(order == B->order || order == 'N')
    {
        info = Array_new_like(A, B);
        if(info != 0)
            return info;
        memcpy((*A)->data, B->data, B->nbytes);
        return 0;
    }
    int j;
    size_t i, ind1 = 0, ind2 = 0;
    ssize_t *coord = malloc(B->ndim*sizeof(*coord));
    for(i = 0; i < B->ndim; i++)
        coord[i] = 0;
    info = Array_new(A, B->ndim, B->shape, B->dtype, order);
    Array *C = *A;
    if(info != 0)
        return info;
    size_t dtype_size = B->dtype_size;
    for(i = 0; i < B->size; i++)
    {
        memcpy(C->data+ind2*dtype_size, B->data+ind1*dtype_size, dtype_size);
        j = B->ndim-1;
        coord[j] += 1;
        ind1 += B->stride[j];
        ind2 += C->stride[j];
        while(coord[j] == B->shape[j] && j > 0)
        {
            ind1 -= B->stride[j]*coord[j];
            ind2 -= C->stride[j]*coord[j];
            coord[j] = 0;
            j -= 1;
            ind1 += B->stride[j];
            ind2 += C->stride[j];
            coord[j] += 1;
        }
    }
    free(coord);
    return 0;
}

int Array_free(Array *A)
// Free memory, consumed by A structure and buffer
{
    if(A == NULL)
    {
        STARS_error("Array_free", "Can not free NULL pointer");
        return 1;
    }
    if(A->data != NULL)
        free(A->data);
    if(A->shape != NULL)
        free(A->shape);
    if(A->stride != NULL)
        free(A->stride);
    free(A);
    return 0;
}

void Array_info(Array *A)
// Print all the data from Array structure
{
    int i;
    printf("<Array at %p of shape (", A);
    if(A->ndim > 0)
    {
        for(i = 0; i < A->ndim; i++)
            printf("%d,", A->shape[i]);
        printf("\b");
    }
    printf("), stride (");
    if(A->ndim > 0)
    {
        for(i = 0; i < A->ndim; i++)
            printf("%zd,", A->stride[i]);
        printf("\b");
    }
    printf("), '%c' order, %zu elements, '%c' dtype, %zu bytes per element, "
            "%zu total bytes per buffer at %p>\n", A->order, A->size,
            A->dtype, A->dtype_size, A->nbytes, A->data);
}

void Array_print(Array *A)
// Print elements of A, different rows of A are printed on different
// rows of output
{
    int j;
    size_t i, row, row_size = A->size/A->shape[0];
    ssize_t offset;
    ssize_t *index = malloc(A->ndim*sizeof(*index));
    if(A->dtype == 's')
    {
        float *data = A->data;
        for(row = 0; row < A->shape[0]; row++)
        {
            index[0] = row;
            for(i = 1; i < A->ndim; i++)
                index[i] = 0;
            for(i = 0; i < row_size; i++)
            {
                offset = 0;
                for(j = 0; j < A->ndim; j++)
                    offset += A->stride[j]*index[j];
                printf(" %.2f(%zd)", data[offset], offset);
                index[1] += 1;
                j = 1;
                while(index[j] == A->shape[j])
                {
                    index[j] = 0;
                    j += 1;
                    index[j] += 1;
                }
            }
            printf("\n");
        }
    }
    else if(A->dtype == 'd')
    {
        double *data = A->data;
        for(row = 0; row < A->shape[0]; row++)
        {
            index[0] = row;
            for(i = 1; i < A->ndim; i++)
                index[i] = 0;
            for(i = 0; i < row_size; i++)
            {
                offset = 0;
                for(j = 0; j < A->ndim; j++)
                    offset += A->stride[j]*index[j];
                printf(" %.2f(%zd)", data[offset], offset);
                index[1] += 1;
                j = 1;
                while(index[j] == A->shape[j])
                {
                    index[j] = 0;
                    j += 1;
                    index[j] += 1;
                }
            }
            printf("\n");
        }
    }
    else if(A->dtype == 'c')
    {
        float complex *data = A->data;
        for(row = 0; row < A->shape[0]; row++)
        {
            index[0] = row;
            for(i = 1; i < A->ndim; i++)
                index[i] = 0;
            for(i = 0; i < row_size; i++)
            {
                offset = 0;
                for(j = 0; j < A->ndim; j++)
                    offset += A->stride[j]*index[j];
                printf(" %.2f%+.2f(%zd)", crealf(data[offset]),
                        cimagf(data[offset]), offset);
                index[1] += 1;
                j = 1;
                while(index[j] == A->shape[j])
                {
                    index[j] = 0;
                    j += 1;
                    index[j] += 1;
                }
            }
            printf("\n");
        }
    }
    else// A->dtype == 'z'
    {
        double complex *data = A->data;
        for(row = 0; row < A->shape[0]; row++)
        {
            index[0] = row;
            for(i = 1; i < A->ndim; i++)
                index[i] = 0;
            for(i = 0; i < row_size; i++)
            {
                offset = 0;
                for(j = 0; j < A->ndim; j++)
                    offset += A->stride[j]*index[j];
                printf(" %.2f%+.2f(%zd)", creal(data[offset]),
                        cimag(data[offset]), offset);
                index[1] += 1;
                j = 1;
                while(index[j] == A->shape[j])
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

int Array_init(Array *A, char *kind)
// Init buffer in a special manner: randn, rand, ones or zeros
{
    if(strcmp(kind, "randn"))
        Array_init_randn(A);
    else if(strcmp(kind, "rand"))
        Array_init_rand(A);
    else if(strcmp(kind, "zeros"))
        Array_init_zeros(A);
    else if(strcmp(kind, "ones"))
        Array_init_ones(A);
    else
    {
        STARS_error("Array_init", "illegal value of kind");
        return 1;
    }
    return 0;
}

void Array_init_randn(Array *A)
// Init buffer of A with random numbers of normal (0,1) distribution
{
    size_t i;
    if(A->dtype == 's')
    {
        float *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = randn();
    }
    else if(A->dtype == 'd')
    {
        double *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = randn();
    }
    else if(A->dtype == 'c')
    {
        float complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = randn()+I*randn();
    }
    else // A->dtype == 'z'
    {
        double complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = randn()+I*randn();
    }
}

void Array_init_rand(Array *A)
// Init buffer with random numbers of uniform [0,1] distribution
{
    size_t i;
    if(A->dtype == 's')
    {
        float *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = (double)rand()/(double)RAND_MAX;
    }
    else if(A->dtype == 'd')
    {
        double *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = (double)rand()/(double)RAND_MAX;
    }
    else if(A->dtype == 'c')
    {
        float complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = (double)rand()/(double)RAND_MAX+
                I*(double)rand()/(double)RAND_MAX;
    }
    else // A->dtype == 'z'
    {
        double complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = (double)rand()/(double)RAND_MAX+
                I*(double)rand()/(double)RAND_MAX;
    }
}

void Array_init_zeros(Array *A)
// Set all elements to 0.0
{
    size_t i;
    if(A->dtype == 's')
    {
        float *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 0.;
    }
    else if(A->dtype == 'd')
    {
        double *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 0.;
    }
    else if(A->dtype == 'c')
    {
        float complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 0.;
    }
    else // A->dtype == 'z'
    {
        double complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 0.;
    }
}

void Array_init_ones(Array *A)
// Set all elements to 1.0
{
    size_t i;
    if(A->dtype == 's')
    {
        float *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 1.;
    }
    else if(A->dtype == 'd')
    {
        double *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 1.;
    }
    else if(A->dtype == 'c')
    {
        float complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 1.;
    }
    else // A->dtype == 'z'
    {
        double complex *data = A->data;
        for(i = 0; i < A->size; i++)
            data[i] = 1.;
    }
}

int Array_tomatrix(Array *A, char kind)
// Convert N-dimensional A to 2-dimensional A (matrix) by
// collapsing dimensions. This collapse can be assumed as attempt to look
// at A as at a matrix with long rows (kind == 'R') or long columns
// (kind == 'C'). If kind is 'R', dimensions from 1 to the last are
// collapsed into columns. If kind is 'C', dimensions from 0 to the last
// minus one are collapsed into rows. Example: A of shape (2,3,4,5)
// will be collapsed to A of shape (2,60) if kind is 'R' or to A of
// shape (24,5) if kind is 'C'.
{
    if(kind != 'R' && kind != 'C')
    {
        STARS_error("Array_tomatrix", "illegal value of kind");
        return 1;
    }
    if(kind == 'R')
    {
        A->shape[1] = A->size/A->shape[0];
        if(A->order == 'C')
            A->stride[1] = 1;
    }
    else // kind == 'C'
    {
        A->shape[1] = A->shape[A->ndim-1];
        A->shape[0] = A->size/A->shape[1];
        if(A->order == 'F')
            A->stride[1] = A->stride[A->ndim-1];
        else // A->order == 'C'
        {
            A->stride[0] = A->stride[A->ndim-2];
            A->stride[1] = A->stride[A->ndim-1];
        }
    }
    A->ndim = 2;
    return 0;
}

void Array_trans_inplace(Array *A)
// Transposition of A. No real transposition is performed, only changes
// shape, stride and order.
{
    int i;
    int *new_shape = malloc(A->ndim*sizeof(*new_shape));
    ssize_t *new_stride = malloc(A->ndim*sizeof(*new_stride));
    for(i = 0; i < A->ndim; i++)
    {
        new_shape[i] = A->shape[A->ndim-i-1];
        new_stride[i] = A->stride[A->ndim-i-1];
    }
    free(A->shape);
    free(A->stride);
    A->shape = new_shape;
    A->stride = new_stride;
    A->order = ('F'-A->order)+'C';
}

int Array_dot(Array* A, Array *B, Array **C)
// GEMM for two As. Multiplication is performed by last dimension of
// A A and first dimension of A B. These dimensions, data types and
// ordering of both As should be equal.
{
    int i;
    if(A->dtype != B->dtype)
    {
        STARS_error("Array_dot", "Data type of each A must be the same.\n");
        return 1;
    }
    if(A->order != B->order)
    {
        STARS_error("Array_dot", "Order of each A must be the same.\n");
        return 1;
    }
    if(A->shape[A->ndim-1] != B->shape[0])
    {
        STARS_error("Array_dot", "Can not multiply As of shapes (");
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
        return 1;
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
    int info;
    int *new_shape = malloc(new_ndim*sizeof(*new_shape));
    for(i = 0; i < A->ndim-1; i++)
        new_shape[i] = A->shape[i];
    for(i = 0; i < B->ndim-1; i++)
        new_shape[i+A->ndim-1] = B->shape[i+1];
    info = Array_new(C, new_ndim, new_shape, A->dtype, A->order);
    Array *D = *C;
    if(info != 0)
        return info;
    int ldc = D->stride[0]* D->stride[D->ndim-1];
    if(D->dtype == 's')
    {
        cblas_sgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, 1., A->data,
                lda, B->data, ldb, 0., D->data, ldc);
    }
    else if(D->dtype == 'd')
    {
        cblas_dgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, 1., A->data,
                lda, B->data, ldb, 0., D->data, ldc);
    }
    else if(D->dtype == 'c')
    {
        float complex one = 1.0, zero = 0.0;
        cblas_cgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, &one,
                A->data, lda, B->data, ldb, &zero, D->data, ldc);
    }
    else // C->dtype == 'z'
    {
        double complex one = 1.0, zero = 0.0;
        cblas_zgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, &one,
                A->data, lda, B->data, ldb, &zero, D->data, ldc);
    }
    return 0;
}

int Array_SVD(Array *A, Array **U, Array **S, Array **V)
// Compute SVD of a given 2-dimensional A.
{
    if(A->ndim != 2)
    {
        STARS_error("Array_SVD", "Input A must be 2-dimensional to perform"
                " SVD");
        return 1;
    }
    char uv_dtype = A->dtype;
    int order, lda, ldu, ldv, info;
    int tmp_shape[2];
    int mn = A->shape[0];
    char s_dtype;
    if(uv_dtype == 's' || uv_dtype == 'c')
        s_dtype = 's';
    else // uv_dtype 'd' || uv_dtype == 'z'
        s_dtype = 'd';
    if(mn > A->shape[1])
        mn = A->shape[1];
    if(A->order == 'C')
    {
        order = LAPACK_ROW_MAJOR;
        lda = A->shape[1];
        ldu = mn;
        ldv = lda;
    }
    else // A->order == 'F'
    {
        order = LAPACK_COL_MAJOR;
        lda = A->shape[0];
        ldu = lda;
        ldv = mn;
    }
    tmp_shape[0] = A->shape[0];
    tmp_shape[1] = mn;
    info = Array_new(U, 2, tmp_shape, uv_dtype, A->order);
    if(info != 0)
        return info;
    tmp_shape[1] = A->shape[1];
    tmp_shape[0] = mn;
    info = Array_new(V, 2, tmp_shape, uv_dtype, A->order);
    if(info != 0)
        return info;
    info = Array_new(S, 1, tmp_shape, s_dtype, A->order);
    if(info != 0)
        return info;
    if(uv_dtype == 's')
        return LAPACKE_sgesdd(order, 'S', A->shape[0], A->shape[1], A->data,
                lda, (*S)->data, (*U)->data, ldu, (*V)->data, ldv);
    else if(uv_dtype == 'd')
        return LAPACKE_dgesdd(order, 'S', A->shape[0], A->shape[1], A->data,
                lda, (*S)->data, (*U)->data, ldu, (*V)->data, ldv);
    else if(uv_dtype == 'c')
        return LAPACKE_cgesdd(order, 'S', A->shape[0], A->shape[1], A->data,
                lda, (*S)->data, (*U)->data, ldu, (*V)->data, ldv);
    else // uv_dtype == 'z'
        return LAPACKE_zgesdd(order, 'S', A->shape[0], A->shape[1], A->data,
                lda, (*S)->data, (*U)->data, ldu, (*V)->data, ldv);
}

int SVD_get_rank(Array *S, double tol, char type, int *rank)
// Returns rank by given A of singular values, tolerance and type of norm
// ('2' for spectral norm, 'F' for Frobenius norm)
{
    if(type != 'F' && type != '2')
    {
        STARS_error("SVD_get_rank", "illegal value of type");
        return 1;
    }
    size_t i, size = S->size;
    if(S->dtype == 's')
    {
        float stol, tmp, *S2, *Sbuf = S->data;
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
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 1;
            while(i < size && Sbuf[i] > stol)
                i++;
        }
    }
    else// S->dtype == 'd'
    {
        double stol, tmp, *S2, *Sbuf = S->data;
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
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 1;
            while(i < size && Sbuf[i] > stol)
                i++;
        }
    }
    *rank = i;
    return 0;
}

int Array_scale(Array *A, char kind, Array *S)
// Apply row or column scaling to A
{
    if(kind != 'R' && kind != 'C')
    {
        STARS_error("Array_scale", "illegal value of kind");
        return 1;
    }
    if(A->dtype != S->dtype)
    {
        STARS_error("Array_scale", "Data types of input arrays should be the"
                " same");
        return 1;
    }
    if(S->ndim != 1)
    {
        STARS_error("Array_scale", "Factor should be 1-dimensional A");
        return 1;
    }
    if(kind == 'R' && S->shape[0] != A->shape[0])
    {
        STARS_error("Array_scale", "Input arrays should have same number "
                "of rows");
        return 1;
    }
    if(kind == 'C' && S->shape[0] != A->shape[A->ndim-1])
    {
        STARS_error("Array_scale", "Input arrays should have same number "
                "of columns");
        return 1;
    }
    int i, m = A->shape[0], n = A->shape[A->ndim-1];
    int mn = m < n ? m : n, info;
    if(A->dtype == 's')
        for(i = 0; i < mn; i++)
            cblas_sscal(m, ((float *)S->data)[i],
                    (float *)A->data+i*A->stride[1], A->stride[0]);
    else if(A->dtype == 'd')
        for(i = 0; i < mn; i++)
            cblas_dscal(m, ((double *)S->data)[i],
                    (double *)A->data+i*A->stride[1], A->stride[0]);
    else if(A->dtype == 'c')
        for(i = 0; i < mn; i++)
            cblas_cscal(m, ((float complex *)S->data)+i,
                    (float complex *)A->data+i*A->stride[1], A->stride[0]);
    else // A->dtype == 'z'
        for(i = 0; i < mn; i++)
            cblas_zscal(m, ((double complex *)S->data)+i,
                    (double complex *)A->data+i*A->stride[1], A->stride[0]);
    return info;
}

int Array_diff(Array *A, Array *B, double *result)
// Measure Frobenius error of approximation of A by B
{
    if(A->dtype != B->dtype)
    {
        STARS_error("Array_diff", "Data types of input arrays must be the "
                "same");
        return 1;
    }
    if(A->ndim != B->ndim)
    {
        STARS_error("Array_diff", "Number of dimensions of arrays must be the"
                " same");
        return 1;
    }
    else
    {
        for(int i = 0; i < A->ndim; i++)
            if(A->shape[i] != B->shape[i])
            {
                STARS_error("Array_diff", "Shapes of arrays must be the same");
                return 1;
            }
    }
    double diff = 0;
    void *tmp_buf = malloc(A->nbytes);
    int copied = 0, info;
    if(A->order != B->order)
    {
        STARS_warning("Array_diff", "input arrays have different data layout "
                "(one is 'C'-order, another is 'F'-order). Creating copy of "
                "2nd A with data layout of 1st A");
        info = Array_new_copy(&B, B, A->order);
        if(info != 0)
            return info;
        copied = 1;
    }
    if(A->dtype == 's')
    {
        cblas_scopy(A->size, B->data, 1, tmp_buf, 1);
        cblas_saxpy(A->size, -1.0, A->data, 1, tmp_buf, 1);
        diff = cblas_snrm2(A->size, tmp_buf, 1);
    }
    else if(A->dtype == 'd')
    {
        cblas_dcopy(A->size, B->data, 1, tmp_buf, 1);
        cblas_daxpy(A->size, -1.0, A->data, 1, tmp_buf, 1);
        diff = cblas_dnrm2(A->size, tmp_buf, 1);
    }
    else if(A->dtype == 'c')
    {
        float complex one = -1;
        cblas_ccopy(A->size, B->data, 1, tmp_buf, 1);
        cblas_caxpy(A->size, &one, A->data, 1, tmp_buf, 1);
        diff = cblas_scnrm2(A->size, tmp_buf, 1);
    }
    else// A->dtype == 'z'
    {
        double complex one = -1;
        cblas_zcopy(A->size, B->data, 1, tmp_buf, 1);
        cblas_zaxpy(A->size, &one, A->data, 1, tmp_buf, 1);
        diff = cblas_dznrm2(A->size, tmp_buf, 1);
    }
    free(tmp_buf);
    if(copied == 1)
        Array_free(B);
    *result = diff;
    return 0;
}

int Array_norm(Array *A, double *result)
{
    if(A->dtype == 's')
    {
        *result = cblas_snrm2(A->size, A->data, 1);
    }
    else if(A->dtype == 'd')
    {
        *result = cblas_dnrm2(A->size, A->data, 1);
    }
    else if(A->dtype == 'c')
    {
        *result = cblas_scnrm2(A->size, A->data, 1);
    }
    else// A->dtype == 'z'
    {
        *result = cblas_dznrm2(A->size, A->data, 1);
    }
    return 0;
}

int Array_convert(Array **A, Array *B, char dtype)
// Copy B and convert data type
{
    if(dtype != 's' && dtype != 'd' && dtype != 'c' && dtype != 'z')
    {
        STARS_error("Array_convert", "illegal value of dtype");
        return 1;
    }
    int info;
    if(B->dtype == dtype)
    {
        info = Array_new_copy(A, B, 'N');
        return info;
    }
    size_t i;
    if(dtype == 's')
    {
        float *dest = malloc(B->size*sizeof(*dest));
        if(B->dtype == 's')
        {
            float *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'd')
        {
            double *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'c')
        {
            float complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else// B->dtype == 'z'
        {
            double complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        info = Array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
    if(dtype == 'd')
    {
        double *dest = malloc(B->size*sizeof(*dest));
        if(B->dtype == 's')
        {
            float *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'd')
        {
            double *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'c')
        {
            float complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else// B->dtype == 'z'
        {
            double complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        info = Array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
    if(dtype == 'c')
    {
        float complex *dest = malloc(B->size*sizeof(*dest));
        if(B->dtype == 's')
        {
            float *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'd')
        {
            double *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'c')
        {
            float complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else// B->dtype == 'z'
        {
            double complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        info = Array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
    else// dtype == 'z'
    {
        double complex *dest = malloc(B->size*sizeof(*dest));
        if(B->dtype == 's')
        {
            float *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'd')
        {
            double *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else if(B->dtype == 'c')
        {
            float complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        else// B->dtype == 'z'
        {
            double complex *src = B->data;
            for(i = 0; i < B->size; i++)
            {
                dest[i] = src[i];
            }
        }
        info = Array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
    return info;
}

int Array_Cholesky(Array *A, char uplo)
// Cholesky factoriation for an A
{
    if(A->ndim != 2)
    {
        STARS_error("Array_Cholesky", "Input A must be 2-dimensional");
        return 1;
    }
    if(A->shape[0] != A->shape[1])
    {
        fprintf(stderr, "Input A must be square.\n");
        return 1;
    }
    if(uplo != 'U' && uplo != 'L')
    {
        STARS_error("Array_Cholesky", "illegal value of uplo");
        return 1;
    }
    int order;
    if(A->order == 'C')
    {
        order = LAPACK_ROW_MAJOR;
    }
    else// A->order =='F'
    {
        order = LAPACK_COL_MAJOR;
    }
    if(A->dtype == 's')
    {
        return LAPACKE_spotrf(order, uplo, A->shape[0], A->data, A->shape[0]);
    }
    else if(A->dtype == 'd')
    {
        return LAPACKE_dpotrf(order, uplo, A->shape[0], A->data, A->shape[0]);
    }
    else if(A->dtype == 'c')
    {
        return LAPACKE_cpotrf(order, uplo, A->shape[0], A->data, A->shape[0]);
    }
    else// A->dtype == 'z'
    {
        return LAPACKE_zpotrf(order, uplo, A->shape[0], A->data, A->shape[0]);
    }
}
