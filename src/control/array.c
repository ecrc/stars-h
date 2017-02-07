#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <mkl.h>
#include "starsh.h"

int array_from_buffer(Array **A, int ndim, int *shape, char dtype,
        char order, void *data)
// Init `A` from given buffer. Check if all parameters are good and proceed.
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(ndim < 0)
    {
        STARSH_ERROR("invalid value of `ndim`");
        return 1;
    }
    if(order != 'F' && order != 'C')
    {
        STARSH_ERROR("invalid value of `order`");
        return 1;
    }
    if(dtype != 's' && dtype != 'd' && dtype != 'c' && dtype != 'z')
    {
        STARSH_ERROR("invalid value of `dtype`");
        return 1;
    }
    STARSH_MALLOC(*A, 1);
    Array *A2 = *A;
    size_t dtype_size = 0, size = 1;
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
        A2->ndim = ndim;
        A2->shape = NULL;
        A2->stride = NULL;
        A2->order = 'F';
        A2->size = 0;
        A2->dtype = dtype;
        A2->dtype_size = dtype_size;
        A2->nbytes = 0;
        A2->data_nbytes = sizeof(*A2);
        A2->data = data;
        return 0;
    }
    int i;
    ssize_t *stride;
    STARSH_MALLOC(stride, ndim);
    int *newshape;
    STARSH_MALLOC(newshape, ndim);
    // Compute strides
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
    A2->ndim = ndim;
    A2->shape = newshape;
    A2->stride = stride;
    A2->order = order;
    A2->size = size;
    A2->dtype = dtype;
    A2->dtype_size = dtype_size;
    A2->data_nbytes = size*dtype_size;
    A2->nbytes = A2->data_nbytes+ndim*(sizeof(*newshape)+sizeof(*stride))+
            sizeof(*A2);
    A2->data = data;
    return 0;
}

int array_new(Array **A, int ndim, int *shape, char dtype, char order)
// Init `A` with NULL buffer and then allocate it
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    int info = array_from_buffer(A, ndim, shape, dtype, order, NULL);
    if(info != 0)
        return info;
    STARSH_MALLOC((*A)->data, (*A)->data_nbytes);
    return 0;
}

int array_new_like(Array **A, Array *B)
// Initialize new `A` with exactly the same shape, dtype and so on, but
// with a different memory buffer
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(B == NULL)
    {
        STARSH_ERROR("invalid value of `B`");
        return 1;
    }
    STARSH_MALLOC(*A, 1);
    Array *A2 = *A;
    A2->ndim = B->ndim;
    STARSH_MALLOC(A2->shape, B->ndim);
    memcpy(A2->shape, B->shape, B->ndim*sizeof(*B->shape));
    STARSH_MALLOC(A2->stride, B->ndim);
    memcpy(A2->stride, B->stride, B->ndim*sizeof(*B->stride));
    A2->order = B->order;
    A2->size = B->size;
    A2->dtype = B->dtype;
    A2->dtype_size = B->dtype_size;
    A2->nbytes = B->nbytes;
    A2->data_nbytes = B->data_nbytes;
    STARSH_MALLOC(A2->data, B->data_nbytes);
    return 0;
}

int array_new_copy(Array **A, Array *B, char order)
// Create `A` as a copy of `B` with given data layout or keeping layout if
// order == 'N'
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(B == NULL)
    {
        STARSH_ERROR("invalid value of `B`");
        return 1;
    }
    if(order != 'F' && order != 'C' && order != 'N')
    {
        STARSH_ERROR("invalid value of `order`");
        return 1;
    }
    int info;
    // If only simple copy isrequired
    if(order == B->order || order == 'N')
    {
        info = array_new_like(A, B);
        if(info != 0)
            return info;
        memcpy((*A)->data, B->data, B->data_nbytes);
        return 0;
    }
    int j;
    size_t i, ind1 = 0, ind2 = 0;
    ssize_t *coord;
    STARSH_MALLOC(coord, B->ndim);
    for(i = 0; i < B->ndim; i++)
        coord[i] = 0;
    info = array_new(A, B->ndim, B->shape, B->dtype, order);
    if(info != 0)
        return info;
    Array *A2 = *A;
    size_t dtype_size = B->dtype_size;
    // Compute offset of each element in `A` and `B` and then copy data
    for(i = 0; i < B->size; i++)
    {
        memcpy(A2->data+ind2*dtype_size, B->data+ind1*dtype_size, dtype_size);
        j = B->ndim-1;
        coord[j] += 1;
        ind1 += B->stride[j];
        ind2 += A2->stride[j];
        while(coord[j] == B->shape[j] && j > 0)
        {
            ind1 -= B->stride[j]*coord[j];
            ind2 -= A2->stride[j]*coord[j];
            coord[j] = 0;
            j -= 1;
            ind1 += B->stride[j];
            ind2 += A2->stride[j];
            coord[j] += 1;
        }
    }
    free(coord);
    return 0;
}

int array_free(Array *A)
// Free memory, consumed by structure and buffer of `A`
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
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

int array_info(Array *A)
// Print all the data from Array structure `A`
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
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
            "%zu bytes of data at %p, %zu total bytes>\n", A->order, A->size,
            A->dtype, A->dtype_size, A->data_nbytes, A->data, A->nbytes);
    return 0;
}

int array_print(Array *A)
// Print elements of `A`, different rows of `A` are printed on different
// rows of output
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    int j;
    size_t i, row, row_size = A->size/A->shape[0];
    ssize_t offset;
    ssize_t *index;
    STARSH_MALLOC(index, A->ndim);
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
    return 0;
}

int array_to_matrix(Array *A, char kind)
// Convert N-dimensional `A` to 2-dimensional `A` (matrix) by
// collapsing dimensions. This collapse can be assumed as attempt to look
// at `A` as at a matrix with long rows (`kind` == 'R') or long columns
// (`kind` == 'C'). If `kind` is 'R', dimensions from 1 to the last are
// collapsed into columns. If `kind` is 'C', dimensions from 0 to the last
// minus one are collapsed into rows. Example: `A` of shape (2,3,4,5)
// will be collapsed to `A` of shape (2,60) if `kind` is 'R' or to `A` of
// shape (24,5) if `kind` is 'C'.
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(kind != 'R' && kind != 'C')
    {
        STARSH_ERROR("invalid value of `kind`");
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

int array_trans_inplace(Array *A)
// Transposition of `A`. No real transposition is performed, only changes
// shape, stride and order.
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    int i;
    int *new_shape;
    STARSH_MALLOC(new_shape, A->ndim);
    ssize_t *new_stride;
    STARSH_MALLOC(new_stride, A->ndim);
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
    return 0;
}

int array_dot(Array* A, Array *B, Array **C)
// GEMM for `A` and `B`. Multiplication is performed by last dimension of
// `A` and first dimension of `B`. These dimensions, data types and
// ordering of `A` and `B` should be equal.
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(B == NULL)
    {
        STARSH_ERROR("invalid value of `B`");
        return 1;
    }
    if(C == NULL)
    {
        STARSH_ERROR("invalid value of `C`");
        return 1;
    }
    int i;
    if(A->dtype != B->dtype)
    {
        STARSH_ERROR("`dtype` of `A` and `B` should be equal");
        return 1;
    }
    if(A->order != B->order)
    {
        STARSH_ERROR("`order` of `A` and `B` should be equal");
        return 1;
    }
    if(A->shape[A->ndim-1] != B->shape[0])
    {
        STARSH_ERROR("non-multiplicative shapes of `A` and `B`");
        return 1;
    }
    int order, new_ndim = A->ndim+B->ndim-2, info;
    if(A->order == 'C')
        order = LAPACK_ROW_MAJOR;
    else // A->order == 'F'
        order = LAPACK_COL_MAJOR;
    size_t m = A->size/A->shape[A->ndim-1];
    size_t n = B->size/B->shape[0];
    int k = B->shape[0];
    ssize_t lda = A->stride[0]*A->stride[A->ndim-1];
    ssize_t ldb = B->stride[0]*B->stride[B->ndim-1];
    int *new_shape;
    STARSH_MALLOC(new_shape, new_ndim);
    for(i = 0; i < A->ndim-1; i++)
        new_shape[i] = A->shape[i];
    for(i = 0; i < B->ndim-1; i++)
        new_shape[i+A->ndim-1] = B->shape[i+1];
    info = array_new(C, new_ndim, new_shape, A->dtype, A->order);
    if(info != 0)
        return info;
    free(new_shape);
    Array *C2 = *C;
    ssize_t ldc = C2->stride[0]*C2->stride[C2->ndim-1];
    if(C2->dtype == 's')
    {
        cblas_sgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, 1., A->data,
                lda, B->data, ldb, 0., C2->data, ldc);
    }
    else if(C2->dtype == 'd')
    {
        cblas_dgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, 1., A->data,
                lda, B->data, ldb, 0., C2->data, ldc);
    }
    else if(C2->dtype == 'c')
    {
        float complex one = 1.0, zero = 0.0;
        cblas_cgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, &one,
                A->data, lda, B->data, ldb, &zero, C2->data, ldc);
    }
    else // C->dtype == 'z'
    {
        double complex one = 1.0, zero = 0.0;
        cblas_zgemm(order, CblasNoTrans, CblasNoTrans, m, n, k, &one,
                A->data, lda, B->data, ldb, &zero, C2->data, ldc);
    }
    return 0;
}

int array_SVD(Array *A, Array **U, Array **S, Array **V)
// Compute SVD of a given 2-dimensional `A`.
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(U == NULL)
    {
        STARSH_ERROR("invalid value of `U`");
        return 1;
    }
    if(S == NULL)
    {
        STARSH_ERROR("invalid value of `S`");
        return 1;
    }
    if(V == NULL)
    {
        STARSH_ERROR("invalid value of `V`");
        return 1;
    }
    if(A->ndim != 2)
    {
        STARSH_ERROR("`A` must be 2-dimensional");
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
    info = array_new(U, 2, tmp_shape, uv_dtype, A->order);
    if(info != 0)
        return info;
    tmp_shape[1] = A->shape[1];
    tmp_shape[0] = mn;
    info = array_new(V, 2, tmp_shape, uv_dtype, A->order);
    if(info != 0)
        return info;
    info = array_new(S, 1, tmp_shape, s_dtype, A->order);
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

int svd_get_rank(Array *S, double tol, char type, int *rank)
// Returns rank by given singular values `S`, tolerance and type of norm
// ('2' for spectral norm, 'F' for Frobenius norm)
{
    if(S == NULL)
    {
        STARSH_ERROR("invalid value of `S`");
        return 1;
    }
    if(rank == NULL)
    {
        STARSH_ERROR("invalid value of `rank`");
        return 1;
    }
    if(type != 'F' && type != '2')
    {
        STARSH_ERROR("invalid value of `type`");
        return 1;
    }
    size_t i, size = S->size;
    if(S->dtype == 's')
    {
        float stol, tmp, *S2, *Sbuf = S->data;
        if(type == 'F')
        {
            STARSH_MALLOC(S2, size);
            i = size-1;
            tmp = Sbuf[i];
            S2[i] = tmp*tmp;
            for(i = size-1; i > 0; i--)
            {
                tmp = Sbuf[i-1];
                S2[i-1] = tmp*tmp+S2[i];
            }
            stol = S2[0]*tol*tol;
            i = 0;
            while(i < size && S2[i] > stol)
                i++;
            free(S2);
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 0;
            while(i < size && Sbuf[i] > stol)
                i++;
        }
    }
    else// S->dtype == 'd'
    {
        double stol, tmp, *S2, *Sbuf = S->data;
        if(type == 'F')
        {
            STARSH_MALLOC(S2, size);
            i = size-1;
            tmp = Sbuf[i];
            S2[i] = tmp*tmp;
            for(i = size-1; i > 0; i--)
            {
                tmp = Sbuf[i-1];
                S2[i-1] = tmp*tmp+S2[i];
            }
            stol = S2[0]*tol*tol;
            i = 0;
            while(i < size && S2[i] > stol)
                i++;
            free(S2);
        }
        else// type == '2'
        {
            stol = Sbuf[0]*tol;
            i = 0;
            while(i < size && Sbuf[i] > stol)
                i++;
        }
    }
    *rank = i;
    return 0;
}

int array_scale(Array *A, char kind, Array *S)
// Apply row or column scaling to A
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(S == NULL)
    {
        STARSH_ERROR("invalid value of `S`");
        return 1;
    }
    if(kind != 'R' && kind != 'C')
    {
        STARSH_ERROR("invalid value of `kind`");
        return 1;
    }
    if(A->dtype != S->dtype)
    {
        STARSH_ERROR("`dtype` of `A` and `S` should be equal");
        return 1;
    }
    if(S->ndim != 1)
    {
        STARSH_ERROR("`S` should be 1-dimensional");
        return 1;
    }
    if(kind == 'R' && S->shape[0] != A->shape[0])
    {
        STARSH_ERROR("`A` and `S` should have equal number of rows");
        return 1;
    }
    if(kind == 'C' && S->shape[0] != A->shape[A->ndim-1])
    {
        STARSH_ERROR("`A` and `S` should have equal number of "
                "columns");
        return 1;
    }
    int i, m = A->shape[0], n = A->shape[A->ndim-1];
    int mn = m < n ? m : n;
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
    return 0;
}

int array_diff(Array *A, Array *B, double *result)
// Measure Frobenius error of approximation of `A` by `B`
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(B == NULL)
    {
        STARSH_ERROR("invalid value of `B`");
        return 1;
    }
    if(result == NULL)
    {
        STARSH_ERROR("invalid value of `result`");
        return 1;
    }
    if(A->dtype != B->dtype)
    {
        STARSH_ERROR("`dtype` of `A` and `B` should be equal");
        return 1;
    }
    if(A->ndim != B->ndim)
    {
        STARSH_ERROR("`ndim` of `A` and `B` should be equal");
        return 1;
    }
    else
    {
        for(int i = 0; i < A->ndim; i++)
            if(A->shape[i] != B->shape[i])
            {
                STARSH_ERROR("Shapes of `A` and `B` should be equal");
                return 1;
            }
    }
    double diff = 0;
    void *tmp_buf;
    STARSH_MALLOC(tmp_buf, A->data_nbytes);
    int copied = 0, info;
    if(A->order != B->order)
    {
        STARSH_WARNING("`A` and `B` have different data layout "
                "(one is 'C'-order, another is 'F'-order). Creating copy of "
                "`B` with data layout of `A`");
        info = array_new_copy(&B, B, A->order);
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
        array_free(B);
    *result = diff;
    return 0;
}

int array_norm(Array *A, double *result)
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(result == NULL)
    {
        STARSH_ERROR("invalid value of `result`");
        return 1;
    }
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

int array_convert(Array **A, Array *B, char dtype)
// Create `A` as a copy of `B` with different data type
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(B == NULL)
    {
        STARSH_ERROR("invalid value of `B`");
        return 1;
    }
    if(dtype != 's' && dtype != 'd' && dtype != 'c' && dtype != 'z')
    {
        STARSH_ERROR("invalid value of `dtype`");
        return 1;
    }
    if(B->dtype == dtype)
    {
        return array_new_copy(A, B, 'N');
    }
    size_t i;
    if(dtype == 's')
    {
        float *dest;
        STARSH_MALLOC(dest, B->size);
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
        return array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
    if(dtype == 'd')
    {
        double *dest;
        STARSH_MALLOC(dest, B->size);
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
        return array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
    if(dtype == 'c')
    {
        float complex *dest;
        STARSH_MALLOC(dest, B->size);
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
        return array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
    else// dtype == 'z'
    {
        double complex *dest;
        STARSH_MALLOC(dest, B->size);
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
        return array_from_buffer(A, B->ndim, B->shape, dtype, B->order, dest);
    }
}

int array_cholesky(Array *A, char uplo)
// Cholesky factoriation for `A`
{
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(A->ndim != 2)
    {
        STARSH_ERROR("`A` should be 2-dimensional");
        return 1;
    }
    if(A->shape[0] != A->shape[1])
    {
        STARSH_ERROR("`A` must be square");
        return 1;
    }
    if(uplo != 'U' && uplo != 'L')
    {
        STARSH_ERROR("invalid value of `uplo`");
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
