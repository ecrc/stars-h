#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include "stars.h"
#include "stars-misc.h"

STARS_Array *STARS_Array_new(int ndim, int *shape, char dtype)
{
    int size = 1, i, elem_size;
    int *stride = (int *)malloc(ndim*sizeof(int));
    int *newshape = (int *)malloc(ndim*sizeof(int));
    stride[0] = 1;
    newshape[0] = shape[0];
    for(i = 1; i < ndim; i++)
    {
        newshape[i] = shape[i];
        stride[i] = stride[i-1]*shape[i-1];
    }
    size = stride[ndim-1]*shape[ndim-1];
    if(dtype == 's')
        elem_size = sizeof(float);
    else if(dtype == 'd')
        elem_size = sizeof(double);
    else if(dtype == 'c')
        elem_size = sizeof(float complex);
    else if(dtype == 'z')
        elem_size = sizeof(double complex);
    else
    {
        printf("Data type is not recognized, returning NULL");
        return NULL;
    }
    STARS_Array *array = (STARS_Array *)malloc(sizeof(STARS_Array));
    array->ndim = ndim;
    array->shape = newshape;
    array->stride = stride;
    array->order = 'F';
    array->dtype = dtype;
    array->size = size;
    array->nbytes = size*elem_size;
    array->buffer = malloc(array->nbytes);
    return array;
}

STARS_Array *STARS_Array_new_like(STARS_Array *array)
{
    STARS_Array *array2 = (STARS_Array *)malloc(sizeof(STARS_Array));
    array2->ndim = array->ndim;
    array2->size = array->size;
    array2->dtype = array->dtype;
    array2->nbytes = array->nbytes;
    array2->shape = (int *)malloc(array->ndim*sizeof(int));
    array2->stride = (int *)malloc(array->ndim*sizeof(int));
    array2->order = array->order;
    array2->buffer = malloc(array->nbytes);
    memcpy(array2->shape, array->shape, array->ndim);
    memcpy(array2->stride, array->stride, array->ndim);
    return array2;
}

STARS_Array *STARS_Array_copy(STARS_Array *array)
{
    STARS_Array *array2 = STARS_Array_new_like(array);
    memcpy(array2->buffer, array->buffer, array->nbytes);
    return array2;
}

void STARS_Array_free(STARS_Array *array)
{
    free(array->buffer);
    free(array->shape);
    free(array->stride);
    free(array);
}

void STARS_Array_info(STARS_Array *array)
{
    int i;
    printf("<STARS_Array at %p of shape (", (char *)array);
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
    printf("), %d elements, %c dtype, %c order>\n", array->size, array->dtype,
            array->order);
}

void STARS_Array_print(STARS_Array *array)
{
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
                printf(" %.0lf(%d)", buffer[offset], offset);
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

void STARS_Array_init(STARS_Array *array, char *kind)
{
    int i;
    if(strcmp(kind, "randn"))
        STARS_Array_init_randn(array);
    else if(strcmp(kind, "rand"))
        STARS_Array_init_rand(array);
    else if(strcmp(kind, "zeros"))
        STARS_Array_init_zeros(array);
    else if(strcmp(kind, "ones"))
        STARS_Array_init_ones(array);
    else
    {
        printf("Parameter kind should be \"randn\", \"rand\", \"zeros\" or "
                "\"ones\", proceeding without changes.\n");
    }
}

void STARS_Array_init_randn(STARS_Array *array)
{
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
    else if(array->dtype == 'z')
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = rand()/RAND_MAX+I*rand()/RAND_MAX;
    }
}

void STARS_Array_init_rand(STARS_Array *array)
{
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
    else if(array->dtype == 'z')
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = randn()+I*randn();
    }
}

void STARS_Array_init_zeros(STARS_Array *array)
{
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
    else if(array->dtype == 'z')
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 0.;
    }
}

void STARS_Array_init_ones(STARS_Array *array)
{
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
    else if(array->dtype == 'z')
    {
        double complex *buffer = (double complex *)array->buffer;
        for(i = 0; i < array->size; i++)
            buffer[i] = 1.;
    }
}

void STARS_Array_tomatrix(STARS_Array *array, char kind)
{
    if(kind != 'R' && kind != 'C')
    {
        printf("Parameter kind must be equal to 'R' or 'C'.\n");
        return;
    }
    if(kind == 'R')
    {
        array->shape[1] = array->size/array->shape[0];
        if(array->order == 'C')
            array->stride[1] = 1;
    }
    else
    {
        array->shape[1] = array->shape[array->ndim-1];
        array->shape[0] = array->size/array->shape[1];
        if(array->order == 'F')
            array->stride[1] = array->stride[array->ndim-1];
        else if(array->order == 'C')
        {
            array->stride[0] = array->stride[array->ndim-2];
            array->stride[1] = array->stride[array->ndim-1];
        }
    }
    array->ndim = 2;
}

void STARS_Array_trans(STARS_Array *array)
{
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
