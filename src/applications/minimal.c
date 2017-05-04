#include "common.h"
#include "starsh.h"
#include "starsh-minimal.h"

static void starsh_mindata_block_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
{
    int i, j;
    STARSH_mindata *data = row_data;
    int n = data->count;
    double *buffer = result;
    //#pragma omp simd
    for(int j = 0; j < ncols; j++)
        for(int i = 0; i < nrows; i++)
        {
            if(irow[i] == icol[j])
                buffer[j*nrows+i] = n+1;
            else
                buffer[j*nrows+i] = 1.0;
        }
}

int starsh_mindata_new(STARSH_mindata **data, int n, char dtype)
{
    STARSH_MALLOC(*data, 1);
    (*data)->count = n;
    (*data)->dtype = dtype;
    return 0;
}

int starsh_mindata_new_va(STARSH_mindata **data, int n, char dtype,
        va_list args)
{
    int arg_type;
    if((arg_type = va_arg(args, int)) != 0)
    {
        STARSH_ERROR("Wrong parameter");
        return 1;
    }
    return starsh_mindata_new(data, n, dtype);
}

int starsh_mindata_new_el(STARSH_mindata **data, int n, char dtype, ...)
{
    va_list args;
    va_start(args, dtype);
    int info = starsh_mindata_new_va(data, n, dtype, args);
    va_end(args);
    return info;
}

void starsh_mindata_free(STARSH_mindata *data)
{
    if(data != NULL)
        free(data);
}

int starsh_mindata_get_kernel(STARSH_kernel *kernel, STARSH_mindata *data,
        int type)
{
    *kernel = starsh_mindata_block_kernel;
    return 0;
}
