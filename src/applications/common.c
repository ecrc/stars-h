#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "starsh.h"
#include "starsh-spatial.h"
#include <stdarg.h>
#include <string.h>
#include <math.h>

int starsh_application(void **data, STARSH_kernel *kernel, int n, char dtype,
        const char *problem_type, const char *kernel_type, ...)
{
    va_list args;
    va_start(args, kernel_type);
    if(!strcmp(problem_type, "spatial"))
    {
        starsh_ssdata_new_va((STARSH_ssdata **)data, n, dtype, args);
        starsh_ssdata_get_kernel(kernel, kernel_type, dtype);
    }
    else if(!strcmp(problem_type, "spatial1d"))
    {
        starsh_ssdata_new_1d_va((STARSH_ssdata **)data, n, dtype, args);
        starsh_ssdata_1d_get_kernel(kernel, kernel_type, dtype);
    }
    else
        STARSH_ERROR("Wrong value of problem_type");
    va_end(args);
    return 0;
}
