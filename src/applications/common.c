#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-minimal.h"
#include "starsh-rndtiled.h"

int starsh_application(void **data, STARSH_kernel *kernel, int n, char dtype,
        const char *problem_type, const char *kernel_type, ...)
{
    va_list args;
    va_start(args, kernel_type);
    int info = 0;
    if(!strcmp(problem_type, "spatial"))
    {
        info = starsh_ssdata_new_va((STARSH_ssdata **)data, n, dtype, args);
        info |= starsh_ssdata_get_kernel(kernel, kernel_type, dtype);
    }
    else if(!strcmp(problem_type, "spatial1d"))
    {
        info = starsh_ssdata_new_1d_va((STARSH_ssdata **)data, n, dtype, args);
        info |= starsh_ssdata_1d_get_kernel(kernel, kernel_type, dtype);
    }
    else if(!strcmp(problem_type, "spatial3d"))
    {
        info = starsh_ssdata_new_3d_va((STARSH_ssdata **)data, n, dtype, args);
        info |= starsh_ssdata_3d_get_kernel(kernel, kernel_type, dtype);
    }
    else if(!strcmp(problem_type, "minimal"))
    {
        info = starsh_mindata_new_va((STARSH_mindata **)data, n, dtype, args);
        info |= starsh_mindata_get_kernel(kernel, kernel_type, dtype);
    }
    else if(!strcmp(problem_type, "rndtiled"))
    {
        info = starsh_rndtiled_new_va((STARSH_rndtiled **)data, n, dtype,
                args);
        info |= starsh_rndtiled_get_kernel(kernel, kernel_type, dtype);
    }
    else
    {
        STARSH_ERROR("Wrong value of problem_type");
        return 1;
    }
    va_end(args);
    return info;
}
