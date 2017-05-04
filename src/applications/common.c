#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-minimal.h"
#include "starsh-rndtiled.h"

int starsh_application(void **data, STARSH_kernel *kernel, int n, char dtype,
        int problem_type, int kernel_type, ...)
{
    va_list args;
    va_start(args, kernel_type);
    int info = 0;
    switch(problem_type)
    {
        case STARSH_MINIMAL:
            info = starsh_mindata_new_va((STARSH_mindata **)data, n, dtype,
                    args);
            info |= starsh_mindata_get_kernel(kernel, *data, kernel_type);
            break;
        case STARSH_RNDTILED:
            info = starsh_rndtiled_new_va((STARSH_rndtiled **)data, n, dtype,
                    args);
            info |= starsh_rndtiled_get_kernel(kernel, *data, kernel_type);
            break;
        case STARSH_SPATIAL:
            info = starsh_ssdata_new_va((STARSH_ssdata **)data, n, dtype,
                    args);
            info |= starsh_ssdata_get_kernel(kernel, *data, kernel_type);
            break;
        default:
            STARSH_ERROR("Wrong value of problem_type");
            return 1;
    }
    va_end(args);
    return info;
}
