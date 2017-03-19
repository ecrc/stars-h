#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "starsh.h"
#include "starsh-spatial.h"
#include <stdarg.h>
#include <string.h>
#include <math.h>

int starsh_application(void **data, STARSH_kernel *kernel, char *type, ...)
{
    va_list args;
    va_start(args, type);
    char *arg_type;
    if(!strcmp(type, "spatial"))
    {
        int n = 0, sqrtn = 0;
        double beta = 0.1;
        arg_type = va_arg(args, char *);
        while(arg_type != NULL)
        {
            if(!strcmp(arg_type, "N"))
            {
                n = va_arg(args, int);
                sqrtn = sqrt(n);
                if(sqrtn*sqrtn != n)
                    STARSH_ERROR("Parameter \"N\" must be square integer");
            }
            else if(!strcmp(arg_type, "Beta"))
            {
                beta = va_arg(args, double);
            }
            arg_type = va_arg(args, char *);
        }
        starsh_gen_ssdata(data, kernel, sqrtn, beta);
    }
    va_end(args);
    return 0;
}
