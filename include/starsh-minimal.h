#ifndef __STARSH_MINIMAL_H__
#define __STARSH_MINIMAL_H__

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_mindata
{
    size_t count;
    char dtype;
} STARSH_mindata;

int starsh_mindata_new(STARSH_mindata **data, int n, char dtype);
int starsh_mindata_new_va(STARSH_mindata **data, int n, char dtype,
        va_list args);
int starsh_mindata_new_el(STARSH_mindata **data, int n, char dtype, ...);
void starsh_mindata_free(STARSH_mindata *data);
int starsh_mindata_get_kernel(STARSH_kernel *kernel, STARSH_mindata *data,
        int type);

#endif // __STARSH_MINIMAL_H__
