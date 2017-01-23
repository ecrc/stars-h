#ifndef _MISC_H_
#define _MISC_H_

#include <stdio.h>
#include <stdlib.h>

double randn();

#define STARSH_MALLOC_FAILED 1

#define STARSH_ERROR(format, ...)\
{\
    fprintf(stderr, "STARS ERROR: %s(): ", __func__);\
    fprintf(stderr, format, ##__VA_ARGS__);\
    fprintf(stderr, "\n");\
}

#define STARSH_WARNING(format, ...)\
{\
    fprintf(stderr, "STARS WARNING: %s(): ", __func__);\
    fprintf(stderr, format, ##__VA_ARGS__);\
    fprintf(stderr, "\n");\
}

#define STARSH_MALLOC(var, expr_nitems)\
{\
    var = malloc(sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("line %d: malloc() failed", __LINE__);\
        return 1;\
    }\
}

#define STARSH_REALLOC(var, expr_nitems)\
{\
    var = realloc(var, sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        return 1;\
    }\
}

#define STARSH_PMALLOC(var, expr_nitems, var_info)\
{\
    var = malloc(sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        var_info = 1;\
    }\
}

#define STARSH_PREALLOC(var, expr_nitems, var_info)\
{\
    var = realloc(var, sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        var_info = 1;\
    }\
}

#endif
