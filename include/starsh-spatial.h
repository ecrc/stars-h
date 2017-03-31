#include <stdint.h>
#include "starsh.h"

typedef struct starsh_ssdata
//! Structure for Spatial Statistics problems.
{
    size_t count;
    //!< Number of spatial points.
    int ndim;
    //!< Dimensionality of the problem.
    double *point;
    //!< Coordinates of spatial points.
    double beta;
    //!< Characteristical length of covariance. Notes as l in some papers.
    double nu;
    //!< Order of Matern kernel.
} STARSH_ssdata;

int starsh_ssdata_new_1d(STARSH_ssdata **data, int sqrtn, char dtype, double beta,
        double nu);
int starsh_ssdata_new(STARSH_ssdata **data, int sqrtn, char dtype, double beta,
        double nu);
int starsh_ssdata_new_1d_va(STARSH_ssdata **data, const int n, char dtype,
        va_list args);
int starsh_ssdata_new_va(STARSH_ssdata **data, const int n, char dtype,
        va_list args);
int starsh_ssdata_new_el(STARSH_ssdata **data, const int n, char dtype, ...);
int starsh_ssdata_new_1d_el(STARSH_ssdata **data, const int n, char dtype, ...);
void starsh_ssdata_free(STARSH_ssdata *data);

int starsh_ssdata_get_kernel(STARSH_kernel *kernel, const char *type, char dtype);
int starsh_ssdata_1d_get_kernel(STARSH_kernel *kernel, const char *type, char dtype);
