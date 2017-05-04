#include <stdint.h>
#include "starsh.h"

typedef struct starsh_ssdata
//! Structure for Spatial Statistics problems.
{
    size_t count;
    //!< Number of spatial points.
    int ndim;
    //!< Dimensionality of the problem.
    char dtype;
    //!< Precision of each matrix element (double, single etc)
    double *point;
    //!< Coordinates of spatial points.
    double beta;
    //!< Characteristical length of covariance. Notes as l in some papers.
    double nu;
    //!< Order of Matern kernel.
    double noise;
    //!< Noise and regularization parameter.
} STARSH_ssdata;

enum STARSH_SPATIAL_KERNEL
{
    STARSH_SPATIAL_EXP = 1,
    STARSH_SPATIAL_SQREXP = 2,
    STARSH_SPATIAL_MATERN = 3,
    STARSH_SPATIAL_EXP_SIMD = 11,
    STARSH_SPATIAL_SQREXP_SIMD = 12,
    STARSH_SPATIAL_MATERN_SIMD = 13
};

enum STARSH_SPATIAL_PARAM
{
    STARSH_SPATIAL_NDIM = 1,
    STARSH_SPATIAL_BETA = 2,
    STARSH_SPATIAL_NU = 3,
    STARSH_SPATIAL_NOISE = 4
};

int starsh_ssdata_new(STARSH_ssdata **data, int n, char dtype, int ndim,
        double beta, double nu, double noise);
int starsh_ssdata_new_va(STARSH_ssdata **data, const int n, char dtype,
        va_list args);
int starsh_ssdata_new_el(STARSH_ssdata **data, const int n, char dtype, ...);
int starsh_ssdata_get_kernel(STARSH_kernel *kernel, STARSH_ssdata *data,
        int type);
void starsh_ssdata_free(STARSH_ssdata *data);
