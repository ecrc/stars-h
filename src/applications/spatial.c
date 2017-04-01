#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-spatial.h"

#ifdef GSL
    #include <gsl/gsl_sf.h>
#endif

static void starsh_ssdata_block_exp_kernel_1d(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
/*! Exponential kernel for spatial statistics problem
 *
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 */
{
    // Block kernel for spatial statistics
    // Returns exp^{-r/beta}, where r is a distance between particles in 1D
    int i, j;
    STARSH_ssdata *data = row_data;
    double tmp, dist, beta = -data->beta;
    double *x = data->point;
    double *buffer = result;
    //#pragma omp parallel
    //printf("myid %d\n", omp_get_thread_num());
    //#pragma omp parallel for private(tmp, dist, i, j)
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = fabs(x[irow[i]]-x[icol[j]]);
            dist = tmp/beta;
            buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

static void starsh_ssdata_block_exp_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
/*! Exponential kernel for spatial statistics problem
 *
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 */
{
    // Block kernel for spatial statistics
    // Returns exp^{-r/beta}, where r is a distance between particles in 2D
    int i, j;
    STARSH_ssdata *data = row_data;
    double tmp, dist, beta = -data->beta;
    double *x = data->point, *y = x+data->count;
    double *buffer = result;
    //#pragma omp parallel
    //printf("myid %d\n", omp_get_thread_num());
    //#pragma omp parallel for private(tmp, dist, i, j)
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

static void starsh_ssdata_block_exp_kernel_3d(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
/*! Exponential kernel for spatial statistics problem
 *
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 */
{
    // Block kernel for spatial statistics
    // Returns exp^{-r/beta}, where r is a distance between particles in 2D
    int i, j;
    STARSH_ssdata *data = row_data;
    double tmp, dist, beta = -data->beta;
    double *x = data->point, *y = x+data->count, *z = y+data->count;
    double *buffer = result;
    //#pragma omp parallel
    //printf("myid %d\n", omp_get_thread_num());
    //#pragma omp parallel for private(tmp, dist, i, j)
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

static void starsh_ssdata_block_sqr_exp_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
/*! Square exponential kernel for spatial statistics problem
 *
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 */
{
    // Block kernel for spatial statistics
    // Returns exp^{-r^2/(2 beta^2)}, where r is a distance between particles
    // in 2D
    int i, j;
    STARSH_ssdata *data = row_data;
    double tmp, dist, beta = -2*data->beta*data->beta;
    double *x = data->point, *y = x+data->count;
    double *buffer = result;
    //#pragma omp parallel
    //printf("myid %d\n", omp_get_thread_num());
    //#pragma omp parallel for private(tmp, dist, i, j)
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = dist/beta;
            buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

#ifdef GSL
    static void starsh_ssdata_block_matern_kernel(int nrows, int ncols,
            int *irow, int *icol, void *row_data, void *col_data, void *result)
    /*! Matern kernel for spatial statistics problem
     *
     * @param[in] nrows: Number of rows of corresponding array.
     * @param[in] ncols: Number of columns of corresponding array.
     * @param[in] irow: Array of row indexes.
     * @param[in] icol: Array of column indexes.
     * @param[in] row_data: Pointer to physical data.
     * @param[in] col_data: Pointer to physical data.
     * @param[out] result: Where to write elements of an array.
     */
    {
        // Block kernel for spatial statistics
        // Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=sqrt(2*nu)*r/beta
        // and r is a distance between particles in 2D
        int i, j;
        STARSH_ssdata *data = row_data;
        double tmp, dist, beta = data->beta, nu = data->nu;
        double *x = data->point, *y = x+data->count;
        double *buffer = result;
        //#pragma omp parallel
        //printf("myid %d\n", omp_get_thread_num());
        //#pragma omp parallel for private(tmp, dist, i, j)
        for(j = 0; j < ncols; j++)
            for(i = 0; i < nrows; i++)
            {
                tmp = x[irow[i]]-x[icol[j]];
                dist = tmp*tmp;
                tmp = y[irow[i]]-y[icol[j]];
                dist += tmp*tmp;
                dist = sqrt(2*nu*dist)/beta;
                if(dist == 0)
                    buffer[j*nrows+i] = 1.0;
                else
                    buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                        pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
            }
    }
#endif

static uint32_t Part1By1(uint32_t x)
{
  x &= 0x0000ffff;
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff;
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f;
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333;
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555;
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

static uint32_t Compact1By1(uint32_t x)
{
  x &= 0x55555555;
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333;
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f;
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff;
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff;
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

static uint64_t Part1By3(uint64_t x)
{
    x &= 0x000000000000ffff;
    // x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 24)) & 0x000000ff000000ff;
    // x = ---- ---- ---- ---- ---- ---- fedc ba98 ---- ---- ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 12)) & 0x000f000f000f000f;
    // x = ---- ---- ---- fedc ---- ---- ---- ba98 ---- ---- ---- 7654 ---- ---- ---- 3210
    x = (x ^ (x << 6)) & 0x0303030303030303;
    // x = ---- --fe ---- --dc ---- --ba ---- --98 ---- --76 ---- --54 ---- --32 ---- --10
    x = (x ^ (x << 3)) & 0x1111111111111111;
    // x = ---f ---e ---d ---c ---b ---a ---9 ---8 ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
    return x;
}

static uint64_t Compact1By3(uint64_t x)
{
    x &= 0x1111111111111111;
    // x = ---f ---e ---d ---c ---b ---a ---9 ---8 ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
    x = (x ^ (x >> 3)) & 0x0303030303030303;
    // x = ---- --fe ---- --dc ---- --ba ---- --98 ---- --76 ---- --54 ---- --32 ---- --10
    x = (x ^ (x >> 6)) & 0x000f000f000f000f;
    // x = ---- ---- ---- fedc ---- ---- ---- ba98 ---- ---- ---- 7654 ---- ---- ---- 3210
    x = (x ^ (x >> 12)) & 0x000000ff000000ff;
    // x = ---- ---- ---- ---- ---- ---- fedc ba98 ---- ---- ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 24)) & 0x000000000000ffff;
    // x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

static uint32_t EncodeMorton2(uint32_t x, uint32_t y)
{
    return (Part1By1(y) << 1) + Part1By1(x);
}

static uint64_t EncodeMorton3(uint64_t x, uint64_t y, uint64_t z)
{
    return (Part1By3(z) << 2) + (Part1By3(y) << 1) + Part1By3(x);
}

static uint32_t DecodeMorton2X(uint32_t code)
{
    return Compact1By1(code >> 0);
}

static uint32_t DecodeMorton2Y(uint32_t code)
{
    return Compact1By1(code >> 1);
}

static uint64_t DecodeMorton3X(uint64_t code)
{
    return Compact1By3(code >> 0);
}

static uint64_t DecodeMorton3Y(uint64_t code)
{
    return Compact1By3(code >> 1);
}

static uint64_t DecodeMorton3Z(uint64_t code)
{
    return Compact1By3(code >> 2);
}

static int compare_uint32(const void *a, const void *b)
{
    uint32_t _a = *(uint32_t *)a;
    uint32_t _b = *(uint32_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static int compare_uint64(const void *a, const void *b)
{
    uint64_t _a = *(uint64_t *)a;
    uint64_t _b = *(uint64_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static void zsort(int n, double *points)
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y;
    uint32_t z[n];
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(points[i]*(double)UINT16_MAX +.5);
        y = (uint16_t)(points[i+n]*(double)UINT16_MAX +.5);
        //printf("%f %f -> %u %u\n", points[i], points[i+n], x, y);
        z[i] = EncodeMorton2(x, y);
    }
    qsort(z, n, sizeof(uint32_t), compare_uint32);
    for(i = 0; i < n; i++)
    {
        x = DecodeMorton2X(z[i]);
        y = DecodeMorton2Y(z[i]);
        points[i] = (double)x/(double)UINT16_MAX;
        points[i+n] = (double)y/(double)UINT16_MAX;
        //printf("%lu (%u %u) -> %f %f\n", z[i], x, y, points[i], points[i+n]);
    }
}

static void zsort3(int n, double *points)
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y, z;
    uint64_t Z[n];
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(points[i]*(double)UINT16_MAX + 0.5);
        y = (uint16_t)(points[i+n]*(double)UINT16_MAX + 0.5);
        z = (uint16_t)(points[i+2*n]*(double)UINT16_MAX + 0.5);
        Z[i] = EncodeMorton3(x, y, z);
    }
    qsort(Z, n, sizeof(uint64_t), compare_uint64);
    for(i = 0; i < n; i++)
    {
        points[i] = (double)DecodeMorton3X(Z[i])/(double)UINT16_MAX;
        points[i+n] = (double)DecodeMorton3Y(Z[i])/(double)UINT16_MAX;
        points[i+2*n] = (double)DecodeMorton3Z(Z[i])/(double)UINT16_MAX;
    }
}

static void gen_points(int n, double *points)
//! Generate particles in 2-dimensional grid plus random noise.
{
    int i, j;
    double *A = points+n*n;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            points[i*n+j] = (j+0.5-0.4*rand()/(double)RAND_MAX)/n;
            A[i*n+j] = (i+0.5-0.4*rand()/(double)RAND_MAX)/n;
        }
    }
}

int starsh_ssdata_new_1d(STARSH_ssdata **data, int n, char dtype, double beta,
        double nu)
//! Generate spatial statistics data.
/*! @param[out] data: Address of pointer to `STARSH_ssdata` object.
 * @param[in] sqrtn: Number of grid steps in one dimension. Total number of
 *     elements will be `sqrtn^2`.
 * @param[in] beta: Parameter for kernel.
 * @return Error code.
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("invalid value of `data`");
        return 1;
    }
    if(n < 0)
    {
        STARSH_ERROR("invalid value of `sqrtn`");
        return 1;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("invalid value iof `beta`");
        return 1;
    }
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    *data = malloc(sizeof(**data));
    double *point;
    STARSH_MALLOC(point, n);
    for(int i = 0; i < n; i++)
        point[i] = (i+0.5-0.4*rand()/(double)RAND_MAX)/n;
    (*data)->point = point;
    (*data)->count = n;
    (*data)->ndim = 1;
    (*data)->beta = beta;
    (*data)->nu = nu;
    return 0;
}

int starsh_ssdata_new_1d_va(STARSH_ssdata **data, int n, char dtype,
        va_list args)
{
    char *arg_type;
    double beta = 0.1;
    double nu = 0.5;
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    while((arg_type = va_arg(args, char *)) != NULL)
    {
        if(!strcmp(arg_type, "beta"))
            beta = va_arg(args, double);
        else if(!strcmp(arg_type, "nu"))
            nu = va_arg(args, double);
        else
            STARSH_ERROR("Wrong parameter name %s", arg_type);
    }
    starsh_ssdata_new_1d(data, n, dtype, beta, nu);
    return 0;
}

int starsh_ssdata_new_1d_el(STARSH_ssdata **data, int n, char dtype, ...)
{
    va_list args;
    va_start(args, dtype);
    int info = starsh_ssdata_new_1d_va(data, n, dtype, args);
    va_end(args);
    return info;
}


int starsh_ssdata_new(STARSH_ssdata **data, int sqrtn, char dtype,
        double beta, double nu)
//! Generate spatial statistics data.
/*! @param[out] data: Address of pointer to `STARSH_ssdata` object.
 * @param[in] sqrtn: Number of grid steps in one dimension. Total number of
 *     elements will be `sqrtn^2`.
 * @param[in] beta: Parameter for kernel.
 * @return Error code.
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("invalid value of `data`");
        return 1;
    }
    if(sqrtn < 0)
    {
        STARSH_ERROR("invalid value of `sqrtn`");
        return 1;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("invalid value iof `beta`");
        return 1;
    }
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    *data = malloc(sizeof(**data));
    double *point;
    int n = sqrtn*sqrtn;
    STARSH_MALLOC(point, 2*n);
    gen_points(sqrtn, point);
    zsort(n, point);
    (*data)->point = point;
    (*data)->count = n;
    (*data)->ndim = 2;
    (*data)->beta = beta;
    (*data)->nu = nu;
    return 0;
}

int starsh_ssdata_new_3d(STARSH_ssdata **data, int cbrtn, char dtype,
        double beta, double nu)
//! Generate spatial statistics data.
/*! @param[out] data: Address of pointer to `STARSH_ssdata` object.
 * @param[in] sqrtn: Number of grid steps in one dimension. Total number of
 *     elements will be `sqrtn^2`.
 * @param[in] beta: Parameter for kernel.
 * @return Error code.
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("invalid value of `data`");
        return 1;
    }
    if(cbrtn < 0)
    {
        STARSH_ERROR("invalid value of `sqrtn`");
        return 1;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("invalid value iof `beta`");
        return 1;
    }
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    *data = malloc(sizeof(**data));
    double *point;
    int n = cbrtn*cbrtn*cbrtn;
    STARSH_MALLOC(point, 3*n);
    double *x = point, *y = x+n, *z = y+n;
    for(int i = 0; i < cbrtn; i++)
        for(int j = 0; j < cbrtn; j++)
            for(int k = 0; k < cbrtn; k++)
            {
                int ind = (i*cbrtn + j)*cbrtn + k;
                x[ind] = (i+0.5-0.4*rand()/(double)RAND_MAX)/cbrtn;
                y[ind] = (j+0.5-0.4*rand()/(double)RAND_MAX)/cbrtn;
                z[ind] = (k+0.5-0.4*rand()/(double)RAND_MAX)/cbrtn;
            }
    //gen_points(sqrtn, point);
    zsort3(n, point);
    (*data)->point = point;
    (*data)->count = n;
    (*data)->ndim = 3;
    (*data)->beta = beta;
    (*data)->nu = nu;
    return 0;
}

int starsh_ssdata_new_3d_va(STARSH_ssdata **data, int n, char dtype,
        va_list args)
{
    char *arg_type;
    double beta = 0.1;
    double nu = 0.5;
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    while((arg_type = va_arg(args, char *)) != NULL)
    {
        if(!strcmp(arg_type, "beta"))
            beta = va_arg(args, double);
        else if(!strcmp(arg_type, "nu"))
            nu = va_arg(args, double);
        else
            STARSH_ERROR("Wrong parameter name %s", arg_type);
    }
    int cbrtn = cbrt(n);
    if(cbrtn*cbrtn*cbrtn != n)
        STARSH_ERROR("Parameter n must be square of integer");
    starsh_ssdata_new_3d(data, cbrtn, dtype, beta, nu);
    return 0;
}

int starsh_ssdata_new_3d_el(STARSH_ssdata **data, int n, char dtype, ...)
{
    va_list args;
    va_start(args, dtype);
    int info = starsh_ssdata_new_3d_va(data, n, dtype, args);
    va_end(args);
    return info;
}

int starsh_ssdata_new_va(STARSH_ssdata **data, int n, char dtype,
        va_list args)
{
    char *arg_type;
    double beta = 0.1;
    double nu = 0.5;
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    while((arg_type = va_arg(args, char *)) != NULL)
    {
        if(!strcmp(arg_type, "beta"))
            beta = va_arg(args, double);
        else if(!strcmp(arg_type, "nu"))
            nu = va_arg(args, double);
        else
            STARSH_ERROR("Wrong parameter name %s", arg_type);
    }
    int sqrtn = sqrt(n);
    if(sqrtn*sqrtn != n)
        STARSH_ERROR("Parameter n must be square of integer");
    starsh_ssdata_new(data, sqrtn, dtype, beta, nu);
    return 0;
}

int starsh_ssdata_new_el(STARSH_ssdata **data, int n, char dtype, ...)
{
    va_list args;
    va_start(args, dtype);
    int info = starsh_ssdata_new_va(data, n, dtype, args);
    va_end(args);
    return info;
}

void starsh_ssdata_free(STARSH_ssdata *data)
//! Free data.
{
    if(data == NULL)
    {
        fprintf(stderr, "Data for spatial statistics problem was not "
                "generated\n");
        return;
    }
    free(data->point);
    free(data);
}

int starsh_ssdata_get_kernel(STARSH_kernel *kernel, const char *type,
        char dtype)
{
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    if(!strcmp(type, "exp"))
        *kernel = starsh_ssdata_block_exp_kernel;
    else if(!strcmp(type, "sqrexp"))
        *kernel = starsh_ssdata_block_sqr_exp_kernel;
    else if(!strcmp(type, "Matern"))
        *kernel = starsh_ssdata_block_matern_kernel;
    else
        STARSH_ERROR("Wrong type of kernel");
    return 0;
}

int starsh_ssdata_1d_get_kernel(STARSH_kernel *kernel, const char *type,
        char dtype)
{
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    if(!strcmp(type, "exp"))
        *kernel = starsh_ssdata_block_exp_kernel_1d;
    else
        STARSH_ERROR("Wrong type of kernel");
    return 0;
}

int starsh_ssdata_3d_get_kernel(STARSH_kernel *kernel, const char *type,
        char dtype)
{
    if(dtype != 'd')
        STARSH_ERROR("Only dtype='d' is supported");
    if(!strcmp(type, "exp"))
        *kernel = starsh_ssdata_block_exp_kernel_3d;
    else
        STARSH_ERROR("Wrong type of kernel");
    return 0;
}
