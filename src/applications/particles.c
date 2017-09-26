/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/particles.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-22
 */

#include "starsh.h"
#include "starsh-particles.h"
#include "applications/particles.h"
#include "common.h"

void starsh_particles_free(STARSH_particles *data)
//! Free memory of STARSH_particles object, pointer itself is also freed
/*! @ingroup applications
 * */
{
    if(data != NULL)
    {
        if(data->point != NULL)
            free(data->point);
        free(data);
    }
}

int starsh_particles_generate(STARSH_particles **data, size_t count, int ndim,
        enum STARSH_PARTICLES_PLACEMENT ptype)
//! Generate STARSH_particles with required distribution of particles
/*! @ingroup applications
 * @param[out] data: Pointer to pointer to STARSH_particles to generate
 * @param[in] count: Amount of particles to generate
 * @param[in] ndim: Dimensionality of space
 * @param[in] ptype: How to place particles. Possible values are
 *  STARSH_PARTICLES_RAND, STARSH_PARTICLES_UNIFORM, STARSH_PARTICLES_RANDGRID,
 *  STARSH_PARTICLES_QUASIUNIFORM1, STARSH_PARTICLES_QUASIUNIFORM2,
 *  STARSH_PARTICLES_OBSOLETE1, STARSH_PARTICLES_OBSOLETE2
 *
 * For more details, take a look at special generation functions
 * starsh_particles_generate_rand(), starsh_particles_generate_uniform(),
 * starsh_particles_generate_randgrid(),
 * starsh_particles_generate_quasiuniform1(),
 * starsh_particles_generate_quasiuniform2(),
 * starsh_particles_generate_obsolete1(),
 * starsh_particles_generate_obsolete2()
 * */
{
    int info = STARSH_SUCCESS;
    switch(ptype)
    {
        case STARSH_PARTICLES_RAND:
            info = starsh_particles_generate_rand(data, count, ndim);
            break;
        /* Not yet implemented
        case STARSH_PARTICLES_RANDN:
            info = starsh_particles_generate_randn(data, count, ndim);
            break;
        */
        case STARSH_PARTICLES_UNIFORM:
            info = starsh_particles_generate_uniform(data, count, ndim);
            break;
        case STARSH_PARTICLES_RANDGRID:
            info = starsh_particles_generate_randgrid(data, count, ndim);
            break;
        /* Not yet implemented
        case STARSH_PARTICLES_RANDNGRID:
            info = starsh_particles_generate_randngrid(data, count, ndim);
            break;
        */
        case STARSH_PARTICLES_QUASIUNIFORM1:
            info = starsh_particles_generate_quasiuniform1(data, count, ndim);
            break;
        case STARSH_PARTICLES_QUASIUNIFORM2:
            info = starsh_particles_generate_quasiuniform2(data, count, ndim);
            break;
        case STARSH_PARTICLES_OBSOLETE1:
            info = starsh_particles_generate_obsolete1(data, count, ndim);
            break;
        case STARSH_PARTICLES_OBSOLETE2:
            info = starsh_particles_generate_obsolete2(data, count, ndim);
            break;
        default:
            return STARSH_WRONG_PARAMETER;
    };
    return info;
}

int starsh_particles_generate_rand(STARSH_particles **data, size_t count,
        int ndim)
//! Generate particles with [0,1] uniform random distribution
/*! @ingroup applications
 * @param[out] data: Pointer to pointer to STARSH_particles to generate
 * @param[in] count: Amount of particles to generate
 * @param[in] ndim: Dimensionality of space
 * */
{
    size_t i;
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->ndim = ndim;
    double *point;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    // Generate particle coordinates randomly
    for(i = 0; i < nelem; i++)
        point[i] = (double)rand()/RAND_MAX;
    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}

/* Need to implement randn function for normal distribution
int starsh_particles_generate_randn(STARSH_particles **data, size_t count,
        int ndim)
{
    size_t i;
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->ndim = ndim;
    double *point;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}
*/

int starsh_particles_generate_randgrid(STARSH_particles **data, size_t count,
        int ndim)
//! Generate a grid on randomly distributed grid coordinates
/*! @ingroup applications
 * @param[out] data: Pointer to pointer to STARSH_particles to generate
 * @param[in] count: Amount of particles to generate
 * @param[in] ndim: Dimensionality of space
 * 
 * Minimal grid, containing all `count` particles, is selected
 * */
{
    size_t i, j, side, total;
    double *point, *ptr, val;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    ptr = point;
    side = ceil(pow(count, 1.0/ndim));
    total = pow(side, ndim);
    if(total < count)
    {
        fprintf(stderr, "Inner error of %s\n", __func__);
        free(point);
        return STARSH_UNKNOWN_ERROR;
    }
    double *coord;
    size_t ncoords = ndim*side;
    STARSH_MALLOC(coord, ncoords);
    // Generate grid coordinates randomly
    for(i = 0; i < ncoords; i++)
        coord[i] = (double)rand()/RAND_MAX;
    size_t *pivot;
    STARSH_MALLOC(pivot, ndim);
    // Store coordinates inside a grid as a set of integers
    for(i = 0; i < ndim; i++)
        pivot[i] = 0;
    for(i = 0; i < count; i++)
    {
        // Get real values of grid coordinates in correspondance to location of
        // particle
        for(j = 0; j < ndim; j++)
            point[i+j*count] = coord[pivot[j]+j*side];
        j = ndim-1;
        // Get next location
        pivot[j]++;
        while(pivot[j] == side)
        {
            pivot[j] = 0;
            if(j > 0)
            {
                j--;
                pivot[j]++;
            }
        }
    }
    free(pivot);
    free(coord);
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->ndim = ndim;
    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}

/* Need to implement randn function for normal distribution
int starsh_particles_generate_randngrid(STARSH_particles **data, size_t count,
        int ndim)
{
    size_t i;
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->ndim = ndim;
    double *point;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}
i*/

int starsh_particles_generate_uniform(STARSH_particles **data, size_t count,
        int ndim)
//! Generate a uniform grid of particles
/*! @ingroup applications
 * @param[out] data: pointer to pointer to STARSH_particles to generate
 * @param[in] count: amount of particles to generate
 * @param[in] ndim: dimensionality of space
 *
 * Minimal grid, containing all of `count` particles, is selected
 * */
{
    size_t i, j, side, total;
    double *point, *ptr, val;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    ptr = point;
    side = ceil(pow(count, 1.0/ndim));
    total = side;
    for(i = 1; i < ndim; i++)
        total *= side;
    if(total < count)
    {
        fprintf(stderr, "Inner error of %s\n", __func__);
        free(point);
        return STARSH_UNKNOWN_ERROR;
    }
    double *coord;
    size_t ncoords = ndim*side;
    STARSH_MALLOC(coord, ncoords);
    // Generate grid coordinates uniformly
    for(i = 0; i < side; i++)
    {
        val = (double)i/side;
        for(j = 0; j < ndim; j++)
            coord[j*side+i] = val;
    }
    size_t *pivot;
    STARSH_MALLOC(pivot, ndim);
    // Store coordinates inside a grid as a set of integers
    for(i = 0; i < ndim; i++)
        pivot[i] = 0;
    for(i = 0; i < count; i++)
    {
        // Get real values of grid coordinates in correspondance to location of
        // particle
        for(j = 0; j < ndim; j++)
            point[i+j*count] = coord[pivot[j]+j*side];
        j = ndim-1;
        // Get next location
        pivot[j]++;
        while(pivot[j] == side)
        {
            pivot[j] = 0;
            if(j > 0)
            {
                j--;
                pivot[j]++;
            }
        }
    }
    free(pivot);
    free(coord);
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->ndim = ndim;
    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}

int starsh_particles_generate_quasiuniform1(STARSH_particles **data,
        size_t count, int ndim)
//! Generate a uniform grid of particles with random shift of each particle
/*! @ingroup applications
 * @param[out] data: pointer to pointer to STARSH_particles to generate
 * @param[in] count: amount of particles to generate
 * @param[in] ndim: dimensionality of space
 *
 * Minimal grid, containing all of `count` particles, is selected
 * */
{
    size_t i, j, side, total;
    double *point, *ptr, val;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    ptr = point;
    side = ceil(pow(count, 1.0/ndim));
    total = pow(side, ndim);
    if(total < count)
    {
        fprintf(stderr, "Inner error of %s\n", __func__);
        free(point);
        return STARSH_UNKNOWN_ERROR;
    }
    double *coord;
    size_t ncoords = ndim*side;
    STARSH_MALLOC(coord, ncoords);
    // Generate grid coordinates uniformly
    for(i = 0; i < side; i++)
    {
        val = (double)i/side;
        for(j = 0; j < ndim; j++)
        coord[j*side+i] = val;
    }
    size_t *pivot;
    STARSH_MALLOC(pivot, ndim);
    // Store coordinates inside a grid as a set of integers
    for(i = 0; i < ndim; i++)
        pivot[i] = 0;
    double mult = 0.8/side/RAND_MAX;
    double add = 0.1/side;
    for(i = 0; i < count; i++)
    {
        // Get real values of grid coordinates in correspondance to location of
        // particle and apply small random shift
        for(j = 0; j < ndim; j++)
            point[i+j*count] = coord[pivot[j]+j*side]+add+mult*rand();
        j = ndim-1;
        // Get next location
        pivot[j]++;
        while(pivot[j] == side)
        {
            pivot[j] = 0;
            if(j > 0)
            {
                j--;
                pivot[j]++;
            }
        }
    }
    free(pivot);
    free(coord);
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->ndim = ndim;
    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}

int starsh_particles_generate_quasiuniform2(STARSH_particles **data,
        size_t count, int ndim)
//! Generate a uniform grid of particles with random shift of grid coordinates
/*! @ingroup applications
 * @param[out] data: pointer to pointer to STARSH_particles to generate
 * @param[in] count: amount of particles to generate
 * @param[in] ndim: dimensionality of space
 *
 * Minimal grid, containing all of `count` particles, is selected
 * */
{
    size_t i, j, side, total;
    double *point, *ptr, val;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    ptr = point;
    side = ceil(pow(count, 1.0/ndim));
    total = pow(side, ndim);
    if(total < count)
    {
        fprintf(stderr, "Inner error of %s\n", __func__);
        free(point);
        return STARSH_UNKNOWN_ERROR;
    }
    double *coord;
    size_t ncoords = ndim*side;
    STARSH_MALLOC(coord, ncoords);
    // Generate grid coordinates uniformly
    double mult = 0.8/side/RAND_MAX;
    double add = 0.1/side;
    for(i = 0; i < side; i++)
    {
        val = (double)i/side;
        for(j = 0; j < ndim; j++)
        coord[j*side+i] = val+add+mult*rand();
    }
    size_t *pivot;
    STARSH_MALLOC(pivot, ndim);
    // Store coordinates inside a grid as a set of integers
    for(i = 0; i < ndim; i++)
        pivot[i] = 0;
    for(i = 0; i < count; i++)
    {
        // Get real values of grid coordinates in correspondance to location of
        // particle and apply small random shift
        for(j = 0; j < ndim; j++)
            point[i+j*count] = coord[pivot[j]+j*side];
        j = ndim-1;
        // Get next location
        pivot[j]++;
        while(pivot[j] == side)
        {
            pivot[j] = 0;
            if(j > 0)
            {
                j--;
                pivot[j]++;
            }
        }
    }
    free(pivot);
    free(coord);
    STARSH_MALLOC(*data, 1);
    (*data)->count = count;
    (*data)->ndim = ndim;
    (*data)->point = point;
    starsh_particles_zsort_inplace(*data);
    return STARSH_SUCCESS;
}

static void zsort(int n, double *points);
static void zsort3(int n, double *points);

int starsh_particles_generate_obsolete1(STARSH_particles **data, size_t count,
        int ndim)
//! Generate a uniform grid of particles with random shift of each particle
/*! @ingroup applications
 * @param[out] data: pointer to pointer to STARSH_particles to generate
 * @param[in] count: amount of particles to generate
 * @param[in] ndim: dimensionality of space
 *
 * Similar to starsh_particles_generate_quasiuniform1(), but works only for 1D,
 * 2D and 3D. Parameter `count` must be square of integer if `ndim`=2 and cube
 * if integer if `ndim`=3.
 * */
{
    size_t i, j, k;
    double *point;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    if(ndim == 1)
    {
        for(i = 0; i < count; i++)
            point[i] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/count;
    }
    else if(ndim == 2)
    {
        size_t sqrtn = floor(sqrt(count)+0.1);
        if(sqrtn*sqrtn != count)
        {
            STARSH_ERROR("parameter `count` must be square of some integer");
            return STARSH_WRONG_PARAMETER;
        }
        double *x = point, *y = x+count;
        for(i = 0; i < sqrtn; i++)
            for(j = 0; j < sqrtn; j++)
            {
                size_t ind = i*sqrtn + j;
                x[ind] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/sqrtn;
                y[ind] = (j+0.1+0.8*rand()/(1.0+RAND_MAX))/sqrtn;
            }
        zsort(count, point);
    }
    else
    {
        size_t cbrtn = floor(cbrt(count)+0.1); 
        if(cbrtn*cbrtn*cbrtn != count)
        {
            STARSH_ERROR("parameter `count` must be cube of some integer");
            return STARSH_WRONG_PARAMETER;
        }
        double *x = point, *y = x+count, *z = y+count;
        for(i = 0; i < cbrtn; i++)
            for(j = 0; j < cbrtn; j++)
                for(k = 0; k < cbrtn; k++)
                {
                    size_t ind = (i*cbrtn + j)*cbrtn + k;
                    x[ind] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/cbrtn;
                    y[ind] = (j+0.1+0.8*rand()/(1.0+RAND_MAX))/cbrtn;
                    z[ind] = (k+0.1+0.8*rand()/(1.0+RAND_MAX))/cbrtn;
                }
        zsort3(count, point);
    }
    STARSH_MALLOC(*data, 1);
    (*data)->point = point;
    (*data)->count = count;
    (*data)->ndim = ndim;
    return STARSH_SUCCESS;
}

int starsh_particles_generate_obsolete2(STARSH_particles **data, size_t count,
        int ndim)
//! Generate a uniform grid of particles with random shift of grid coordinates
/*! @ingroup applications
 * @param[out] data: pointer to pointer to STARSH_particles to generate
 * @param[in] count: amount of particles to generate
 * @param[in] ndim: dimensionality of space
 *
 * Similar to starsh_particles_generate_quasiuniform2(), but works only for 1D,
 * 2D and 3D. Parameter `count` must be square of integer if `ndim`=2 and cube
 * if integer if `ndim`=3.
 * */
{
    size_t i, j, k;
    double *point;
    size_t nelem = count*ndim;
    STARSH_MALLOC(point, nelem);
    if(ndim == 1)
    {
        for(i = 0; i < count; i++)
            point[i] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/count;
    }
    else if(ndim == 2)
    {
        size_t sqrtn = floor(sqrt(count)+0.1);
        if(sqrtn*sqrtn != count)
        {
            STARSH_ERROR("parameter `count` must be square of some integer");
            return STARSH_WRONG_PARAMETER;
        }
        double *x = point, *y = x+count;
        for(i = 0; i < sqrtn; i++)
            x[i] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/sqrtn;
        for(i = 0; i < sqrtn; i++)
            y[i*sqrtn] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/sqrtn;
        for(i = 0; i < sqrtn; i++)
            for(j = 0; j < sqrtn; j++)
            {
                size_t ind = i*sqrtn + j;
                x[ind] = x[j];
                y[ind] = y[i*sqrtn];
            }
        zsort(count, point);
    }
    else
    {
        size_t cbrtn = floor(cbrt(count)+0.1); 
        if(cbrtn*cbrtn*cbrtn != count)
        {
            STARSH_ERROR("parameter `count` must be cube of some integer");
            return STARSH_WRONG_PARAMETER;
        }
        double *x = point, *y = x+count, *z = y+count;
        for(i = 0; i < cbrtn; i++)
            x[i] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/cbrtn;
        for(i = 0; i < cbrtn; i++)
            y[i*cbrtn] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/cbrtn;
        for(i = 0; i < cbrtn; i++)
            z[i*cbrtn*cbrtn] = (i+0.1+0.8*rand()/(1.0+RAND_MAX))/cbrtn;
        for(i = 0; i < cbrtn; i++)
            for(j = 0; j < cbrtn; j++)
                for(k = 0; k < cbrtn; k++)
                {
                    size_t ind = (i*cbrtn + j)*cbrtn + k;
                    x[ind] = x[k];
                    y[ind] = y[j*cbrtn];
                    z[ind] = z[i*cbrtn*cbrtn];
                }
        zsort3(count, point);
    }
    STARSH_MALLOC(*data, 1);
    (*data)->point = point;
    (*data)->count = count;
    (*data)->ndim = ndim;
    return 0;
}

int starsh_particles_read_from_file(STARSH_particles **data, const char *fname,
        const enum STARSH_FILE_TYPE ftype)
//! Read configuration of STARSH_particles from file
/*! @ingroup applications
 * @param[out] data: Pointer to pointer to STARSH_particles to read from file
 * @param[in] fname: Name of file to read from
 * @param[in] ftype: Choose file type: ASCII or binary. Possible values are
 *  STARSH_ASCII and STARSH_BINARY
 * */
{
    FILE *fp;
    int info;
    switch(ftype)
    {
        case STARSH_ASCII:
            fp = fopen(fname, "r");
            if(fp == NULL)
                return STARSH_FILE_NOT_EXIST;
            info = starsh_particles_read_from_file_pointer_ascii(data, fp);
            break;
        case STARSH_BINARY:
            fp = fopen(fname, "rb");
            if(fp == NULL)
                return STARSH_FILE_NOT_EXIST;
            info = starsh_particles_read_from_file_pointer_binary(data, fp);
            break;
        default:
            return STARSH_WRONG_PARAMETER;
    }
    fclose(fp);
    return info;
}

int starsh_particles_read_from_file_pointer(STARSH_particles **data,
        FILE *fp, const enum STARSH_FILE_TYPE ftype)
//! Read configuration of STARSH_particles from file pointer
/*! @ingroup applications
 * @param[out] data: Pointer to pointer to STARSH_particles to read from file
 * @param[in] fp: File pointer to read from, must be open before
 * @param[in] ftype: Choose file type: ASCII or binary. Possible values are
 *  STARSH_ASCII and STARSH_BINARY
 *
 * After finishing, file pointer `fp` will still be open
 * */
{
    switch(ftype)
    {
        case STARSH_ASCII:
            return starsh_particles_read_from_file_pointer_ascii(data, fp);
        case STARSH_BINARY:
            return starsh_particles_read_from_file_pointer_binary(data, fp);
        default:
            return STARSH_WRONG_PARAMETER;
    }
}

int starsh_particles_read_from_file_pointer_ascii(STARSH_particles **data,
        FILE *fp)
//! Read configuration of STARSH_particles from file pointer in ASCII format
/*! @ingroup applications
 * @param[out] data: Pointer to pointer to STARSH_particles to read from file
 * @param[in] fp: File pointer to read from, must be open before
 *
 * After finishing, file pointer `fp` will still be open
 * */
{
    int info, ndim;
    size_t count, i, j;
    double *point, *ptr1, *ptr2;
    info = fscanf(fp, "count=%zu, ndim=%d", &count, &ndim);
    if(info != 2)
        return STARSH_FILE_WRONG_INPUT;
    STARSH_MALLOC(*data, 1);
    STARSH_MALLOC(point, count*ndim);
    (*data)->point = point;
    (*data)->count = count;
    (*data)->ndim = ndim;
    ptr1 = point;
    ptr2 = point+count;
    for(i = 0; i < count; i++)
    {
        info = fscanf(fp, "%lf", ptr1);
        if(info != 1)
            break;
        for(j = 0; j < ndim-1; j++)
        {
            info = fscanf(fp, ",%lf", ptr2+j*count);
            if(info != 1)
                break;
        }
        if(info != 1)
            break;
        ptr1++;
        ptr2++;
    }
    if(info != 1)
    {
        free(*data);
        *data = NULL;
        free(point);
        return STARSH_FILE_WRONG_INPUT;
    }
    return STARSH_SUCCESS;
}

int starsh_particles_read_from_file_pointer_binary(STARSH_particles **data,
        FILE *fp)
//! Read configuration of STARSH_particles from file pointer in binary format
/*! @ingroup applications
 * @param[out] data: Pointer to pointer to STARSH_particles to read from file
 * @param[in] fp: File pointer to read from, must be open before
 *
 * After finishing, file pointer `fp` will still be open
 * */
{
    int ndim;
    size_t count, nmemb, info;
    double *point;
    info = fread(&count, sizeof(count), 1, fp);
    if(info != 1)
        return STARSH_FILE_WRONG_INPUT;
    info = fread(&ndim, sizeof(ndim), 1, fp);
    if(info != 1)
        return STARSH_FILE_WRONG_INPUT;
    STARSH_MALLOC(*data, 1);
    STARSH_MALLOC(point, count*ndim);
    (*data)->point = point;
    (*data)->count = count;
    (*data)->ndim = ndim;
    nmemb = count*ndim;
    info = fread(point, sizeof(*point), nmemb, fp);
    if(info != nmemb)
    {
        free(*data);
        *data = NULL;
        free(point);
        return STARSH_FILE_WRONG_INPUT;
    }
    return STARSH_SUCCESS;
}

int starsh_particles_write_to_file(const STARSH_particles *data,
        const char *fname, const enum STARSH_FILE_TYPE ftype)
//! Write configuration of STARSH_particles to file
/*! @ingroup applications
 * @param[out] data: Pointer to data to write to file
 * @param[in] fname: Name of file to write to
 * @param[in] ftype: Choose file type: ASCII or binary. Possible values are
 *  STARSH_ASCII and STARSH_BINARY
 * */
{
    FILE *fp;
    int info;
    switch(ftype)
    {
        case STARSH_ASCII:
            fp = fopen(fname, "w");
            if(fp == NULL)
                return STARSH_FILE_NOT_EXIST;
            info = starsh_particles_write_to_file_pointer_ascii(data, fp);
            break;
        case STARSH_BINARY:
            fp = fopen(fname, "wb");
            if(fp == NULL)
                return STARSH_FILE_NOT_EXIST;
            info = starsh_particles_write_to_file_pointer_binary(data, fp);
            break;
        default:
            return STARSH_WRONG_PARAMETER;
    }
    fclose(fp);
    return info;
}

int starsh_particles_write_to_file_pointer(const STARSH_particles *data,
        FILE *fp, const enum STARSH_FILE_TYPE ftype)
//! Write configuration of STARSH_particles to file pointer
/*! @ingroup applications
 * @param[out] data: Pointer to data to write to file
 * @param[in] fp: File pointer to write to, must be open before
 * @param[in] ftype: Choose file type: ASCII or binary. Possible values are
 *  STARSH_ASCII and STARSH_BINARY
 *
 * After finishing, file pointer `fp` will still be open
 * */
{
    switch(ftype)
    {
        case STARSH_ASCII:
            return starsh_particles_write_to_file_pointer_ascii(data, fp);
        case STARSH_BINARY:
            return starsh_particles_write_to_file_pointer_binary(data, fp);
        default:
            return STARSH_WRONG_PARAMETER;
    }
}

int starsh_particles_write_to_file_pointer_ascii(const STARSH_particles *data,
        FILE *fp)
//! Write configuration of STARSH_particles to file pointer in ASCII format
/*! @ingroup applications
 * @param[out] data: Pointer to data to write to file
 * @param[in] fp: File pointer to write to, must be open before
 *
 * After finishing, file pointer `fp` will still be open
 * */
{
    int info, ndim = data->ndim;
    size_t count = data->count, i, j;
    double *point = data->point, *ptr1, *ptr2;
    info = fprintf(fp, "count=%zu, ndim=%d\n", count, ndim);
    if(info < 0)
        return STARSH_FPRINTF_ERROR;
    ptr1 = point;
    ptr2 = point+count;
    for(i = 0; i < count; i++)
    {
        info = fprintf(fp, "%lf", *ptr1);
        if(info < 0)
            break;
        for(j = 0; j < ndim-1; j++)
        {
            info = fprintf(fp, ", %lf", ptr2[j*count]);
            if(info < 0)
                break;
        }
        if(info < 0)
            break;
        info = fprintf(fp, "\n");
        if(info < 0)
            break;
        ptr1++;
        ptr2++;
    }
    if(info < 0)
        return STARSH_FPRINTF_ERROR;
    return STARSH_SUCCESS;
}

int starsh_particles_write_to_file_pointer_binary(const STARSH_particles *data,
        FILE *fp)
//! Write configuration of STARSH_particles to file pointer in binary format
/*! @ingroup applications
 * @param[out] data: Pointer to data to write to file
 * @param[in] fp: File pointer to write to, must be open before
 *
 * After finishing, file pointer `fp` will still be open
 * */
{
    int ndim = data->ndim;
    size_t count = data->count, nmemb, info;
    double *point = data->point;
    info = fwrite(&count, sizeof(count), 1, fp);
    if(info != 1)
        return STARSH_FWRITE_ERROR;
    info = fwrite(&ndim, sizeof(ndim), 1, fp);
    if(info != 1)
        return STARSH_FWRITE_ERROR;
    nmemb = count*ndim;
    info = fwrite(point, sizeof(*point), nmemb, fp);
    if(info != nmemb)
        return STARSH_FWRITE_ERROR;
    return STARSH_SUCCESS;
}

/*
static int qsort_dimension(double *data, size_t count, int ndim, int sdim,
        size_t lo, size_t hi)
//! Inplace sort of particles by given dimension
/*! @param[in] data: Pointer to array of coordinates of particles
 * @param[in] count: Amount of particles
 * @param[in] ndim: Dimensionality of space
 * @param[in] sdim: Sorting dimension
 * @param[in] lo: First element to sort
 * @param[in] hi: Last element to sort
 * /
{
    size_t i, j;
    int k;
    double *sort_data = data+count*sdim;
    if(lo < hi)
    {
        pivot = sort_data[lo];
        while(1)
        {
            while(sort_data[i] < pivot)
                i++;
            i--;
            while(sort_data[j] > pivot)
                j--;
            j++;
            if(i >= j)
            {
                p = ;
                break;
            }
            for(
        }
    }
    return STARSH_SUCCESS;
}
*/

int starsh_particles_zsort_inplace(STARSH_particles *data)
//! Sort particles in Z-order (Morton order)
{
    int i;
    size_t j, new_j, tmp_j;
    size_t count = data->count;
    int ndim = data->ndim;
    int info;
    double *point = data->point;
    double *ptr1;
    double *minmax; // min is stored in lower part, max is stored in upper part
    double tmp_x;
    STARSH_MALLOC(minmax, count*2);
    for(i = 0; i < ndim; i++)
    {
        ptr1 = point+i*count; // i-th dimension
        minmax[i] = ptr1[0];
        minmax[i+ndim] = minmax[i];
        for(j = 1; j < count; j++)
        {
            if(minmax[i] > ptr1[j])
                minmax[i] = ptr1[j];
            else if (minmax[i+ndim] < ptr1[j])
                minmax[i+ndim] = ptr1[j];
        }
    }
    // Now minmax[0:ndim] and minmax[ndim:2*ndim] store minimal and maximal
    // values of coordinates
    uint32_t *uint_point;
    STARSH_MALLOC(uint_point, count*ndim);
    uint32_t *uint_ptr1;
    double min, range;
    for(i = 0; i < ndim; i++)
    {
        uint_ptr1 = uint_point+i*count;
        ptr1 = point+i*count;
        min = minmax[i];
        range = minmax[i+ndim]-min;
        for(j = 0; j < count; j++)
            uint_ptr1[j] = (ptr1[j]-min)/range*UINT32_MAX;
    }
    free(minmax);
    // Now uint_ptr1 contains initial coordinates, rescaled to range
    // [0, UINT32_MAX] and converted to uint32_t type to use special radix sort
    // Prepare indexes to store sort order
    size_t *order;
    STARSH_MALLOC(order, count);
    for(j = 0; j < count; j++)
        order[j] = j;
    info = radix_sort(uint_point, count, ndim, order);
    if(info != STARSH_SUCCESS)
    {
        free(uint_point);
        free(order);
        return info;
    }
    double *new_point;
    STARSH_MALLOC(new_point, count*ndim);
    for(j = 0; j < count; j++)
    {
        for(i = 0; i < ndim; i++)
            new_point[count*i+j] = point[count*i+order[j]];
    }
    data->point = new_point;
    free(point);
    free(uint_point);
    free(order);
    return STARSH_SUCCESS;
}

static int radix_sort(uint32_t *data, size_t count, int ndim, size_t *order)
//! Auxiliary sorting function for Z-sort
{
    size_t *tmp_order;
    STARSH_MALLOC(tmp_order, count);
    radix_sort_recursive(data, count, ndim, order, tmp_order, ndim-1, 31, 0,
            count-1);
    free(tmp_order);
    return STARSH_SUCCESS;
}

static void radix_sort_recursive(uint32_t *data, size_t count, int ndim,
        size_t *order, size_t *tmp_order, int sdim, int sbit,
        size_t lo, size_t hi)
//! Hierarchical radix sort to get Z-order of particles
{
    size_t i, lo_last = lo, hi_last = hi;
    uint32_t *sdata = data+sdim*count;
    uint32_t check = 1 << sbit;
    for(i = lo; i <= hi; i++)
    {
        if((sdata[order[i]] & check) == 0)
        {
            tmp_order[lo_last] = order[i];
            lo_last++;
        }
        else
        {
            tmp_order[hi_last] = order[i];
            hi_last--;
        }
    }
    for(i = lo; i <= hi; i++)
        order[i] = tmp_order[i];
    if(sdim > 0)
    {
        if(lo_last-lo > 1)
            radix_sort_recursive(data, count, ndim, order, tmp_order, sdim-1,
                    sbit, lo, lo_last-1);
        if(hi-hi_last > 1)
            radix_sort_recursive(data, count, ndim, order, tmp_order, sdim-1,
                    sbit, hi_last+1, hi);
    }
    else if(sbit > 0)
    {
        if(lo_last-lo > 1)
            radix_sort_recursive(data, count, ndim, order, tmp_order, ndim-1,
                    sbit-1, lo, lo_last-1);
        if(hi-hi_last > 1)
            radix_sort_recursive(data, count, ndim, order, tmp_order, ndim-1,
                    sbit-1, hi_last+1, hi);
    }
}


/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
// THIS CODE HERE ONLY FOR COMPATIBILITY WITH OLD CODE, IT WILL BE REMOVED AT
// CERTAIN POINT



static uint32_t Part1By1(uint32_t x)
//! Spread lower bits of input
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
//! Collect every second bit into lower part of input
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
//! Spread lower bits of input
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
//! Collect every 4-th bit into lower part of input
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
//! Encode two inputs into one
{
    return (Part1By1(y) << 1) + Part1By1(x);
}

static uint64_t EncodeMorton3(uint64_t x, uint64_t y, uint64_t z)
//! Encode 3 inputs into one
{
    return (Part1By3(z) << 2) + (Part1By3(y) << 1) + Part1By3(x);
}

static uint32_t DecodeMorton2X(uint32_t code)
//! Decode first input
{
    return Compact1By1(code >> 0);
}

static uint32_t DecodeMorton2Y(uint32_t code)
//! Decode second input
{
    return Compact1By1(code >> 1);
}

static uint64_t DecodeMorton3X(uint64_t code)
//! Decode first input
{
    return Compact1By3(code >> 0);
}

static uint64_t DecodeMorton3Y(uint64_t code)
//! Decode second input
{
    return Compact1By3(code >> 1);
}

static uint64_t DecodeMorton3Z(uint64_t code)
//! Decode third input
{
    return Compact1By3(code >> 2);
}

static int compare_uint32(const void *a, const void *b)
//! Compare two uint32_t
{
    uint32_t _a = *(uint32_t *)a;
    uint32_t _b = *(uint32_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static int compare_uint64(const void *a, const void *b)
//! Compare two uint64_t
{
    uint64_t _a = *(uint64_t *)a;
    uint64_t _b = *(uint64_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static void zsort(int n, double *points)
//! Sort in Morton order (input points must be in [0;1]x[0;1] square])
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y;
    uint32_t z[n];
    // Encode data into vector z
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(points[i]*(double)UINT16_MAX +.5);
        y = (uint16_t)(points[i+n]*(double)UINT16_MAX +.5);
        //printf("%f %f -> %u %u\n", points[i], points[i+n], x, y);
        z[i] = EncodeMorton2(x, y);
    }
    // Sort vector z
    qsort(z, n, sizeof(uint32_t), compare_uint32);
    // Decode data from vector z
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
//! Sort in Morton order for 3D
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y, z;
    uint64_t Z[n];
    // Encode data into vector Z
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(points[i]*(double)UINT16_MAX + 0.5);
        y = (uint16_t)(points[i+n]*(double)UINT16_MAX + 0.5);
        z = (uint16_t)(points[i+2*n]*(double)UINT16_MAX + 0.5);
        Z[i] = EncodeMorton3(x, y, z);
    }
    // Sort Z
    qsort(Z, n, sizeof(uint64_t), compare_uint64);
    // Decode data from vector Z
    for(i = 0; i < n; i++)
    {
        points[i] = (double)DecodeMorton3X(Z[i])/(double)UINT16_MAX;
        points[i+n] = (double)DecodeMorton3Y(Z[i])/(double)UINT16_MAX;
        points[i+2*n] = (double)DecodeMorton3Z(Z[i])/(double)UINT16_MAX;
    }
}
