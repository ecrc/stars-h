/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * Generate different functions for different dimensions. This hack improves
 * performance in certain cases. Value 'n' stands for general case, whereas all
 * other values correspond to static values of dimensionality.
 * will be replace by proposed values. If you want to use this file outside
 * STARS-H, simply do substitutions yourself.
 *
 * @file src/applications/spatial/kernel_exp.c
 * @version 0.1.1
 * @author Aleksandr Mikhalev
 * @date 2018-11-06
 */

#include "common.h"
#include "starsh.h"
#include "starsh-rbf.h"
#include <inttypes.h>

static uint32_t Part1By1(uint32_t x)
// Spread lower bits of input
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
// Collect every second bit into lower part of input
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
// Spread lower bits of input
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
// Collect every 4-th bit into lower part of input
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
// Encode two inputs into one
{
    return (Part1By1(y) << 1) + Part1By1(x);
}

static uint64_t EncodeMorton3(uint64_t x, uint64_t y, uint64_t z)
// Encode 3 inputs into one
{
    return (Part1By3(z) << 2) + (Part1By3(y) << 1) + Part1By3(x);
}

static uint32_t DecodeMorton2X(uint32_t code)
// Decode first input
{
    return Compact1By1(code >> 0);
}

static uint32_t DecodeMorton2Y(uint32_t code)
// Decode second input
{
    return Compact1By1(code >> 1);
}

static uint64_t DecodeMorton3X(uint64_t code)
// Decode first input
{
    return Compact1By3(code >> 0);
}

static uint64_t DecodeMorton3Y(uint64_t code)
// Decode second input
{
    return Compact1By3(code >> 1);
}

static uint64_t DecodeMorton3Z(uint64_t code)
// Decode third input
{
    return Compact1By3(code >> 2);
}

static int compare_uint32(const void *a, const void *b)
// Compare two uint32_t
{
    uint32_t _a = *(uint32_t *)a;
    uint32_t _b = *(uint32_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static int compare_uint64(const void *a, const void *b)
// Compare two uint64_t
{
    uint64_t _a = *(uint64_t *)a;
    uint64_t _b = *(uint64_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static void starsh_morton_zsort(int n, double *points)
// Sort in Morton order (input points must be in [0;1]x[0;1] square])
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

static void starsh_morton_zsort3(int n, double *points)
// Sort in Morton order for 3D
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


void starsh_generate_3d_rbf_mesh_coordinates(STARSH_mddata **data, char *file_name, STARSH_int mesh_points, int ndim, int kernel, 
                          int numobj, int isreg, double reg, double rad, int mordering){

   
  //double *mesh = (double *) malloc
  STARSH_particles *particles;
  STARSH_MALLOC(particles, 1);
  (particles)->count = mesh_points;
  (particles)->ndim = ndim;

  double *mesh;
  size_t nelem = mesh_points*ndim;
  STARSH_MALLOC(mesh, nelem);

	FILE *p_file = fopen(file_name,"r");
	char line[100];
	int i=0;
	if(!p_file)
	{
		printf("\n File missing or error when reading file:");
		return 0;
	}

	while(fgets(line,100,p_file) != NULL)
	{
		char *p = strtok(line, ",");
		while(p)
		{
			mesh[i]=atof(p);
			p=strtok(NULL, ",");
			i++;
		}
               
	}
  
    if(mordering==1){
         starsh_morton_zsort3(mesh_points, mesh);
    }

    (particles)->point = mesh;
    (particles)->count = mesh_points;
    (particles)->ndim = ndim;    

    STARSH_MALLOC(*data, 1);
    (*data)->particles = *particles;
    free(particles);
    (*data)->reg = reg;
    (*data)->isreg = isreg;
    (*data)->numobj = numobj; //For example number of viruses within population
    (*data)->mordering = mordering;
    (*data)->kernel = kernel;
    (*data)->rad = rad;
}

