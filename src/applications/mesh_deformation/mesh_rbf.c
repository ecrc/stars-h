/*! @copyright (c) 2020 King Abdullah University of Science and
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
 * @file src/applications/mesh_deformation/mesh_rbf.c
 * @version 0.1.1
 * @author Rabab Alomairy
 * @author Aleksandr Mikhalev
 * @date 2020-05-09 
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
	// Some sorting, required by mesh deformation
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

/*! It reads mesh pointd fron file
 *
 * @param[inout] data: STARSH_mddata mesh deformation 
 * @param[in] file_name: path to mesh file
 * @param[in] mesh points: number of mesh points
 * @param[in] ndim: problem dimension.
 * @param[in] kernel_type: kernel (0:).
 * @param[in] numobj: how many objects (e.g. number of viurese)
 * @param[in] isreg:  it is either 0 or 1 if you want to add regularizer
 * @param[in] reg:  regularization value
 * @param[in] rad: RBF scaling factor 
 * @param[in] denst: density scaling factor  
 * @param[in] mordering: 0: no ordering, 1: Morton ordering.
 * */
int starsh_generate_3d_rbf_mesh_coordinates_virus(STARSH_mddata **data, char *file_name, STARSH_int mesh_points, int ndim, int kernel, 
		int numobj, int isreg, double reg, double rad, double denst, int mordering){


	STARSH_particles *particles;
	STARSH_MALLOC(particles, 1);
	(particles)->count = mesh_points;
	(particles)->ndim = ndim;

	double *mesh;
	size_t nelem = mesh_points*ndim;
	STARSH_MALLOC(mesh, nelem);

	FILE *p_file = fopen(file_name,"r");
	char line[100];
	int i=0, j=0;
	if(!p_file)
	{
		printf("\n File missing or error when reading file:");
		return 0;
	}

	while(fgets(line,100,p_file) != NULL && j < mesh_points)
	{
		char *p = strtok(line, ",");
		while(p)
		{
			mesh[i]=atof(p);
			p=strtok(NULL, ",");
			i++;
		}
          j++;
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
        (*data)->denst = denst;
       
        return STARSH_SUCCESS;

}


/*! It reads mesh pointd fron file
 *
 * @param[inout] data: STARSH_mddata mesh deformation 
 * @param[in] mesh points: number of mesh points
 * @param[in] ndim: problem dimension.
 * @param[in] kernel_type: kernel (0:).
 * @param[in] isreg:  it is either 0 or 1 if you want to add regularizer
 * @param[in] reg:  regularization value
 * @param[in] rad: RBF scaling factor 
 * @param[in] mordering: 0: no ordering, 1: Morton ordering.
 * */
int starsh_generate_3d_rbf_mesh_coordinates_cube(STARSH_mddata **data, STARSH_int mesh_points, int ndim, int kernel, 
	 int isreg, double reg, double rad, int mordering){


	STARSH_particles *particles;
	STARSH_MALLOC(particles, 1);
	(particles)->count = mesh_points;
	(particles)->ndim = ndim;

	double *mesh;
	size_t nelem = mesh_points*ndim;
	STARSH_MALLOC(mesh, nelem);

	int i;


	int n = floor(1 + sqrt(1 - (8 - mesh_points) / (double)6)) + 1;
	int nb = (int)(6 * pow(n, 2) - 12 * n + 8);
	double L = 0.5*n; // 0.5 : minimal distance between two neighboring mesh points

	for (i=0; i<mesh_points; i++)
	{
		cube(&(mesh[i*3]), i, L, n);
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
	(*data)->numobj = 1; //For example number of viruses within population
	(*data)->mordering = mordering;
	(*data)->kernel = kernel;
	(*data)->rad = rad;
        (*data)->denst = -1;
       
        return STARSH_SUCCESS;

}

void starsh_mddata_free(STARSH_mddata *data)
	//! Free memory of @ref STARSH_mddata object.
	/*! @sa starsh_mddata_new(), starsh_mddata_init(), starsh_mddata_generate().
	 * @ingroup app-spatial
	 * */
{
	starsh_particles_free(&data->particles);
}

