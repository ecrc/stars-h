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
 * @file src/applications/mesh_deformation/cube.c
 * @version 0.1.2
 * @author Rabab Alomairy
 * @date 2020-06-09
 */

#include "common.h"
#include "starsh.h"
#include "starsh-rbf.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>



/*! Computing cube Coordinates
 * @param[inout] v: Mesh Coordinates
 * @param[in] index: element position
 * @param[in] L: edge length of cube
 * @param[in] n:  overall size of cube //TODO ask application pp
 */

static void cube(double* v, int index, double L, int n)
{
	double step = 1 / (double)(n - 1);
	double x = -1;
	double y = -1;
	double z = -1;

	if (index < 2 * pow(n, 2))
	{
		z = index / (int)pow(n, 2);
		int ind = index - z*pow(n, 2);

		x = (ind / n)* step;
		y = (ind % n)* step;
	}
	else if ((index >= 2 * pow(n, 2)) && (index < 4 * (pow(n, 2) - n)))
	{
		int ind = index - 2 * pow(n, 2);
		x = ind / (int)(pow(n, 2) - 2 * n);
		ind =(int)(ind - x*(pow(n, 2) - 2 * n));

		y = (ind % n)* step;
		z = ((ind / n) + 1)* step;
	}
	else if ((index >= 4 * (pow(n, 2) - n)) && (index < 6 * pow(n, 2) - 12 * n + 8))
	{
		int ind = index - 4 * (pow(n, 2) - n);
		y = ind / (int)(pow(n, 2) - 4 * n + 4);
		ind =(int)(ind - y*(pow(n, 2) - 4 * n + 4));

		x = ((ind / (n - 2)) + 1)* step;
		z = ((ind % (n - 2)) + 1)* step;
	}
	v[0] = x*L;
	v[1] = y*L;
	v[2] = z*L;

}

/*! Fills matrix \f$ A \f$ with values
 * @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (\ref STARSH_mddata object).
 * @param[in] col_data: Pointer to physical data (\ref STARSH_mddata object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * */

void starsh_generate_3d_cube(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int lda)
{
	double vi[3];
	double vj[3];
	STARSH_mddata *data = row_data;
	STARSH_int i0, j0;
	int n = floor(1 + sqrt(1 - (8 - data->mesh_points) / (double)6)) + 1;
	int nb = (int)(6 * pow(n, 2) - 12 * n + 8);
	double L = 0.5*n; // 0.5 : minimal distance between two neighboring mesh points
	int m, k;
	double *A = (double *) result;
	for(m=0;m<nrows;m++){
		i0 = irow[m];
		cube(vi, i0, L, n);
		for(k=0;k<ncols;k++){
			j0 = icol[k];
			cube(vj, j0, L, n);
			double d = diff(vi, vj) / (double) data->rad;
			switch(data->kernel){
				case 0: A[lda*k+m]=Gaussian(d);
					break;
				case 1: A[lda*k+m]=Expon(d);
					break;
				case 2: A[lda*k+m]=InvQUAD(d);
					break;
				case 3: A[lda*k+m]=InvMQUAD(d);
					break;
				case 4: A[lda*k+m]=Maternc1(d);
					break;
				case 5: A[lda*k+m]=Maternc2(d);
					break;
				case 6: A[lda*k+m]=TPS(d);
					break;
				case 7: A[lda*k+m]=CTPS(d);
					break;
				case 8: A[lda*k+m]=QUAD(d);
					break;
				default: A[lda*k+m]=Wendland(d);
					 break;
			}
			if( i0==j0 && (data->isreg)) A[lda*k+m] += (data->reg);
		}
	}

}


/*! Fills matrix (RHS) \f$ A \f$ with values
 * @param[in] ld: Total number of mesh points.
 * @param[inout] ld: Pointer to memory of \f$ A \f$..    
 */
void starsh_generate_3d_cube_rhs(STARSH_int mesh_points, double *A)
{

	int i;

	for(i=0;i<mesh_points;i++)
		A[i] = 1;
	for(i=mesh_points;i<2*mesh_points;i++)
		A[i] = 0;
	for(i=2*mesh_points;i<3*mesh_points;i++)
		A[i] = 0;
}
