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
 * @file src/applications/mesh_deformation/virus.c
 * @version 0.1.1
 * @author Rabab Alomairy
 * @date 2020-05-09
 */

#include "common.h"
#include "starsh.h"
#include "starsh-rbf.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#define pi 3.14159265358979323846

/*! Fills matrix \f$ A \f$ with values
 *
 * @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (\ref STARSH_mddata object).
 * @param[in] col_data: Pointer to physical data (\ref STARSH_mddata object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * */


void starsh_generate_3d_virus(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int lda)
{
	int m, k;
	STARSH_mddata *data = row_data;
	double *mesh = data->particles.point;
	double rad = data->rad;

	if((data->numobj)>1 && (data->rad)<0 && (data->denst)<0){
                rad=0.25*(data->numobj)*sqrt(3); // For uniform dist
        }else if((data->numobj)>1 && (data->rad)<0 && (data->denst)>0){
           rad= (sqrt(3)) * (pow(((4 * pi * (0.09 * 0.09 * 0.09) * (data->numobj)/ 3) / (data->denst) ), (1/3)) + 0.18); // For sphere packing
        }

	double *A= (double *)result;

	for(m=0;m<nrows;m++){
		int i0=irow[m];
		int posi=i0*3;
		double vi[3] = {mesh[posi], mesh[posi+1], mesh[posi+2]};

		for(k=0;k<ncols;k++){
			int j0=icol[k];
			int posj=j0*3;
			double vj[3] = {mesh[posj], mesh[posj+1], mesh[posj+2]};
			double d = diff(vi, vj) / rad;
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
				case 5: A[lda*k+m]=Maternc1(d);
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
			if( i0==j0 && (data->isreg)) A[lda*k+m]+=(data->reg);
		}
	}

}

/*! Fills matrix (RHS) \f$ A \f$ with values
 * @param[in] ld: Total number of mesh points.
 * @param[inout] ld: Pointer to memory of \f$ A \f$..    
 */
void starsh_generate_3d_virus_rhs(STARSH_int mesh_points, double *A)
{
	int i;

	for(i=0;i<mesh_points;i++)
		A[i]=0.01;
	for(i=mesh_points;i<2*mesh_points;i++)
		A[i]=-0.019;
	for(i=2*mesh_points;i<3*mesh_points;i++)
		A[i]=0.021;
}
