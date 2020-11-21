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
 * @file src/applications/acoustic/acoustic.c
 * @version 0.1.1
 * @author Rabab Alomairy
 * @date 2020-05-09
 */

#include "common.h"
#include "starsh.h"
#include "starsh-acoustic.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>


/*! Fills matrix \f$ A \f$ with values
 *
 * @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (\ref STARSH_acdata object).
 * @param[in] col_data: Pointer to physical data (\ref STARSH_acdata object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * */


void starsh_generate_3d_acoustic(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int lda)
{
	int m, k;
	STARSH_acdata *data = row_data;
        int local_nt = nrows/data->nipp;
	double _Complex *zz= (double _Complex *)result;
       
        int i = (int) (irow[0]/nrows);
        int j = (int) (icol[0]/nrows);
        int p=i*local_nt+1;
        int q=j*local_nt+1;

        acoustic_generate_kernel(&(data->nipp), &(data->train), zz, &p, &q, &local_nt, &nrows);

}

/*! Fills matrix (RHS) \f$ A \f$ with values
 * @param[in] train:  number of traingles
 * @param[in] nip:  number of quadrature points
 * @param[out] crhs: Pointer to memory of \f$ RHS \f$.
 * @param[in] m: Number of rows of \f$ RHS \f$.
 * @param[in] n: Number of columns of \f$ RHS \f$.
 * @param[in] local_nt: number of triangles interactions \f$ RHS \f$.
 * @param[in] nb: tile size \f$ RHS \f$ 
*/
void starsh_generate_acoustic_rhs(int nip, int ntrain, double _Complex *crhs, int m, int n, int local_nt, int nb)
{
  
      acoustic_generate_rhs(&nip, &ntrain, crhs, &m, &n, &local_nt, &nb);
}


/*! Fills matrix (RHS) \f$ A \f$ with near scattered values
 * @param[out] x: Pointer to memory of \f$ solution \f$.
 * @param[in] nipp:  number of quadrature points
 * @param[in] train:  number of traingles
 * @param[in] nip:  number of quadrature points
*/
void starsh_generate_acoustic_near_sca(double _Complex *x, int nip, int ntrian)
{
      acoustic_generate_near_sca(x, &nip, &ntrian);

}
