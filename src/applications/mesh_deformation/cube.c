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
 * @author Rabab Alomairy
 * @date 2020-05-09
 */

#include "common.h"
#include "starsh.h"
#include "starsh-rbf.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>



static double Gaussian(double x)
{
        return exp(-pow(x, 2));
}

static double Expon(double x)
{       
        return exp(-x);
}

static double Maternc1(double x)
{
        return exp(-x)+(1+x);
}

static double Maternc2(double x)
{
        return exp(-x)+(3+3*x+pow(x,2));
}

static double QUAD(double x)
{
        return 1 + pow(x, 2);
}

static double InvQUAD(double x)
{
	return 1 / (1 + pow(x, 2));
}

static double InvMQUAD(double x)
{
        return 1 / sqrt(1 + pow(x, 2));
}

static double TPS(double x)
{
        return pow(x, 2) * log(x);
}

static double Wendland(double x)
{
        if (x > 1)
                return 0;
        else
                return pow(1 - x, 4)*(4 * x + 1);
}

static double CTPS(double x)
{
        if (x > 1)
                return 0;
        else
				return pow(1 - x, 5);
}

static double diff(double*x, double*y)
{
        double r = 0;
        for (int i = 0; i < 3; i++)
                r = r + pow(x[i] - y[i], 2);
        return pow(r, 0.5);
}

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
void starsh_generate_3d_cube(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int lda)
{
        double vi[3];
        double vj[3];
        STARSH_mddata *data = row_data;
        int i0, j0;
        int n = floor(1 + pow(1 - (8 - data->mesh_points) / (double)6, 0.5)) + 1;
        int nb = (int)(6 * pow(n, 2) - 12 * n + 8);
        double L = 0.5*n; // 0.5 : minimal distance between two neighboring mesh points
        int m, k;
        double *A = (double *) result;
        for(m=0;m<nrows;m++){
                i0 = irow*nrows+m;
                cube(vi, i0, L, n);
                for(k=0;k<ncols;k++){
                        j0 = icol*ncols+k;
                        cube(vj, j0, L, n);
                        double d = diff(vi, vj) / (double)rad;
                       //A[lda*k+m]=Gaussian3(d);
                        switch(kernel){
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
                           default: A[lda*k+m]=Wendland(d);
                                   break;
                        }
                        if( i0==j0 && (data->isreg)) A[lda*k+m] + = (data->reg);
                }
        }

}

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
