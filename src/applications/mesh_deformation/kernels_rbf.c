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

/*! RBF Gaussian basis function
 * @param[in] x: Euclidean distance
 */
double Gaussian(double x)
{
        return exp(-pow(x, 2));
}

/*! RBF Exponential basis function
 * @param[in] x: Euclidean distance
 */
double Expon(double x)
{       
        return exp(-x);
}
/*! RBF Maternc1 basis function
 * @param[in] x: Euclidean distance
 */
double Maternc1(double x)
{
        return exp(-x)+(1+x);
}
/*! RBF Maternc2 basis function
 * @param[in] x: Euclidean distance
 */
double Maternc2(double x)
{
        return exp(-x)+(3+3*x+pow(x,2));
}

/*! RBF Quadratic basis function
 * @param[in] x: Euclidean distance
 */
double QUAD(double x)
{
        return 1 + pow(x, 2);
}

/*! RBF Inverse Quadratic basis function
 * @param[in] x: Euclidean distance
 */
double InvQUAD(double x)
{
	return 1 / (1 + pow(x, 2));
}

/*! RBF Inverse Multi-Quadratic basis function
 * @param[in] x: Euclidean distance
 */
double InvMQUAD(double x)
{
        return 1 / sqrt(1 + pow(x, 2));
}

/*! RBF Thin plate spline basis function
 * @param[in] x: Euclidean distance
 */
double TPS(double x)
{
        return pow(x, 2) * log(x);
}

/*! RBF Wendland basis function
 * @param[in] x: Euclidean distance
 */
double Wendland(double x)
{
        if (x > 1)
                return 0;
        else
                return pow(1 - x, 4)*(4 * x + 1);
}

/*! RBF Continuous Thin plate spline basis function
 * @param[in] x: Euclidean distance
 */
double CTPS(double x)
{
        if (x > 1)
                return 0;
        else
                return pow(1 - x, 5);
}

/*! Computing Euclidean distance
 * @param[in] x: Mesh Coordinates along x-axis
 * @param[in] y: Mesh Coordinates along y-axis
 */

double diff(double*x, double*y)
{
        double r = 0;
        for (int i = 0; i < 3; i++)
                r = r + pow(x[i] - y[i], 2);
        return pow(r, 0.5);
}
