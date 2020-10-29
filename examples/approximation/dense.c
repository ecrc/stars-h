/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/approximation/dense.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>

int main(int argc, char **argv)
{
    // Matrix size
    int N = 2500;
    int shape[2] = {N, N}; // shape of corresponding array
    char dtype = 'd'; // 'd' for double precision array
    char order = 'F'; // Fortran order (column-major)
    char symm = 'N'; // 'S' for symmetric problem, 'N' for nonsymmetric
    int info;
    int maxrank = 30; // Maximum rank to be used
    int oversample = 10; // Parameter for randomized SVD (extra random vectors)
    double tol = 1e-3; // Error threshold
    int onfly = 1; // Do not store dense blocks (since they are stored in data)
    // Size of tile
    int block_size = 250;
    // Allocate memory for dense matrix
    double *data = malloc(N*N*sizeof(*data));
    if(data == NULL)
    {
        printf("Error in allocation of memory for data buffer\n");
        exit(1);
    }
    // Iterate over columns
    for(int j = 0; j < N; j++)
        // Iterate over rows
        for(int i = 0; i < N; i++)
        {
            // Fill matrix in Fortran order (column-major)
            data[j*N+i] = 1.0;
            // Make this matrix diagonally dominant
            if(i == j)
                data[j*N+i] += N;
        }
    // Create structure array to cover double *data
    Array *array;
    info = array_from_buffer(&array, 2, shape, dtype, order, data);
    if(info != 0)
    {
        printf("Error in creation of Array structure\n");
        exit(info);
    }
    // Init STARS-H
    starsh_init();
    // Set up problem
    STARSH_problem *problem;
    info = starsh_problem_from_array(&problem, array, symm);
    if(info != 0)
    {
        printf("Error in creation of problem\n");
        exit(info);
    }
    // Set up clusterization (divide rows and columns into blocks)
    STARSH_cluster *cluster;
    info = starsh_cluster_new_plain(&cluster, array, N, block_size);
    if(info != 0)
    {
        printf("Error in creation of cluster\n");
        exit(info);
    }
    // Set up format (divide matrix into tiles)
    STARSH_blrf *format;
    info = starsh_blrf_new_tlr(&format, problem, symm, cluster, cluster);
    if(info != 0)
    {
        printf("Error in creation of format\n");
        exit(info);
    }
    // Approximate with tile low-rank matrix
    STARSH_blrm *matrix;
    info = starsh_blrm_approximate(&matrix, format, maxrank, tol,
            onfly);
    if(info != 0)
    {
        printf("Error in approximation\n");
        exit(info);
    }
    printf("Done with approximation!\n");
    starsh_blrf_info(format); // Show info about format
    starsh_blrm_info(matrix); // Show info about approximation
    // Show realtive error of approximation in Frobenius norm
    double rel_err = starsh_blrm__dfe_omp(matrix);
    printf("Relative error in Frobenius norm: %e\n", rel_err);
    return 0;
}
