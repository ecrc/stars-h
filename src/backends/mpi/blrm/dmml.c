#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>
#include "starsh.h"

int starsh_blrm__dmml_mpi(STARSH_blrm *M, int nrhs, double alpha, double *A,
        int lda, double beta, double *B, int ldb)
//! Double precision Multiply by dense Matrix, blr-matrix is on Left side.
/*! Performs `B=alpha*M*A+beta*B` using OpenMP */
{
    STARSH_blrf *F = M->format;
    STARSH_problem *P = F->problem;
    STARSH_kernel kernel = P->kernel;
    int nrows = P->shape[0];
    int ncols = P->shape[P->ndim-1];
    // Shorcuts to information about clusters
    STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
    void *RD = R->data, *CD = C->data;
    // Number of far-field and near-field blocks
    size_t nblocks_far_local = F->nblocks_far_local;
    size_t nblocks_near_local = F->nblocks_near_local;
    size_t lbi;
    char symm = F->symm;
    int maxrank = 0;
    for(size_t lbi = 0; lbi < nblocks_far_local; lbi++)
        if(maxrank < M->far_rank[lbi])
            maxrank = M->far_rank[lbi];
    int maxnb = nrows/F->nbrows;
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    for(int i = 0; i < nrhs; i++)
        MPI_Bcast(A+i*lda, ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double *temp_D, *temp_B;
    int num_threads;
    #pragma omp parallel
    #pragma omp master
    num_threads = omp_get_num_threads();
    if(M->onfly == 0)
    {
        STARSH_MALLOC(temp_D, num_threads*nrhs*maxrank);
    }
    else
    {
        STARSH_MALLOC(temp_D, num_threads*maxnb*maxnb);
    }
    STARSH_MALLOC(temp_B, num_threads*nrhs*nrows);
    // Setting temp_B=beta*B for master thread of root node and B=0 otherwise
    #pragma omp parallel
    {
        double *out = temp_B+omp_get_thread_num()*nrhs*nrows;
        for(int j = 0; j < nrhs*nrows; j++)
            out[j] = 0.;
    }
    if(beta != 0. && mpi_rank == 0)
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < nrhs; j++)
                temp_B[j*ldb+i] = beta*B[j*ldb+i];
    int ldout = nrows;
    // Simple cycle over all far-field admissible blocks
    #pragma omp parallel for schedule(dynamic, 1)
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        size_t bi = F->block_far_local[lbi];
        // Get indexes of corresponding block row and block column
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        // Get sizes and rank
        int nrows = R->size[i];
        int ncols = C->size[j];
        int rank = M->far_rank[lbi];
        // Get pointers to data buffers
        double *U = M->far_U[lbi]->data, *V = M->far_V[lbi]->data;
        int info = 0;
        double *D = temp_D+omp_get_thread_num()*nrhs*maxrank;
        double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                rank, alpha, U, nrows, D, rank, 1.0, out+R->start[i], ldout);
        if(i != j && symm == 'S')
        {
            // Multiply low-rank matrix in V*U^T format by a dense matrix
            // U and V are simply swapped in case of symmetric block
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                    nrows, 1.0, U, nrows, A+R->start[i], lda, 0.0, D, rank);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ncols,
                    nrhs, rank, alpha, V, ncols, D, rank, 1.0,
                    out+C->start[j], ldout);
        }
    }
    if(M->onfly == 1)
        // Simple cycle over all near-field blocks
        #pragma omp parallel for schedule(dynamic, 1)
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            size_t bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            int info = 0;
            double *D = temp_D+omp_get_thread_num()*maxnb*maxnb;
            double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
            // Fill temporary buffer with elements of corresponding block
            kernel(nrows, ncols, R->pivot+R->start[i],
                    C->pivot+C->start[j], RD, CD, D);
            // Multiply 2 dense matrices
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                    out+R->start[i], ldout);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncols,
                        nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                        1.0, out+C->start[j], ldout);
            }
        }
    else
        // Simple cycle over all near-field blocks
        #pragma omp parallel for schedule(dynamic, 1)
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            size_t bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Get pointers to data buffers
            double *D = M->near_D[lbi]->data;
            double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
            // Multiply 2 dense matrices
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                    out+R->start[i], ldout);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncols,
                        nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                        1.0, out+C->start[j], ldout);
            }
        }
    // Reduce result to temp_B, corresponding to master openmp thread
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < ldout; i++)
        for(int j = 0; j < nrhs; j++)
            for(int k = 1; k < num_threads; k++)
                temp_B[j*ldout+i] += temp_B[(k*nrhs+j)*ldout+i];
    // Since I keep result only on root node, following code is commented
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Allreduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM,
    //            MPI_COMM_WORLD);
    for(int i = 0; i < nrhs; i++)
        MPI_Reduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
    free(temp_B);
    free(temp_D);
    return 0;
}



int starsh_blrm__dmml_mpi_tiled(STARSH_blrm *M, int nrhs, double alpha,
        double *A, int lda, double beta, double *B, int ldb)
//! Double precision Multiply by dense Matrix, blr-matrix is on Left side.
/*! Performs `B=alpha*M*A+beta*B` using OpenMP */
{
    STARSH_blrf *F = M->format;
    STARSH_problem *P = F->problem;
    STARSH_kernel kernel = P->kernel;
    int nrows = P->shape[0];
    int ncols = P->shape[P->ndim-1];
    // Shorcuts to information about clusters
    STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
    void *RD = R->data, *CD = C->data;
    // Number of far-field and near-field blocks
    size_t nblocks_far_local = F->nblocks_far_local;
    size_t nblocks_near_local = F->nblocks_near_local;
    size_t lbi;
    char symm = F->symm;
    int maxrank = 0;
    for(size_t lbi = 0; lbi < nblocks_far_local; lbi++)
        if(maxrank < M->far_rank[lbi])
            maxrank = M->far_rank[lbi];
    int maxnb = nrows/F->nbrows;
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int grid_nx = 1, grid_ny, grid_x, grid_y;
    int pow = 0, val = 1;
    while(val < mpi_size)
    {
        pow++;
        val *= 4;
        grid_nx *= 2;
    }
    if(val != mpi_size)
        STARSH_ERROR("MPI SIZE MUST BE POWER OF 4!");
    grid_ny = mpi_size / grid_nx;
    grid_x = mpi_rank % grid_nx;
    grid_y = mpi_rank / grid_nx;
    for(int i = 0; i < nrhs; i++)
        MPI_Bcast(A+i*lda, ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double *temp_D, *temp_B;
    int num_threads;
    #pragma omp parallel
    #pragma omp master
    num_threads = omp_get_num_threads();
    if(M->onfly == 0)
    {
        STARSH_MALLOC(temp_D, num_threads*nrhs*maxrank);
    }
    else
    {
        STARSH_MALLOC(temp_D, num_threads*maxnb*maxnb);
    }
    STARSH_MALLOC(temp_B, num_threads*nrhs*nrows);
    // Setting temp_B=beta*B for master thread of root node and B=0 otherwise
    #pragma omp parallel
    {
        double *out = temp_B+omp_get_thread_num()*nrhs*nrows;
        for(int j = 0; j < nrhs*nrows; j++)
            out[j] = 0.;
    }
    if(beta != 0. && mpi_rank == 0)
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < nrhs; j++)
                temp_B[j*ldb+i] = beta*B[j*ldb+i];
    int ldout = nrows;
    // Simple cycle over all far-field admissible blocks
    #pragma omp parallel for schedule(dynamic, 1)
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        size_t bi = F->block_far_local[lbi];
        // Get indexes of corresponding block row and block column
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        // Get sizes and rank
        int nrows = R->size[i];
        int ncols = C->size[j];
        int rank = M->far_rank[lbi];
        // Get pointers to data buffers
        double *U = M->far_U[lbi]->data, *V = M->far_V[lbi]->data;
        int info = 0;
        double *D = temp_D+omp_get_thread_num()*nrhs*maxrank;
        double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                rank, alpha, U, nrows, D, rank, 1.0, out+R->start[i], ldout);
        if(i != j && symm == 'S')
        {
            // Multiply low-rank matrix in V*U^T format by a dense matrix
            // U and V are simply swapped in case of symmetric block
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                    nrows, 1.0, U, nrows, A+R->start[i], lda, 0.0, D, rank);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ncols,
                    nrhs, rank, alpha, V, ncols, D, rank, 1.0,
                    out+C->start[j], ldout);
        }
    }
    if(M->onfly == 1)
        // Simple cycle over all near-field blocks
        #pragma omp parallel for schedule(dynamic, 1)
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            size_t bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            int info = 0;
            double *D = temp_D+omp_get_thread_num()*maxnb*maxnb;
            double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
            // Fill temporary buffer with elements of corresponding block
            kernel(nrows, ncols, R->pivot+R->start[i],
                    C->pivot+C->start[j], RD, CD, D);
            // Multiply 2 dense matrices
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                    out+R->start[i], ldout);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncols,
                        nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                        1.0, out+C->start[j], ldout);
            }
        }
    else
        // Simple cycle over all near-field blocks
        #pragma omp parallel for schedule(dynamic, 1)
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            size_t bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Get pointers to data buffers
            double *D = M->near_D[lbi]->data;
            double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
            // Multiply 2 dense matrices
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                    out+R->start[i], ldout);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncols,
                        nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                        1.0, out+C->start[j], ldout);
            }
        }
    // Reduce result to temp_B, corresponding to master openmp thread
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < ldout; i++)
        for(int j = 0; j < nrhs; j++)
            for(int k = 1; k < num_threads; k++)
                temp_B[j*ldout+i] += temp_B[(k*nrhs+j)*ldout+i];
    // Since I keep result only on root node, following code is commented
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Allreduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM,
    //            MPI_COMM_WORLD);
    for(int i = 0; i < nrhs; i++)
        MPI_Reduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
    free(temp_B);
    free(temp_D);
    return 0;
}
