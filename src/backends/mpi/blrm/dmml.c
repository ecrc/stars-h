#include "common.h"
#include "starsh.h"

int starsh_blrm__dmml_mpi(STARSH_blrm *M, int nrhs, double alpha, double *A,
        int lda, double beta, double *B, int ldb)
//! Multiply by dense Matrix, blr-matrix is on Left side.
/*! Performs `B=alpha*M*A+beta*B` using MPI+OpenMP */
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
        if(rank == 0)
            continue;
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
//! Multiply by dense Matrix, tiled blr-matrix is on Left side.
/*! Performs `B=alpha*M*A+beta*B` using MPI+OpenMP */
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
    int grid_nx = sqrt(mpi_size), grid_ny = grid_nx, grid_x, grid_y;
    if(grid_nx*grid_ny != mpi_size)
        STARSH_ERROR("MPI SIZE MUST BE SQUARE OF INTEGER!");
    grid_ny = mpi_size / grid_nx;
    grid_x = mpi_rank / grid_nx;
    grid_y = mpi_rank % grid_nx;
    MPI_Group mpi_leadingx_group, mpi_leadingy_group, mpi_world_group;
    MPI_Comm mpi_splitx, mpi_splity, mpi_leadingx, mpi_leadingy;
    MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);
    int group_rank[grid_nx];
    for(int i = 0; i < grid_ny; i++)
        group_rank[i] = i;
    MPI_Group_incl(mpi_world_group, grid_ny, group_rank, &mpi_leadingy_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, mpi_leadingy_group, 0,
            &mpi_leadingy);
    for(int i = 0; i < grid_nx; i++)
        group_rank[i] = i*grid_ny;
    MPI_Group_incl(mpi_world_group, grid_nx, group_rank, &mpi_leadingx_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, mpi_leadingx_group, 0,
            &mpi_leadingx);
    MPI_Comm_split(MPI_COMM_WORLD, grid_x, mpi_rank, &mpi_splitx);
    MPI_Comm_split(MPI_COMM_WORLD, grid_y, mpi_rank, &mpi_splity);
    int mpi_leadingx_rank=-1, mpi_leadingx_size=-1;
    int mpi_leadingy_rank=-1, mpi_leadingy_size=-1;
    int mpi_splitx_rank, mpi_splitx_size;
    int mpi_splity_rank, mpi_splity_size;
    if(mpi_leadingx != MPI_COMM_NULL)
    {
        MPI_Comm_rank(mpi_leadingx, &mpi_leadingx_rank);
        MPI_Comm_size(mpi_leadingx, &mpi_leadingx_size);
    }
    if(mpi_leadingy != MPI_COMM_NULL)
    {
        MPI_Comm_rank(mpi_leadingy, &mpi_leadingy_rank);
        MPI_Comm_size(mpi_leadingy, &mpi_leadingy_size);
    }
    MPI_Comm_rank(mpi_splitx, &mpi_splitx_rank);
    MPI_Comm_size(mpi_splitx, &mpi_splitx_size);
    MPI_Comm_rank(mpi_splity, &mpi_splity_rank);
    MPI_Comm_size(mpi_splity, &mpi_splity_size);
    /*
    STARSH_WARNING("MPI: GLOBAL=%d/%d LEADX=%d/%d LEADY=%d/%d SPLITX=%d/%d "
            "SPLITY=%d/%d", mpi_rank, mpi_size, mpi_leadingx_rank,
            mpi_leadingx_size, mpi_leadingy_rank, mpi_leadingy_size,
            mpi_splitx_rank, mpi_splitx_size, mpi_splity_rank,
            mpi_splity_size);
    */
    int grid_block_size = maxnb*grid_nx;
    int ld_temp_A = (F->nbcols+grid_nx-1-grid_x)/grid_nx*maxnb;
    double *temp_A;
    STARSH_MALLOC(temp_A, nrhs*ld_temp_A);
    if(mpi_leadingx != MPI_COMM_NULL)
    {
        for(int i = 0; i < F->nbcols/grid_nx; i++)
        {
            double *src = A+i*grid_block_size;
            double *recv = temp_A+i*maxnb;
            for(int j = 0; j < nrhs; j++)
            {
                MPI_Scatter(src+j*lda, maxnb, MPI_DOUBLE, recv+j*ld_temp_A,
                        maxnb, MPI_DOUBLE, 0, mpi_leadingx);
            }
        }
        int i = F->nbcols/grid_nx;
        int remain = F->nbcols-i*grid_nx;
        if(remain > 0)
        {
            double *src = A+i*grid_block_size;
            double *recv = temp_A+i*maxnb;
            if(mpi_rank == 0)
            {
                int sendcounts[grid_nx], displs[grid_nx];
                for(int j = 0; j < remain; j++)
                    sendcounts[j] = maxnb;
                for(int j = remain; j < grid_nx; j++)
                    sendcounts[j] = 0;
                displs[0] = 0;
                for(int j = 1; j < grid_nx; j++)
                    displs[j] = displs[j-1]+sendcounts[j-1];
                for(int j = 0; j < nrhs; j++)
                    MPI_Scatterv(src+j*lda, sendcounts, displs, MPI_DOUBLE,
                            recv+j*ld_temp_A, maxnb, MPI_DOUBLE, 0,
                            mpi_leadingx);
            }
            else
            {
                int recvcount = 0;
                if(grid_x < remain)
                    recvcount = maxnb;
                for(int j = 0; j < nrhs; j++)
                    MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                            recv+j*ld_temp_A, recvcount, MPI_DOUBLE, 0,
                            mpi_leadingx);
            }
        }
    }
    MPI_Bcast(temp_A, nrhs*ld_temp_A, MPI_DOUBLE, 0, mpi_splitx);
    //if(mpi_rank == 0)
    //    STARSH_WARNING("DATA DISTRIBUTED!");
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Bcast(A+i*lda, ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    int ldout = (F->nbrows+grid_ny-1-grid_y)/grid_ny*maxnb;
    //STARSH_WARNING("MPI=%d ldA=%d ldB=%d", mpi_rank, ld_temp_A, ldout);
    STARSH_MALLOC(temp_B, num_threads*nrhs*ldout);
    // Setting temp_B=beta*B for master thread of root node and B=0 otherwise
    #pragma omp parallel
    {
        double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
        for(int j = 0; j < nrhs*ldout; j++)
            out[j] = 0.;
    }
    if(beta != 0. && mpi_leadingy != MPI_COMM_NULL)
    {
        for(int i = 0; i < F->nbrows/grid_ny; i++)
        {
            double *src = B+i*maxnb*grid_ny;
            double *recv = temp_B+i*maxnb;
            for(int j = 0; j < nrhs; j++)
                MPI_Scatter(src+j*ldb, maxnb, MPI_DOUBLE, recv+j*ldout, maxnb,
                        MPI_DOUBLE, 0, mpi_leadingy);
        }
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < ldout; i++)
            for(int j = 0; j < nrhs; j++)
                temp_B[j*ldb+i] *= beta;
    }
    //if(mpi_rank == 0)
    //    STARSH_WARNING("MORE DATA DISTRIBUTED");
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
        if(rank == 0)
            continue;
        // Get pointers to data buffers
        double *U = M->far_U[lbi]->data, *V = M->far_V[lbi]->data;
        int info = 0;
        double *D = temp_D+omp_get_thread_num()*nrhs*maxrank;
        double *out = temp_B+omp_get_thread_num()*nrhs*ldout;
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
        //        ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                ncols, 1.0, V, ncols, temp_A+(j/grid_nx)*maxnb, ld_temp_A, 0.0,
                D, rank);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                rank, alpha, U, nrows, D, rank, 1.0, out+i/grid_ny*maxnb,
                ldout);
    }
    //STARSH_WARNING("NODE %d DONE WITH FAR", mpi_rank);
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
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
            //        out+R->start[i], ldout);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, temp_A+(j/grid_nx)*maxnb,
                    ld_temp_A, 1.0, out+i/grid_ny*maxnb, ldout);
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
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
            //        out+R->start[i], ldout);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, temp_A+(j/grid_nx)*maxnb,
                    ld_temp_A, 1.0, out+i/grid_ny*maxnb, ldout);
        }
    // Reduce result to temp_B, corresponding to master openmp thread
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < ldout; i++)
        for(int j = 0; j < nrhs; j++)
            for(int k = 1; k < num_threads; k++)
                temp_B[j*ldout+i] += temp_B[(k*nrhs+j)*ldout+i];
    //STARSH_WARNING("NODE %d DONE WITH OMP REDUCTION", mpi_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    // Since I keep result only on root node, following code is commented
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Allreduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM,
    //            MPI_COMM_WORLD);
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Reduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM, 0,
    //            MPI_COMM_WORLD);
    double *final_B = NULL;
    if(mpi_leadingy != MPI_COMM_NULL)
    {
        STARSH_MALLOC(final_B, nrhs*ldout);
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < nrhs*ldout; i++)
            final_B[i] = 0.0;
    }
    MPI_Reduce(temp_B, final_B, nrhs*ldout, MPI_DOUBLE, MPI_SUM, 0,
            mpi_splity);
    //STARSH_WARNING("REDUCE(%d): %f", mpi_rank, temp_B[0]);
    //if(mpi_splity_rank == 0)
    //    STARSH_WARNING("RESULT(%d): %f", mpi_rank, final_B[0]);
    if(mpi_leadingy != MPI_COMM_NULL)
    {
        for(int i = 0; i < F->nbrows/grid_ny; i++)
        {
            double *src = final_B+i*maxnb;
            double *recv = B+i*maxnb*grid_ny;
            for(int j = 0; j < nrhs; j++)
                MPI_Gather(src+j*ldout, maxnb, MPI_DOUBLE, recv+j*ldb, maxnb,
                        MPI_DOUBLE, 0, mpi_leadingy);
        }
        int i = F->nbrows/grid_ny;
        int remain = F->nbrows-i*grid_ny;
        if(remain > 0)
        {
            double *src = final_B+i*maxnb;
            double *recv = B+i*maxnb*grid_ny;
            if(mpi_rank == 0)
            {
                int recvcounts[grid_ny], displs[grid_ny];
                for(int j = 0; j < remain; j++)
                    recvcounts[j] = maxnb;
                for(int j = remain; j < grid_ny; j++)
                    recvcounts[j] = 0;
                displs[0] = 0;
                for(int j = 1; j < grid_ny; j++)
                    displs[j] = displs[j-1]+recvcounts[j-1];
                for(int j = 0; j < nrhs; j++)
                    MPI_Gatherv(src+j*ldout, maxnb, MPI_DOUBLE, recv+j*ldb,
                            recvcounts, displs, MPI_DOUBLE, 0, mpi_leadingy);
            }
            else
            {
                int sendcount = 0;
                if(grid_y < remain)
                    sendcount = maxnb;
                for(int j = 0; j < nrhs; j++)
                    MPI_Gatherv(src+j*ldout, sendcount, MPI_DOUBLE, NULL, NULL,
                            NULL, MPI_DOUBLE, 0, mpi_leadingy);
            }
        }
        MPI_Comm_free(&mpi_leadingy);
        free(final_B);
    }
    if(mpi_leadingx != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_leadingx);
    MPI_Comm_free(&mpi_splitx);
    MPI_Comm_free(&mpi_splity);
    free(temp_A);
    free(temp_B);
    free(temp_D);
    return 0;
}
