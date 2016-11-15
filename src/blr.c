#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <string.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"
#include "lapacke.h"


STARS_BLRF *STARS_BLRF_init(STARS_Problem *problem, char symm,
        STARS_Cluster *row_cluster, STARS_Cluster *col_cluster,
        int admissible_nblocks, int *ibrow_admissible_start,
        int *ibcol_admissible_start, int *ibrow_admissible_size,
        int *ibcol_admissible_size, int *ibrow_admissible,
        int *ibcol_admissible, STARS_BlockStatus *ibrow_admissible_status,
        STARS_BlockStatus *ibcol_admissible_status, STARS_BLRF_Type type)
// Initialization of structure STARS_BLRF
// Parameters:
//   problem: pointer to a structure, holding all the information about problem
//   symm: 'S' if problem and division into blocks are both symmetric, 'N'
//     otherwise
//   row_cluster: clusterization of rows into block rows.
//   col_cluster: clusterization of columns into block columns
//   admissible_nblocks: number of admissible blocks all in all
//   ibrow_admissible_start: starting point of list of admissible blocks
//     columns for a given block row in array ibrow_admissible. Not copied.
//   ibcol_admissible_start: starting point of list of admissible blocks
//     rows for a given block column in array ibcol_admissible. Not copied.
//   ibrow_admissible_size: number of admissible block columns for a given
//     block row. Not copied.
//   ibcol_admissible_size: number of admissible block rows for a given block
//     column. Not copied.
//   ibrow_admissible: array to store indexes of admissible block columns for
//     each block row. Not copied.
//   ibcol_admissible: array to store indexes of admissible block rows for each
//     block column. Not copied.
//   ibrow_admissible_status: status of each admissible pair of block row and
//     block column. STARS_Dense for guaranteed dense, STARS_LowRank for
//     guaranteed low-rank or STARS_Unknown if not known a priori. Not copied.
//   ibcol_admissible_status: status of each admissible pair of block column
//     and block row. STARS_Dense for guaranteed dense, STARS_LowRank for
//     guaranteed low-rank or STARS_Unknown if not known a priori. Not copied.
{
    STARS_BLRF *blrf = malloc(sizeof(*blrf));
    blrf->problem = problem;
    blrf->symm = symm;
    blrf->row_cluster = row_cluster;
    blrf->nbrows = row_cluster->nblocks;
    blrf->admissible_nblocks = admissible_nblocks;
    blrf->ibrow_admissible_start = ibrow_admissible_start;
    blrf->ibrow_admissible_size = ibrow_admissible_size;
    blrf->ibrow_admissible = ibrow_admissible;
    blrf->ibrow_admissible_status = ibrow_admissible_status;
    if(symm == 'N')
    {
        blrf->col_cluster = col_cluster;
        blrf->nbcols = col_cluster->nblocks;
        blrf->ibcol_admissible_start = ibcol_admissible_start;
        blrf->ibcol_admissible_size = ibcol_admissible_size;
        blrf->ibcol_admissible = ibcol_admissible;
        blrf->ibcol_admissible_status = ibcol_admissible_status;
    }
    else
    {
        blrf->col_cluster = blrf->row_cluster;
        blrf->nbcols = blrf->nbrows;
        blrf->ibcol_admissible_start = blrf->ibrow_admissible_start;
        blrf->ibcol_admissible_size = blrf->ibrow_admissible_size;
        blrf->ibcol_admissible = blrf->ibrow_admissible;
        blrf->ibcol_admissible_status = blrf->ibrow_admissible_status;
    }
    blrf->type = type;
    return blrf;
}

void STARS_BLRF_free(STARS_BLRF *blrf)
// Free memory, used by block low rank format (partitioning of array into
// blocks)
{
    if(blrf == NULL)
    {
        fprintf(stderr, "STARS_BLRF instance is NOT initialized\n");
        return;
    }
    free(blrf->ibrow_admissible_start);
    free(blrf->ibrow_admissible_size);
    free(blrf->ibrow_admissible);
    free(blrf->ibrow_admissible_status);
    if(blrf->symm == 'N')
    {
        free(blrf->ibcol_admissible_start);
        free(blrf->ibcol_admissible_size);
        free(blrf->ibcol_admissible);
        free(blrf->ibcol_admissible_status);
    }
    free(blrf);
}

void STARS_BLRF_info(STARS_BLRF *blrf)
// Print short info on block partitioning
{
    if(blrf == NULL)
    {
        fprintf(stderr, "STARS_BLRF instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRF at %p, '%c' symmetric, %d block rows, %d "
            "block columns, %d admissible blocks\n", blrf, blrf->symm,
            blrf->nbrows, blrf->nbcols, blrf->admissible_nblocks);
}

void STARS_BLRF_print(STARS_BLRF *blrf)
// Print full info on block partitioning
{
    int i, j, k;
    if(blrf == NULL)
    {
        printf("STARS_BLRF instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRF instance at %p, '%c' symmetric, %d block rows, %d "
            "block columns, %d admissible blocks\n", blrf, blrf->symm,
            blrf->nbrows, blrf->nbcols, blrf->admissible_nblocks);
    for(i = 0; i < blrf->nbrows; i++)
    {
        j = blrf->ibrow_admissible_start[i];
        if(blrf->ibrow_admissible_size[i] > 0)
            printf("Admissible block columns for block row %d: %d", i,
                    blrf->ibrow_admissible[j]);
        for(k = 1; k < blrf->ibrow_admissible_size[i]; k++)
        {
            printf(" %d", blrf->ibrow_admissible[j+k]);
        }
        if(blrf->ibrow_admissible_size[i] > 0)
            printf("\n");
    }
    if(blrf->symm == 'N')
        for(i = 0; i < blrf->nbcols; i++)
        {
            j = blrf->ibcol_admissible_start[i];
            if(blrf->ibcol_admissible_size[i] > 0)
                printf("Admissible block rows for block column %d: %d", i,
                        blrf->ibcol_admissible[j]);
            for(k = 1; k < blrf->ibcol_admissible_size[i]; k++)
            {
                printf(" %d", blrf->ibcol_admissible[j+k]);
            }
            if(blrf->ibcol_admissible_size[i] > 0)
                printf("\n");
        }
}

STARS_BLRF *STARS_BLRF_init_tiled(STARS_Problem *problem, STARS_Cluster
        *row_cluster, STARS_Cluster *col_cluster, char symm)
// Create plain division into tiles/blocks using plain cluster trees for rows
// and columns without actual pivoting
{
    if(symm == 'S' && problem->symm == 'N')
    {
        fprintf(stderr, "Since problem is NOT symmetric, can not proceed with "
                "symmetric flag on in STARS_BLRF_plain\n");
        exit(1);
    }
    if(symm == 'S' && row_cluster != col_cluster)
    {
        fprintf(stderr, "Since problem is symmetric, clusters should be the "
                "same (both pointers should be equal)\n");
        exit(1);
    }
    int nbrows = row_cluster->nblocks, nbcols = col_cluster->nblocks;
    int admissible_nblocks = nbrows*nbcols;
    int *ibrow_admissible_start = malloc(nbrows*sizeof(int));
    int *ibcol_admissible_start = ibrow_admissible_start;
    int *ibrow_admissible_size = malloc(nbrows*sizeof(int));
    int *ibcol_admissible_size = ibrow_admissible_size;
    int *ibrow_admissible = malloc(admissible_nblocks*sizeof(int));
    int *ibcol_admissible = ibrow_admissible;
    STARS_BlockStatus *ibrow_admissible_status = malloc(admissible_nblocks*
            sizeof(STARS_BlockStatus));
    STARS_BlockStatus *ibcol_admissible_status = ibrow_admissible_status;
    int i, j;
    for(i = 0; i < nbrows; i++)
    {
        ibrow_admissible_start[i] = i*nbcols;
        ibrow_admissible_size[i] = nbcols;
        for(j = 0; j < nbcols; j++)
        {
            ibrow_admissible[i*nbcols+j] = j;
            ibrow_admissible_status[i*nbcols+j] = STARS_Unknown;
        }
    }
    if(symm == 'N')
    {
        ibcol_admissible_start = malloc(nbcols*sizeof(int));
        ibcol_admissible_size = malloc(nbcols*sizeof(int));
        ibcol_admissible = malloc(admissible_nblocks*sizeof(int));
        ibcol_admissible_status = malloc(admissible_nblocks*
                sizeof(STARS_BlockStatus));
        for(i = 0; i < nbcols; i++)
        {
            ibcol_admissible_start[i] = i*nbrows;
            ibcol_admissible_size[i] = nbrows;
            for(j = 0; j < nbrows; j++)
            {
                ibcol_admissible[i*nbrows+j] = j;
                ibcol_admissible_status[i*nbrows+j] = STARS_Unknown;
            }
        }
    }
    return STARS_BLRF_init(problem, symm, row_cluster, col_cluster,
            admissible_nblocks, ibrow_admissible_start, ibcol_admissible_start,
            ibrow_admissible_size, ibcol_admissible_size, ibrow_admissible,
            ibcol_admissible, ibrow_admissible_status,
            ibcol_admissible_status, STARS_BLRF_Tiled);
}



/*
int batched_lowrank_approximation(STARS_BLRFmatrix *mat, int count, int *id,
        int maxrank, double tol, void **UV, int *rank)
{
    int bi, i, j;
    STARS_BLRF *format = mat->format;
    block_kernel kernel = format->problem->kernel;
    int max_rows = 0, max_cols = 0;
    for(i = 0; i < format->nbrows; i++)
        if(format->ibrow_size[i] > max_rows)
            max_rows = format->ibrow_size[i];
    for(i = 0; i < format->nbcols; i++)
        if(format->ibcol_size[i] > max_cols)
            max_cols = format->ibcol_size[i];
    int mx = max_cols > max_rows ? max_cols : max_rows;
    int dtype_size = format->problem->dtype_size;
    int block_size = max_rows*max_cols*dtype_size;
    int tlwork = (4*mx+7)*mx;
    int lwork = tlwork*dtype_size;
    int liwork = 8*mx*dtype_size;
    void *block, *work, *iwork, *U, *S, *V;
    int S_dtype_size;
    if(format->problem->dtype == 's')
        S_dtype_size = sizeof(float);
    else if(format->problem->dtype == 'd')
        S_dtype_size = sizeof(double);
    else if(format->problem->dtype == 'c')
        S_dtype_size = sizeof(float);
    else
        S_dtype_size = sizeof(double);
    //omp_set_max_active_levels(2);
    //printf("count=%d\n", count);
    #pragma omp parallel shared(block, work, iwork, U, V, S) private(i, j, bi)
    {
        int nthreads;
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
            //printf("Total threads %d\n", nthreads);
            block = malloc(nthreads*block_size);
            work = malloc(nthreads*lwork);
            iwork = malloc(nthreads*liwork);
            U = malloc(nthreads*block_size);
            V = malloc(nthreads*block_size);
            S = malloc(nthreads*mx*S_dtype_size);
            //printf("block_size %d, S_size %d\n", block_size,
            //mx*S_dtype_size);
        }
        #pragma omp barrier
        int tid = omp_get_thread_num();
        void *tblock = block+block_size*tid;
        void *twork = work+lwork*tid;
        void *tiwork = iwork+liwork*tid;
        void *tU = U+block_size*tid;
        void *tV = V+block_size*tid;
        void *tS = S+mx*S_dtype_size*tid;
        int tinfo = 0, rows, cols, mn, trank, k, l, bid;
        //printf("Work in thread %d\n", tid);
        #pragma omp for
        for(bi = 0; bi < count; bi++)
        {
            bid = id[bi];
            i = mat->bindex[2*bid];
            j = mat->bindex[2*bid+1];
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            mn = rows > cols ? cols : rows;
            //printf("%d %d %d\n", bi, rows, cols);
            kernel(rows, cols, format->row_pivot+
                    format->ibrow_start[i],
                    format->col_pivot+format->ibcol_start[j],
                    format->problem->row_data,
                    format->problem->col_data, tblock);
            tinfo = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', rows, cols,
                    tblock, rows, tS, tU, rows, tV, cols, twork, tlwork,
                    tiwork);
            double Sthresh = 0., Scur = 0.;
            double *ptrS = tS, *ptr, *ptrV = tV;
            for(k = 0; k < mn; k++)
                Sthresh += ptrS[k]*ptrS[k];
            Sthresh *= tol*tol;
            trank = 1;
            for(k = mn-1; k >= 1; k--)
            {
                Scur += ptrS[k]*ptrS[k];
                if(Sthresh < Scur)
                {
                    trank = k+1;
                    break;
                }
            }
            if(2*trank < mn)
            {
                cblas_dcopy(rows*trank, tU, 1, UV[bi], 1);
                ptr = UV[bi]+sizeof(double)*rows*trank;
                for(k = 0; k < cols; k++)
                    for(l = 0; l < trank; l++)
                    {
                        ptr[k*trank+l] = ptrS[l]*ptrV[k*mn+l];
                    }
                rank[bi] = trank;
            }
            else
                rank[bi] = -1;
        }
        #pragma omp master
        {
            free(block);
            free(work);
            free(iwork);
            free(U);
            free(V);
            free(S);
        }
        //#pragma omp barrier
    }
    return 0;
}

int batched_get_block(STARS_BLRFmatrix *mat, int count, int *id, void **A)
{
    STARS_BLRF *format = mat->format;
    block_kernel kernel = format->problem->kernel;
    #pragma omp parallel
    {
        int bi, i, j, rows, cols, nthreads, tid, bid;
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
            //printf("Total threads %d\n", nthreads);
        }
        #pragma omp barrier
        tid = omp_get_thread_num();
        //printf("Work in thread %d\n", tid);
        #pragma omp for
        for(bi = 0; bi < count; bi++)
        {
            bid = id[bi];
            i = mat->bindex[2*bid];
            j = mat->bindex[2*bid+1];
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            kernel(rows, cols, format->row_pivot+
                    format->ibrow_start[i],
                    format->col_pivot+format->ibcol_start[j],
                    format->problem->row_data,
                    format->problem->col_data, A[bi]);
        }
        //#pragma omp barrier
    }
    return 0;
}

STARS_BLRFmatrix *STARS_blrf_batched_algebraic_compress(STARS_BLRF *format,
        int maxrank, double tol)
{
    int i, j, bi, mn;
    char symm = format->symm;
    int num_blocks = format->nbrows*format->nbcols;
    int total_blocks = num_blocks;
    if(symm == 'S')
        num_blocks = (format->nbrows+1)*format->nbrows/2;
    //printf("X %d %d\n", format->nbrows, num_blocks);
    int *block_id = (int *)malloc(num_blocks*sizeof(int));
    int batched_id = 0;
    STARS_BLRFmatrix *mat = (STARS_BLRFmatrix *)malloc(sizeof(STARS_BLRFmatrix));
    mat->bcount = total_blocks;
    mat->format = format;
    mat->bindex = (int *)malloc(2*total_blocks*sizeof(int));
    mat->brank = (int *)malloc(total_blocks*sizeof(int));
    int *rank = (int *)malloc(num_blocks*sizeof(int));
    mat->U = (Array **)malloc(total_blocks*sizeof(Array *));
    mat->V = (Array **)malloc(total_blocks*sizeof(Array *));
    mat->A = (Array **)malloc(total_blocks*sizeof(Array *));
    for(i = 0; i < format->nbrows; i++)
        for(j = 0; j < format->nbcols; j++)
        {
            bi = i * format->nbcols + j;
            mat->bindex[2*bi] = i;
            mat->bindex[2*bi+1] = j;
            if(i < j && symm == 'S')
            {
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = NULL;
                mat->brank[bi] = -1;
                continue;
            }
            //printf("bid %d\n", batched_id);
            block_id[batched_id] = bi;
            batched_id++;
        }
    int rows = format->ibrow_size[0], cols = format->ibcol_size[0];
    int uv_size = (rows+cols)*maxrank*sizeof(double);
    void *UV_alloc = malloc(uv_size*num_blocks);
    mat->UV_alloc = UV_alloc;
    void **UV = malloc(num_blocks*sizeof(void *));
    for(bi = 0; bi < num_blocks; bi++)
        UV[bi] = UV_alloc+uv_size*bi;
    //printf("num_blocks=%d\n", num_blocks);
    batched_lowrank_approximation(mat, num_blocks, block_id, maxrank,
            tol, UV, rank);
    //printf("num_blocks=%d\n", num_blocks);
    int shape[2], bid;
    int num_fullrank = 0;
    for(bi = 0; bi < num_blocks; bi++)
    {
        bid = block_id[bi];
        mat->brank[bid] = rank[bi];
        mat->A[bid] = NULL;
        i = mat->bindex[2*bid];
        j = mat->bindex[2*bid+1];
        if(rank[bi] == -1)
        {
            mat->U[bid] = NULL;
            mat->V[bid] = NULL;
            block_id[num_fullrank] = bid;
            num_fullrank++;
        }
        else
        {
            shape[0] = format->ibrow_size[i];
            shape[1] = rank[bi];
            //if(rank[bi] > maxrank)
            //    printf("DDDDD\n");
            mat->U[bid] = Array_from_buffer(2, shape, 'd', 'F', UV[bi]);
            shape[0] = shape[1];
            shape[1] = format->ibcol_size[j];
            mat->V[bid] = Array_from_buffer(2, shape, 'd', 'F', UV[bi]+
                    mat->U[bid]->nbytes);
        }
    }
    //printf("33\n");
    free(UV);
    int a_size = rows*cols*(format->problem->dtype_size);
    //printf("a_size %d, %d\n", a_size, num_fullrank);
    void *A_alloc = malloc(num_fullrank*a_size);
    //printf("A_alloc %p\n", A_alloc);
    mat->A_alloc = A_alloc;
    void **A = malloc(num_fullrank*sizeof(void *));
    for(bi = 0; bi < num_fullrank; bi++)
    {
        bid = block_id[bi];
        A[bi] = A_alloc+a_size*bi;
        shape[0] = format->ibrow_size[mat->bindex[2*bid]];
        shape[1] = format->ibcol_size[mat->bindex[2*bid+1]];
        //printf("(%d,%d)\n", shape[0], shape[1]);
        mat->A[bid] = Array_from_buffer(2, shape, format->problem->dtype,
                'F', A[bi]);
    } 
    //printf("GOING DEEP\n");
    batched_get_block(mat, num_fullrank, block_id, A);
    return mat;
}


void STARS_BLRFmatrix_info(STARS_BLRFmatrix *mat)
    // Print information on each block of block low-rank matrix.
{
    int i, bi, bj, r;
    if(mat == NULL)
    {
        printf("STARS_BLRFmatrix NOT initialized\n");
        return;
    }
    for(i = 0; i < mat->bcount; i++)
    {
        bi = mat->bindex[2*i];
        bj = mat->bindex[2*i+1];
        r = mat->brank[i];
        if(r != -1)
        {
            printf("block (%i, %i) U: ", bi, bj);
            Array_info(mat->U[i]);
            printf("block (%i, %i) V: ", bi, bj);
            Array_info(mat->V[i]);
        }
        else
        {
            printf("block (%i, %i): ", bi, bj);
            Array_info(mat->A[i]);
        }
    }
}

void STARS_BLRFmatrix_free(STARS_BLRFmatrix *mat)
    // Free memory, used by matrix
{
    int bi;
    char symm = mat->format->symm;
    if(mat == NULL)
    {
        printf("STARS_BLRFmatrix NOT initialized\n");
        return;
    }
    for(bi = 0; bi < mat->bcount; bi++)
    {
        if(mat->A[bi] != NULL)
            Array_free(mat->A[bi]);
        if(mat->U[bi] != NULL)
            Array_free(mat->U[bi]);
        if(mat->V[bi] != NULL)
            Array_free(mat->V[bi]);
    }
    free(mat->A);
    free(mat->U);
    free(mat->V);
    free(mat->bindex);
    free(mat->brank);
    free(mat);
}



*/
/*
void STARS_BLRFmatrix_getblock(STARS_BLRFmatrix *mat, int i, int j, int pivot,
        int *shape, int *rank, void **U, void **V, void **A)
// PLEASE CLEAN MEMORY AFTER USE
{
    if(pivot != 'C' && pivot != 'F')
    {
        fprintf(stderr, "Parameter pivot should be 'C' or 'F', not '%c'\n",
                pivot);
        exit(1);
    }
    int bi = i * mat->format->nbcols + j;
    Array *tmp;
    *rank = mat->brank[bi];
    shape[0] = mat->format->ibrow_size[i];
    shape[1] = mat->format->ibcol_size[j];
    *U = NULL;
    *V = NULL;
    *A = NULL;
    if(mat->U[bi] != NULL)
    {
        tmp = Array_copy(mat->U[bi], pivot);
        *U = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->V[bi] != NULL)
    {
        tmp = Array_copy(mat->V[bi], pivot);
        *V = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->A[bi] != NULL)
    {
        tmp = Array_copy(mat->A[bi], pivot);
        *A = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
}

void STARS_BLRF_getblock(STARS_BLRF *format, int i, int j, int pivot, int *shape,
        void **A)
// PLEASE CLEAN MEMORY POINTER AFTER USE
{
    if(pivot != 'C' && pivot != 'F')
    {
        fprintf(stderr, "Parameter pivot should be 'C' or 'F', not '%c'\n",
                pivot);
        exit(1);
    }
    if(pivot == 'C')
    {
        fprintf(stderr, "pivot 'C' is not supported anymore\n");
        exit(1);
    }
    int rows = format->ibrow_size[i];
    int cols = format->ibcol_size[j];
    int info;
    shape[0] = rows;
    shape[1] = cols;
    *A = malloc(format->problem->dtype_size*rows*cols);
    info = (format->problem->kernel)(rows, cols, format->row_pivot +
            format->ibrow_start[i], format->col_pivot +
            format->ibcol_start[j], format->problem->row_data,
            format->problem->col_data, *A);
}

void STARS_BLRFmatrix_printKADIR(STARS_BLRFmatrix *mat)
{
    int i, j, bi;
    for(bi = 0; bi < mat->bcount; bi++)
    {
        i = mat->bindex[2*bi];
        j = mat->bindex[2*bi+1];
        if(i < j)
            continue;
        printf("BLOCK %d %d:\n", i, j);
        if(mat->A[bi] == NULL)
        {
            Array_info(mat->U[bi]);
            Array_print(mat->U[bi]);
            Array_info(mat->V[bi]);
            Array_print(mat->V[bi]);
        }
        else
        {
            Array_info(mat->A[bi]);
            Array_print(mat->A[bi]);
        }
        printf("\n");
    }
}

void STARS_BLRFmatrix_heatmap(STARS_BLRFmatrix *mat, char *fname)
{
    int i, j, bi;
    STARS_BLRF *format = mat->format;
    FILE *fd = fopen(fname, "w");
    fprintf(fd, "%d %d\n", format->nbrows, format->nbcols);
    for(i = 0; i < format->nbrows; i++)
    {
        for(j = 0; j < format->nbrows; j++)
        {
            bi = i * format->nbcols + j;
            if(format->symm == 'S' && i < j)
                bi = j * format->nbcols + i;
            fprintf(fd, " %d", mat->brank[bi]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
*/
