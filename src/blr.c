#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"
#include "lapacke.h"


int batched_lowrank_approximation(STARS_BLRmatrix *mat, int count, int *id,
        int maxrank, double tol, void **UV, int *rank)
{
    int bi, i, j;
    STARS_BLR *format = mat->format;
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
            kernel(rows, cols, format->row_order+
                    format->ibrow_start[i],
                    format->col_order+format->ibcol_start[j],
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
                /*
                for(k = 0; k < trank; k++)
                    cblas_dcopy(cols, tV+k, rows, UV[bi]+sizeof(double)*
                            (rows*trank+k*cols), 1);
                for(k = 0; k < cols; k++)
                    cblas_dscal(trank, ptrS[k], UV[bi]+sizeof(double)*
                            (rows*trank+k), cols);
                */
                /*
                for(k = 0; k < cols; k++)
                {
                    cblas_dscal(trank, ptrS[k], tV+k*rows, 1);
                    cblas_dcopy(trank, tV+k*rows, 1, UV[bi]+sizeof(double)*
                            trank*(rows+k), 1);
                }
                */
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

int batched_get_block(STARS_BLRmatrix *mat, int count, int *id, void **A)
{
    STARS_BLR *format = mat->format;
    block_kernel kernel = mat->problem->kernel;
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
            kernel(rows, cols, format->row_order+
                    format->ibrow_start[i],
                    format->col_order+format->ibcol_start[j],
                    format->problem->row_data,
                    format->problem->col_data, A[bi]);
        }
        //#pragma omp barrier
    }
    return 0;
}

STARS_BLRmatrix *STARS_blr_batched_algebraic_compress(STARS_BLR *format,
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
    STARS_BLRmatrix *mat = (STARS_BLRmatrix *)malloc(sizeof(STARS_BLRmatrix));
    mat->problem = format->problem;
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

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        int maxrank, double tol, int KADIR)
    // Private function of STARS-H
    // Uses SVD to acquire rank of each block, compresses given matrix (given
    // by block kernel, which returns submatrices) with relative accuracy tol
    // or with given maximum rank (if maxrank <= 0, then tolerance is used)
{
    int i, j, k, l, bi, rows, cols, mn, rank, tmprank, error = 0, info;
    int shape[2];
    char symm = format->symm;
    //printf("'%c' format->symm\n", symm);
    Array *block, *block2;
    double *U, *S, *V, *ptr;
    double norm;
    STARS_BLRmatrix *mat = (STARS_BLRmatrix *)malloc(sizeof(STARS_BLRmatrix));
    mat->format = format;
    mat->problem = format->problem;
    mat->bcount = format->nbrows * format->nbcols;
    mat->bindex = (int *)malloc(2*mat->bcount*sizeof(int));
    mat->brank = (int *)malloc(mat->bcount*sizeof(int));
    mat->U = (Array **)malloc(mat->bcount*sizeof(Array *));
    mat->V = (Array **)malloc(mat->bcount*sizeof(Array *));
    mat->A = (Array **)malloc(mat->bcount*sizeof(Array *));
    for(i = 0; i < format->nbrows; i++)
        for(j = 0; j < format->nbcols; j++)
            // Cycle over every block
        {
            bi = i * format->nbcols + j;
            mat->bindex[2*bi] = i;
            mat->bindex[2*bi+1] = j;
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            mn = rows > cols ? cols : rows;
            if((i < j && symm == 'S') || error == 1)
            {
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = NULL;
                mat->brank[bi] = mn;
                continue;
            }
            shape[0] = rows;
            shape[1] = cols;
            block = Array_new(2, shape, mat->problem->dtype, 'F');
            info = (mat->problem->kernel)(rows, cols, format->row_order +
                    format->ibrow_start[i], format->col_order +
                    format->ibcol_start[j], format->problem->row_data,
                    format->problem->col_data, block->buffer);
            block2 = Array_copy(block, 'N');
            dmatrix_lr(rows, cols, block->buffer, tol, &rank, &U, &S, &V);
            //Array_SVD(block, &U, &S, &V);
            //rank = SVD_get_rank(S, tol, 'F');
            Array_free(block);
            if((KADIR == 0 && rank < mn/2) || (KADIR == 1 && i != j))
                // If block is low-rank
            {
                if(maxrank > 0)
                    // If block is low-rank and maximum rank is upperbounded,
                    // then rank should be equal to maxrank
                    rank = maxrank;
                mat->A[bi] = NULL;
                shape[0] = rows;
                shape[1] = rank;
                mat->U[bi] = Array_new(2, shape, 'd', 'F');
                shape[0] = rank;
                shape[1] = cols;
                mat->V[bi] = Array_new(2, shape, 'd', 'F');
                cblas_dcopy(rank*rows, U, 1, mat->U[bi]->buffer, 1);
                ptr = mat->V[bi]->buffer;
                for(k = 0; k < cols; k++)
                    for(l = 0; l < rank; l++)
                    {
                        ptr[k*rank+l] = S[l]*V[k*mn+l];
                    }
                mat->brank[bi] = rank;
                Array_free(block2);
            }
            else
                // If block is NOT low-rank
            {
                if(i != j && KADIR == 1)
                {
                    printf("FULL rank offdiagonal (%i,%i)\n", i, j);
                    error = 1;
                }
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = block2;
                mat->brank[bi] = mn;
            }
            free(U);
            free(S);
            free(V);
        }
    if(error == 1)
    {
        STARS_BLRmatrix_free(mat);
        return NULL;
    }
    return mat;
}

void STARS_BLRmatrix_info(STARS_BLRmatrix *mat)
    // Print information on each block of block low-rank matrix.
{
    int i, bi, bj, r;
    if(mat == NULL)
    {
        printf("STARS_BLRmatrix NOT initialized\n");
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

void STARS_BLRmatrix_free(STARS_BLRmatrix *mat)
    // Free memory, used by matrix
{
    int bi;
    char symm = mat->format->symm;
    if(mat == NULL)
    {
        printf("STARS_BLRmatrix NOT initialized\n");
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

void STARS_BLR_info(STARS_BLR *format)
    // Print onfo on block partitioning
{
    int i;
    if(format == NULL)
    {
        printf("STARS_BLR NOT initialized\n");
        return;
    }
    if(format->symm == 'S')
        printf("Symmetric partitioning into blocks\n(blocking for columns "
                "is the same, as blocking for rows)\n");
    printf("Block rows (start, end):");
    i = 0;
    if(format->nbrows > 0)
        printf(" (%i, %i)", format->ibrow_start[i],
                format->ibrow_start[i]+format->ibrow_size[i]);
    for(i = 1; i < format->nbrows; i++)
    {
        printf(", (%i, %i)", format->ibrow_start[i],
                format->ibrow_start[i]+format->ibrow_size[i]);
    }
    printf("\n");
    if(format->symm == 'N')
    {
        printf("Block columns (start, end):");
        i = 0;
        if(format->nbcols > 0)
            printf(" (%i, %i)", format->ibcol_start[i],
                    format->ibcol_start[i]+format->ibcol_size[i]);
        for(i = 0; i < format->nbcols; i++)
        {
            printf("(%i, %i), ", format->ibcol_start[i],
                    format->ibcol_start[i]+format->ibcol_size[i]);
        }
        printf("\n");
    }
}

void STARS_BLR_free(STARS_BLR *format)
    // Free memory, used by block partitioning data
{
    if(format == NULL)
    {
        printf("STARS_BLR NOT initialized\n");
        return;
    }
    free(format->row_order);
    free(format->ibrow_start);
    free(format->ibrow_size);
    if(format->symm == 'N')
    {
       free(format->col_order);
       free(format->ibcol_start);
       free(format->ibcol_size);
    }
    free(format);
}

void STARS_BLRmatrix_error(STARS_BLRmatrix *mat)
{
    int bi, i, j;
    double diff = 0., norm = 0., tmpnorm, tmpdiff, tmperr, maxerr = 0.;
    int rows, cols, info, shape[2];
    STARS_BLR *format = mat->format;
    STARS_Problem *problem = format->problem;
    //Array *(*kernel)(int, int, int *, int *, void *, void *) =
    //    mat->format->problem->kernel;
    Array *block, *block2;
    char symm = format->symm;
    for(bi = 0; bi < mat->bcount; bi++)
    {
        i = mat->bindex[2*bi];
        j = mat->bindex[2*bi+1];
        if(i < j && symm == 'S')
            continue;
        rows = format->ibrow_size[i];
        cols = format->ibcol_size[j];
        shape[0] = rows;
        shape[1] = cols;
        block = Array_new(2, shape, problem->dtype, 'F');
        info = (problem->kernel)(rows, cols, format->row_order +
                format->ibrow_start[i], format->col_order +
                format->ibcol_start[j], problem->row_data,
                problem->col_data, block->buffer);
        tmpnorm = Array_norm(block);
        norm += tmpnorm*tmpnorm;
        if(i != j && symm == 'S')
            norm += tmpnorm*tmpnorm;
        if(mat->A[bi] == NULL)
        {
            block2 = Array_dot(mat->U[bi], mat->V[bi]);
            tmpdiff = Array_diff(block, block2);
            Array_free(block2);
            diff += tmpdiff*tmpdiff;
            if(i != j && symm == 'S')
                diff += tmpdiff*tmpdiff;
            tmperr = tmpdiff/tmpnorm;
            if(tmperr > maxerr)
                maxerr = tmperr;
        }
        /*
        if(mat->A[bi] != NULL)
        {
            block2 = mat->A[bi];
            tmpdiff = Array_diff(block, block2);
            diff += tmpdiff*tmpdiff;
        }
        */
        Array_free(block);
    }
    printf("Relative error of approximation of full matrix: %e\n",
            sqrt(diff/norm));
    printf("Maximum relative error of per-block approximation: %e\n", maxerr);
}

void STARS_BLRmatrix_getblock(STARS_BLRmatrix *mat, int i, int j, int order,
        int *shape, int *rank, void **U, void **V, void **A)
// PLEASE CLEAN MEMORY AFTER USE
{
    if(order != 'C' && order != 'F')
    {
        fprintf(stderr, "Parameter order should be 'C' or 'F', not '%c'\n",
                order);
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
        tmp = Array_copy(mat->U[bi], order);
        *U = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->V[bi] != NULL)
    {
        tmp = Array_copy(mat->V[bi], order);
        *V = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->A[bi] != NULL)
    {
        tmp = Array_copy(mat->A[bi], order);
        *A = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
}

void STARS_BLR_getblock(STARS_BLR *format, int i, int j, int order, int *shape,
        void **A)
// PLEASE CLEAN MEMORY POINTER AFTER USE
{
    if(order != 'C' && order != 'F')
    {
        fprintf(stderr, "Parameter order should be 'C' or 'F', not '%c'\n",
                order);
        exit(1);
    }
    if(order == 'C')
    {
        fprintf(stderr, "Order 'C' is not supported anymore\n");
        exit(1);
    }
    int rows = format->ibrow_size[i];
    int cols = format->ibcol_size[j];
    int info;
    shape[0] = rows;
    shape[1] = cols;
    *A = malloc(format->problem->dtype_size*rows*cols);
    info = (format->problem->kernel)(rows, cols, format->row_order +
            format->ibrow_start[i], format->col_order +
            format->ibcol_start[j], format->problem->row_data,
            format->problem->col_data, *A);
}

void STARS_BLRmatrix_printKADIR(STARS_BLRmatrix *mat)
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

void STARS_BLRmatrix_heatmap(STARS_BLRmatrix *mat, char *fname)
{
    int i, j, bi;
    STARS_BLR *format = mat->format;
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
