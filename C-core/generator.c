#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct
{
    int block_size, block_count;
    double *U;
    double *S;
    double *V;
} flatH;

void dcopy_(int *, double *, int *, double *, int *);
void dscal_(int *, double *, double *, int *);
void dgeqrf_(int *, int *, double *, int *, double *, double *, int *, int *);
void dorgqr_(int *, int *, int *, double *, int *, double *, double *, int *,
        int *);
void dgemm_(char *, char *, int *, int *, int *, double *, double *, int *,
        double *, int *, double *, double *, int *);
void dgesdd_(char *, int *, int *, double *, int *, double *, double *, int *,
        double *, int *, double *, int *, int *, int *);

double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if(phase == 0)
    {
        do
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
    {
        X = V2 * sqrt(-2 * log(S) / S);
    }
    phase = 1 - phase;
    return X;
}

void random_orthogonal(int n, int m, double **A)
{
    int i, info, lwork=m;
    int size = n*m;
    if(*A == NULL)
    {
        *A = (double *)malloc(size*sizeof(double));
    }
    double *a = *A;
    double *tau = (double *)malloc(m*sizeof(double));
    double *work = (double *)malloc(lwork*sizeof(double));
    for(i = 0; i < size; i++)
    {
        a[i] = gaussrand();
    }
    dgeqrf_(&n, &m, a, &n, tau, work, &lwork, &info);
    dorgqr_(&n, &m, &m, a, &n, tau, work, &lwork, &info);
    free(tau);
    free(work);
}

void test_random_orthogonal()
{
    int n = 5, m = 3;
    int i, j;
    char cN = 'N', cT = 'T';
    double one = 1.0, zero = 0.0;
    srand(time(0));
    double *a = NULL;
    random_orthogonal(n, m, &a);
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < m; j++)
        {
            printf("%f ", a[j*n+i]);
        }
        printf("\n");
    }
    printf("Orthogonality check:\n");
    double *work = (double *)malloc(m*m*sizeof(double));
    dgemm_(&cT, &cN, &m, &m, &n, &one, a, &n, a, &n, &zero, work, &m);
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < m; j++)
        {
            printf("%f ", work[j*m+i]);
        }
        printf("\n");
    }
    free(work);
    free(a);
}

int block_generate(int block_size, double *U, double *S, double *V, double *A,
        int lda, double *work)
// Generation of block inside a matrix with given singular vectors and values
// A: output, result is written into data (column-major order)
// block_size: size of block to generate (block is square)
// lda: leading dimension of A (column-major order, memory offset between
// columns)
// U: left (row) singular vectors
// S: singular vectors
// V: right (column) singular vectors
// work: temporary array
{
    int i, incx = 1;
    char cN = 'N';
    double one = 1.0, zero = 0.0;
    int size = block_size*block_size;
    dcopy_(&size, U, &incx, work, &incx);
    for(i = 0; i < block_size; i++)
    {
        dscal_(&block_size, S+i, work+block_size*i, &incx);
    }
    dgemm_(&cN, &cN, &block_size, &block_size, &block_size, &one, work,
            &block_size, V, &block_size, &zero, A, &lda);
    return 0;
}

int matrix_generate(flatH *A, double **mat)
{
    int i, j;
    int block_count = A->block_count;
    int block_size = A->block_size;
    int size = block_count*block_size;
    int size2 = block_size*size;
    int bsize = block_size*block_size;
    if(*mat == NULL)
    {
        *mat = (double *)malloc(size*size*sizeof(double));
    }
    double *ptr;
    double *U = A->U;
    double *S = A->S;
    double *V = A->V;
    double *work = (double *)malloc(bsize*sizeof(double));
    for(i = 0; i < block_count; i++)
    {
        ptr = *mat+block_size*i;
        for(j = 0; j < block_count; j++)
        {
            block_generate(block_size, U+j*bsize, S, V+i*bsize,
                    ptr+j*size2, size, work);
        }
    }
    free(work);
    return 0;
}

void generate_structure(int block_size, int block_count, double decay,
        flatH *A)
{
    int i;
    int bsize = block_size*block_size;
    int msize = bsize*block_count;
    int hsize = block_size*block_count;
    double *U = (double *)malloc(msize*sizeof(double));
    double *V = (double *)malloc(msize*sizeof(double));
    double *S = (double *)malloc(hsize*sizeof(double));
    double *ptr;
    for(i = 0; i < block_count; i++)
    {
        ptr = U+i*bsize;
        random_orthogonal(block_size, block_size, &ptr);
        ptr = V+i*bsize;
        random_orthogonal(block_size, block_size, &ptr);
    }
    S[0] = 1.0;
    for(i = 1; i < block_size; i++)
    {
        S[i] = S[i-1]/decay;
    }
    A->block_size = block_size;
    A->block_count = block_count;
    A->U = U;
    A->V = V;
    A->S = S;
}

void test_generate_matrix()
{
    int block_size = 2;
    int block_count = 2;
    double decay = 2.0;
    flatH A;
    srand(time(0));
    generate_structure(block_size, block_count, decay, &A);
    double *mat = NULL;
    matrix_generate(&A, &mat);
    int i, j, size = block_size*block_count;
    int size2 = size*size, incx = 1;
    printf("Initial singular values:\n   ");
    for(i = 0; i < block_size; i++)
    {
        printf(" %f", A.S[i]);
    }
    printf("\n");
    printf("Per-block singular values\n");
    char cN = 'N';
    int lwork = 10*block_size, info;
    double *S = (double *)malloc(block_size*sizeof(double));
    double *work = (double *)malloc(lwork*sizeof(double));
    int *iwork = (int *)malloc(8*block_size*sizeof(int));
    int k;
    double *mat2 = (double *)malloc(size*size*sizeof(double));
    dcopy_(&size2, mat, &incx, mat2, &incx);
    for(i = 0; i < block_count; i++)
    {
        for(j = 0; j < block_count; j++)
        {
            dgesdd_(&cN, &block_size, &block_size,
                    mat2+(j*size+i)*block_size,
                    &size, S, NULL, &block_size, NULL, &block_size, work,
                    &lwork, iwork, &info);
            printf("%d %d:\n   ", i, j);
            for(k = 0; k < block_size; k++)
            {
                printf(" %f", S[k]);
            }
            printf("\n");
        }
    }
    dgesdd_(&cN, &size, &block_size, mat, &size, S, NULL, &block_size, NULL,
            &block_size, work, &lwork, iwork, &info);
    printf("Block-column singular values (should NOT scale the same way, as"
        " above):\n   ");
    for(k = 0; k < block_size; k++)
    {
        printf(" %f", S[k]);
    }
    printf("\n");
    free(S);
    free(iwork);
    free(work);
}
