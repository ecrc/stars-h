#include <stdio.h>
#include "astronomy.h"
#include "stars.h"
#include "stars-astronomy.h"

void reconstructor(double *Cmm,double *Ctm, long M, long N, long lda, long ldb){

    int info= LAPACKE_dpotrf(LAPACK_ROW_MAJOR ,'L',N,Cmm,lda);
    if(info!=0){
        printf("An error occured in dpotrf: %d\n", info);
        return;
    }
    cblas_dtrsm(CblasRowMajor,CblasRight,CblasLower,CblasTrans  ,CblasNonUnit,M,N,1.,Cmm,lda,Ctm,ldb);
    cblas_dtrsm(CblasRowMajor,CblasRight,CblasLower,CblasNoTrans,CblasNonUnit,M,N,1.,Cmm,lda,Ctm,ldb);

}



void compute_Cee_Cvv(int nMeasTS, int nMeas, int nact, double *Cmm, double *Cpp, double *Cpm, double *R, double *Dx, double *Cee, double *Cvv){
/**
*compute the matrix Cee Cvv
*
*imput:
*   int nMeasTS: number of measurements of the truth sensor
*   int nMeas   :number of measurements
*   double *Cmm :covariance matrix between the sensors                              Cmm(nMeas,nMeas)
*   double *Cpp :covariance matrix of the truth sensor                              Cpp(nMeasTS,nMeasTS)
*   double *Cpm :covariance matrix between the truth sensors and the other sensors  Cpm(nMeas,nMeasTS)
*   double *R   :tomographic reconstructor                                          R(nMeas,nMeasTS))
*   double *Dx  :                                                                   Dx(nact,nMeasTS)
*
*output:
*   double *Cee : tomographic error
*   double *Cvv :
*
*/
    double *Tmp=(double*)malloc(nact*nMeasTS*sizeof(double));

    double alpha=-1.;
    double beta =1.;
    cblas_dsyr2k(CblasRowMajor,CblasUpper,CblasNoTrans,nMeasTS,nMeas,alpha,Cpm,nMeas,R,nMeas,beta,Cpp,nMeasTS);


    alpha=1.;
    beta =0.;
    cblas_dsymm(CblasRowMajor,CblasRight,CblasUpper,nMeasTS,nMeas,alpha,Cmm,nMeas,R,nMeas,beta,Cpm,nMeas);


    LAPACKE_dlacpy(LAPACK_COL_MAJOR,'L',nMeasTS,nMeasTS,Cpp,nMeasTS,Cee,nMeasTS);


    alpha=1.0;
    beta=1.0;
    add('T',nMeasTS,nMeasTS,alpha,Cee,beta,Cee);

    alpha=0.5;
    scaldiag(nMeasTS,alpha,Cee);

    alpha=1.;
    beta =1.;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,nMeasTS,nMeasTS,nMeas,alpha,Cpm,nMeas,R,nMeas,beta,Cee,nMeasTS);

    alpha=1.;
    beta =0.;
    cblas_dsymm(CblasRowMajor,CblasRight,CblasLower,nact,nMeasTS,alpha,Cee,nMeasTS,Dx,nMeasTS,beta,Tmp,nMeasTS);

    alpha=1.;
    beta =0.;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,nact,nact,nMeasTS,alpha,Tmp,nMeasTS,Dx,nMeasTS,beta,Cvv,nact);

    free(Tmp);
}



/**
*restore the symmetry of a square matrix
*
*char       uplo    : part of the matrix from which the symmetry is build
*int        M       : order of the matrix
*double*    A       : data
*int        lda     : leading dimension of A
*
*/
void restore_symm(char uplo,int M, double *A,int lda){
    int i,j;
    if(M>lda){
        fprintf(stderr,"error: M>lda");
        return;
    }
    if(uplo=='U'){
        for(i=0;i<M;i++)
            for(j=0;j<i;j++)
                A[i*lda+j]=A[j*lda+i];
    }
    if(uplo=='L'){
        for(i=0;i<M;i++)
            for(j=i+1;j<M;j++)
                A[i*lda+j]=A[j*lda+i];
    }

}


/**
*addition of 2 matrices
*return alpha*op(A)+beta*B
*
*op(A):(M,N)
*B    :(M,N)
*
*op(A)=A            if trans='N'
*      transpose(A) if Trans='T'
*
*/
void add(char Trans,int M,int N,double alpha,double *A,double beta,double *B){

    int i,j;

    if(Trans=='N'){
        for(i=0;i<M*N;i++)
            B[i]=alpha*A[i]+beta*B[i];
    }
    else if(Trans=='T'){
        for(i=0;i<M;i++){
            for(j=0;j<N;j++){
                B[i*N+j]=alpha*A[i+j*M]+beta* B[i*N+j];
            }
        }
    }
}

/**
*scale the diagonnal of a square matrix
*
*int        M       : order of the matrix
*double     alpha   : scaling factor
*double*    A       : data
*/
void scaldiag(int M,double alpha, double *A){
    int i;
    for(i=0;i<M;i++)
        A[i*(M+1)]*=alpha;
}


//for cython use
//void _dtrsm(int M, int N, double *A, int lda, double *B, int ldb){
//    //op( A )*X = alpha*B
//    cblas_dtrsm(CblasRowMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,M,N,1.,A,lda,B,ldb);
//}
//
//void _dsyr2k(int N, int M, double alpha, double *A, int lda, double *B, int ldb, double beta,double *C, int ldc){
//    //alpha*A*B' + alpha*B*A' + beta*C
//    cblas_dsyr2k(CblasRowMajor,CblasLower,CblasNoTrans,M,N,alpha,A,lda,B,ldb,beta,C,ldc);
//}
//
//void _dsymm(int M, int N, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc){
//    //C := alpha*B*A + beta*C
//    cblas_dsymm(CblasRowMajor,CblasRight,CblasLower,M,N,alpha,A,lda,B,ldb,beta,C,ldc);
//}
//
//void _dlacpy(int M,int N, double *A, int lda, double *B, int ldb){
//    //copy A in B
//    //cblas_dlacpy(CblasRowMajor,CblasLower,M,N,A,lda,B,ldb);
//    LAPACKE_dlacpy(CblasRowMajor,CblasLower,M,N,A,lda,B,ldb);
//}
//
//void _dgemm(int M,int N,int K,double alpha, double *A, int lda, double *B,int ldb,double beta, double *C,int ldc){
//
//    //C := alpha* A * B.T  + beta*C
//    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
//
//}

int main(int argc, char **argv){

    if(argc < 6)
    {
        printf("./astronomy.out files_path block_size maxrank tol heatmap_filename\n");
        exit(1);
    }
    STARS_tomo *tomo;
    char *files_path = argv[1];
    int night_idx=1;
    int snapshots_per_night=1;
    int snapshot_idx=1;
    int obs_idx=0;
    double alphaX=0.0;
    double alphaY=0.0;
    char *heatmap_fname = argv[5];
    int block_size = atoi(argv[2]);
    int maxrank = atoi(argv[3]);
    double tol = atof(argv[4]);
    tomo = STARS_gen_aodata(files_path, night_idx, snapshots_per_night,
            snapshot_idx, obs_idx, alphaX, alphaY);
    STARS_Problem *problem = STARS_gen_aoproblem(tomo);
    STARS_BLR *format = STARS_gen_ao_blrformat(problem, block_size);
    STARS_BLRmatrix *mat = STARS_blr_batched_algebraic_compress(format, maxrank, tol);
    STARS_BLRmatrix_error(mat);
    //STARS_BLRmatrix_heatmap(mat, heatmap_fname);
    //STARS_BLRmatrix_free(mat);
    /*
    int nbrows = format->nbrows, shape[2], i;
    char order = 'C';
    double *buffer = NULL;
    FILE *fd;
    Array *U, *S, *V, *block;
    STARS_BLR_getblock(format, nbrows/2, nbrows/2, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'C', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_getblock(format, 5*nbrows/8, 3*nbrows/8, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'C', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_getblock(format, 3*nbrows/4, nbrows/4, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'C', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_getblock(format, nbrows-1, 0, order, shape, &buffer);
    block = Array_from_buffer(2, shape, 'd', 'C', buffer);
    Array_SVD(block, &U, &S, &V);
    fd = fopen(heatmap_fname, "a");
    buffer = S->buffer;
    for(i = 0; i < S->size; i++)
        fprintf(fd, "%.12e ", buffer[i]);
    fprintf(fd, "\n");
    fclose(fd);
    Array_free(block);
    Array_free(U);
    Array_free(S);
    Array_free(V);
    STARS_BLR_free(format);
    */
    free(tomo);
    return 0;
}

