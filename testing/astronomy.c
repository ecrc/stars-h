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

int main(){

    STARS_tomo *tomo, *tomo2;
    //TODO put the following variables as parameter
    long nssp=10;
    char files_path[]="./";
    int night_idx=1;
    int snapshots_per_night=1;
    int snapshot_idx=1;
    int obs_idx=0;
    double alphaX=0.0;
    double alphaY=0.0;
    int nact=100;
    Array *matrix, *matrix2;
    int *order;
    int i;

    //matcov_init_tomo_tiled(&tomo, nssp, files_path, night_idx,
    //        snapshots_per_night, snapshot_idx, obs_idx, alphaX, alphaY);
    tomo2 = STARS_gen_aodata(nssp, files_path, night_idx, snapshots_per_night,
            snapshot_idx, obs_idx, alphaX, alphaY, nact);

    //int nmeas = matcov_getNumMeasurements(tomo2);
    //int shape[2] = {nmeas, nmeas};
    //int nmeasts = matcov_getNumMeasurementsTS(&tomo);
    //order = (int *)malloc(nmeas*sizeof(int));
    //for(i = 0; i < nmeas; i++)
    //    order[i] = i;

    //matrix = Array_new(2, shape, 'd', 'F');
    //double *Cmm = matrix->buffer;
    //double *R  =(double*)malloc(nmeas  *nmeasts*sizeof(double));

    //matcov_comp_tile(Cmm,nmeas,nmeas,0,0,nmeas,&tomo,1);
    //matrix2 = block_astronomy_kernel(nmeas, nmeas, order, order, &tomo, &tomo);
    //printf("Difference: %e\n", Array_error(matrix, matrix2)/Array_norm(matrix));

    //matcov_comp_tile(R,nmeas,nmeasts,0,0,nmeasts,&tomo,3);

    //reconstructor(Cmm,R,nmeas,nmeasts,nmeas,nmeasts);

    //double *Cpm=(double*)malloc(nmeas  *nmeasts *sizeof(double));
    //double *Cpp=(double*)malloc(nmeasts*nmeasts*sizeof(double));
    //double *Cee=(double*)malloc(nmeasts*nmeasts*sizeof(double));
    //double *Cvv=(double*)malloc(nact   *nact   *sizeof(double));
    //double *Dx =(double*)malloc(nact   *nmeasts*sizeof(double));

    //matcov_comp_tile(Cmm,nmeas  ,nmeas  ,0,0,nmeas  ,&tomo,1);
    //matcov_comp_tile(Cpm,nmeas  ,nmeasts,0,0,nmeasts,&tomo,3);
    //matcov_comp_tile(Cpp,nmeasts,nmeasts,0,0,nmeasts,&tomo,3);
    //TODO create Dx

    //compute_Cee_Cvv(nmeasts, nmeas,nact, Cmm, Cpp, Cpm, R, Dx, Cee, Cvv);

    //fprintf(stderr,"done\n");

    //free(Cmm);
    //free(Cpm);
    //free(Cpp);
    //free(R);
    //free(Cee);
    //free(Cvv);
    //free(Dx);

    int block_size = 10;
    int maxrank = 0;
    double tol = 1e-3;
    
    /*
    STARS_Problem *problem = (STARS_Problem *)malloc(sizeof(STARS_Problem));
    problem->nrows = 12;
    problem->ncols = problem->nrows;
    problem->symm = 'S';
    problem->dtype = 'd';
    problem->row_data = NULL;
    problem->col_data = NULL;
    problem->kernel = NULL;
    */
    STARS_Problem *problem = STARS_gen_aoproblem(tomo2);
    STARS_BLR *format = STARS_gen_ao_blrformat(problem, block_size);
    STARS_BLRmatrix *mat = STARS_blr__compress_algebraic_svd(format, maxrank, tol, 0);
    STARS_BLRmatrix_free(mat);
    STARS_BLR_free(format);
    //STARS_Problem_free(problem);
    free(tomo2);
    return 0;
}

