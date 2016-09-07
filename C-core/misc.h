double randn();
void dmatrix_print(int m, int n, double *A);
void dmatrix_randn(int m, int n, double *A);
void dmatrix_copy(int m, int n, double *A, double *B);
void dmatrix_qr(int m, int n, double *A, double *Q, double *R);
void dmatrix_lr(int m, int n, double *A, double tol, int *rank, double **U,
        double **S, double **V);
void gen_points(int n, double *points);
