/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_H__
#define __STARSH_H__

// Add definitions for size_t and ssize_t.
#include <sys/types.h>

// Add definition for va_args.
#include <stdarg.h>

// Add definitions for enumerated constants.
#include "starsh-constants.h"


///////////////////////////////////////////////////////////////////////////////
//                                ENVIRONMENT                                //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup environment Environment
 * @brief Routines for environmentally controlled parameters.
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

//! Structure for built-in STARS-H parameters.
struct starsh_params
{
    enum STARSH_BACKEND backend;
    //!< What backend to use (e.g. MPI or STARPU).
    enum STARSH_LRENGINE lrengine;
    //!< What low-rank engine to use (e.g. RSVD).
    int oversample;
    //!< Oversampling parameter for RSVD and RRQR.
};

//! Built-in parameters of STARS-H, accessible through environment.
extern struct starsh_params starsh_params;

int starsh_init();
int starsh_set_backend(const char *string);
int starsh_set_lrengine(const char *string);
int starsh_set_oversample(const char *string);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                                 TYPEDEF                                   //
///////////////////////////////////////////////////////////////////////////////

//! STARSH signed integer to support more, than MAX_INT rows/columns.
typedef ssize_t STARSH_int;

//! Typedef for kernels
//! @ingroup applications
typedef void STARSH_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

//! Typedef for @ref ::array
//! @ingroup array
typedef struct array Array;

//! Typedef for [problem](@ref ::starsh_problem)
//! @ingroup problem
typedef struct starsh_problem STARSH_problem;

//! Typedef for [cluster](@ref ::starsh_cluster)
//! @ingroup cluster
typedef struct starsh_cluster STARSH_cluster;

//! Typedef for [block-wise low-rank format](@ref ::starsh_blrf)
//! @ingroup blrf
typedef struct starsh_blrf STARSH_blrf;

//! Typedef for [block-wise low-rank matrix](@ref ::starsh_blrm)
//! @ingroup blrm
typedef struct starsh_blrm STARSH_blrm;


///////////////////////////////////////////////////////////////////////////////
//                               APPLICATIONS                                //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup applications Applications
 * @brief Set of applications
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

int starsh_application(void **data, STARSH_kernel **kernel, STARSH_int count,
        char dtype, int problem_type, int kernel_type, ...);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                                   ARRAY                                   //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup array Array
 * @brief Routines for n-dimensional arrays
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

struct array
//! N-dimensional array.
/*! Simplifies debugging.
 * */
{
    int ndim;
    //!< Number of dimensions of array.
    int *shape;
    //!< Shape of array.
    ssize_t *stride;
    //!< Strides of array
    char order;
    //!< Ordering of array.
    /*!< `'C'` for C order (row-major), `'F'` for Fortran order
     * (column-major).
     * */
    size_t size;
    //!< Number of elements of array.
    char dtype;
    //!< Precision of array.
    /*!< Possile value is `'s'`, `'d'`, `'c'` or `'z'`, much like
     * in names of **LAPACK** routines.
     * */
    size_t dtype_size;
    //!< Size of one element of array in bytes.
    size_t nbytes;
    //!< Size of data buffer and array structure together in bytes.
    size_t data_nbytes;
    //!< Size of data buffer in bytes.
    void *data;
    //!< Pointer to data buffer.
};

int array_from_buffer(Array **A, int ndim, int *shape, char dtype, char order,
        void *buffer);
int array_new(Array **A, int ndim, int *shape, char dtype, char order);
int array_new_like(Array **A, Array *B);
int array_new_copy(Array **A, Array *B, char order);
void array_free(Array *A);
void array_info(Array *A);
void array_print(Array *A);
int array_to_matrix(Array *A, char kind);
int array_trans_inplace(Array *A);
int array_dot(Array* A, Array *B, Array **C);
int array_SVD(Array *A, Array **U, Array **S, Array **V);
int SVD_get_rank(Array *S, double tol, char type, int *rank);
int array_scale(Array *A, char kind, Array *S);
int array_diff(Array *A, Array *B, double *result);
int array_norm(Array *A, double *result);
int array_convert(Array **A, Array *B, char dtype);
int array_cholesky(Array *A, char uplo);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                                  PROBLEM                                  //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup problem Problem
 * @brief Routines for @ref ::STARSH_problem
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

struct starsh_problem
//! Matrix/tensor in matrix/tensor-free form.
/*! Structure, storing all the necessary data for reconstruction of
 * array, generated by given kernel. This array may be not
 * `2`-dimensional (e.g. for astrophysics problem, where each matrix
 * entry is a vector of `3` elements). Array elements are not stored in
 * memory, but computed on demand. Rows correspond to first dimension
 * of the array and columns correspond to last dimension of the array.
 * */
{
    int ndim;
    //!< Number of dimensions of corresponding problem.
    /*!< Real dimensionality of corresponding array. `ndim=2` for
     * problems with scalar kernel. `ndim=3` for astrophysics problem.
     * Can not be less, than `2`.
     * */
    STARSH_int *shape;
    //!< Shape of corresponding array.
    /*!< In case of non-scalar kernel `shape[0]` stands for number of
     * rows of array, `shape[ndim-1]` stands for number of columns of
     * array and `(ndim-2)`-dimensional tuple `shape[1:ndim-2]` stands
     * for kernel shape.
     * */
    char symm;
    //!< `'S'` if problem is symmetric, and `'N'` otherwise.
    char dtype;
    //!< Precision of problem and corresponding array.
    /*!< Possile value is `'s'`, `'d'`, `'c'` or `'z'`, much like in
     * names of **LAPACK** routines.
     * */
    size_t dtype_size;
    //!< Size of element of array in bytes.
    size_t entry_size;
    //!< Size of subarray, corresponding to kernel shape, in bytes.
    /*!< Corresponds to size of subarray on a single row on a single
     * column. Equal to size of array element, multiplied by total
     * number of elements and divided by number of rows and number of
     * columns.
     * */
    void *row_data;
    //!< Pointer to data, corresponding to rows.
    void *col_data;
    //!< Pointer to data, corresponding to columns.
    STARSH_kernel *kernel;
    //!< Pointer to a kernel.
    /*!< Kernel computes elements of a submatrix on intersection of
     * given rows and columns. Rows stand for first dimension and
     * columns stand for last dimension.
     * */
    char *name;
    //!< Name of corresponding problem.
};

int starsh_problem_new(STARSH_problem **problem, int ndim, STARSH_int *shape,
        char symm, char dtype, void *row_data, void *col_data,
        STARSH_kernel *kernel, char *name);
void starsh_problem_free(STARSH_problem *problem);
void starsh_problem_info(STARSH_problem *problem);
int starsh_problem_get_block(STARSH_problem *problem, int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, Array **A);
int starsh_problem_from_array(STARSH_problem **problem, Array *A, char symm);
int starsh_problem_to_array(STARSH_problem *problem, Array **A);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                                  CLUSTER                                  //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup cluster Cluster
 * @brief Clusterization routines
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

struct starsh_cluster
//! Info about clusterization of physical data.
{
    void *data;
    //!< Pointer to structure, holding physical data.
    STARSH_int ndata;
    //!< Number of discrete elements, corresponding to physical data.
    /*!< Discrete element can be anything, i.e. particles, grid nodes,
     * edges or mesh elements.
     * */
    STARSH_int *pivot;
    //!< Pivoting of discrete elements for clusterization.
    /*!< After pivoting discrete elements of a single cluster are one
     * after another. Used by field `start`.
     * */
    STARSH_int nblocks;
    //!< Total number of subclusters (blocks) of discrete elements.
    STARSH_int nlevels;
    //!< Number of levels of hierarchy.
    /*!< `0` in case of tiled clusterization.
     * */
    STARSH_int *level;
    //!< Index of first cluster for each level of hierarchy.
    /*!< All subclusters, corresponding to given level of hierarchy are
     * stored in compressed format: subclusters with indexes from
     * `level[i]` to `level[i+1]-1` inclusively correspond to `i`-th
     * level of hierarchy. Field `level` is an array with `nlevels+1`
     * elements in hierarchical case and is `NULL` in tiled case.
     * */
    STARSH_int *start;
    //!< Index of first pivoted discrete element of a cluster.
    /*!< Indexes of discrete elements, corresponding to a single
     * cluster, are located one after another in array `pivot`.
     * */
    STARSH_int *size;
    //!< Number of discrete elements in each cluster.
    STARSH_int *parent;
    //!< Parent cluster for each subcluster.
    /*!< Root node is `0` node and it has no parent, so `parent[0] = -1`.
     * */
    STARSH_int *child_start;
    //!< Start index of `child` for each cluster.
    /*!< Clusters with indexes from `child[child_start[i]]` to
     * `child[child_start[i+1]]-1`inclusively are children for a
     * cluster `i`. In case of tiled clusterization `child_start` is
     * `NULL`.
     * */
    STARSH_int *child;
    //!< Children clusters of each cluster.
    /*!< Clusters with indexes from `child[child_start[i]]` to
     * `child[child_start[i+1]]-1`inclusively are children for a
     * cluster `i`. In case of tiled clusterization `child` is
     * `NULL`.
     * */
    enum STARSH_CLUSTER_TYPE type;
    //!< Type of cluster (tiled or hierarchical).
};

int starsh_cluster_new(STARSH_cluster **cluster, void *data, STARSH_int ndata,
        STARSH_int *pivot, STARSH_int nblocks, STARSH_int nlevels,
        STARSH_int *level, STARSH_int *start, STARSH_int *size,
        STARSH_int *parent, STARSH_int *child_start, STARSH_int *child,
        enum STARSH_CLUSTER_TYPE type);
void starsh_cluster_free(STARSH_cluster *cluster);
void starsh_cluster_info(STARSH_cluster *cluster);
int starsh_cluster_new_plain(STARSH_cluster **cluster, void *data,
        STARSH_int ndata, STARSH_int block_size);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                                H-FORMAT                                   //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup blrf H-format
 * @brief Routines to partition matrix into low-rank blocks.
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

struct starsh_blrf
//! Non-nested block-wise low-rank format.
/*! Stores non-nested division of problem into set of admissible pairs of
 * block rows and block columns. For simplicity, such admissible pairs are
 * called admissible blocks, since they stand as submatrix of corresponding
 * matrix. Each admissible block is either far-field or near-field. Each
 * far-field block can be approximated by some low-rank matrix with any given
 * precision, whereas each near-field block is assumed to be dense.
 *
 * Virtually, there is matrix of size total number of block rows by total
 * number of block columns with elements, equal to 0 for non-admissible
 * blocks and 1 for admissible far-field blocks. This is matrix of admissible
 * far-field blocks. Correspondingly, there is matrix of admissible near-field
 * blocks. These matrices are virtual, so we name them as lists of admissible
 * far-field and near-field blocks. These lists are sparse, so they are stored
 * in 3 different sparse formats: COO, CSR and CSC. For example, fields
 * `nblocks_far` and `block_far` represent COO format, fields `brow_far_start`
 * and `brow_far` represent CSR format and fields `bcol_near_start` and
 * `bcol_near` represent CSC format. We use these lists to flatten hierarchical
 * algorithms to get good performance in parallel implementations.
 *
 * Function starsh_blrf_generate() computes CSR and CSC formats out of given
 * COO format.
 *
 * @sa starsh_blrf_generate().
 * */
{
    STARSH_problem *problem;
    //!< Corresponding problem.
    char symm;
    //!< 'S' if format is symmetric and 'N' otherwise.
    STARSH_cluster *row_cluster;
    //!< Clusterization of rows into subclusters (block rows).
    STARSH_cluster *col_cluster;
    //!< Clusterization of columns into subclusters (block columns).
    STARSH_int nbrows;
    //!< Number of block rows (clusters of rows).
    STARSH_int nbcols;
    //!< Number of block columns (clusters of columns).
    STARSH_int nblocks_far;
    //!< Number of admissible far-field blocks.
    STARSH_int *block_far;
    //!< Coordinates of far-field admissible blocks.
    /*!< `block_far[2*i]` is an index of block row (row cluster) and
     * `block_far[2*i+1]` is an index of block column (column cluster).
     * */
    STARSH_int *brow_far_start;
    //!< Start indexes of admissibly far block columns for each block row.
    /*!< Admissibly far blocks for a given block `i` are elements of array
     * `brow_far` from index `brow_far_start[i]` to `brow_far_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int *brow_far;
    //!< List of admissibly far block columns for each block row.
    /*!< Admissibly far blocks for a given block `i` are elements of array
     * `brow_far` from index `brow_far_start[i]` to `brow_far_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int *bcol_far_start;
    //!< Start indexes of admissibly far block rows for each block column.
    /*!< Admissibly far blocks for a given block `i` are elements of array
     * `bcol_far` from index `bcol_far_start[i]` to `bcol_far_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int *bcol_far;
    //!< List of admissibly far block rows for each block column.
    /*!< Admissibly far blocks for a given block `i` are elements of array
     * `bcol_far` from index `bcol_far_start[i]` to `bcol_far_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int nblocks_far_local;
    //!< Number of far-field blocks, stored locally on MPI node.
    STARSH_int *block_far_local;
    //!< List of far-field blocks, stored locally on MPI node.
    STARSH_int nblocks_near;
    //!< Number of admissible near-field blocks.
    /*!< Near-field blocks are dense. */
    STARSH_int *block_near;
    //!< Coordinates of near-field admissible blocks.
    /*!< `block_near[2*i]` is an index of block row (row cluster) and
     * `block_near[2*i+1]` is an index of block column (column cluster).
     * */
    STARSH_int *brow_near_start;
    //!< Start indexes of admissibly near block colums for each block row.
    /*!< Admissibly near blocks for a given block `i` are elements of array
     * `brow_near` from index `brow_near_start[i]` to `brow_near_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int *brow_near;
    //!< List of admissibly near block columns for each block row.
    /*!< Admissibly near blocks for a given block `i` are elements of array
     * `brow_near` from index `brow_near_start[i]` to `brow_near_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int *bcol_near_start;
    //!< Start indexes of admissibly near block rows for each block column.
    /*!< Admissibly near blocks for a given block `i` are elements of array
     * `bcol_near` from index `bcol_near_start[i]` to `bcol_near_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int *bcol_near;
    //!< List of admissibly near block rows for each block column.
    /*!< Admissibly near blocks for a given block `i` are elements of array
     * `bcol_near` from index `bcol_near_start[i]` to `bcol_near_start[i+1]-1`
     * inclusively.
     * */
    STARSH_int nblocks_near_local;
    //!< Number of near-field blocks, stored locally on MPI node.
    STARSH_int *block_near_local;
    //!< List of near-field blocks, stored locally on MPI node.
    enum STARSH_BLRF_TYPE type;
    //!< Type of format.
};

int starsh_blrf_new(STARSH_blrf **format, STARSH_problem *problem, char symm,
        STARSH_cluster *row_cluster, STARSH_cluster *col_cluster,
        STARSH_int nbrows, STARSH_int nbcols, STARSH_int nblocks_far,
        STARSH_int *block_far, STARSH_int *brow_far_start,
        STARSH_int *brow_far, STARSH_int *bcol_far_start,
        STARSH_int *bcol_far, STARSH_int nblocks_far_local,
        STARSH_int *block_far_local, STARSH_int nblocks_near,
        STARSH_int *block_near, STARSH_int *brow_near_start,
        STARSH_int *brow_near, STARSH_int *bcol_near_start,
        STARSH_int *bcol_near, STARSH_int nblocks_near_local,
        STARSH_int *block_near_local, enum STARSH_BLRF_TYPE type);
int starsh_blrf_new_from_coo(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster,
        STARSH_int nblocks_far, STARSH_int *block_far,
        STARSH_int nblocks_near, STARSH_int *block_near,
        enum STARSH_BLRF_TYPE type);
int starsh_blrf_new_tlr(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster);
void starsh_blrf_free(STARSH_blrf *format);
void starsh_blrf_info(STARSH_blrf *format);
void starsh_blrf_print(STARSH_blrf *format);
int starsh_blrf_get_block(STARSH_blrf *format, STARSH_int i, STARSH_int j,
        int *shape, void **D);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                                H-MATRIX                                   //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup blrm H-matrix
 * @brief Block-wise Low-Rank Matrix/Tensor
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

struct starsh_blrm
//! Non-nested block low-rank matrix.
/*! Stores approximation or dense form of each admissible blocks.Division into
 * blocks is defined by corresponding block low-rank format.
 * */
{
    STARSH_blrf *format;
    //!< Pointer to block low-rank format.
    int *far_rank;
    //!< Rank of each far-field block.
    Array **far_U;
    //!< Low rank factor of each far-field block.
    /*!< Multiplication of `far_U[i]` by transposed `far_V[i]` is an
     * approximation of `i`-th far-field block.
     * */
    Array **far_V;
    //!< Low rank factor of each far-field block.
    /*!< Multiplication of `far_U[i]` by transposed `far_V[i]` is an
     * approximation of `i`-th far-field block.
     * */
    int onfly;
    //!< Equal to `1` to store dense blocks, `0` to compute it on demand.
    Array **near_D;
    //!< Array of pointers to dense near-field blocks.
    void *alloc_U;
    //!< Pointer to memory buffer, holding all `far_U`.
    void *alloc_V;
    //!< Pointer to memory buffer, holding all `far_V`.
    void *alloc_D;
    //!< Pointer to memory buffer, holding all `near_D`.
    char alloc_type;
    //!< Type of memory allocation.
    /*!< Equal to `1` if allocating 3 big buffers `U_alloc`, `V_alloc` and
     * `D_alloc`; `2` if allocating many small buffers for each `far_U`,
     * `far_V` and `near_D`.
     * */
    size_t nbytes;
    //!< Total size of block low-rank matrix, including auxiliary buffers.
    size_t data_nbytes;
    //!< Size of low-rank factors and dense blocks in block low-rank matrix.
};

int starsh_blrm_new(STARSH_blrm **matrix, STARSH_blrf *format, int *far_rank,
        Array **far_U, Array **far_V, int onfly, Array **near_D, void *alloc_U,
        void *alloc_V, void *alloc_D, char alloc_type);
void starsh_blrm_free(STARSH_blrm *matrix);
void starsh_blrm_info(STARSH_blrm *matrix);
int starsh_blrm_get_block(STARSH_blrm *matrix, STARSH_int i, STARSH_int j,
        int *shape, int *rank, void **U, void **V, void **D);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                            APPROXIMATIONS                                 //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup approximations Approximation routines
 * @brief Approximation schemes for dense matrices
 * @ingroup blrm
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

//! Typedef for prototype of approximation routine
typedef int STARSH_blrm_approximate(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
//! Approximtion routine, chosen by @ref ::starsh_init().
STARSH_blrm_approximate *starsh_blrm_approximate;

int starsh_blrm__dsdd(STARSH_blrm **matrix, STARSH_blrf *format, int maxrank,
        double tol, int onfly);
int starsh_blrm__drsdd(STARSH_blrm **matrix, STARSH_blrf *format, int maxrank,
        double tol, int onfly);
int starsh_blrm__dqp3(STARSH_blrm **matrix, STARSH_blrf *format, int maxrank,
        double tol, int onfly);
//int starsh_blrm__dna(STARSH_blrm **matrix, STARSH_blrf *format, int maxrank,
//        double tol, int onfly);

int starsh_blrm__dsdd_omp(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__drsdd_omp(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__dqp3_omp(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
//int starsh_blrm__dna_omp(STARSH_blrm **matrix, STARSH_blrf *format,
//        int maxrank, double tol, int onfly);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                  MATRIX-MATRIX MULTIPLICATION                             //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup matmul GEMM
 * @brief H-by-dense matrix multiplication
 * @ingroup blrm
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

int starsh_blrm__dmml(STARSH_blrm *matrix, int nrhs, double alpha,
        double *A, int lda, double beta, double *B, int ldb);
int starsh_blrm__dmml_omp(STARSH_blrm *matrix, int nrhs, double alpha,
        double *A, int lda, double beta, double *B, int ldb);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                   MEASURE APPROXIMATION ERROR                             //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup approximation_error Error of approximation
 * @brief Routines to measure error of approximation
 * @ingroup blrm
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

double starsh_blrm__dfe(STARSH_blrm *matrix);
double starsh_blrm__dfe_omp(STARSH_blrm *matrix);

// This function should not be in this group, but it is for now.
int starsh_blrm__dca(STARSH_blrm *matrix, Array *A);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                  LOW-RANK ROUTINES FOR DENSE                              //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup lrdense Low-rank dense routines
 * @brief Set of low-rank routines for dense matrices
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

int starsh_dense_dsvfr(int size, double *S, double tol);

void starsh_dense_dlrsdd(int nrows, int ncols, double *D, int ldD, double *U,
        int ldU, double *V, int ldV, int *rank, int maxrank, double tol,
        double *work, int lwork, int *iwork);
void starsh_dense_dlrrsdd(int nrows, int ncols, double *D, int ldD, double *U,
        int ldU, double *V, int ldV, int *rank, int maxrank, int oversample,
        double tol, double *work, int lwork, int *iwork);
void starsh_dense_dlrqp3(int nrows, int ncols, double *D, int ldD, double *U,
        int ldU, double *V, int ldV, int *rank, int maxrank, int oversample,
        double tol, double *work, int lwork, int *iwork);
void starsh_dense_dlrna(int nrows, int ncols, double *D, double *U, double *V,
        int *rank, int maxrank, double tol, double *work, int lwork,
        int *iwork);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                            ITERATIVE SOLVERS                              //
///////////////////////////////////////////////////////////////////////////////

/*! @defgroup iter Iterative solvers
 * @brief Set of iterative solvers
 * */
//! @{
// This will automatically include all entities between @{ and @} into group.

int starsh_itersolvers__dcg_omp(STARSH_blrm *matrix, int nrhs, double *B,
        int ldb, double *X, int ldx, double tol, double *work);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                       HEADERS FOR OTHER BACKENDS                          //
///////////////////////////////////////////////////////////////////////////////

// This headers should be at the end, since all Doxygen groups must be defined
// before (and they are defined above).
#include "starsh-mpi.h"
#include "starsh-starpu.h"
#include "starsh-mpi-starpu.h"


#endif // __STARSH_H__

