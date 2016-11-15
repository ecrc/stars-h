#ifndef __MATCOV_TILED_H__
#define __MATCOV_TILED_H__

#include<stdio.h>
#include<ctype.h>
#include<stdlib.h>
#include<math.h>

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//code fom original matcov.h with additional fields for matcov_tiled

typedef struct tomo_struct {
    int count; // size of corresponding matrix Cmm
    long Nw;    // number of wavefront sensors

    // pointers on arrays containing corrdinates of sub-apertures
    // X and Y are biiiiig arrays from yorick containing all the subap
    // coordinates of all the WFSs, one after the other.
    double *X;
    double *Y;

    double DiamTel; // telescope Diameter

    double obs;//telescope obscuration

    // array of the number of subap of each WFS, contains Nw elements
    long *Nsubap;

    // array of the number of subap of each WFS along the telescop diameter, contains Nw elements
    long *Nssp;

    // array of the inverse of the guide star altitude (in 1/meters), contains Nw elements
    double *GsAlt;

    // type of WFS, 0, 1, 2 or 3. 0 is unused, 1=NGS, 2=LGS, 3=TipTilt-guide star
    int *type;

    // Pointing directions of WFS
    double *alphaX;                     // pointing direction in X, radian
    double *alphaY;                     // pointing direction in Y, radian

    // Deviations of WFSs
    double *XPup;                         // pupil shift of the WFS, in meters
    double *YPup;                         // pupil shift of the WFS, in meters
    double *thetaML;                    // rotation of microlenses
    double *thetaCam;                 // rotation of camera
    double *sensibilite;            // sensitivity coeff of this WFS
    double *diamPup;                    // magnification factor of this WFS
    double *sspSize;                    // subaperture size of this WFS

    // PROFILE
    long Nlayer;                            // number of layers in the profile
    double r0;                                // r0 at wfs lambda
    double *cn2;                            // profile strengh, units as in    E-SPE-ESO-276-0206_atmosphericparameters
    double *h;                                // altitude of layers (meters)
    double *L0;                             // outer scale (meters)

    double rmax;                            // maximum distance between subapertures (computed with yorick)
    double *tracking;                 // telescope tracking error parameters (x^2, y^2 and xy), units : arcsec^2

    double pasDPHI;                     //Precision of DPHI precomputation.
    int ncpu;                                 //Number of CPU used (only with openMP)
    int part;                                 //Computed part of the cov. matrix. 0: complete 1: cmm 2: cpp 3: cpm ??
    int Nx;
    int Nslopes;
    int nlgs;

    double lgs_cst;
    double noise_var;
    double spot_width;
    double lgs_alt;
    double lgs_depth;

    //Ali
    long     *indexL0;
    double *L0diff;
    double *tabDPHI;
    double *u;
    double *v;
    double *sspSizeL;
    long        nsubaps_offaxis;

    char        files_path[512];
} STARS_aodata;




//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int matcov_init_tomo_tiled(struct tomo_struct *tomo, char* files_path,
        int night_idx, int snapshots_per_night, int snapshot_idx, int obs_idx,
        double alphaX, double alphaY);
void matcov_update_tomo_tiled(struct tomo_struct *tomo);
void matcov_free_tomo_tiled(struct tomo_struct *tomo);

int matcov_getNumMeasurements(struct tomo_struct *tomo);
int matcov_getNumMeasurementsTS(struct tomo_struct *tomo);

void matcov_set_gal_coords(struct tomo_struct *tomo, double alphaX,
        double alphaY);

void matcov_comp_tile(double* data, int nrows, int ncols, int xoffset,
        int yoffset, int lda,
        struct tomo_struct *tomo, int part);

int matcov_update_atm_params(struct tomo_struct *tomo, int night_idx,
        int snapshots_per_night, int snapshot_idx, int obs_idx);

int STARS_aodata_block_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result);
STARS_aodata *STARS_gen_aodata(char *files_path, int night_idx,
        int snapshots_per_night, int snapshot_idx, int obs_idx, double alphaX,
        double alphaY);
void STARS_aodata_free(STARS_aodata *data);
//STARS_Problem *STARS_gen_aoproblem(STARS_tomo *tomo);
//STARS_BLR *STARS_gen_ao_blrformat(STARS_Problem *problem, int block_size);
#endif
