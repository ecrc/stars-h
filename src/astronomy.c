#include <string.h>
#include "stars.h"
#include "stars-astronomy.h"
//code imported from original matcov.cpp

//#undef DEBUG_MSG

#ifdef DEBUG_MSG
#define FPRINTF fprintf
#else
#define FPRINTF //
#endif

void read_paraml(FILE *file, long *par) {
    char line[128];
    fgets(line, sizeof(line), file);
    //printf("%s", line);
    fgets(line, sizeof(line), file);
    long tmpi;
    if( sscanf(line, "%ld", &tmpi) == 1 ) *par = tmpi;
    //printf("%ld\n", tmpi);
}

void read_parami(FILE *file, int *par) {
    char line[128];
    fgets(line, sizeof(line), file);
    //printf("%s", line);
    fgets(line, sizeof(line), file);
    int tmpi;
    if( sscanf(line, "%d", &tmpi) == 1 ) *par = tmpi;
    //printf("%d\n", tmpi);
}

void read_arrayl(FILE *file, long *arr, int nmax) {
    char line[128];
    char *tail = line;
    int i = 0;
    fgets(line, sizeof(line), file);
    //printf("%s", line);
    fgets(line, sizeof(line), file);
    while((*tail) && (i < nmax)){
        while (isspace (*tail)) tail++;
        arr[i++] = strtol(tail,&tail,10);
    }
    //for(i=0; i<nmax; i++) printf("%ld ",arr[i]);
    //printf("\n");
}

void read_arrayi(FILE *file, int *arr, int nmax) {
    char line[128];
    char *tail = line;
    int i = 0;
    fgets(line, sizeof(line), file);
    //printf("%s", line);
    fgets(line, sizeof(line), file);
    while((*tail) && (i < nmax)){
        while (isspace (*tail)) tail++;
        arr[i++] = (int)strtol(tail,&tail,10);
    }
    //for(i=0; i<nmax; i++) printf("%d ",arr[i]);
    //printf("\n");
}

void read_paramd(FILE *file, double *par) {
    char line[128];
    fgets(line, sizeof(line), file);
    //printf("%s", line);
    fgets(line, sizeof(line), file);
    float tmpf;
    if( sscanf(line, "%f", &tmpf) == 1 ) *par = tmpf;
    //printf("%f\n", tmpf);
}

void read_arrayd(FILE *file, double *arr, int nmax) {
    char line[128];
    char *tail = line;
    int i = 0;
    fgets(line, sizeof(line), file);
    //printf("%s", line);
    fgets(line, sizeof(line), file);
    while((*tail) && (i < nmax)){
        while (isspace (*tail)) tail++;
        arr[i++] = strtod(tail,&tail);
    }
    //for(i=0; i<nmax; i++) printf("%f ",arr[i]);
    //printf("\n");
}

//------------------------------------------------------------------------------------
void generateXY(struct tomo_struct *tomo)
/* DOCUMENT    generateXY(struct tomo_struct tomo, double *Nsubap)
 <tomo>                             :    structure with all the needed information
 <tomo.X> & <tomo.Y>                        :     arrays containing all the sub-apertures coordinates of all WFS, one after the other
<tomo.Nsubap>                            :    number of subaperture of ezach WFS
 Generate the position (X,Y) of each subapertures of each WFS on the telescope pupil and the number of subaperture of ezach WFS (Nsubap) 
 */
{
    const double bornemin = -tomo->DiamTel / 2.;
    const double Rtel2 = (tomo->DiamTel * tomo->DiamTel) / 4.;
    long NsubapTot = 0;
    long n;

    //Total number of subapertures (without obstruction)
    for (n = 0; n < tomo->Nw; n++) {
        NsubapTot += tomo->Nssp[n] * tomo->Nssp[n];
    }

    const long cNsubapTot = NsubapTot;
    double x[cNsubapTot], y[cNsubapTot];
    int index[cNsubapTot];

    int cpt = 0;
    int ioff = 0;

    //Computation of all the subapertures' positions
    for (n = 0; n < tomo->Nw; n++) {
        long Nsap = 0;
        double pas = tomo->DiamTel / (1. * tomo->Nssp[n]);
        int i;
        double Robs2;

        // to avoid some bug that eliminates useful central subapertures when obs=0.286
        if (tomo->Nssp[n] != 7 || (tomo->obs <= 0.285 || tomo->obs >= 0.29)) {
            Robs2 = tomo->DiamTel * tomo->obs / 2. * tomo->DiamTel * tomo->obs / 2.;
        } else {
            Robs2 = tomo->DiamTel * 0.285 / 2. * tomo->DiamTel * 0.285 / 2.;
        }

        if (tomo->Nssp[n] != 1) {
            for (i = 0; i < tomo->Nssp[n]; i++) {
                double tp = bornemin + pas / 2. * (2. * i + 1.); // y-coord of current subap
                int j;

                for (j = 0; j < tomo->Nssp[n]; j++) {
                    x[ioff + j] = bornemin + pas / 2. * (2. * j + 1.); // x-coord of current subap
                    y[ioff + j] = tp;

                    double r2 = x[ioff + j] * x[ioff + j] + y[ioff + j] * y[ioff + j];

                    //Search the non-valid subapertures
                    if (r2 < Robs2 || r2 >= Rtel2) {
                        index[cpt] = j + ioff; //list of the useless subapertures index
                        cpt++;
                    }
                    else {
                        Nsap++;
                    }
                }
                ioff += tomo->Nssp[n];
            }
            tomo->Nsubap[n] = Nsap;
     } else { //Special case (Nssp = 1)
            x[ioff] = 0.; // x-coord of current subap
            y[ioff] = 0.;
            ioff += tomo->Nssp[n];
            tomo->Nsubap[n] = 1;
        }
    }

    tomo->X=(double*)malloc((cNsubapTot-cpt)*sizeof(double));
    tomo->Y=(double*)malloc((cNsubapTot-cpt)*sizeof(double));
    tomo->Nx = cNsubapTot-cpt;
    
    int a = 0;
    int off = 0;
    int borne = 0;
    int i;
    //Suppress the non-valid subapertures
    while (a <= cpt) {

        if (a == cpt) {
            borne = cNsubapTot;
        } else {
            borne = index[a];
        }

        for (i = off; i < borne; i++) {
            tomo->X[i - a] = x[i];
            tomo->Y[i - a] = y[i];
        }

        off = index[a] + 1;
        a++;
    }
}

//------------------------------------------------------------------------------------
int init_tomo_sys(struct tomo_struct *tomo){
    int i;

    char sys_filename[512];
    sprintf(sys_filename, "%ssys-params.txt", tomo->files_path);
    FPRINTF(stdout, "opening file %s ", sys_filename);
    
    FILE *file = fopen(sys_filename, "r");
    if(!file){
        fprintf(stderr, "ERROR: not able to open file %s\n", sys_filename);
        return 0;
    }
    read_paramd(file,&(tomo->DiamTel));
 
    read_paramd(file,&(tomo->obs));

    read_paraml(file,&(tomo->Nw)); // number of wavefront sensors

    // array of the number of subap of each WFS along the telescop diameter, contains Nw elements
    long nssp_tmp;
    read_paraml(file,&nssp_tmp); // number of wavefront sensors
    tomo->Nssp = (long*)malloc(tomo->Nw*sizeof(long));
    //read_arrayl(file,tomo->Nssp,tomo->Nw);

    // array of the number of subap of each WFS, contains Nw elements
    tomo->Nsubap = (long*)malloc(tomo->Nw*sizeof(long));

    // array of the inverse of the guide star altitude (in 1/meters), contains Nw elements
    tomo->GsAlt = (double*)malloc(tomo->Nw*sizeof(double));
    read_arrayd(file,tomo->GsAlt,tomo->Nw);

    // type of WFS, 0, 1, 2 or 3. 0 is unused, 1=NGS, 2=LGS, 3=TipTilt-guide star
    tomo->type = (int*)malloc(tomo->Nw*sizeof(int));
    read_arrayi(file,tomo->type,tomo->Nw);

    read_parami(file,&(tomo->nlgs));

    // Pointing directions of WFS
    tomo->alphaX = (double*)malloc(tomo->Nw*sizeof(double));
    read_arrayd(file,tomo->alphaX,tomo->Nw);
    tomo->alphaY = (double*)malloc(tomo->Nw*sizeof(double));
    read_arrayd(file,tomo->alphaY,tomo->Nw);

    // Deviations of WFSs
    tomo->XPup = (double*)malloc(tomo->Nw*sizeof(double)); // pupil shift of the WFS, in meters
    read_arrayd(file,tomo->XPup,tomo->Nw);
    tomo->YPup = (double*)malloc(tomo->Nw*sizeof(double)); // pupil shift of the WFS, in meters
    read_arrayd(file,tomo->YPup,tomo->Nw);
    tomo->thetaML = (double*)malloc(tomo->Nw*sizeof(double));    // rotation of microlenses
    read_arrayd(file,tomo->thetaML,tomo->Nw);
    tomo->thetaCam = (double*)malloc(tomo->Nw*sizeof(double)); // rotation of microlenses
    read_arrayd(file,tomo->thetaCam,tomo->Nw);
    tomo->sensibilite = (double*)malloc(tomo->Nw*sizeof(double)); // sensitivity coeff of this WFS
    read_arrayd(file,tomo->sensibilite,tomo->Nw);

    tomo->diamPup = (double*)malloc(tomo->Nw*sizeof(double)); 
    tomo->sspSize = (double*)malloc(tomo->Nw*sizeof(double)); // subaperture size of this WFS

    for(i=0; i<tomo->Nw; i++){
        tomo->Nssp[i]        = nssp_tmp; // all wfs have same number of subaps from the parameter file
        tomo->alphaX[i] /= 206265.0; // convert to radian
        tomo->alphaY[i] /= 206265.0; 
        tomo->diamPup[i] =(double)tomo->Nssp[i];
        tomo->sspSize[i] =(double)tomo->DiamTel/tomo->diamPup[i];
    }

    // telescope tracking error parameters (x^2, y^2 and xy), units : arcsec^2
    tomo->tracking = (double*)malloc(3*sizeof(double));
    read_arrayd(file,tomo->tracking,3);

    read_paramd(file,&(tomo->pasDPHI));

    read_parami(file,&(tomo->ncpu));
    /*
            noise stuff
     */

    read_paramd(file,&(tomo->lgs_cst));

    read_paramd(file,&(tomo->noise_var)); 

    read_paramd(file,&(tomo->spot_width)); 

    read_paramd(file,&(tomo->lgs_alt));

    read_paramd(file,&(tomo->lgs_depth));

    //Generate the subapertures positions and fill tomo.Nsubap
    generateXY(tomo);

    int Nslopes = 0;
    for (int i = 0; i < tomo->Nw; i++)
        Nslopes += tomo->Nsubap[i]*2;
    tomo->Nslopes = Nslopes;

    fclose(file);
    //printf("initialized %lu wfs \n", tomo->Nw);
    return 1;
}

//------------------------------------------------------------------------------------
int init_tomo_atm(struct tomo_struct *tomo, int night_idx, int snapshots_per_night, int snapshot_idx, int obs_idx){
    int i;
    char atm_filename[512];
    sprintf(atm_filename, "%sprof%d-atmos-night%d.txt", tomo->files_path, snapshot_idx * snapshots_per_night + obs_idx, night_idx);
    //sprintf(atm_filename, "%sprof%d-atmos-night%d.txt", tomo->files_path, snapshot_idx, night_idx);

    FPRINTF(stdout, "opening file %s ", atm_filename);
    
    FILE *file = fopen(atm_filename, "r");
    if(!file){
        fprintf(stderr, "ERROR: not able to open file %s!\n", atm_filename);
        return 0;
    }
    read_paraml(file,&(tomo->Nlayer)); // number of layers in the profile
    read_paramd(file,&(tomo->r0)); // number of layers in the profile
 
    // PROFILE
    tomo->cn2 = (double*)malloc(tomo->Nlayer*sizeof(double)); // profile strengh, units TBD ..
    read_arrayd(file,tomo->cn2,tomo->Nlayer);
    tomo->h = (double*)malloc(tomo->Nlayer*sizeof(double)); // altitude of layers (meters)
    read_arrayd(file,tomo->h,tomo->Nlayer);
    tomo->L0 = (double*)malloc(tomo->Nlayer*sizeof(double)); // outer scale (meters)
    read_arrayd(file,tomo->L0,tomo->Nlayer);
    /*
    double scn2 = 0.0d;
    for (int cc=0;cc<tomo->Nw;cc++) {
        scn2 += tomo->cn2[cc];
    }
    for (int cc=0;cc<tomo->Nw;cc++) {
        tomo->cn2[cc] /= scn2;
        tomo->cn2[cc] /= pow(tomo->r0,5.0/3.0);
    }
    */
    //rmax = max(abs(tomo.wfs.x,tomo.wfs.y))*2*max(tomo.learn.altitude)/206265.+tomo.tel.diam;
    double dmax = 0.0;
    double maxalt = tomo->h[tomo->Nlayer-1];
    int minssp = tomo->Nssp[0];
    for (int cc=0;cc<tomo->Nw;cc++) {
        double tmp = sqrtf(tomo->alphaX[cc]*tomo->alphaX[cc] + tomo->alphaY[cc]*tomo->alphaY[cc]);
        if (tmp > dmax) dmax = tmp;
        if(tomo->Nssp[cc] < minssp) minssp = tomo->Nssp[cc];
    }
    tomo->rmax = dmax * 2 * maxalt + (1 + 1./minssp) * tomo->DiamTel;

    fclose(file);
    //printf("initialized %lu layers \n", tomo->Nlayer);
    return 1;
}

//------------------------------------------------------------------------------------
void free_tomo(struct tomo_struct *tomo){
    free(tomo->Nssp);

    // array of the number of subap of each WFS, contains Nw elements
    free(tomo->Nsubap);

    // array of the inverse of the guide star altitude (in 1/meters), contains Nw elements
    free(tomo->GsAlt);

    // type of WFS, 0, 1, 2 or 3. 0 is unused, 1=NGS, 2=LGS, 3=TipTilt-guide star
    free(tomo->type);

    free(tomo->alphaX);
    free(tomo->alphaY);

    // Deviations of WFSs
    free(tomo->XPup); // pupil shift of the WFS, in meters
    free(tomo->YPup); // pupil shift of the WFS, in meters
    free(tomo->thetaML);    // rotation of microlenses
    free( tomo->thetaCam); // rotation of microlenses
    free(tomo->sensibilite); // sensitivity coeff of this WFS

    free(tomo->diamPup); 
    free(tomo->sspSize);

    free(tomo->cn2);
    free(tomo->h);
    free(tomo->L0);
    free(tomo->tracking);

    free(tomo->X);
    free(tomo->Y);
}



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//============================================================================================

//equivalent to init_tomo_gpu_gb()
int init_tomo_tile(struct tomo_struct *tomo){
    
    tomo->indexL0 = (long*)malloc(tomo->Nlayer * sizeof(long));
    
    tomo->u = (double*)malloc(tomo->Nlayer * tomo->Nx * sizeof(double));
    tomo->v = (double*)malloc(tomo->Nlayer * tomo->Nx * sizeof(double));
    /*tomo->u = (double*)malloc(tomo->Nlayer * tomo->Nsubap[0] * tomo->Nw * sizeof(double));
    tomo->v = (double*)malloc(tomo->Nlayer * tomo->Nsubap[0] * tomo->Nw * sizeof(double));*/


    tomo->sspSizeL = (double*)malloc(tomo->Nw * tomo->Nlayer * sizeof(double));

    tomo->L0diff = NULL;
    tomo->tabDPHI = NULL;
    return 1;
}

//------------------------------------------------------------------------------------
//equivalent to update_tomo_sys_gpu_gb()
//void update_tomo_sys_tile(struct tomo_struct *tomo){
//
//}
//------------------------------------------------------------------------------------
void free_tomo_tile(struct tomo_struct *tomo){

    free(tomo->u);
    free(tomo->v);
    free(tomo->sspSizeL);
    free(tomo->indexL0);

    if ((tomo->tabDPHI) != NULL)
        free(tomo->tabDPHI);
    if ((tomo->L0diff) != NULL)
        free(tomo->L0diff);
}
//------------------------------------------------------------------------------------
/*!
 * DOCUMENT                 subap_position(tomo, u, v)
     <tomo>                                : structure with all the needed information.
     <u> and <v>                     : 3d arrays containing the sub-apertures projected coordinates onto all the layers. u[0][2][1] is the X-coordinate of the subap 2 of the WFS 0 on the layer 1.

     Computes the projected coordinates of all subapertures    projected onto all the layer
    */
void subap_position(struct tomo_struct *tomo) {

    FPRINTF(stdout, "starting subap_position...");fflush(stdout);
    long i, tid;
    long n = 0;
    long l;
    const double rad = 3.14159265358979323846 / 180.;
    
    const long int cNsubap0 = tomo->Nsubap[0];
    const long int cNw = tomo->Nw;

    long ioff[tomo->Nw];
    ioff[0] = 0;
    for (i = 1; i < tomo->Nw; i++) {
        ioff[i] = ioff[i-1] + tomo->Nsubap[i-1];
    }
    
    for (tid = 0; tid < tomo->Nlayer * tomo->Nx; tid++) {

        n = 0;
        l = tid / tomo->Nx;
        const int pos = tid - l * tomo->Nx;
        long Nsubapx = tomo->Nsubap[0];

        while(pos >= Nsubapx){
            n++;
            Nsubapx += tomo->Nsubap[n];
        }
        Nsubapx -= tomo->Nsubap[n];

        i = pos - Nsubapx;
        
        const double dX = tomo->alphaX[n] * tomo->h[l];
        const double dY = tomo->alphaY[n] * tomo->h[l];

        const double rr = 1. - tomo->h[l] * tomo->GsAlt[n];

        //const long Nsap = tomo->Nsubap[n];
        const long nssp = tomo->Nssp[n];

        //magnification factor
        const double G = tomo->diamPup[n] / (double) (nssp);

        //rotation angle
        const double th = tomo->thetaML[n] * rad;

        //taking magnification factor into account
        const double xtp = tomo->X[ioff[n] + i] * G;
        const double ytp = tomo->Y[ioff[n] + i] * G;

        //taking rotation into account
        double uu = xtp * cos(th) - ytp * sin(th);
        double vv = xtp * sin(th) + ytp * cos(th);

        //taking pupil offset into account
        uu += tomo->XPup[n];
        vv += tomo->YPup[n];

        //Projection onto    the layer
        tomo->u[tid] = uu * rr + dX;
        tomo->v[tid] = vv * rr + dY;
    }
    /*const long int cNsubap0 = tomo->Nsubap[0];
    const long int cNw = tomo->Nw;

    for (l = 0; l < tomo->Nlayer; l++) {
        long ioff = 0;

        for (n = 0; n < cNw; n++) {

            const double dX = tomo->alphaX[n] * tomo->h[l];
            const double dY = tomo->alphaY[n] * tomo->h[l];

            const double rr = 1. - tomo->h[l] * tomo->GsAlt[n];

            const long Nsap = tomo->Nsubap[n];
            const long nssp = tomo->Nssp[n];

            //magnification factor
            const double G = tomo->diamPup[n] / (double) (nssp);

            //rotation angle
            const double th = tomo->thetaML[n] * rad;

            for (i = 0; i < Nsap; i++) {
                //taking magnification factor into account
                const double xtp = tomo->X[ioff + i] * G;
                const double ytp = tomo->Y[ioff + i] * G;

                //taking rotation into account
                double uu = xtp * cos(th) - ytp * sin(th);
                double vv = xtp * sin(th) + ytp * cos(th);

                //taking pupil offset into account
                uu += tomo->XPup[n];
                vv += tomo->YPup[n];

                //Projection onto    the layer
                tomo->u[l * cNw * cNsubap0 + i * cNw + n] = uu * rr + dX;
                tomo->v[l * cNw * cNsubap0 + i * cNw + n] = vv * rr + dY;
            }
            //index offset
            ioff += Nsap;
        }
    }*/
    FPRINTF(stdout, "done...\n");fflush(stdout);
}

double
macdo_x56(double x, int k)
/* DOCUMENT    macdo_x56(x)

 Computation of the function
 f(x) = x^(5/6)*K_{5/6}(x)
 using a series for the esimation of K_{5/6}, taken from Rod Conan thesis :
 K_a(x)=1/2 \sum_{n=0}^\infty \frac{(-1)^n}{n!}
 \left(\Gamma(-n-a) (x/2)^{2n+a} + \Gamma(-n+a) (x/2)^{2n-a} \right) ,
 with a = 5/6.

 Setting x22 = (x/2)^2, setting uda = (1/2)^a, and multiplying by x^a,
 this becomes :
 x^a * Ka(x) = 0.5 $ -1^n / n! [ G(-n-a).uda x22^(n+a) + G(-n+a)/uda x22^n ]
 Then we use the following recurrence formulae on the following quantities :
 G(-(n+1)-a) = G(-n-a) / -a-n-1
 G(-(n+1)+a) = G(-n+a) /    a-n-1
 (n+1)! = n! * (n+1)
 x22^(n+1) = x22^n * x22
 and at each iteration on n, one will use the values already computed at step (n-1).
 The values of G(a) and G(-a) are hardcoded instead of being computed.

 The first term of the series has also been skipped, as it
 vanishes with another term in the expression of Dphi.

 SEE ALSO:
 */
{
    const double a = 5. / 6.;
    const double x2a = pow(x, (double)2. * a), x22 = x * x / 4.;
    double x2n;                             // x^2.a, etc
    double s = 0.0;
    int n;

    const double Ga[11] = { 0, 12.067619015983075, 5.17183672113560444,
            0.795667187867016068, 0.0628158306210802181, 0.00301515986981185091,
            9.72632216068338833e-05, 2.25320204494595251e-06, 3.93000356676612095e-08,
            5.34694362825451923e-10, 5.83302941264329804e-12 };

    const double Gma[11] = { -3.74878707653729304, -2.04479295083852408,
            -0.360845814853857083, -0.0313778969438136685, -0.001622994669507603,
            -5.56455315259749673e-05, -1.35720808599938951e-06,
            -2.47515152461894642e-08, -3.50257291219662472e-10,
            -3.95770950530691961e-12, -3.65327031259100284e-14 };

    x2n = 0.5;                                                     // init (1/2) * x^0

    s = Gma[0] * x2a;
    s *= x2n;

    // prepare recurrence iteration for next step
    x2n *= x22;        // x^n

    for (n = 1; n <= 10; n++) {

        s += (Gma[n] * x2a + Ga[n]) * x2n;
        // prepare recurrence iteration for next step
        x2n *= x22;        // x^n
    }
    return s;
}

double
asymp_macdo(double x)
/* DOCUMENT asymp_macdo(x)

 Computes a term involved in the computation of the phase struct
 function with a finite outer scale according to the Von-Karman
 model. The term involves the MacDonald function (modified bessel
 function of second kind) K_{5/6}(x), and the algorithm uses the
 asymptotic form for x ~ infinity.
 Warnings :
 - This function makes a doubleing point interrupt for x=0
 and should not be used in this case.
 - Works only for x>0.

 SEE ALSO:
 */
{
    // k2 is the value for
    // gamma_R(5./6)*2^(-1./6)
    const double k2 = 1.00563491799858928388289314170833;
    const double k3 = 1.25331413731550012081;     //    sqrt(pi/2)
    const double a1 = 0.22222222222222222222;     //    2/9
    const double a2 = -0.08641975308641974829;    //    -7/89
    const double a3 = 0.08001828989483310284;     // 175/2187
    double res;
    double x_1;

    x_1 = 1. / x;
    res = k2
            - k3 * exp(-x) * pow(x, (double)(1 / 3.))
                    * (1.0 + x_1 * (a1 + x_1 * (a2 + x_1 * a3)));
    return res;
}

double
rodconan(double r, double L0, int k)
/* DOCUMENT rodconan(r,L0,k=)
 The phase structure function is computed from the expression
 Dphi(r) = k1    * L0^(5./3) * (k2 - (2.pi.r/L0)^5/6 K_{5/6}(2.pi.r/L0))

 For small r, the expression is computed from a development of
 K_5/6 near 0. The value of k2 is not used, as this same value
 appears in the series and cancels with k2.
 For large r, the expression is taken from an asymptotic form.

 SEE ALSO:
 */
{

    const double pi = 3.1415926535897932384626433;
    double res = 0;

    // k1 is the value of :
    // 2*gamma_R(11./6)*2^(-5./6)*pi^(-8./3)*(24*gamma_R(6./5)/5.)^(5./6);
    const double k1 = 0.1716613621245709486;
    const double dprf0 = (2 * pi / L0) * r;
    // k2 is the value for gamma_R(5./6)*2^(-1./6),
    // but is now unused
    // k2 = 1.0056349179985892838;

    // Xlim = 0.75*2*pi;     // = 4.71239
    if (dprf0 > 4.71239) {
        res = asymp_macdo(dprf0);
    } else {
        res = -macdo_x56(dprf0, k);
    }
    res *= k1 * pow(L0, (double)5. / (double)3.0);
    return res;
}

//============================================================================================
    int matcov_init_tomo_tiled(struct tomo_struct *tomo, char* files_path, int night_idx, int snapshots_per_night, int snapshot_idx, int obs_idx, double alphaX, double alphaY){
        
        FPRINTF(stdout, "initializing matcov...");fflush(stdout);

        strcpy( tomo->files_path, files_path);
        //initialize tomo struct on cpu
        if(!init_tomo_sys(tomo)) return 0;
        tomo->alphaX[tomo->Nw-1] = alphaX / 206265.0; // convert to radian;
        tomo->alphaY[tomo->Nw-1] = alphaY / 206265.0; // convert to radian;
        if(!init_tomo_atm(tomo, night_idx, snapshots_per_night, snapshot_idx, obs_idx)) return 0;
                                    
        /*long nsubaps_offaxis;             // number of sub-aperture of offaxis WFSs
        nsubaps_offaxis = 0;
        for (int i = 0; i < tomo->Nw-1; i++)
            nsubaps_offaxis += tomo->Nsubap[i];
        tomo->nsubaps_offaxis = nsubaps_offaxis;*/
        
        //initialize tomo struct tile data
        if(!init_tomo_tile(tomo)) return 0;
        
        //if(!update_tomo_sys_tile(tomo)) return 0;
        matcov_update_tomo_tiled(tomo);
        FPRINTF(stdout, "done...\n");fflush(stdout);
        
        return 1;
    }
    //------------------------------------------------------------------------------------
    //equivalent to update_tomo_atm_gpu_gb()
    void matcov_update_tomo_tiled(struct tomo_struct *tomo){

        FPRINTF(stdout, "starting matcov_update_tomo_tiled...");fflush(stdout);
        const double crmax = tomo->rmax;
        const double pasDPHI = 1./tomo->pasDPHI; //inverse du pas de rr
        const long Ndphi = floor(crmax*pasDPHI)+1;
        const double convert = (double)(Ndphi-1)/(crmax+1./pasDPHI);
        
        FPRINTF(stdout, "starting cc loop...");fflush(stdout);
        for (int cc = 0; cc < tomo->Nw * tomo->Nlayer; cc++) {
            int n = cc / tomo->Nlayer;
            int l = cc - n * tomo->Nlayer;
            if(n >= tomo->Nw) n-=1;
            tomo->sspSizeL[cc] = tomo->sspSize[n] * (1. - tomo->GsAlt[n] * tomo->h[l]);
        }
        FPRINTF(stdout, "done...\n");fflush(stdout);

        //*            
        FPRINTF(stdout, "starting tmp loop...");fflush(stdout);
        //Search the different L0 and build indexL0
        const long cNlayer = tomo->Nlayer;
        long i, j;
        int cpt = 1;
        double tmp[cNlayer];

        tmp[0] = tomo->L0[0];
        tomo->indexL0[0] = 0;
        
        for (i = 1; i < cNlayer; i++) {
            j = 0;
            const double l0 = tomo->L0[i];
            
            while ((j < cpt) && (tmp[j] != l0)) {j++;}
            
            tomo->indexL0[i] = j;
            
            if (j == cpt) {
                tmp[j] = l0;
                cpt++;
            }
        }
        FPRINTF(stdout, "cpt: %d, Ndphi %d...", cpt, Ndphi);fflush(stdout);
        FPRINTF(stdout, "done...\n");fflush(stdout);
        
        FPRINTF(stdout, "starting L0diff loop...");fflush(stdout);
        // allocate space for L0
        if ((tomo->L0diff) != NULL){free(tomo->L0diff);}
        tomo->L0diff = (double*)malloc(cNlayer*sizeof(double));
        //tomo->L0diff = (double*)malloc(cpt*sizeof(double));

        //for (i = 0; i < cpt; i++)
            //tomo->L0diff[i] = tmp[i];
        for (i = 0; i < cNlayer; i++)    {
            tomo->L0diff[i] = tomo->L0[i];
        }
        FPRINTF(stdout, "done...\n");fflush(stdout);
        
        FPRINTF(stdout, "starting tabDPHI loop...");fflush(stdout);
        //précalcul de DPHI : que pour chaque différent L0
        if ((tomo->tabDPHI) != NULL){free(tomo->tabDPHI);}
        FPRINTF(stdout, "allocating tabDPHI of size %d...", 2);fflush(stdout);
        tomo->tabDPHI = (double*)malloc(2*sizeof(double));
        /*FPRINTF(stdout, "allocating tabDPHI of size %d...", cpt*Ndphi);fflush(stdout);
        tomo->tabDPHI = (double*)malloc(cpt*Ndphi*sizeof(double));

        for (int l = 0; l < cpt; l++) {
            for (j = 0; j < Ndphi; j++) {
                tomo->tabDPHI[l*Ndphi+j] = rodconan((double)j / convert, tomo->L0diff[l], 10);
            }
        }*/
        FPRINTF(stdout, "done...\n");fflush(stdout);
        //
        //Computes    u and v
        subap_position(tomo);
        FPRINTF(stdout, "done...\n");fflush(stdout);
    }
    //------------------------------------------------------------------------------------
    void matcov_free_tomo_tiled(struct tomo_struct *tomo){

        free_tomo_tile(tomo);
        free_tomo(tomo);
    }
    //------------------------------------------------------------------------------------
    int matcov_getNumMeasurements(struct tomo_struct *tomo){
        return tomo->Nx * 2;
        //return tomo->nsubaps_offaxis * 2;
    }
    //------------------------------------------------------------------------------------
    int matcov_getNumMeasurementsTS(struct tomo_struct *tomo){
        return tomo->Nsubap[tomo->Nw-1] * 2;
    }
    //------------------------------------------------------------------------------------
    void matcov_set_gal_coords(struct tomo_struct *tomo, double alphaX, double alphaY){
        FPRINTF(stdout, "setting matcov galaxy params...");fflush(stdout);
        tomo->alphaX[tomo->Nw-1] = alphaX/ 206265.0; // convert to radian;
        tomo->alphaY[tomo->Nw-1] = alphaY/ 206265.0; // convert to radian;
        
        double dmax = 0.0;
        double maxalt = tomo->h[tomo->Nlayer-1];
        for (int cc=0;cc<tomo->Nw;cc++) {
            double tmp = sqrtf(tomo->alphaX[cc]*tomo->alphaX[cc] + tomo->alphaY[cc]*tomo->alphaY[cc]);
            if (tmp > dmax) dmax = tmp;
        }
        tomo->rmax = dmax * 2 * maxalt + tomo->DiamTel;
        
        //subap_position(tomo);
        matcov_update_tomo_tiled(tomo);
        FPRINTF(stdout, "done...\n");fflush(stdout);
    }
    //------------------------------------------------------------------------------------
    int matcov_update_atm_params(struct tomo_struct *tomo, int night_idx, int snapshots_per_night, int snapshot_idx, int obs_idx){
        FPRINTF(stdout, "updating matcov atm-params...");fflush(stdout);

        long int Nlayer;
        char atm_filename[512];
        //sprintf(atm_filename, "%sprof%d-atmos-night%d.txt", tomo->files_path, snapshot_idx, night_idx);
        sprintf(atm_filename, "%sprof%d-atmos-night%d.txt", tomo->files_path, snapshot_idx * snapshots_per_night + obs_idx, night_idx);

        FPRINTF(stdout, "opening file %s ", atm_filename);
        
        FILE *file = fopen(atm_filename, "r");
        if(!file){
            fprintf(stderr, "ERROR: not able to open file %s!\n", atm_filename);
            return 0;
        }
            

        read_paraml(file,&Nlayer); // number of layers in the profile
        if(Nlayer != tomo->Nlayer){
            fprintf(stderr, "ERROR: case not supported yet, Nlayer is supposed to be the same for across simulation \n");
            return 0;
        }
        read_paramd(file,&(tomo->r0)); 
        
        read_arrayd(file,tomo->cn2,tomo->Nlayer);
        
        fclose(file);
        FPRINTF(stdout, "done...\n");fflush(stdout);

        return 1;
    }

//============================================================================================
/*!
 * dphi = DPHI(x,y,indexL0,rr,tabDPHI,convert) * r0^(-5./3)
 <x> & <y>                 :    separation between apertures
 <indexL0>                 :    index for the L0 taken into account
 <rr>                            :    array of distance between apertures
 <tabDPHI>                 :    array of precomputed DPHI
 <convert>                 :    relation between the index on tabDPHI and (x,y)

 Computes the phase structure function for a separation (x,y).
 The r0 is not taken into account : the final result of DPHI(x,y,L0)
 has to be scaled with r0^-5/3, with r0 expressed in meters, to get
 the right value.

 SEE ALSO:
 */
//
double DPHI_gb(double x, double y, double indexL0, double *tabDPHI, double convert, int Ndphi)
{
    double r = sqrt(x * x + y * y);
    
    return rodconan(r, indexL0, 10);
    /*long i0 = (long) (r * convert);
    long i1 = i0 + 1;

    return ((r - (double)i0 / convert) * tabDPHI[indexL0 * Ndphi + i1]
                    + ((double)i1 / convert - r) * tabDPHI[indexL0 * Ndphi + i0]);*/
            
}
/*

double DPHI_gb(double x, double y, long indexL0, double *tabDPHI, double convert, int Ndphi)
{
    return 6.88*pow( x * x + y * y , (double)5./6. );
    //
    //double r = sqrt(x * x + y * y);
    //return ((r - (double)i0 / convert) * 6.88 * pow((double)i1/convert,(double)5./6.)
    //                + ((double)i1 / convert - r) * 6.88 * pow((double)i0/convert,(double)5./6.));
    //
}*/

//------------------------------------------------------------------------------------
/*!
 *    Compute the XX-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
 */
double cov_XX_gb(double du, double dv, double ac, double ad, double bc, double bd, double *tabDPHI, long indexL0, double convert, int Ndphi)

{
    return -DPHI_gb(du + ac, dv, indexL0, tabDPHI, convert, Ndphi)
        + DPHI_gb(du + ad, dv, indexL0, tabDPHI, convert, Ndphi)
        + DPHI_gb(du + bc, dv, indexL0, tabDPHI, convert, Ndphi)
        - DPHI_gb(du + bd, dv, indexL0, tabDPHI, convert, Ndphi);
}

//------------------------------------------------------------------------------------
/*!
 *    Compute the YY-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
 */
double cov_YY_gb(double du, double dv, double ac, double ad, double bc, double bd, double *tabDPHI, long indexL0, double convert, int Ndphi)
{ 
    return    -DPHI_gb(du, dv + ac, indexL0, tabDPHI, convert, Ndphi)
        + DPHI_gb(du, dv + ad, indexL0, tabDPHI, convert, Ndphi)
        + DPHI_gb(du, dv + bc, indexL0, tabDPHI, convert, Ndphi)
        - DPHI_gb(du, dv + bd, indexL0, tabDPHI, convert, Ndphi);
}

//------------------------------------------------------------------------------------
/*!
 * Compute the XY-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
 */
double cov_XY_gb(double du, double dv, double s0, double *tabDPHI, long indexL0, double convert, int Ndphi)
{
    return -DPHI_gb(du + s0, dv - s0, indexL0, tabDPHI, convert, Ndphi)
        + DPHI_gb(du + s0, dv + s0, indexL0, tabDPHI, convert, Ndphi)
        + DPHI_gb(du - s0, dv - s0, indexL0, tabDPHI, convert, Ndphi)
        - DPHI_gb(du - s0, dv + s0, indexL0, tabDPHI, convert, Ndphi);
}

//------------------------------------------------------------------------------------
/*! Covariance matrix per-element generation ***
*     Arguments
*     =========
*             ipos:                     Integer: global x-coordinate of the element w.r.t. the entire matrix
*             jpos:                     Integer: global y-coordinate of the element w.r.t. the entire matrix
*/
double compute_element_tiled_4(
        int ipos, int jpos, double convert, double *sspSizeL, long *Nssp, double *u, double *v, double *X, double *Y,
        double pasDPHI, double *tabDPHI, double *indexL0, double *cn2, int Ndphi, int Nw, int Nlayer,
        /*int Nsubap*/long * Nsubap_wfs, long Nx, double *alphaX, double *alphaY, double lgs_cst, double noise_var, double spotWidth,
        double dH_lgs, double alt_lgs, int type_mat, int nlgs, double teldiam)
{
    
    const double lambda2 = 0.00026942094446267851;
    //WFS m
    //int m = ipos / (2 * Nsubap);
    long Nsubapx = Nsubap_wfs[0];
    int m = 0;
    if (type_mat == 3){
             m = Nw-1;
             Nsubapx=Nx;
             ipos+=2*(Nx-Nsubap_wfs[m]);
    }
    else{
            while((ipos / (2 * Nsubapx)) >= 1){
                                    m++;
                                    Nsubapx += Nsubap_wfs[m];
        }
    }
    Nsubapx -= Nsubap_wfs[m];
    
    //WFS n
    //int n = jpos / (2 * Nsubap);
    long Nsubapy = Nsubap_wfs[0];
    int n = 0;
    if (type_mat == 2){
            n = Nw-1;
             Nsubapy=Nx;
             jpos+=2*(Nx-Nsubap_wfs[m]);
    }
    else{
            while((jpos / (2 * Nsubapy)) >= 1){
                n++;
                Nsubapy += Nsubap_wfs[n];
            }
    }
    Nsubapy -= Nsubap_wfs[n];

    //subap i
    //int i = ipos % (2 * Nsubap); 
    int i = ipos - 2 * Nsubapx;
    //subap j
    //int j = jpos % (2 * Nsubap);
    int j = jpos - 2 * Nsubapy;
    //xy i
    int xy_i;
    //xy j
    int xy_j;
    /*if (i>=Nsubap) {
        i-= Nsubap;
        xy_i = 1;
    } else xy_i = 0;
    if (j>=Nsubap) {
        j-= Nsubap;
        xy_j = 1;
    } else xy_j = 0;*/
    if (i>=Nsubap_wfs[m]) {
        i-= Nsubap_wfs[m];
        xy_i = 1;
    } else xy_i = 0;
    if (j>=Nsubap_wfs[n]) {
        j-= Nsubap_wfs[n];
        xy_j = 1;
    } else xy_j = 0;

    const double sspSizem = teldiam / Nssp[m];
    const double sspSizen = teldiam / Nssp[n];
    
    const double kk = lambda2 / (sspSizem * sspSizen);
        
    int type = xy_i * 2 + xy_j;

    //Layer l
    double covar = 0.0;
    for (int l = 0; l < Nlayer; l++) 
    {
        double sspSizeml = sspSizeL[m * Nlayer + l];
        double sspSizenl = sspSizeL[n * Nlayer + l];
        //test if the altitude layers is not higher than the LGS altitude
        if ((sspSizeml > 0) && (sspSizenl > 0)) 
        {
            /*int pos1 = m + i * Nw + l * Nw * Nsubap;
            int pos2 = n + j * Nw + l * Nw * Nsubap;*/
            int pos1 = i + Nsubapx + l * Nx;
            int pos2 = j + Nsubapy + l * Nx;
            //if(threadIdx.x == 6 && threadIdx.y == 0 && blockIdx.x == 6 && blockIdx.y == 1)
            //if((pos1 >= 6840) || (pos2 >= 6839))
            //{
            //                printf("================ pos1 = %d, pos2 = %d \n", pos1, pos2);
            //}
            //(6,0,0) in block (0,2,0);
            double du =    u[pos1] - u[pos2];                 
            double dv =    v[pos1] - v[pos2];
            
            double s1 = sspSizeml * 0.5;
            double s2 = sspSizenl * 0.5;
            
            double ac = s1 - s2;
            double ad = s1 + s2;
            double bc = -ad;     // initially -s1-s2;
            double bd = -ac;     // initially -s1+s2;

            if (type == 0) covar += 0.5 /* pasDPHI*/ * cov_XX_gb(du,dv,ac,ad,bc,bd,tabDPHI,indexL0[l],convert,Ndphi) * kk * cn2[l];
            else if (type == 3) covar += 0.5 /* pasDPHI*/ * cov_YY_gb(du,dv,ac,ad,bc,bd,tabDPHI,indexL0[l],convert,Ndphi) * kk * cn2[l];
            else //if ((type == 1) || (type == 2)) 
            {
                double s0 = sqrt(s1 * s1 + s2 * s2); //half size of the subaperture equivalent to a convolution by s1 and s2
                double dd = (s1 > s2) ? 1. - s2 / s1 : 1. - s1 / s2; // Nono's style ....
                covar += 0.25 /* pasDPHI*/ * cov_XY_gb(du,dv,s0,tabDPHI,indexL0[l],convert,Ndphi) * kk * cn2[l] * (1. - dd * dd);
            }
        }
    }

    // adding noise

    if (m == n) {
        if (m < nlgs) {
            if (i == j) {
                // lgs case
                //const int pos1 = m + i * Nw;
                const int pos1 = i + Nsubapx;
                double x = X[pos1];                     
                double y = Y[pos1];
                double xwfs = alphaX[m] * 206265;                         
                double ywfs = alphaY[m] * 206265;
                double lltx = 0;                            
                double llty = 0;
                const double lltnorm = sqrtf(xwfs*xwfs + ywfs*ywfs);
                if (lltnorm != 0) {
                    lltx = xwfs / lltnorm * teldiam / 2.0;
                    llty = ywfs / lltnorm * teldiam / 2.0;
                }
                x -= lltx;
                y -= llty;
                x    = 206265. * dH_lgs * x / alt_lgs / alt_lgs;     // extension at Fwhm, in arcsec
                y    = 206265. * dH_lgs * y / alt_lgs / alt_lgs;     // extension at Fwhm, in arcsec
                double lgsExt = sqrtf(x * x + y * y);     // lengh of the extension
                double lgsTheta = x != 0 ? atan2( y , x) : 0.0;     // angle of extension
                double totalExt = sqrtf( lgsExt *    lgsExt + spotWidth * spotWidth); 
                // lengh of the extension including seeing, laser size, ...
                double ratio = totalExt / spotWidth;
                double noiseLongAxis = noise_var * ratio * ratio;
                if (type == 0) covar += noiseLongAxis * cos(lgsTheta) * cos(lgsTheta) + 
                                                 noise_var * sin(lgsTheta) * sin(lgsTheta);
                else if (type == 3) covar += noiseLongAxis * sin(lgsTheta) * sin(lgsTheta) + 
                                                            noise_var * cos(lgsTheta) * cos(lgsTheta);
                else covar += (noiseLongAxis-noise_var) * sin(lgsTheta) * cos(lgsTheta);
                //if ((type == 0) || (type == 3))
                //    covar += lgs_cst;
            }
            if ((type == 0) || (type == 3))
                covar += lgs_cst;
        } else {
        // ngs case
            if (i==j) {
                if ((type == 0) || (type == 3)) {
                    covar += noise_var;
                }
            }
        }
    }

    return (double)covar; 
}


//============================================================================================
/*!
 * covariance matrix generation kernel ***
 *        The kernel generates the element values in a given matrix/submatrix
 *     The generation function can be any function, as long as each element
 *     can be computed both individually and independently
 *
 *        see argument description in the kernel driver
 */
void matcov_kernel_4(
    char uplo, char copy, double* data, int nrows, int ncols, int xoffset, int yoffset, int lda,
    double convert, double *sspSizeL, long *Nssp, double *u, double *v,double *X, double *Y,
    double pasDPHI, double *tabDPHI, double *indexL0, double *cn2, int Ndphi, int Nw, int Nlayer,
    /*int Nsubap*/long *Nsubap, long Nx, double *alphaX, double *alphaY, double lgs_cst, double noise_var, double spotWidth,
    double dH_lgs, double alt_lgs, int type_mat, int nlgs, double teldiam, int lx, int ly)
{
                
    /*/ local thread coordinates w.r.t. thread block
    const int tx_ = threadIdx.x;
    const int ty_ = threadIdx.y;
                
    // local thread block coordinates w.r.t. kernel grid
    const int bx_ = blockIdx.x;
    const int by_ = blockIdx.y;
                
    // local coordinates of the element w.r.t. submatrix
    int lx = bx_ * blockDim.x + tx_;
    int ly = by_ * blockDim.y + ty_;*/
                
    // global coordinates of the elemnt w.r.t. the entire matrix
    int gx = lx + xoffset;
    int gy = ly + yoffset;
                
    // out-of-bound threads should terminate
    //if( (lx >= nrows) || (ly >= ncols) ) return;
                
    // Advance the data pointer accordingly
    //data += ly * lda + lx;
                
    double value;
    if(uplo == 'l')
    {
        if(gy <= gx)
        {
            value = compute_element_tiled_4(
                                    gx, gy, convert, sspSizeL, Nssp, u, v,X,Y, pasDPHI, tabDPHI, indexL0, cn2,
                                    Ndphi, Nw, Nlayer, Nsubap, Nx, alphaX, alphaY, lgs_cst, noise_var, spotWidth,
                                    dH_lgs, alt_lgs, type_mat, nlgs, teldiam);
            data[lx * lda + ly] = value;
            if(copy == 'c') data[ly * lda + lx] = value;
        }
    }
    else if (uplo == 'u') // upper
    {
        if(gx <= gy)
        {
            value = compute_element_tiled_4(
                                    gx, gy, convert, sspSizeL, Nssp, u, v,X,Y, pasDPHI, tabDPHI, indexL0, cn2,
                                    Ndphi, Nw, Nlayer, Nsubap, Nx, alphaX, alphaY, lgs_cst, noise_var, spotWidth,
                                    dH_lgs, alt_lgs, type_mat, nlgs, teldiam);
            data[lx * lda + ly] = value;
            if(copy == 'c') data[ly * lda + lx] = value;
        }
    }
    else    // uplo = 'f' full generation
    {
        value = compute_element_tiled_4(
                                    gx, gy, convert, sspSizeL, Nssp, u, v,X,Y, pasDPHI, tabDPHI, indexL0, cn2,
                                    Ndphi, Nw, Nlayer, Nsubap, Nx, alphaX, alphaY, lgs_cst, noise_var, spotWidth,
                                    dH_lgs, alt_lgs, type_mat, nlgs, teldiam);
        data[lx * lda + ly] = value;
    }
}


//============================================================================================
/*!
 * ** Covariance matrix per-element generation ***
 *     Arguments
 *     =========
 *             ipos:                     Integer: global x-coordinate of the element w.r.t. the entire matrix
 *             jpos:                     Integer: global y-coordinate of the element w.r.t. the entire matrix
 */
double compute_element_ts_tile(
    int ipos, int jpos, double convert, double *X, double *Y,
    long *Nssp, double pasDPHI, double *tabDPHI, double *L0diff, double *cn2, 
    int Ndphi, int Nw, int Nlayer, int Nsubap, double teldiam)
{
    //(lambda/(2*pi)*rasc)**2    with lambda=0.5e-6 m
    //TODO generalize lambda
    double lambda2 = 0.00026942094446267851;
    //WFS Nw-1
     //subap i
    int i = ipos < Nsubap ? ipos : ipos - Nsubap;
    //subap j
    int j = jpos < Nsubap ? jpos : jpos - Nsubap;
    //xy i
    int xy_i = ipos < Nsubap ? 0 : 1;
    //xy j
    int xy_j = jpos < Nsubap ? 0 : 1;
    
    double sspSize = teldiam / Nssp[Nw-1];
    
    double kk = lambda2 / (sspSize * sspSize);
        
    int type = xy_i * 2 + xy_j;

    double s = sspSize * 0.5;
    
    double ac = 0.0;
    double ad = 2.0 * s;
    double bc = -ad;     
    double bd = 0.0;     

        //TODO: valable uniquement si Nsubap constant
    double du = X[(Nsubap*(Nw-1)+i)] - X[(Nsubap*(Nw-1)+j)];                        
    double dv = Y[(Nsubap*(Nw-1)+i)] - Y[(Nsubap*(Nw-1)+j)];
    
    //if(ipos < 10)printf("ipos = %d - %d\n", ipos, (Nsubap*(Nw-1)+i));
    //if(jpos < 10)printf("jpos = %d - %d\n", jpos, (Nsubap*(Nw-1)+j));
    
    //const double du = X[0] - X[1];                        
    //const double dv = Y[0] - Y[1];
    
    //Layer l
    double covar = 0.0;
    #pragma unroll
    for (int l = 0; l < Nlayer; l++) 
    {
         //test if the altitude layers is not higher than the LGS altitude
        if (sspSize > 0) 
        {
            if (type == 0) covar += 0.5 /* pasDPHI*/ * cov_XX_gb(du,dv,ac,ad,bc,bd,tabDPHI,L0diff[l],convert,Ndphi) * kk * cn2[l];
            else if (type == 3) covar += 0.5 /* pasDPHI*/ * cov_YY_gb(du,dv,ac,ad,bc,bd,tabDPHI,L0diff[l],convert,Ndphi) * kk * cn2[l];
            else 
            {
                double s0 = 1.41421*s; //half size of the subaperture equivalent to a convolution by s1 and s2
                double dd = 0;
                covar += 0.25 /* pasDPHI*/ * cov_XY_gb(du,dv,s0,tabDPHI,L0diff[l],convert,Ndphi) * kk * cn2[l] * (1. - dd * dd);
            }
        }
    }
    return (double)covar; 
}
//--------------------------------------------------------------------------------------------
/*!
 *** covariance matrix generation kernel ***
 *        The kernel generates the element values in a given matrix/submatrix
 *     The generation function can be any function, as long as each element
 *     can be computed both individually and independently
 *
 *        see argument description in the kernel driver
 */
void matcov_ts_kernel_tile(
    double* data, int nrows, int ncols, int xoffset, int yoffset, int lda,
    double convert, double *X, double *Y, long *Nssp, double pasDPHI,double *tabDPHI, 
    double *indexL0, double *cn2, int Ndphi, int Nw, int Nlayer, int Nsubap, double teldiam,
    int lx, int ly)
{
                
    /*/ local thread coordinates w.r.t. thread block
    const int tx_ = threadIdx.x;
    const int ty_ = threadIdx.y;
                
    // local thread block coordinates w.r.t. kernel grid
    const int bx_ = blockIdx.x;
    const int by_ = blockIdx.y;
                
    // local coordinates of the element w.r.t. submatrix
    int lx = bx_ * blockDim.x + tx_;
    int ly = by_ * blockDim.y + ty_;*/
                
    // global coordinates of the elemnt w.r.t. the entire matrix
    int gx = lx + xoffset;
    int gy = ly + yoffset;
                
    // out-of-bound threads should terminate
    if( (lx >= nrows) || (ly >= ncols) ) return;
                
    // Advance the data pointer accordingly
    data += lx * lda + ly;
                
    // call the generation function
    data[0] = compute_element_ts_tile(gx, gy, convert,X, Y,Nssp,pasDPHI,tabDPHI, indexL0,cn2,Ndphi,Nw,Nlayer,Nsubap,teldiam);
        //printf("gx = %d, gy = %d ----- %.2f \n", gx, gy, data[0]);
}

//--------------------------------------------------------------------------------------------
void matcov_cpp_tiled(
    double* data, int nrows, int ncols, int xoffset, int yoffset, int lda,
    struct tomo_struct *tomo, int type_mat){
                
    const long Nw = tomo->Nw;
    const double crmax = tomo->rmax;
    const double pasDPHI = 1./tomo->pasDPHI; //inverse du pas de rr
    const long Ndphi = floor(crmax*pasDPHI)+1;
    const double convert = (double)(Ndphi-1)/(crmax+1./pasDPHI);
    const long Nsubap = tomo->Nsubap[Nw-1];
    
    /*#ifdef USE_OPENMP
    #pragma omp parallel num_threads(tomo->ncpu)
    #pragma omp for nowait
    #endif*/
    for(int lx = 0; lx < nrows; lx++){
        for(int ly = 0; ly < ncols; ly++){
            matcov_ts_kernel_tile(
                    data, nrows, ncols, xoffset, yoffset, lda,
                    convert, tomo->X, tomo->Y, tomo->Nssp,
                    pasDPHI, tomo->tabDPHI, tomo->L0diff, tomo->cn2,
                    Ndphi, tomo->Nw, tomo->Nlayer, Nsubap, tomo->DiamTel,
                    lx, ly);
        }
    }

    /*matcov_ts_kernel<<<dimGrid, dimBlock, 0, tomo_gpu->matcov_stream>>>(
            data, nrows, ncols, xoffset, yoffset, lda,
            convert,tomo_gpu->X_d,tomo_gpu->Y_d,tomo_gpu->Nssp_d,
            pasDPHI,tomo_gpu->tabDPHI_d,tomo_gpu->indexL0_d,tomo_gpu->cn2_d,
            Ndphi,tomo.Nw,tomo.Nlayer,Nsubap,tomo.DiamTel);*/
}


//============================================================================================
/*! matcov tile driver
*    Arguments
*    ==========
*    data                 double pointer: A pointer to the matrix/submatrix to be generated. It
*                                             should always point to the first element in a matrix/submatrix
*
*    nrows                integer: The number of rows of the matrix/submatrix to be generated
*
*    ncols                integer: The number of columns of the matrix/submatrix to be generated
*
*    xoffset            integer: The x-offset of the submatrix, must be zero if the entire matrix
*                                             is generated. Its the x-coordinate of the first element in the matrix/submatrix
*
*    yoffset    integer: The y-offset of the submatrix, must be zero if the entire matrix
*                                             is generated. Its the y-coordinate of the first element in the matrix/submatrix
*
*    lda                    integer: The leading dimension of the matrix/submatrix
*/
void matcov_comp_tile(
    double* data, int nrows, int ncols, int xoffset, int yoffset, int lda,
    struct tomo_struct *tomo, int type_mat)
{

    char uplo, copy;

    uplo = 'f';     // full generation is enabled by default
    copy = 'c';

    //int type_mat = tomo->part;

    if(type_mat == 1) // Caa matrix
    {
        // check if a square diagonal tile is generated then we set uplo to 'l' or 'u'
        // and then enable the copy
        // This also applies if the entire matrix will be generated
        // otherwise (off diagonal tile or non square submatrix) - full generation is assumed
        if((xoffset == yoffset) && (nrows == ncols))        // if sqaure & diagonal
        {
            uplo = 'l';
            copy = 'c';
        }
        else        // full generation, copy is ignored
        {
            uplo = 'f';
        }
    }
    //else if(type_mat == 2) //
    else
    if(type_mat == 2 || type_mat == 3) // Cmaa matrix
    {
        uplo = 'f';                         // full generation, copy is ignored
    }
    else
    if(type_mat != 4)
    {
        fprintf(stderr, "ERROR: unrecognized type_mat %d \n", type_mat);
        exit(1);
    }
    // %%%%%%% Pre-computation of DPHI %%%%%%%%%%
    //Computes an array of DPHI (tabDPHI) for an array of subaperture distance rr for each DIFFERENT L0
    const double crmax = tomo->rmax;
    const double pasDPHI = 1./tomo->pasDPHI; //inverse du pas de rr
    const long Ndphi = floor(crmax*pasDPHI)+1;
    const double convert = (double)(Ndphi-1)/(crmax+tomo->pasDPHI);
    //const double convert = (double)(Ndphi-1)/(crmax+1./pasDPHI);

    if(type_mat == 4){
        const long Nw = tomo->Nw;
        const long Nsubap = tomo->Nsubap[Nw-1];

        for(int lx = 0; lx < nrows; lx++){
            for(int ly = 0; ly < ncols; ly++){
                matcov_ts_kernel_tile(
                        data, nrows, ncols, xoffset, yoffset, lda,
                        convert, tomo->X, tomo->Y, tomo->Nssp,
                        pasDPHI, tomo->tabDPHI, tomo->L0diff, tomo->cn2,
                        Ndphi, tomo->Nw, tomo->Nlayer, Nsubap, tomo->DiamTel,
                        lx, ly);
            }
        }
    }else{
        //const long Nsubap = tomo->Nsubap[0];

        /*#ifdef USE_OPENMP
        #pragma omp parallel num_threads(tomo->ncpu)
        #pragma omp for nowait
        #endif*/
        for(int lx = 0; lx < nrows; lx++){
            for(int ly = 0; ly < ncols; ly++){
                matcov_kernel_4(
                        uplo, copy, data, nrows, ncols, xoffset, yoffset, lda, convert, tomo->sspSizeL,
                        tomo->Nssp, tomo->u, tomo->v,tomo->X,tomo->Y, pasDPHI, tomo->tabDPHI,
                        tomo->L0diff/*tomo->indexL0*/, tomo->cn2, Ndphi, tomo->Nw, tomo->Nlayer,
                        /*Nsubap*/tomo->Nsubap, tomo->Nx, tomo->alphaX, tomo->alphaY, tomo->lgs_cst, tomo->noise_var,
                        tomo->spot_width, tomo->lgs_depth, tomo->lgs_alt, type_mat, tomo->nlgs, tomo->DiamTel,
                        lx, ly);
            }
        }
    }

}

int STARS_aodata_block_kernel(size_t nrows, size_t ncols, size_t *irow,
        size_t *icol, void *row_data, void *col_data, void *result)
{
    struct tomo_struct *data = row_data;
    size_t i, j;
    double *buffer = result;
    double crmax = data->rmax;
    double pasDPHI = 1./data->pasDPHI; //inverse du pas de rr
    long Ndphi = floor(crmax*pasDPHI)+1;
    double convert = (double)(Ndphi-1)/(crmax+data->pasDPHI);
    int type_mat = 1;
    #pragma omp parallel for private(i, j)
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
            buffer[j*nrows+i] = compute_element_tiled_4(irow[i], icol[j],
                    convert, data->sspSizeL, data->Nssp, data->u, data->v,
                    data->X, data->Y, pasDPHI, data->tabDPHI, data->L0diff,
                    data->cn2, Ndphi, data->Nw, data->Nlayer,
                    data->Nsubap, data->Nx, data->alphaX, data->alphaY,
                    data->lgs_cst, data->noise_var, data->spot_width,
                    data->lgs_depth, data->lgs_alt, type_mat, data->nlgs,
                    data->DiamTel);
    return 0;
}

STARS_aodata *STARS_gen_aodata(char *files_path, int night_idx,
        int snapshots_per_night, int snapshot_idx, int obs_idx, double alphaX,
        double alphaY)
{
    STARS_aodata *data = malloc(sizeof(*data));
    matcov_init_tomo_tiled(data, files_path, night_idx,
            snapshots_per_night, snapshot_idx, obs_idx, alphaX, alphaY);
    data->count = matcov_getNumMeasurements(data)-
            matcov_getNumMeasurementsTS(data);
    return data;
}

void STARS_aodata_free(STARS_aodata *data)
{
    matcov_free_tomo_tiled(data);
    free(data);
}
/*
STARS_Problem *STARS_gen_aoproblem(STARS_tomo *tomo)
{
    int ndim = 2;
    int shape[2];
    shape[0] = matcov_getNumMeasurements(tomo)-
        matcov_getNumMeasurementsTS(tomo);
    //printf("%d %d\n", matcov_getNumMeasurements(tomo),
    //        matcov_getNumMeasurementsTS(tomo));
    shape[1] = shape[0];
    //printf("%d %d\n", shape[0], shape[1]);
    STARS_Problem *problem = STARS_Problem_init(ndim, shape, 'S', 'd', tomo,
            tomo, block_astronomy_kernel_noalloc, "Astronomy");
    return problem;
}

STARS_BLR *STARS_gen_ao_blrformat(STARS_Problem *problem, int block_size)
{
    return STARS_BLR_plain(problem, 'S', block_size);
}
*/
