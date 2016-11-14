#include <stdio.h>
#include "astronomy.h"
#include "stars.h"
#include "stars-astronomy.h"


int main(int argc, char **argv){

    if(argc < 6)
    {
        printf("./astronomy.out files_path block_size maxrank tol "
                "heatmap_filename\n");
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
    STARS_Problem_info(problem);
    //STARS_BLR *blr = STARS_gen_ao_blrformat(problem, block_size);
    //STARS_BLR_info(blr);
    //STARS_BLR_print(blr);
    //STARS_BLR_free(blr);
    //STARS_Problem_free(problem);
    free(tomo);
    return 0;
}

