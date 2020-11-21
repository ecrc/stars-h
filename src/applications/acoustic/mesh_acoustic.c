/*! @copyright (c) 2020 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * Generate different functions for different dimensions. This hack improves
 * performance in certain cases. Value 'n' stands for general case, whereas all
 * other values correspond to static values of dimensionality.
 * will be replace by proposed values. If you want to use this file outside
 * STARS-H, simply do substitutions yourself.
 *
 * @file src/applications/acoustic/mesh_acoustic.c
 * @version 0.1.1
 * @author Rabab Alomairy
 * @author Aleksandr Mikhalev
 * @date 2020-05-09 
 */

#include "common.h"
#include "starsh.h"
#include "starsh-acoustic.h"
#include <inttypes.h>
#include <string.h>

/*! It reads mesh points from file
 *
 * @param[inout] data: STARSH_acdata acoustic scattering
 * @param[in] mesh points: number of mesh points
 * @param[in] ndim: problem dimension.
 * @param[in] train:  number of traingles
 * @param[in] nipp:  number of quadrature points
 * @param[in] mordering: 0: no ordering, 1: Morton ordering.
 * */
int starsh_generate_3d_acoustic_coordinates(STARSH_acdata **data, STARSH_int mesh_points, 
                                 int ndim, int trian, int nipp, int mordering, char* file_name, char* file_name_interpl
){

        int filelength1=strlen(file_name);
        int filelength2=strlen(file_name_interpl);
        generate_mesh_points_serials(&nipp, &trian, file_name, &filelength1, file_name_interpl, &filelength2);        

	STARSH_MALLOC(*data, 1);
	(*data)->train = trian;
	(*data)->nipp = nipp;
	(*data)->mordering = mordering;

       if(nipp!=3 || nipp !=6 || nipp!=12){
             STARSH_ERROR("Wrong parameter type, number of quadrature points are 3, 6, or 12");
             return STARSH_WRONG_PARAMETER;
        }

        return STARSH_SUCCESS;
}




