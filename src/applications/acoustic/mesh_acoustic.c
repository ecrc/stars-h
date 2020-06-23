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


/*! It reads mesh points fron file
 *
 * @param[inout] data: STARSH_acdata acoustic scattering
 * @param[in] mesh points: number of mesh points
 * @param[in] ndim: problem dimension.
 * @param[in] train:  number of traingles
 * @param[in] nipp:  number of quadrature points
 * @param[in] mordering: 0: no ordering, 1: Morton ordering.
 * */
int starsh_generate_3d_acoustic_coordinates(STARSH_acdata **data, STARSH_int mesh_points, 
                                 int ndim, int trian, int nipp, int mordering){


        generate_mesh_points_serials(&nipp, &trian);        

	STARSH_MALLOC(*data, 1);
	(*data)->train = trian;
	(*data)->nipp = nipp;
	(*data)->mordering = mordering;

        return STARSH_SUCCESS;
}




