# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file CMakeLists.txt
# @version 0.3.0
# @author Eduardo Gonzalez Fisher
# @author Aleksandr Mikhalev
# @date 2020-06-09
# Add public headers files to installation

set(STARSH_PUB_HDR
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-mpi.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-starpu.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-mpi-starpu.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-spatial.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-rbf.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-spatial-gsl.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-electrostatics.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-electrodynamics.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-randtlr.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-minimal.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-constants.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-particles.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-cauchy.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-acoustic.h"
)

# Add public headers to doxygen input
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${STARSH_PUB_HDR} PARENT_SCOPE)

# Install public headers
install(FILES ${STARSH_PUB_HDR} DESTINATION ${INCDIR})
