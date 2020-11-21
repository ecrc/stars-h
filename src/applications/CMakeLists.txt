# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/applications/CMakeLists.txt
# @version 0.3.0
# @author Aleksandr Mikhalev
# @date 2020-06-09


# set the values of the variable in the parent scope
add_subdirectory("spatial")
add_subdirectory("electrostatics")
add_subdirectory("electrodynamics")

# Collect sources
set(STARSH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/common.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/electrodynamics.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/randtlr.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/minimal.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/particles.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/cauchy.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh_deformation/cube.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh_deformation/mesh_rbf.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh_deformation/virus.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh_deformation/kernels_rbf.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/curve_para.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/frequency.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/gaussintegraldata.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/global_com.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/global_dim.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/global_geom.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/hard_nys.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/inci_rhs.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic-kernel/near_scafield.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/mesh_acoustic.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/acoustic/acoustic.c"
    ${STARSH_SRC})
set(STARSH_SRC ${STARSH_SRC} PARENT_SCOPE)
