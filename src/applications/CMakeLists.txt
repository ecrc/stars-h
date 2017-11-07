# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/applications/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


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
    ${STARSH_SRC})
set(STARSH_SRC ${STARSH_SRC} PARENT_SCOPE)
