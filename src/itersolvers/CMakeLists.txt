# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/itersolvers/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


# set the values of the variable in the parent scope
set(STARSH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cg.c"
    ${STARSH_SRC})
set(STARSH_SRC ${STARSH_SRC} PARENT_SCOPE)
