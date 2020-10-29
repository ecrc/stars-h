# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/backends/starpu/dense/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


# set the values of the variable in the parent scope
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/dqp3.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/drsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dgemm.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/fake_init.c"
    ${SRC} PARENT_SCOPE)
