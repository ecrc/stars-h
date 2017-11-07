# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/backends/sequential/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


set(SRC)
add_subdirectory("blrm")
add_subdirectory("dense")
add_library(backends_sequential OBJECT ${SRC})
target_include_directories(backends_sequential PRIVATE
    "${CMAKE_SOURCE_DIR}/include")

# Put doxygen input to parent scope
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${SRC} PARENT_SCOPE)
