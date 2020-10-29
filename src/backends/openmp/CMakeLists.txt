# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/backends/openmp/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


# Collect sources for documentation and compilation
set(SRC)
add_subdirectory("blrm")

# Compilation of OpenMP is always required
add_library(backends_openmp OBJECT ${SRC})
target_compile_options(backends_openmp PUBLIC "${OpenMP_C_FLAGS}")

# Put doxygen input to parent scope
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${SRC} PARENT_SCOPE)
