# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/backends/starpu/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


set(DOXYGEN_INPUT ${DOXYGEN_INPUT})

# Collect sources for documentation and compilation
set(SRC)
add_subdirectory("blrm")
add_subdirectory("dense")

# If compilation is requried
if(STARPU)
    add_library(backends_starpu OBJECT ${SRC})
endif()

# Put doxygen input to parent scope
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${SRC} PARENT_SCOPE)
