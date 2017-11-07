# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/backends/mpi/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


# Collect sources for documentation and compilation
set(SRC)
add_subdirectory("blrm")

# If compilation is requried
if(MPI)
    add_library(backends_mpi OBJECT ${SRC})
    set_target_properties(backends_mpi PROPERTIES COMPILE_FLAGS
        "${MPI_C_COMPILE_FLAGS}")
    if(OPENMP)
        target_compile_options(backends_mpi PUBLIC "${OpenMP_C_FLAGS}")
    endif(OPENMP)
endif()

# Put doxygen input to parent scope
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${SRC} PARENT_SCOPE)
