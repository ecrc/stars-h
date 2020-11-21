# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/backends/CMakeLists.txt
# @version 0.3.0
# @author Aleksandr Mikhalev
# @date 2017-11-07
# Add sequential code

# If FULL documentation is required, we generate all sources for all backends,
# but we limit compilation only to selected backends

add_subdirectory("sequential")
set(BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_sequential>)

# List StarPU sources for docs or build
if(STARPU OR DOCS STREQUAL "FULL")
    add_subdirectory("starpu")
    if(KBLAS OR DOCS STREQUAL "FULL")
        add_subdirectory("starpu_kblas")
        add_subdirectory("starpu_kblas2")
        add_subdirectory("starpu_kblas3_spatial")
    endif()
    if(CUDA OR DOCS STREQUAL "FULL")
        add_subdirectory("starpu_cuda")
    endif()
endif()

# Add StarPU backend
if(STARPU)
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_starpu>)
    if(KBLAS)
        list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_starpu_kblas>)
        list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_starpu_kblas2>)
        list(APPEND BACKENDS_OBJECTS
            $<TARGET_OBJECTS:backends_starpu_kblas3_spatial>)
    endif()
    if(CUDA)
        list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_starpu_cuda>)
    endif()
endif()

# List OpenMP sources for docs or build
if(OPENMP OR DOCS STREQUAL "FULL")
    add_subdirectory("openmp")
endif()

# Add OpenMP backend
if(OPENMP)
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_openmp>)
endif()

# List MPI sources for docs or build
if(MPI OR DOCS STREQUAL "FULL")
    add_subdirectory("mpi")
endif()

# Add MPI backend
if(MPI)
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_mpi>)
endif()

# List MPI+STARPU sources for docs or build
if((MPI AND STARPU) OR DOCS STREQUAL "FULL")
    add_subdirectory("mpi_starpu")
endif()

# Add MPI+StarPU backend
if(MPI AND STARPU)
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_mpi_starpu>)
endif()

# List MPI+STARPU+KBLAS sources for docs or build
if((MPI AND STARPU AND KBLAS) OR DOCS STREQUAL "FULL")
    add_subdirectory("mpi_starpu_kblas")
    add_subdirectory("mpi_starpu_kblas2")
    add_subdirectory("mpi_starpu_kblas3_spatial")
endif()

# Add MPI+StarPU+KBLAS backend
if(MPI AND STARPU AND KBLAS)
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_mpi_starpu_kblas>)
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_mpi_starpu_kblas2>)
    list(APPEND BACKENDS_OBJECTS
        $<TARGET_OBJECTS:backends_mpi_starpu_kblas3_spatial>)
endif()

# Move all selected backends to parent directory
set(BACKENDS_OBJECTS ${BACKENDS_OBJECTS} PARENT_SCOPE)

# Put doxygen input to parent scope
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} PARENT_SCOPE)
