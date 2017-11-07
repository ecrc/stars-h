# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


# Non-backend sources of STARS-H
set(STARSH_SRC)

# Define the source files
add_subdirectory("control")
# Get code for applications
add_subdirectory("applications")
# Add some tools like iterative solvers
add_subdirectory("itersolvers")

# Set backends and check if there is support for them
set(BACKENDS_OBJECTS)
add_subdirectory("backends")

# Put all doxygen inputs back to parent
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${STARSH_SRC} PARENT_SCOPE)

# Define STARS-H library
add_library(starsh STATIC ${STARSH_SRC} ${BACKENDS_OBJECTS})

# Install STARS-H library
install(TARGETS starsh DESTINATION ${LIBDIR})

# Add compile and link flags for STARS-H
if(OPENMP)
    target_compile_options(starsh PUBLIC "${OpenMP_C_FLAGS}")
    target_link_libraries(starsh PUBLIC "${OpenMP_C_FLAGS}")
endif(OPENMP)
if(STARPU)
    target_link_libraries(starsh PUBLIC ${STARPU_LIBRARIES})
endif(STARPU)
if(GSL_FOUND)
    target_link_libraries(starsh PUBLIC ${GSL_LIBRARIES})
endif(GSL_FOUND)
if(MPI)
    set_target_properties(starsh PROPERTIES LINK_FLAGS "${MPI_C_LINK_FLAGS}")
    target_link_libraries(starsh PUBLIC ${MPI_C_LIBRARIES})
endif(MPI)
