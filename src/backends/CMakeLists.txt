# Add sequential code
add_subdirectory("sequential")
set(BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_sequential>)

# Add StarPU code
if(STARPU)
    add_subdirectory("starpu")
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_starpu>)
endif()

# Add OpenMP code
if(OPENMP)
    add_subdirectory("openmp")
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_openmp>)
endif()

# Add MPI code
if(MPI)
    add_subdirectory("mpi")
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_mpi>)
endif()

# Add MPI+StarPU code
if(MPI AND STARPU)
    add_subdirectory("mpi_starpu")
    list(APPEND BACKENDS_OBJECTS $<TARGET_OBJECTS:backends_mpi_starpu>)
endif()

set(BACKENDS_OBJECTS ${BACKENDS_OBJECTS} PARENT_SCOPE)
