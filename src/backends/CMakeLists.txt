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

set(BACKENDS_OBJECTS ${BACKENDS_OBJECTS} PARENT_SCOPE)
