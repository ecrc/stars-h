# Add sequential code
add_subdirectory(sequential)

# Add StarPU code
if(STARPU)
    add_subdirectory("starpu")
endif()

# Add OpenMP code
if(OPENMP)
    add_subdirectory("openmp")
endif()

# set the values of the variable in the parent scope
set( STARSH_SRC ${STARSH_SRC} PARENT_SCOPE)
