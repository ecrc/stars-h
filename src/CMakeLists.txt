set(STARSH_SRC)

# Define the source files
add_subdirectory("control")
# Get code for applications
add_subdirectory("applications")
# Connection with other libraries (PLASMA)
add_subdirectory("external")
# Add some tools loke iterative solvers
add_subdirectory("itersolvers")

# Set backends and check if there is support for them
set(BACKENDS_OBJECTS)
add_subdirectory("backends")

add_library(starsh STATIC ${STARSH_SRC} ${BACKENDS_OBJECTS})
if(OPENMP)
    target_link_libraries(starsh PRIVATE "${OpenMP_C_FLAGS}")
endif(OPENMP)
if(STARPU)
    target_link_libraries(starsh PRIVATE ${STARPU_LIBRARIES})
endif(STARPU)
if(GSL_FOUND)
    target_link_libraries(starsh PRIVATE ${GSL_LIBRARIES})
endif(GSL_FOUND)

# Install STARSH library
install(TARGETS starsh
    DESTINATION ${LIBDIR})
