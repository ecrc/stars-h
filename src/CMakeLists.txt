set(STARSH_SRC)

# Define the source files
add_subdirectory("control")
add_subdirectory("applications")
add_subdirectory("itersolvers")

# Set backends and check if there is support for them
set(BACKENDS_OBJECTS)
add_subdirectory("backends")

add_library(starsh STATIC ${STARSH_SRC} ${BACKENDS_OBJECTS})
if(OPENMP)
    target_link_libraries(starsh PRIVATE "${OpenMP_C_FLAGS}")
endif()

# Install STARSH library
install(TARGETS starsh
    DESTINATION ${LIBDIR})
