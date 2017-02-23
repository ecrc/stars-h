# Define the source files

set(STARSH_SRC)
add_subdirectory(control)
add_subdirectory(applications)
add_subdirectory(itersolvers)

# Set backends and check if there is support for them
set(BACKENDS "sequential" "openmp" "starpu")


add_subdirectory(backends)


include_directories(${CMAKE_SOURCE_DIR}/include)


# Defines the STARS-H library to build
add_library(starsh STATIC ${STARSH_SRC})

install( TARGETS starsh
         DESTINATION lib )

