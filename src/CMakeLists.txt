# Define the source files
#file(GLOB_RECURSE STARSH_SRC "*.c")
set ( STARSH_SRC control/cluster.c
                 control/blrf.c
                 control/blrm.c
                 control/array.c
                 control/problem.c
                 applications/spatial.c
                 applications/electrostatics.c
                 applications/astronomy.c
                 misc.c
                 )

set( BACKEND_TYPE "sequential" CACHE STRING "Options are: sequential, omp, starpu" )
file(GLOB_RECURSE BACKEND_SRC "backends/${BACKEND_TYPE}/*.c")
list(APPEND STARSH_SRC ${BACKEND_SRC} )

include_directories( ${PROJECT_SOURCE_DIR}/include)

if( "${BACKEND_TYPE}" STREQUAL "starpu" )
	find_package(STARPU)
	if( STARPU_FOUND )
		include_directories( ${STARPU_INCLUDE_DIRS})
	else()
	endif()
endif()


# Defines the STARS-H library to build
add_library(starsh STATIC ${STARSH_SRC})

# Add public headers files to installation
set( STARSH_PUB_HDR ${PROJECT_SOURCE_DIR}/include/stars.h ${PROJECT_SOURCE_DIR}/include/stars-spatial.h )
set_target_properties( starsh PROPERTIES PUBLIC_HEADER "${STARSH_PUB_HDR}" )


option( BUILD_SHARED_LIBS "Generates shared library" OFF)
if( BUILD_SHARED_LIBS )
	add_library ( starsh-so SHARED ${STARSH_SRC} )
	target_link_libraries(starsh-so ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES} )
	set_target_properties(starsh-so PROPERTIES OUTPUT_NAME starsh )
	set_target_properties(starsh-so PROPERTIES VERSION ${STARSH_VERSION} )
	set_target_properties(starsh-so PROPERTIES SOVERSION ${STARSH_VERSION_MAJOR} )
	install( TARGETS  starsh-so DESTINATION lib )
endif( BUILD_SHARED_LIBS )

install( TARGETS starsh
         DESTINATION lib
         PUBLIC_HEADER DESTINATION include         )

