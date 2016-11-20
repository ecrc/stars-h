# Define the source files
file(GLOB STARSH_SRC "*.c")
# set ( STARSH_SRC array.c
#                  blr.c
#                  electrostatics.c
#                  misc.c
#                  problem.c
#                  spatial.c
#                  )

include_directories( ${PROJECT_SOURCE_DIR}/include)

# Defines the STARS-H library to build
add_library(starsh STATIC ${STARSH_SRC})

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
         )

