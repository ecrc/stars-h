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

install( TARGETS starsh
         DESTINATION lib
         )
