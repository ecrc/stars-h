# Add public headers files to installation
set(STARSH_PUB_HDR
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-spatial.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-electrostatics.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-electrodynamics.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-randtlr.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-minimal.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-constants.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-particles.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-cauchy.h"
)

# Add public headers if MPI is enabled
if(MPI)
    list(APPEND STARSH_PUB_HDR "${CMAKE_CURRENT_SOURCE_DIR}/starsh-mpi.h")
endif()

# Add public headers if STARPU is enabled
if(STARPU)
    list(APPEND STARSH_PUB_HDR "${CMAKE_CURRENT_SOURCE_DIR}/starsh-starpu.h")
endif()

# Add public headers if MPI+STARPU are enabled
if(MPI AND STARPU)
    list(APPEND STARSH_PUB_HDR "${CMAKE_CURRENT_SOURCE_DIR}/starsh-mpi-starpu.h")
endif()

# Add public headers to doxygen input
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${STARSH_PUB_HDR} PARENT_SCOPE)

# Install public headers
install(FILES ${STARSH_PUB_HDR} DESTINATION ${INCDIR})
