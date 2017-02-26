# Add public headers files to installation
set(STARSH_PUB_HDR
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-spatial.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/starsh-rndtiled.h"
    ${STARSH_PUB_HDR} PARENT_SCOPE)
