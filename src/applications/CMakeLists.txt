# set the values of the variable in the parent scope
set(STARSH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/common.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/rndtiled.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/minimal.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/particles.c"
    ${STARSH_SRC} PARENT_SCOPE)
