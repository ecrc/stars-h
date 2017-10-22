# set the values of the variable in the parent scope
add_subdirectory("spatial")
add_subdirectory("electrostatics")
add_subdirectory("electrodynamics")

set(STARSH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/common.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/randtlr.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/minimal.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/particles.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/cauchy.c"
    ${STARSH_SRC} PARENT_SCOPE)
