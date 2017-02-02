# set the values of the variable in the parent scope
add_subdirectory(blrm)
add_subdirectory(kernels)
set ( STARSH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/blrm.c
                 ${STARSH_SRC}
                 PARENT_SCOPE )