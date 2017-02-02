# set the values of the variable in the parent scope
set ( STARSH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/spatial.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/astronomy.c
                 ${STARSH_SRC}
                 PARENT_SCOPE )