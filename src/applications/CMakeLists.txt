# set a new variable in the local scoppe, with the values of the parent scope
set ( LOCAL_SRC ${STARSH_SRC} )
# set the values of the variable in the parent scope
set ( STARSH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/spatial.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/astronomy.c
                 ${LOCAL_SRC}
                 PARENT_SCOPE )