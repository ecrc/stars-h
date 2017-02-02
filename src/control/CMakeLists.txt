# set a new variable in the local scoppe, with the values of the parent scope
set ( LOCAL_SRC ${STARSH_SRC} )
# set the values of the variable in the parent scope
set ( STARSH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cluster.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/blrf.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/blrm.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/array.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/problem.c
                 ${LOCAL_SRC}
                 PARENT_SCOPE )