# set the values of the variable in the parent scope
set ( STARSH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cluster.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/blrf.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/blrm.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/array.c
                 ${CMAKE_CURRENT_SOURCE_DIR}/problem.c
                 ${STARSH_SRC}
                 PARENT_SCOPE )