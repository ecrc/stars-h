# set the values of the variable in the parent scope
set(STARSH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/dqp3.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/drsdd2.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/drsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dsdd.c"
    "${STARSH_SRC}" PARENT_SCOPE)
