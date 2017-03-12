# set the values of the variable in the parent scope
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/dca.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dfe.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dmml.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dqp3.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/drsdd2.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/drsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dsdd.c"
    ${SRC} PARENT_SCOPE)
