# set the values of the variable in the parent scope
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/dqp3.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/drsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dgemm.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/fake_init.c"
    ${SRC} PARENT_SCOPE)
