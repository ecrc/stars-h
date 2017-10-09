# set the values of the variable in the parent scope
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/dqp3.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/drsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dsdd.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dsvfr.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/dna.c"
    ${SRC} PARENT_SCOPE)
