foreach(BACKEND ${BACKEND_TYPE})
    add_subdirectory(${BACKEND})
endforeach()
# set the values of the variable in the parent scope
set( STARSH_SRC ${STARSH_SRC} PARENT_SCOPE)
