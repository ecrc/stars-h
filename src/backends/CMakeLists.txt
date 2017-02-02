add_subdirectory( ${BACKEND_TYPE} )
# set the values of the variable in the parent scope
set( STARSH_SRC ${STARSH_SRC} PARENT_SCOPE)