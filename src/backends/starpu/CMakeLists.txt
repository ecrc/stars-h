# This project needs GNU99 standard to compile properly with StarPU
# (could have used CMAKE_C_FLAGS instead)
add_definitions("-std=gnu99")
# set the values of the variable in the parent scope
add_subdirectory("blrm")
add_subdirectory("kernels")
set(STARSH_SRC "${STARSH_SRC}" PARENT_SCOPE)
