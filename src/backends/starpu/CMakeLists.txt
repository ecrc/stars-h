set(SRC)
add_subdirectory("blrm")
add_subdirectory("kernels")
add_library(backends_starpu OBJECT ${SRC})
