set(SRC)
add_subdirectory("blrm")
add_subdirectory("dense")
add_library(backends_starpu OBJECT ${SRC})
