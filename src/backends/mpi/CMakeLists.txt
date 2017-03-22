set(SRC)
add_subdirectory("blrm")
add_library(backends_mpi OBJECT ${SRC})
