set(SRC)
add_subdirectory("blrm")
add_library(backends_mpi_starpu OBJECT ${SRC})
set_target_properties(backends_mpi_starpu PROPERTIES COMPILE_FLAGS
    "${MPI_C_COMPILE_FLAGS}")
