set(SRC)
add_subdirectory("blrm")
add_library(backends_mpi OBJECT ${SRC})
set_target_properties(backends_mpi PROPERTIES COMPILE_FLAGS
    "${MPI_C_COMPILE_FLAGS}")
if(OPENMP)
    target_compile_options(backends_mpi PUBLIC "${OpenMP_C_FLAGS}")
endif(OPENMP)
