set(SRC)
add_subdirectory("blrm")
add_library(backends_openmp OBJECT ${SRC})
target_compile_options(backends_openmp PUBLIC "${OpenMP_C_FLAGS}")
