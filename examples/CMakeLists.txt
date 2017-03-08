# Defines the source files of the tests. Each file generates 1 test
set(tests_files "spatial.c" "rndtiled.c")

include_directories("${CMAKE_SOURCE_DIR}/include")

# Uses RUNPATH instead of RPATH
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach(test_src ${tests_files})
    get_filename_component(test_exe ${test_src} NAME_WE)
    add_executable(${test_exe} ${test_src})
    target_link_libraries(${test_exe} starsh ${CBLAS_LIBRARIES}
        ${LAPACKE_LIBRARIES})
endforeach()

# Add basic tests. To run them type: 'make test' in build directory
add_test(NAME Spatial_sdd COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 sdd)
add_test(NAME Spatial_omp_sdd COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 omp_sdd)
add_test(NAME Spatial_starpu_sdd COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 starpu_sdd)
add_test(NAME Spatial_qp3 COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 qp3)
add_test(NAME Spatial_omp_qp3 COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 omp_qp3)
add_test(NAME Spatial_starpu_qp3 COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 starpu_qp3)
add_test(NAME Spatial_rsdd COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 rsdd)
add_test(NAME Spatial_omp_rsdd COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 omp_rsdd)
add_test(NAME Spatial_starpu_rsdd COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 starpu_rsdd)
add_test(NAME Spatial_rsdd2 COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 rsdd2)
add_test(NAME Spatial_omp_rsdd2 COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 omp_rsdd2)
add_test(NAME Spatial_starpu_rsdd2 COMMAND
    spatial 100 1000 100 10 1e-12 1e-1 starpu_rsdd2)

