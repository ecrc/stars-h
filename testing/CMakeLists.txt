# Defines the source files of the tests. Each file generates 1 test
set(tests_files
    #"symm_positivity_after_compression.c"
    "mpi_minimal.c"
    "mpi_spatial.c"
    "minimal.c"
    "spatial.c"
    "rndtiled.c"
    )

if( PLASMA_FOUND )
    list(APPEND tests_files "plasma/solve.c")
endif()

# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach(test_src ${tests_files})
    get_filename_component(test_exe ${test_src} NAME_WE)
    add_executable(${test_exe} ${test_src})
    target_link_libraries(${test_exe} starsh ${CBLAS_LIBRARIES}
        ${LAPACKE_LIBRARIES} ${OpenMP_C_FLAGS})
endforeach()

add_test(NAME minimal_rsdd COMMAND minimal 2500 500 omp_rsdd 90 1e-9)
add_test(NAME minimal_sdd COMMAND minimal 2500 500 omp_sdd 90 1e-9)
add_test(NAME minimal_qp3 COMMAND minimal 2500 500 omp_qp3 90 1e-9)
add_test(NAME spatial_2d_exp_rsdd COMMAND
    spatial spatial "exp simd" 0.1 0.5 2500 500 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_2d_exp_sdd COMMAND
    spatial spatial "exp simd" 0.1 0.5 2500 500 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_2d_exp_qp3 COMMAND
    spatial spatial "exp simd" 0.1 0.5 2500 500 omp_qp3 90 1e-9 1 0)
add_test(NAME spatial_2d_sqrexp_rsdd COMMAND
    spatial spatial "sqrexp simd" 0.1 0.5 2500 500 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_2d_sqrexp_sdd COMMAND
    spatial spatial "sqrexp simd" 0.1 0.5 2500 500 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_2d_sqrexp_qp3 COMMAND
    spatial spatial "sqrexp simd" 0.1 0.5 2500 500 omp_qp3 90 1e-9 1 0)
add_test(NAME spatial_2d_matern_rsdd COMMAND
    spatial spatial "Matern simd" 0.1 10 2500 500 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_2d_matern_sdd COMMAND
    spatial spatial "Matern simd" 0.1 10 2500 500 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_2d_matern_qp3 COMMAND
    spatial spatial "Matern simd" 0.1 10 2500 500 omp_qp3 90 1e-9 1 0)
add_test(NAME spatial_3d_exp_rsdd COMMAND
    spatial spatial3d "exp simd" 0.1 0.5 2744 686 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_3d_exp_sdd COMMAND
    spatial spatial3d "exp simd" 0.1 0.5 2744 686 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_3d_exp_qp3 COMMAND
    spatial spatial3d "exp simd" 0.1 0.5 2744 686 omp_qp3 90 1e-9 1 0)
add_test(NAME spatial_3d_sqrexp_rsdd COMMAND
    spatial spatial3d "sqrexp simd" 0.1 0.5 2744 686 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_3d_sqrexp_sdd COMMAND
    spatial spatial3d "sqrexp simd" 0.1 0.5 2744 686 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_3d_sqrexp_qp3 COMMAND
    spatial spatial3d "sqrexp simd" 0.1 0.5 2744 686 omp_qp3 90 1e-9 1 0)
add_test(NAME spatial_3d_matern_rsdd COMMAND
    spatial spatial3d "Matern simd" 0.1 10 2744 686 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_3d_matern_sdd COMMAND
    spatial spatial3d "Matern simd" 0.1 10 2744 686 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_3d_matern_qp3 COMMAND
    spatial spatial3d "Matern simd" 0.1 10 2744 686 omp_qp3 90 1e-9 1 0)
