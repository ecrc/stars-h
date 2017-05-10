# Defines the source files of the tests. Each file generates 1 test
set(tests_files)

# NEED TO PUT EXPLICIT DEPENDENCIES ON OTHER LIBRARIES HERE
# However, it works for static library libstarsh.a and will NOT work for shared

if(OPENMP)
    list(APPEND tests_files
    "minimal.c"
    "spatial.c"
    "rndtiled.c"
    )
endif()

if(MPI)
    list(APPEND tests_files
    "mpi_minimal.c"
    "mpi_spatial.c"
    )
endif()

if(PLASMA)
    #list(APPEND tests_files "plasma/solve.c")
endif()

# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach(test_src ${tests_files})
    get_filename_component(test_exe ${test_src} NAME_WE)
    add_executable(test_${test_exe} ${test_src})
    target_link_libraries(test_${test_exe} starsh ${CBLAS_LIBRARIES}
        ${LAPACKE_LIBRARIES} ${OpenMP_C_FLAGS})
    set_target_properties(test_${test_exe} PROPERTIES OUTPUT_NAME ${test_exe})
endforeach()

add_test(NAME minimal_rsdd COMMAND minimal 2500 500 omp_rsdd 90 1e-9)
add_test(NAME minimal_sdd COMMAND minimal 2500 500 omp_sdd 90 1e-9)
add_test(NAME minimal_qp3 COMMAND minimal 2500 500 omp_qp3 90 1e-9)
add_test(NAME spatial_2d_exp_rsdd COMMAND
    spatial 2 11 0.1 0.5 2500 500 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_2d_exp_sdd COMMAND
    spatial 2 11 0.1 0.5 2500 500 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_2d_exp_qp3 COMMAND
    spatial 2 11 0.1 0.5 2500 500 omp_qp3 90 1e-9 1 0)
add_test(NAME spatial_2d_sqrexp_rsdd COMMAND
    spatial 2 12 0.1 0.5 2500 500 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_2d_sqrexp_sdd COMMAND
    spatial 2 12 0.1 0.5 2500 500 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_2d_sqrexp_qp3 COMMAND
    spatial 2 12 0.1 0.5 2500 500 omp_qp3 90 1e-9 1 0)

if(GSL_FOUND)
    add_test(NAME spatial_2d_matern_rsdd COMMAND
        spatial 2 13 0.1 10 2500 500 omp_rsdd 90 1e-9 1 0)
    add_test(NAME spatial_2d_matern_sdd COMMAND
        spatial 2 13 0.1 10 2500 500 omp_sdd 90 1e-9 1 0)
    add_test(NAME spatial_2d_matern_qp3 COMMAND
        spatial 2 13 0.1 10 2500 500 omp_qp3 90 1e-9 1 0)
    add_test(NAME spatial_2d_matern2_rsdd COMMAND
        spatial 2 14 0.1 10 2500 500 omp_rsdd 90 1e-9 1 0)
    add_test(NAME spatial_2d_matern2_sdd COMMAND
        spatial 2 14 0.1 10 2500 500 omp_sdd 90 1e-9 1 0)
    add_test(NAME spatial_2d_matern2_qp3 COMMAND
        spatial 2 14 0.1 10 2500 500 omp_qp3 90 1e-9 1 0)
endif()

add_test(NAME spatial_3d_exp_rsdd COMMAND
    spatial 3 11 0.1 0.5 2744 686 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_3d_exp_sdd COMMAND
    spatial 3 11 0.1 0.5 2744 686 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_3d_exp_qp3 COMMAND
    spatial 3 11 0.1 0.5 2744 686 omp_qp3 90 1e-9 1 0)
add_test(NAME spatial_3d_sqrexp_rsdd COMMAND
    spatial 3 12 0.1 0.5 2744 686 omp_rsdd 90 1e-9 1 0)
add_test(NAME spatial_3d_sqrexp_sdd COMMAND
    spatial 3 12 0.1 0.5 2744 686 omp_sdd 90 1e-9 1 0)
add_test(NAME spatial_3d_sqrexp_qp3 COMMAND
    spatial 3 12 0.1 0.5 2744 686 omp_qp3 90 1e-9 1 0)

if(GSL_FOUND)
    add_test(NAME spatial_3d_matern_rsdd COMMAND
        spatial 3 13 0.1 10 2744 686 omp_rsdd 90 1e-9 1 0)
    add_test(NAME spatial_3d_matern_sdd COMMAND
        spatial 3 13 0.1 10 2744 686 omp_sdd 90 1e-9 1 0)
    add_test(NAME spatial_3d_matern_qp3 COMMAND
        spatial 3 13 0.1 10 2744 686 omp_qp3 90 1e-9 1 0)
    add_test(NAME spatial_3d_matern2_rsdd COMMAND
        spatial 3 14 0.1 10 2744 686 omp_rsdd 90 1e-9 1 0)
    add_test(NAME spatial_3d_matern2_sdd COMMAND
        spatial 3 14 0.1 10 2744 686 omp_sdd 90 1e-9 1 0)
    add_test(NAME spatial_3d_matern2_qp3 COMMAND
        spatial 3 14 0.1 10 2744 686 omp_qp3 90 1e-9 1 0)
endif()
