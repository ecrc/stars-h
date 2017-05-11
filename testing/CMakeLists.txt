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

include(ProcessorCount)
ProcessorCount(N)
if(N EQUAL 0)
    message(ERROR "Could not get number of processors/cores for tests")
else()
    message(STATUS "Number of processors/cores for tests is ${N}")
endif()
# NOMP is number of threads per MPI process (because we test with 4 MPI
#   processes)
math(EXPR NOMP ${N}/4)

# Set possible approximation schemes
set(SCHEMES "rsdd" "sdd" "qp3")

# Add tests for minimal example
foreach(scheme IN ITEMS ${SCHEMES})
    add_test(NAME minimal_${scheme} COMMAND
        minimal 2500 500 omp_${scheme} 10 1e-9)
        set_tests_properties(minimal_${scheme} PROPERTIES
            ENVIRONMENT "MKL_NUM_THREADS=1")
    if(MPI)
        add_test(NAME mpi_minimal_${scheme} COMMAND
            ./mpi_minimal 2500 500 mpi_${scheme} 10 1e-9)
            set_tests_properties(mpi_minimal_${scheme} PROPERTIES
                ENVIRONMENT "MKL_NUM_THREADS=1")
            set_tests_properties(mpi_minimal_${scheme} PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=${NOMP}")
    endif()
endforeach()

# Add tests for spatial statistics
# At first decide what matrix kernels are supported
set(KERNAMES)
set(KERCODES)
list(APPEND KERNAMES "exp" "sqrexp")
list(APPEND KERCODES "11" "12")
if(GSL_FOUND)
    list(APPEND KERNAMES "matern" "matern2")
    list(APPEND KERCODES "13" "14")
endif()
list(LENGTH KERNAMES NKERNELS)
math(EXPR NKERNELS "${NKERNELS}-1")
# Then cycle over all supported configurations
foreach(scheme IN ITEMS ${SCHEMES})
    foreach(kernel RANGE ${NKERNELS})
        list(GET KERNAMES ${kernel} KERNAME)
        list(GET KERCODES ${kernel} KERCODE)
        add_test(NAME spatial_2d_${KERNAME}_${scheme} COMMAND
            spatial 2 ${KERCODE} 0.1 10 2500 500 omp_${scheme} 90 1e-9 1 0)
        set_tests_properties(spatial_2d_${KERNAME}_${scheme} PROPERTIES
            ENVIRONMENT "MKL_NUM_THREADS=1")
        add_test(NAME spatial_3d_${KERNAME}_${scheme} COMMAND
            spatial 3 ${KERCODE} 0.1 10 3375 675 omp_${scheme} 240 1e-9 1 0)
        set_tests_properties(spatial_3d_${KERNAME}_${scheme} PROPERTIES
            ENVIRONMENT "MKL_NUM_THREADS=1")
        if(MPI)
            add_test(NAME mpi_spatial_2d_${KERNAME}_${scheme} COMMAND
                ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                ./mpi_spatial 2 ${KERCODE} 0.1 10 2500 500 mpi_${scheme} 90
                1e-9 1 0)
            set_tests_properties(mpi_spatial_2d_${KERNAME}_${scheme} PROPERTIES
                ENVIRONMENT "MKL_NUM_THREADS=1")
            set_tests_properties(mpi_spatial_2d_${KERNAME}_${scheme} PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=${NOMP}")
            add_test(NAME mpi_spatial_3d_${KERNAME}_${scheme} COMMAND
                ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                ./mpi_spatial 3 ${KERCODE} 0.1 10 3375 675 mpi_${scheme} 240
                1e-9 1 0)
            set_tests_properties(mpi_spatial_3d_${KERNAME}_${scheme} PROPERTIES
                ENVIRONMENT "MKL_NUM_THREADS=1")
            set_tests_properties(mpi_spatial_3d_${KERNAME}_${scheme} PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=${NOMP}")
        endif()
    endforeach()
endforeach()
