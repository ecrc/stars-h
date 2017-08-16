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

# Set possible approximation lrengines
set(LRENGINES "SVD" "RRQR" "RSVD")

# Add tests for minimal example
foreach(lrengine IN ITEMS ${LRENGINES})
    add_test(NAME minimal_${lrengine} COMMAND
        minimal 2500 500 10 1e-9)
    set(test_env "MKL_NUM_THREADS=1"
        "STARSH_BACKEND=OPENMP"
        "STARSH_LRENGINE=${lrengine}")
    set_tests_properties(minimal_${lrengine} PROPERTIES
        ENVIRONMENT "${test_env}")
    if(MPI)
        add_test(NAME mpi_minimal_${lrengine} COMMAND
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
            ./mpi_minimal 2500 500 10 1e-9)
        set(test_env "MKL_NUM_THREADS=1"
            "OMP_NUM_THREADS=${NOMP}"
            "STARSH_BACKEND=MPI_OPENMP"
            "STARSH_LRENGINE=${lrengine}")
        set_tests_properties(mpi_minimal_${lrengine} PROPERTIES
            ENVIRONMENT "${test_env}")
    endif()
endforeach()

# Add tests for synthetic TLR matrices
foreach(lrengine IN ITEMS ${LRENGINES})
    add_test(NAME rndtiled_${lrengine} COMMAND
        rndtiled 2500 250 0.5 100 1e-9)
    set(test_env "MKL_NUM_THREADS=1"
        "STARSH_BACKEND=OPENMP"
        "STARSH_LRENGINE=${lrengine}")
    set_tests_properties(rndtiled_${lrengine} PROPERTIES
        ENVIRONMENT "${test_env}")
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
foreach(lrengine IN ITEMS ${LRENGINES})
    foreach(kernel RANGE ${NKERNELS})
        list(GET KERNAMES ${kernel} KERNAME)
        list(GET KERCODES ${kernel} KERCODE)
        add_test(NAME spatial_2d_${KERNAME}_${lrengine} COMMAND
            spatial 2 ${KERCODE} 0.1 10 2500 500 90 1e-9 1 0)
        set(test_env "MKL_NUM_THREADS=1"
            "STARSH_BACKEND=OPENMP"
            "STARSH_LRENGINE=${lrengine}")
        set_tests_properties(spatial_2d_${KERNAME}_${lrengine} PROPERTIES
            ENVIRONMENT "${test_env}")
        add_test(NAME spatial_3d_${KERNAME}_${lrengine} COMMAND
            spatial 3 ${KERCODE} 0.1 10 3375 675 240 1e-9 1 0)
        set_tests_properties(spatial_3d_${KERNAME}_${lrengine} PROPERTIES
            ENVIRONMENT "${test_env}")
        if(MPI)
            add_test(NAME mpi_spatial_2d_${KERNAME}_${lrengine} COMMAND
                ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                ./mpi_spatial 2 ${KERCODE} 0.1 10 2500 500 90 1e-9 1 0)
            set(test_env "MKL_NUM_THREADS=1"
                "OMP_NUM_THREADS=${NOMP}"
                "STARSH_BACKEND=MPI_OPENMP"
                "STARSH_LRENGINE=${lrengine}")
            set_tests_properties(mpi_spatial_2d_${KERNAME}_${lrengine} PROPERTIES
                ENVIRONMENT "${test_env}")
            add_test(NAME mpi_spatial_3d_${KERNAME}_${lrengine} COMMAND
                ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                ./mpi_spatial 3 ${KERCODE} 0.1 10 3375 675 240 1e-9 1 0)
            set_tests_properties(mpi_spatial_3d_${KERNAME}_${lrengine} PROPERTIES
                ENVIRONMENT "${test_env}")
        endif()
    endforeach()
endforeach()
