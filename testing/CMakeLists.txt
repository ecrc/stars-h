# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file testing/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


# Defines the source files of the tests. Each file generates 1 test
set(tests_files
    "particles.c")

file(COPY "particles.txt"
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# NEED TO PUT EXPLICIT DEPENDENCIES ON OTHER LIBRARIES HERE
# However, it works for static library libstarsh.a and will NOT work for shared

if(OPENMP)
    list(APPEND tests_files
        "minimal.c"
        "cauchy.c"
        "spatial.c"
        "electrostatics.c"
        "electrodynamics.c"
        "randtlr.c"
        )
endif()

if(MPI)
    list(APPEND tests_files
        "mpi_minimal.c"
        "mpi_cauchy.c"
        "mpi_spatial.c"
        "mpi_electrostatics.c"
        "mpi_electrodynamics.c"
        )
endif()

if(STARPU)
    list(APPEND tests_files
        "starpu_minimal.c"
        "starpu_cauchy.c"
        "starpu_spatial.c"
        "starpu_electrostatics.c"
        "starpu_electrodynamics.c"
        )
endif()

if(MPI AND STARPU)
    list(APPEND tests_files
        "mpi_starpu_minimal.c"
        "mpi_starpu_cauchy.c"
        "mpi_starpu_spatial.c"
        "mpi_starpu_electrostatics.c"
        "mpi_starpu_electrodynamics.c"
        )
endif()

# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach(test_src ${tests_files})
    get_filename_component(test_exe ${test_src} NAME_WE)
    add_executable(test_${test_exe} ${test_src})
    target_link_libraries(test_${test_exe} starsh ${CBLAS_LIBRARIES}
        ${LAPACKE_LIBRARIES} ${OpenMP_C_FLAGS})
    if(test_src MATCHES "starpu_*")
        target_link_libraries(test_${test_exe} ${STARPU_LIBRARIES})
    endif()
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

# Add tests for IO
add_test(NAME particles_io COMMAND particles)

# Add tests for minimal example
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
if(OPENMP)
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
        if(STARPU)
            add_test(NAME starpu_minimal_${lrengine} COMMAND
                starpu_minimal 2500 500 10 1e-9)
            set(test_env "MKL_NUM_THREADS=1"
                "OMP_NUM_THREADS=1"
                "STARSH_BACKEND=STARPU"
                "STARSH_LRENGINE=${lrengine}")
            set_tests_properties(starpu_minimal_${lrengine} PROPERTIES
                ENVIRONMENT "${test_env}")
        endif()
    endforeach()
endif()
if(MPI AND STARPU)
    foreach(lrengine IN ITEMS ${LRENGINES})
        add_test(NAME mpi_starpu_minimal_${lrengine} COMMAND
            mpi_starpu_minimal 2500 500 10 1e-9)
        set(test_env "MKL_NUM_THREADS=1"
            "OMP_NUM_THREADS=1"
            "STARPU_NCPU=${NOMP}"
            "STARSH_BACKEND=MPI_STARPU"
            "STARSH_LRENGINE=${lrengine}")
        set_tests_properties(mpi_starpu_minimal_${lrengine} PROPERTIES
            ENVIRONMENT "${test_env}")
    endforeach()
endif()


# Add tests for Cauchy matrices
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
if(OPENMP)
    foreach(lrengine IN ITEMS ${LRENGINES})
        add_test(NAME cauchy_${lrengine} COMMAND
            cauchy 2500 250 100 1e-9)
        set(test_env "MKL_NUM_THREADS=1"
            "STARSH_BACKEND=OPENMP"
            "STARSH_LRENGINE=${lrengine}")
        set_tests_properties(cauchy_${lrengine} PROPERTIES
            ENVIRONMENT "${test_env}")
        if(MPI)
            add_test(NAME mpi_cauchy_${lrengine} COMMAND
                ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                ./mpi_cauchy 2500 500 10 1e-9)
            set(test_env "MKL_NUM_THREADS=1"
                "OMP_NUM_THREADS=${NOMP}"
                "STARSH_BACKEND=MPI_OPENMP"
                "STARSH_LRENGINE=${lrengine}")
            set_tests_properties(mpi_minimal_${lrengine} PROPERTIES
                ENVIRONMENT "${test_env}")
        endif()
        if(STARPU)
            add_test(NAME starpu_cauchy_${lrengine} COMMAND
                starpu_cauchy 2500 500 10 1e-9)
            set(test_env "MKL_NUM_THREADS=1"
                "OMP_NUM_THREADS=1"
                "STARSH_BACKEND=STARPU"
                "STARSH_LRENGINE=${lrengine}")
            set_tests_properties(starpu_cauchy_${lrengine} PROPERTIES
                ENVIRONMENT "${test_env}")
        endif()
    endforeach()
endif()
if(MPI AND STARPU)
    foreach(lrengine IN ITEMS ${LRENGINES})
        add_test(NAME mpi_starpu_cauchy_${lrengine} COMMAND
            mpi_starpu_cauchy 2500 500 10 1e-9)
        set(test_env "MKL_NUM_THREADS=1"
            "OMP_NUM_THREADS=1"
            "STARPU_NCPU=${NOMP}"
            "STARSH_BACKEND=MPI_STARPU"
            "STARSH_LRENGINE=${lrengine}")
        set_tests_properties(mpi_starpu_cauchy_${lrengine} PROPERTIES
            ENVIRONMENT "${test_env}")
    endforeach()
endif()


# Add tests for synthetic TLR matrices
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
if(OPENMP)
    foreach(lrengine IN ITEMS ${LRENGINES})
        add_test(NAME randtlr_${lrengine} COMMAND
            randtlr 2500 250 0.5 100 1e-9)
        set(test_env "MKL_NUM_THREADS=1"
            "STARSH_BACKEND=OPENMP"
            "STARSH_LRENGINE=${lrengine}")
        set_tests_properties(randtlr_${lrengine} PROPERTIES
            ENVIRONMENT "${test_env}")
    endforeach()
endif()


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
# Set possible generation schemes for particles
set(PLACENAMES "rand" "uniform" "randgrid" "quasiuniform1" "quasiuniform2"
    "obsolete1" "obsolete2")
set(PLACEMENTS "1" "3" "4" "6" "7" "-1" "-2")
list(LENGTH PLACEMENTS NPLACES)
math(EXPR NPLACES "${NPLACES}-1")
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
if(OPENMP)
    # Then cycle over all supported configurations for spatial statistics tests
    foreach(lrengine IN ITEMS ${LRENGINES})
        foreach(kernel RANGE ${NKERNELS})
            foreach(place RANGE ${NPLACES})
                list(GET KERNAMES ${kernel} KERNAME)
                list(GET KERCODES ${kernel} KERCODE)
                list(GET PLACEMENTS ${place} placement)
                list(GET PLACENAMES ${place} placename)
                add_test(NAME spatial_2d_${KERNAME}_${lrengine}_${placename}
                    COMMAND spatial 2 ${placement} ${KERCODE} 0.1 10 2500 500
                    90 1e-9)
                set(test_env "MKL_NUM_THREADS=1"
                    "STARSH_BACKEND=OPENMP"
                    "STARSH_LRENGINE=${lrengine}")
                set_tests_properties(
                    spatial_2d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                add_test(NAME spatial_3d_${KERNAME}_${lrengine}_${placename}
                    COMMAND spatial 3 ${placement} ${KERCODE} 0.1 10 3375 675
                    240 1e-9)
                set_tests_properties(
                    spatial_3d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                if(MPI)
                    add_test(NAME
                        mpi_spatial_2d_${KERNAME}_${lrengine}_${placename}
                        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                        ./mpi_spatial 2 ${placement} ${KERCODE} 0.1 10 2500 500
                        90 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=${NOMP}"
                        "STARSH_BACKEND=MPI_OPENMP"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        mpi_spatial_2d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                    add_test(NAME
                        mpi_spatial_3d_${KERNAME}_${lrengine}_${placename}
                        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                        ./mpi_spatial 3 ${placement} ${KERCODE} 0.1 10 3375 675
                        240 1e-9)
                    set_tests_properties(
                        mpi_spatial_3d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                endif()
                if(STARPU)
                    add_test(NAME
                        starpu_spatial_2d_${KERNAME}_${lrengine}_${placename}
                        COMMAND starpu_spatial 2 ${placement} ${KERCODE} 0.1 10
                        2500 500 90 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=1"
                        "STARSH_BACKEND=STARPU"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        starpu_spatial_2d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                    add_test(NAME
                        starpu_spatial_3d_${KERNAME}_${lrengine}_${placename}
                        COMMAND starpu_spatial 3 ${placement} ${KERCODE} 0.1 10
                        3375 675 240 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=1"
                        "STARSH_BACKEND=STARPU"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        starpu_spatial_3d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                endif()
            endforeach()
        endforeach()
    endforeach()
endif()
# Check if MPI and STARPU are supported
if(MPI AND STARPU)
    # Then cycle over all supported configurations for spatial statistics tests
    foreach(lrengine IN ITEMS ${LRENGINES})
        foreach(kernel RANGE ${NKERNELS})
            foreach(place RANGE ${NPLACES})
                list(GET KERNAMES ${kernel} KERNAME)
                list(GET KERCODES ${kernel} KERCODE)
                list(GET PLACEMENTS ${place} placement)
                list(GET PLACENAMES ${place} placename)
                add_test(NAME
                    mpi_starpu_spatial_2d_${KERNAME}_${lrengine}_${placename}
                    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                    ./mpi_starpu_spatial 2 ${placement} ${KERCODE} 0.1 10 2500
                    500 90 1e-9)
                set(test_env "MKL_NUM_THREADS=1"
                    "OMP_NUM_THREADS=1"
                    "STARPU_NCPU=${NOMP}"
                    "STARSH_BACKEND=MPI_STARPU"
                    "STARSH_LRENGINE=${lrengine}")
                set_tests_properties(
                    mpi_starpu_spatial_2d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                add_test(NAME
                    mpi_starpu_spatial_3d_${KERNAME}_${lrengine}_${placename}
                    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                    ./mpi_starpu_spatial 3 ${placement} ${KERCODE} 0.1 10 3375
                    675 240 1e-9)
                set_tests_properties(
                    mpi_starpu_spatial_3d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
            endforeach()
        endforeach()
    endforeach()
endif()


# Add tests for electrostatics
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
# At first decide what matrix kernels are supported
set(KERNAMES)
set(KERCODES)
list(APPEND KERNAMES "coulomb_potential")
list(APPEND KERCODES "2")
list(LENGTH KERNAMES NKERNELS)
math(EXPR NKERNELS "${NKERNELS}-1")
# Set possible generation schemes for particles
set(PLACENAMES "rand" "uniform" "randgrid" "quasiuniform1" "quasiuniform2"
    "obsolete1" "obsolete2")
set(PLACEMENTS "1" "3" "4" "6" "7" "-1" "-2")
list(LENGTH PLACEMENTS NPLACES)
math(EXPR NPLACES "${NPLACES}-1")
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
if(OPENMP)
    # Then cycle over all supported configurations for electrostatics tests
    foreach(lrengine IN ITEMS ${LRENGINES})
        foreach(kernel RANGE ${NKERNELS})
            foreach(place RANGE ${NPLACES})
                list(GET KERNAMES ${kernel} KERNAME)
                list(GET KERCODES ${kernel} KERCODE)
                list(GET PLACEMENTS ${place} placement)
                list(GET PLACENAMES ${place} placename)
                add_test(NAME
                    electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                    COMMAND electrostatics 2 ${placement} ${KERCODE} 2500 500
                    90 1e-9)
                set(test_env "MKL_NUM_THREADS=1"
                    "STARSH_BACKEND=OPENMP"
                    "STARSH_LRENGINE=${lrengine}")
                set_tests_properties(
                    electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                add_test(NAME
                    electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                    COMMAND electrostatics 3 ${placement} ${KERCODE} 3375 675
                    240 1e-9)
                set_tests_properties(
                    electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                if(MPI)
                    add_test(NAME
                        mpi_electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                        ./mpi_electrostatics 2 ${placement} ${KERCODE} 2500 500
                        90 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=${NOMP}"
                        "STARSH_BACKEND=MPI_OPENMP"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        mpi_electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                    add_test(NAME
                        mpi_electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                        ./mpi_electrostatics 3 ${placement} ${KERCODE} 3375 675
                        240 1e-9)
                    set_tests_properties(
                        mpi_electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                endif()
                if(STARPU)
                    add_test(NAME
                        starpu_electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                        COMMAND starpu_electrostatics 2 ${placement} ${KERCODE}
                        2500 500 90 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=1"
                        "STARSH_BACKEND=STARPU"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        starpu_electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                    add_test(NAME
                        starpu_electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                        COMMAND starpu_electrostatics 3 ${placement} ${KERCODE}
                        3375 675 240 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=1"
                        "STARSH_BACKEND=STARPU"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        starpu_electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                endif()
            endforeach()
        endforeach()
    endforeach()
endif()
# Check if MPI and STARPU are supported
if(MPI AND STARPU)
    # Then cycle over all supported configurations for spatial statistics tests
    foreach(lrengine IN ITEMS ${LRENGINES})
        foreach(kernel RANGE ${NKERNELS})
            foreach(place RANGE ${NPLACES})
                list(GET KERNAMES ${kernel} KERNAME)
                list(GET KERCODES ${kernel} KERCODE)
                list(GET PLACEMENTS ${place} placement)
                list(GET PLACENAMES ${place} placename)
                add_test(NAME
                    mpi_starpu_electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                    ./mpi_starpu_electrostatics 2 ${placement} ${KERCODE} 2500
                    500 90 1e-9)
                set(test_env "MKL_NUM_THREADS=1"
                    "OMP_NUM_THREADS=1"
                    "STARPU_NCPU=${NOMP}"
                    "STARSH_BACKEND=MPI_STARPU"
                    "STARSH_LRENGINE=${lrengine}")
                set_tests_properties(
                    mpi_starpu_electrostatics_2d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                add_test(NAME
                    mpi_starpu_electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                    ./mpi_starpu_electrostatics 3 ${placement} ${KERCODE} 3375
                    675 240 1e-9)
                set_tests_properties(
                    mpi_starpu_electrostatics_3d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
            endforeach()
        endforeach()
    endforeach()
endif()


# Add tests for electrodynamics
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
# At first decide what matrix kernels are supported
set(KERNAMES)
set(KERCODES)
list(APPEND KERNAMES "sin" "cos")
list(APPEND KERCODES "11" "12")
list(LENGTH KERNAMES NKERNELS)
math(EXPR NKERNELS "${NKERNELS}-1")
# Set possible generation schemes for particles
set(PLACENAMES "rand" "uniform" "randgrid" "quasiuniform1" "quasiuniform2"
    "obsolete1" "obsolete2")
set(PLACEMENTS "1" "3" "4" "6" "7" "-1" "-2")
list(LENGTH PLACEMENTS NPLACES)
math(EXPR NPLACES "${NPLACES}-1")
# Check if OPENMP is supported, since we use omp_get_wtime function to measure
# performance
if(OPENMP)
    # Then cycle over all supported configurations for electrodynamics tests
    foreach(lrengine IN ITEMS ${LRENGINES})
        foreach(kernel RANGE ${NKERNELS})
            foreach(place RANGE ${NPLACES})
                list(GET KERNAMES ${kernel} KERNAME)
                list(GET KERCODES ${kernel} KERCODE)
                list(GET PLACEMENTS ${place} placement)
                list(GET PLACENAMES ${place} placename)
                add_test(NAME
                    electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                    COMMAND electrodynamics 2 ${placement} ${KERCODE} 6 1
                    2500 500 90 1e-9)
                set(test_env "MKL_NUM_THREADS=1"
                    "STARSH_BACKEND=OPENMP"
                    "STARSH_LRENGINE=${lrengine}")
                set_tests_properties(
                    electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                add_test(NAME
                    electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                    COMMAND electrodynamics 3 ${placement} ${KERCODE} 6 1
                    3375 675 240 1e-9)
                set_tests_properties(
                    electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                if(MPI)
                    add_test(NAME
                        mpi_electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                        ./mpi_electrodynamics 2 ${placement} ${KERCODE} 6 1
                        2500 500 90 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=${NOMP}"
                        "STARSH_BACKEND=MPI_OPENMP"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        mpi_electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                    add_test(NAME
                        mpi_electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                        ./mpi_electrodynamics 3 ${placement} ${KERCODE} 6 1
                        3375 675 240 1e-9)
                    set_tests_properties(
                        mpi_electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                endif()
                if(STARPU)
                    add_test(NAME
                        starpu_electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                        COMMAND starpu_electrodynamics 2 ${placement}
                        ${KERCODE} 6 1 2500 500 90 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=1"
                        "STARSH_BACKEND=STARPU"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        starpu_electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                    add_test(NAME
                        starpu_electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                        COMMAND starpu_electrodynamics 3 ${placement}
                        ${KERCODE} 6 1 3375 675 240 1e-9)
                    set(test_env "MKL_NUM_THREADS=1"
                        "OMP_NUM_THREADS=1"
                        "STARSH_BACKEND=STARPU"
                        "STARSH_LRENGINE=${lrengine}")
                    set_tests_properties(
                        starpu_electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                        PROPERTIES ENVIRONMENT "${test_env}")
                endif()
            endforeach()
        endforeach()
    endforeach()
endif()
# Check if MPI and STARPU are supported
if(MPI AND STARPU)
    # Then cycle over all supported configurations for spatial statistics tests
    foreach(lrengine IN ITEMS ${LRENGINES})
        foreach(kernel RANGE ${NKERNELS})
            foreach(place RANGE ${NPLACES})
                list(GET KERNAMES ${kernel} KERNAME)
                list(GET KERCODES ${kernel} KERCODE)
                list(GET PLACEMENTS ${place} placement)
                list(GET PLACENAMES ${place} placename)
                add_test(NAME
                    mpi_starpu_electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                    ./mpi_starpu_electrodynamics 2 ${placement} ${KERCODE} 6 1
                    2500 500 90 1e-9)
                set(test_env "MKL_NUM_THREADS=1"
                    "OMP_NUM_THREADS=1"
                    "STARPU_NCPU=${NOMP}"
                    "STARSH_BACKEND=MPI_STARPU"
                    "STARSH_LRENGINE=${lrengine}")
                set_tests_properties(
                    mpi_starpu_electrodynamics_2d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
                add_test(NAME
                    mpi_starpu_electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                    ./mpi_starpu_electrodynamics 3 ${placement} ${KERCODE} 6 1
                    3375 675 240 1e-9)
                set_tests_properties(
                    mpi_starpu_electrodynamics_3d_${KERNAME}_${lrengine}_${placename}
                    PROPERTIES ENVIRONMENT "${test_env}")
            endforeach()
        endforeach()
    endforeach()
endif()
