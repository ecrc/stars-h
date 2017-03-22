# Defines the source files of the tests. Each file generates 1 test
set(tests_files "symm_positivity_after_compression.c"
    "mpi.c")

if( PLASMA_FOUND )
    list(APPEND tests_files "plasma/solve.c")
endif()

# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach(test_src ${tests_files})
    get_filename_component(test_exe ${test_src} NAME_WE)
    add_executable(${test_exe} ${test_src})
    target_link_libraries(${test_exe} starsh ${CBLAS_LIBRARIES}
        ${LAPACKE_LIBRARIES} ${OpenMP_C_FLAGS} ${PLASMA_LIBRARIES})
endforeach()
