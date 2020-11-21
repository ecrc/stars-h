# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file examples/problem/CMakeLists.txt
# @version 0.3.0
# @author Aleksandr Mikhalev
# @date 2020-06-09


set(examples_files
    "spatial.c"
    "spatial_bivariate.c"
    "randtlr.c"
    "minimal.c"
    "dense.c"
    "particles.c"
    "rbf_virus.c"
    "rbf_cube.c"
    "acoustic.c"
    )


file(COPY "particles.txt"
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach(example_src ${examples_files})
    get_filename_component(example ${example_src} NAME_WE)
    add_executable(example_problem_${example} ${example_src})
    target_link_libraries(example_problem_${example} starsh
        ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES} ${OpenMP_C_FLAGS})
    set_target_properties(example_problem_${example} PROPERTIES
        OUTPUT_NAME ${example})
endforeach()
