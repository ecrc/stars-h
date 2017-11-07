# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file examples/approximation/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


set(examples_files)

if(OPENMP)
    list(APPEND examples_files
        "spatial.c"
        "randtlr.c"
        "minimal.c"
        "dense.c"
    )
endif()

# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach(example_src ${examples_files})
    get_filename_component(example ${example_src} NAME_WE)
    add_executable(example_approximation_${example} ${example_src})
    target_link_libraries(example_approximation_${example} starsh
        ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES} ${OpenMP_C_FLAGS})
    set_target_properties(example_approximation_${example} PROPERTIES
        OUTPUT_NAME ${example})
endforeach()
