# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/applications/electrostatics/CMakeLists.txt
# @version 0.1.0
# @author Aleksandr Mikhalev
# @date 2017-11-07

EXECUTE_PROCESS(COMMAND "python"
    "../misc_scripts/code_generation/applications/particles/kernel_nd.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_coulomb_potential.c"
    "${CMAKE_CURRENT_BINARY_DIR}" OUTPUT_VARIABLE generated_files)

#message("${generated_files}")
set(STARSH_SRC
    ${generated_files}
    ${STARSH_SRC} PARENT_SCOPE)
