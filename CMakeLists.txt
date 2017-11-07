# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file CMakeLists.txt
# @version 0.1.0
# @author Eduardo Gonzalez Fisher
# @author Aleksandr Mikhalev
# @date 2017-11-07


###############################################################################
#                       THIS IS A TOP-LEVEL CMAKELISTS.txt                    #
#                                                                             #
#        It is intended to find all dependencies (required or optional)       #
#                    and set up corresponding variables                       #
###############################################################################


###############################################################################
##                            PRELIMINARIES                                  ##
###############################################################################

# Need to identify lowest possible CMake version
cmake_minimum_required(VERSION 3.2.3)

# Restrict building in top-level directory
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed.\nPlease create a "
        "build directory first and execute cmake configuration from this "
        "directory. Example: mkdir build && cd build && cmake ..")
endif()

# Check if ECRC CMake modules are there
# This modules are used to find all required and optional dependencies
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake_modules/ecrc/modules")
    message(FATAL_ERROR "ECRC CMake modules were not found.\nHave you done: "
        "'git submodule update --init'?")
endif()

# Append path to CMake modules and init ECRC CMake modules
set(ECRC_CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake_modules/ecrc/modules")
list(APPEND CMAKE_MODULE_PATH "${ECRC_CMAKE_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")
include(EcrcInit)
include(GenPkgConfig)
include(FindPkgConfig)

# Create project and check C compiler
cmake_policy(SET CMP0048 NEW)
project(STARS-H VERSION 0.1.0 LANGUAGES C)

message(STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION}")


###############################################################################
##                           DEFINE OPTIONS                                  ##
###############################################################################

# Option for build type
set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose type of build, options are: None Debug Release\
    RelWithDebInfo MinSizeRel")

# Options for possible backends
option(OPENMP "Use OpenMP" ON)
option(MPI "Use MPI" ON)
option(STARPU "Use StarPU" ON)

# Option to force using parallel blas instead of sequential
option(USE_PARALLEL_BLAS "Prefer parallel blas libraries" OFF)

# Option for Gnu Scientific Library
option(GSL "Use special functions from Gnu Scientific Library" ON)

# Options for examples
option(EXAMPLES "Generate examples binaries" ON)

# Option for testing
option(TESTING "Generate testing binaries" ON)

# Option for documentation
set(DOCS FULL CACHE STRING
    "Choose kind of documentation to build, options are: OFF SHORT FULL")

# Option for STARSH warnings with debug information
option(STARSH_WARNINGS "Show STARSH warnings with debug information" OFF)
if(STARSH_WARNINGS)
    add_definitions("-DSHOW_WARNINGS")
endif()

# Workaround for C99 standard for Intel compiler
if(NOT ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel"))
    set(CMAKE_C_STANDARD 99)
    set(CMAKE_C_STANDARD_REQUIRED ON)
else()
    add_definitions("-std=c99")
endif()

# Set the RPATH config
# --------------------
# use, i.e. don't skip the full RPATH for the build tree
#set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, use the install RPATH already
# (automated test will need this)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# the RPATH to be used when installing
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Packaging (make package)
#set(CPACK_PACKAGE_VERSION ${STARSH_VERSION})
#set(CPACK_GENERATOR "TGZ")
#include(CPack)
find_package(PkgConfig REQUIRED)

# Set path to header files of STARS-H
include_directories("${CMAKE_SOURCE_DIR}/include")


###############################################################################
##                     CHECK IF OPTIONS ARE SUPPORTED                        ##
###############################################################################

# Gather Doxygen enabled sections for SHORT documentation
set(DOXYGEN_ENABLED_SECTIONS)

# Check if OpenMP is supported and expand compilation flags
# Addition of OpenMP flags will be local in future versions
if(OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        add_definitions("-DOPENMP")
        list(APPEND DOXYGEN_ENABLED_SECTIONS OPENMP)
    else()
        set(OPENMP OFF)
    endif()
endif()

# Check if MPI is available
if(MPI)
    find_package(MPI)
    if(MPI_C_FOUND)
        include_directories(${MPI_C_INCLUDE_PATH})
        add_definitions("-DMPI")
        list(APPEND DOXYGEN_ENABLED_SECTIONS MPI)
    else()
        set(MPI OFF)
    endif()
    if(MPI_C_LIBRARIES)
        # Go through the list of mpi libs and add them to
        # EXTRAPKGLIBS, in the form of "-lnameoflibrary"
        foreach(_mpilib ${MPI_C_LIBRARIES} )
            get_filename_component( _extralib ${_mpilib} NAME_WE)
            string(REGEX REPLACE "^lib" " -l" _extralib ${_extralib})
            list(APPEND EXTRAPKGLIBS ${_extralib})
        endforeach()
    endif()
endif()

# Check if StarPU is available and add links to library
# Addition of StarPU directories and libraries will be local in future versions
if(STARPU)
    find_package(STARPU)
    if(STARPU_FOUND)
        include_directories(${STARPU_INCLUDE_DIRS})
        link_directories(${STARPU_LIBRARY_DIRS})
        add_definitions("-DSTARPU")
        list(APPEND DOXYGEN_ENABLED_SECTIONS STARPU)
    else()
        set(STARPU OFF)
    endif()
endif()

# Check if GNU Scientific Library is available (for Matern kernel and
# Bessel function)
if(GSL)
    find_package(GSL)
    if(GSL_FOUND)
        include_directories(${GSL_INCLUDE_DIRS})
        add_definitions("-DGSL")
        list(APPEND DOXYGEN_ENABLED_SECTIONS GSL)
    else()
        set(GSL OFF)
    endif()
endif()

# Check if testing is required
if(TESTING)
    include(CTest)
    enable_testing()
endif()

# Check if need to process with docs
if(NOT DOCS STREQUAL "OFF")
    find_package(Doxygen)
    if(NOT DOXYGEN_FOUND)
        set(DOCS "OFF")
    endif()
endif()


###############################################################################
##                      BLAS, CBLAS, LAPACK AND LAPACKE                      ##
###############################################################################

# Find BLAS and set it to parallel if USE_PARALLEL_BLAS is ON
find_package(BLASEXT)
if(BLAS_FOUND)
    if(USE_PARALLEL_BLAS AND NOT "${BLAS_PAR_LIBRARIES}" STREQUAL "")
        message(STATUS "Using Parallel Blas")
        set(BLAS_LIBRARIES "${BLAS_PAR_LIBRARIES}")
    endif()
else()
    message(FATAL_ERROR "BLAS library has not been found")
endif()

# Find CBLAS and set flags for compiler and linker
find_package(CBLAS COMPONENTS BLASEXT)
if(BLAS_FOUND)
    if (BLAS_LIBRARY_DIRS)
        # the RPATH to be used when installing
        list(APPEND CMAKE_INSTALL_RPATH "${BLAS_LIBRARY_DIRS}")
    endif()
    if(BLAS_LINKER_FLAGS)
        #list(APPEND CMAKE_EXE_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
    endif()
else()
    message(FATAL_ERROR "BLAS library has not been found")
endif()
if(CBLAS_FOUND)
    include_directories(${CBLAS_INCLUDE_DIRS})
    if(CBLAS_LIBRARY_DIRS)
        # the RPATH to be used when installing
        list(APPEND CMAKE_INSTALL_RPATH "${CBLAS_LIBRARY_DIRS}")
    endif()
else()
    if(ECRC_VERBOSE_FIND_PACKAGE)
        if(CBLAS_STANDALONE OR NOT CBLAS_WORKS)
            if (NOT CBLAS_cblas.h_DIRS)
                Print_Find_Header_Status(cblas cblas.h)
            endif ()
            if (NOT CBLAS_cblas_LIBRARY)
                Print_Find_Library_Status(cblas libcblas)
            endif ()
        endif()
    else()
        message(WARNING "CBLAS library has not been found and "
            "ECRC_VERBOSE_FIND_PACKAGE is set to OFF. Try to "
            "activate ECRC_VERBOSE_FIND_PACKAGE option "
            "(-DECRC_VERBOSE_FIND_PACKAGE=ON) to get some hints "
            "for the detection")
    endif()
    message(FATAL_ERROR "A CBLAS library is required but has "
        "not been found")
endif()

# Find LAPACKE and set flags for compiler and linker
find_package(LAPACKE COMPONENTS LAPACKEXT)
if(LAPACK_FOUND AND LAPACK_LIBRARY_DIRS)
    # the RPATH to be used when installing
    list(APPEND CMAKE_INSTALL_RPATH "${LAPACK_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "A LAPACK library is required but has not "
        "been found")
endif()
if(LAPACKE_FOUND)
    include_directories(${LAPACKE_INCLUDE_DIRS})
    if(LAPACKE_LIBRARY_DIRS)
        # the RPATH to be used when installing
        list(APPEND CMAKE_INSTALL_RPATH "${LAPACKE_LIBRARY_DIRS}")
    endif()
    if(LAPACKE_LINKER_FLAGS)
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} ${LAPACKE_LINKER_FLAGS}")
    endif()
else()
    if(ECRC_VERBOSE_FIND_PACKAGE)
        if (LAPACKE_STANDALONE OR NOT LAPACKE_WORKS)
            if (NOT LAPACKE_lapacke.h_DIRS)
                Print_Find_Header_Status(lapacke lapacke.h)
            endif ()
            if (NOT LAPACKE_lapacke_LIBRARY)
                Print_Find_Library_Status(lapacke liblapacke)
            endif ()
        endif()
    else()
        message(WARNING "LAPACKE library has not been found and "
            "ECRC_VERBOSE_FIND_PACKAGE is set to OFF. Try to "
            "activate ECRC_VERBOSE_FIND_PACKAGE option "
            "(-DECRC_VERBOSE_FIND_PACKAGE=ON) to get some hints "
            "for the detection")
    endif()
    message(FATAL_ERROR "A LAPACKE library is required but has not "
        "been found")
endif()
#list(APPEND REQUIRES ${LAPACKE_LIBRARIES} ${CBLAS_LIBRARIES})

# Check if BLAS and LAPACK are provided by Intel
# (use mkl.h instead of lapacke.h and cblas.h in this case)
if(BLA_VENDOR MATCHES "Intel")
    add_definitions("-DMKL")
endif()


###############################################################################
##                          PRINT CONFIGURATION                              ##
###############################################################################

get_directory_property(STARSH_DEFINITIONS_LIST DIRECTORY ${CMAKE_SOURCE_DIR}
    COMPILE_DEFINITIONS)
include(PrintOpts)


###############################################################################
##                 FILL STARSH.PC FILE FOR PKG_CONFIG                        ##
###############################################################################

# Set parameters for PKGCONFIG
set(PREFIX "${CMAKE_INSTALL_PREFIX}")
set(LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
set(INCDIR "${CMAKE_INSTALL_PREFIX}/include")

#Install Pkg-config file
generate_pkgconfig_file()


###############################################################################
##               LIST FILES FOR DOCUMENTATION AND COMPILATION                ##
###############################################################################

# Collect STARS-H sources + headers
set(DOXYGEN_INPUT)
add_subdirectory("src")
add_subdirectory("include")


###############################################################################
##                  BUILD EXAMPLES, TESTS AND DOCS                           ##
###############################################################################

# If examples option is ON
if(EXAMPLES)
    add_subdirectory("examples")
endif()

# If testing option is ON
if(TESTING)
    add_subdirectory("testing")
endif()

# Build docs, if needed
if(NOT DOCS STREQUAL "OFF")
    add_subdirectory("docs")
endif()

