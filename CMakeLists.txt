###############################################################################
#                       THIS IS A TOP-LEVEL CMAKELISTS.txt                    #
#                                                                             #
#        It is intended to find all dependencies (required or optional)       #
#                    and set up corresponding variables                       #
###############################################################################

# Need to identify lowest possible CMake version
cmake_minimum_required(VERSION 3.1.3)

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

# Create project and check C compiler
cmake_policy(SET CMP0048 NEW)
project(STARS-H VERSION 1.0.0.2 LANGUAGES C)

message(STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION}")

# Set build type for optimization flags
set(CMAKE_BUILD_TYPE Release)

# Set desired backends
# One may use -DSTARPU=OFF to disable StarPU support manually
option(STARPU "Use StarPU backend" ON)
option(OPENMP "Use OpenMP support" ON)

# Set desired external libraries
# One may use -DPLASMA=OFF to disable wrappers for PLASMA
option(PLASMA "Wrappers for PLASMA" ON)

# Options for examples, testing and documentation
option(TESTING "Generate testing binaries" ON)
option(EXAMPLES "Generate example binaries" OFF)
option(DOCS "Build documentation in docs directory" OFF)

# Use STARPU backend only when compiling by gcc
if(NOT ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU"))
    set(STARPU OFF)
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

# Append path to CMake modules and init ECRC CMake modules
set(ECRC_CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake_modules/ecrc/modules")
list(APPEND CMAKE_MODULE_PATH "${ECRC_CMAKE_MODULE_PATH}")
include(EcrcInit)
include(FindPkgConfig)
find_package(PkgConfig QUIET)

# init requirements
set(REQUIRES)

# Set path to header files of STARS-H
include_directories("${CMAKE_SOURCE_DIR}/include")

# Check if OpenMP is supported and expand compilation flags
# Addition of OpenMP flags will be local in future versions
if(OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        add_definitions("-DOPENMP")
    else()
        set(OPENMP OFF)
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
        list(APPEND REQUIRES ${STARPU_LIBRARIES})
    else()
        message("StarPU is NOT found, skipping it")
        set(STARPU OFF)
    endif()
endif()

# Check if PLASMA is available to build wrappers
if(PLASMA)
    pkg_search_module(PLASMA plasma)
    if(PLASMA_FOUND)
        include_directories(${PLASMA_INCLUDE_DIRS})
        link_directories(${PLASMA_LIBRARY_DIRS})
        add_definitions("-DPLASMA")
        list(APPEND REQUIRES ${PLASMA_LIBRARIES})
    else()
        message("PLASMA is NOT found, skipping it")
        set(PLASMA OFF)
    endif()
endif()

# Find LAPACK and BLAS for tests and examples
if(TESTING OR EXAMPLES)
    # Enable testing
    include(CTest)
    enable_testing()
    # Find BLAS
    find_package(BLASEXT)
    if(BLAS_FOUND)
        # Use parallel blas for cblas
        option(USE_PARALLEL_BLAS "Prefer parallel blas libraries" ON)
        if(USE_PARALLEL_BLAS AND NOT "${BLAS_PAR_LIBRARIES}" STREQUAL "")
            message(STATUS "Using Parallel Blas")
            set(BLAS_LIBRARIES "${BLAS_PAR_LIBRARIES}")
        endif()
    else()
        message(FATAL_ERROR "BLAS library has not been found")
    endif()
    # Find CBLAS
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
    ## LAPACKE
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
endif()

if(TESTING)
    add_subdirectory("testing")
endif()

if(EXAMPLES)
    add_subdirectory("examples")
endif()

if(DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory("docs")
    else()
        message(STATUS "Doxygen NOT found, skipping it")
    endif()
endif()

# PKGCONFIG
set(PREFIX "${CMAKE_INSTALL_PREFIX}")
set(LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
set(INCDIR "${CMAKE_INSTALL_PREFIX}/include")


# Collect STARS-H sources + headers
add_subdirectory("src")
add_subdirectory("include")

#Install Pkg-config file
set(PKG_CONFIG_DIR "${LIBDIR}/pkgconfig")
string(REPLACE ";" "," REQUIRES "${REQUIRES}")
configure_file("${CMAKE_SOURCE_DIR}/starsh.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/starsh.pc")
install(FILES "${CMAKE_BINARY_DIR}/starsh.pc"
    DESTINATION "${PKG_CONFIG_DIR}")
