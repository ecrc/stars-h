cmake_minimum_required(VERSION 2.8)
#project(STARS-H C CXX)
project(STARS-H C)


# directly make an error if in-source build
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
	message(FATAL_ERROR "In-source builds are not allowed.\n"
	"Please create a build directory first and execute cmake configuration from "
	"this directory. Example: mkdir build && cd build && cmake ..")
endif()


# Version
# Number 0.1.1 is temptative, may differ from actual release.
SET( STARSH_VERSION_MAJOR 0 )
SET( STARSH_VERSION_MINOR 1 )
SET( STARSH_VERSION_PATCH 1 )
set( STARSH_VERSION "${STARSH_VERSION_MAJOR}.${STARSH_VERSION_MINOR}.${STARSH_VERSION_PATCH}" )


# Set the RPATH config
# --------------------
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, use the install RPATH already
# (automated test will need this)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# Packaging (make package)
SET(CPACK_PACKAGE_VERSION ${STARSH_VERSION})
SET(CPACK_GENERATOR "TGZ")
INCLUDE(CPack)
INCLUDE(CMakePushCheckState)

# This project eeds C99 standard to compile properly
# (could have used CMAKE_C_FLAGS instead)
ADD_DEFINITIONS( -std=gnu99 )

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


## CMAKE MODULES :: ECRC
## REQUIRED FOR TESTS TO LINK LIBRARIES
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/cmake_modules/ecrc/modules" )
	message(FATAL_ERROR "ECRC CMake modules were not found.\n"
	                    "Have you done: 'git submodule init && git submodule update'?" )
else()
	## ECRC INITIALIZATION
	list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
	list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/ecrc/modules")
        set(ECRC_CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules/ecrc/modules )
        include(EcrcInit)
endif()



# STARS-H code
add_subdirectory(src)


# TESTS
option(STARSH_TESTING "Generates testing binaries" OFF)
if( STARSH_TESTING )
	## BLAS
	find_package(BLASEXT)
	if(BLAS_FOUND)
	# Use parallel blas for cblas
	option( USE_PARALLEL_BLAS "If possible, prefer parallel blas libraries" ON )
	if( USE_PARALLEL_BLAS AND NOT "${BLAS_PAR_LIBRARIES}" STREQUAL "" )
		message( STATUS "Using Parallel Blas" )
		set( BLAS_LIBRARIES "${BLAS_PAR_LIBRARIES}" )
	endif()
	else()
		message(FATAL_ERROR "BLAS library has not been found")
	endif()

	# CBLAS
	find_package(CBLAS COMPONENTS BLASEXT)
	if(BLAS_FOUND)
		if (BLAS_LIBRARY_DIRS)
			# the RPATH to be used when installing
			list(APPEND CMAKE_INSTALL_RPATH "${BLAS_LIBRARY_DIRS}")
		endif()
		if(BLAS_LINKER_FLAGS)
			#list(APPEND CMAKE_EXE_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")
			set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
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
			message(WARNING "CBLAS library has not been found and ECRC_VERBOSE_FIND_PACKAGE is set to OFF"
			                "Try to activate ECRC_VERBOSE_FIND_PACKAGE option (-DECRC_VERBOSE_FIND_PACKAGE=ON) to get some hints for the detection")
		endif()
		message(FATAL_ERROR "A CBLAS library is required but has not been found")
	endif()

	## LAPACKE
	find_package(LAPACKE COMPONENTS LAPACKEXT)
	if(LAPACK_FOUND AND LAPACK_LIBRARY_DIRS)
		# the RPATH to be used when installing
		list(APPEND CMAKE_INSTALL_RPATH "${LAPACK_LIBRARY_DIRS}")
	else()
		message(FATAL_ERROR "A LAPACK library is required but has not been found")
	endif()
	if(LAPACKE_FOUND)
		include_directories(${LAPACKE_INCLUDE_DIRS})
		if(LAPACKE_LIBRARY_DIRS)
			# the RPATH to be used when installing
			list(APPEND CMAKE_INSTALL_RPATH "${LAPACKE_LIBRARY_DIRS}")
		endif()
		if(LAPACKE_LINKER_FLAGS)
			set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACKE_LINKER_FLAGS}")
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
			message(WARNING "LAPACKE library has not been found and ECRC_VERBOSE_FIND_PACKAGE is set to OFF"
			                "Try to activate ECRC_VERBOSE_FIND_PACKAGE option (-DECRC_VERBOSE_FIND_PACKAGE=ON) to get some hints for the detection")
		endif()
		message(FATAL_ERROR "A LAPACKE library is required but has not been found")
	endif()

foreach(BACKEND ${BACKEND_TYPE})
    if( "${BACKEND}" STREQUAL "starpu" )
        find_package(STARPU)
        if( STARPU_FOUND )
            include_directories( ${STARPU_INCLUDE_DIRS})
            link_directories( ${STARPU_LIBRARY_DIRS} )
            link_libraries( ${STARPU_LIBRARIES} )
        else()
        endif()
    endif()
endforeach()

	# BUILD TESTS
	enable_testing() # enables ctest
	add_subdirectory(examples)
        add_subdirectory(testing)
endif()

# PKGCONFIG
set(prefix ${CMAKE_INSTALL_PREFIX})
set(libdir ${CMAKE_INSTALL_PREFIX}/lib)
set(includedir ${CMAKE_INSTALL_PREFIX}/include)
set(PKG_CONFIG_DIR ${libdir}/pkgconfig)
configure_file(${CMAKE_SOURCE_DIR}/starsh.pc.in ${CMAKE_CURRENT_BINARY_DIR}/starsh.pc)
install( FILES ${CMAKE_BINARY_DIR}/starsh.pc
         DESTINATION ${PKG_CONFIG_DIR}
       )

