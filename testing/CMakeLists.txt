# Defines the source files of the tests. Each file generates 1 test
set(tests_files symm_positivity_after_compression.c)

include_directories(${CMAKE_SOURCE_DIR}/include)


# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach( test_src ${tests_files} )
	get_filename_component( test_exe ${test_src}  NAME_WE )
	add_executable( ${test_exe} ${test_src} )
	target_link_libraries( ${test_exe} starsh ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES} )
endforeach()

# Copies TXT files needed for testing
file(GLOB tests_files "*.txt")
list(REMOVE_ITEM tests_files "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
foreach( file ${tests_files} )
	get_filename_component( txt ${file} NAME )
	# copy the files before building the last test
	add_custom_command( TARGET ${test_exe} PRE_BUILD
	                    COMMAND ${CMAKE_COMMAND} -E copy
	                    ${CMAKE_CURRENT_SOURCE_DIR}/${txt}
	                    ${CMAKE_CURRENT_BINARY_DIR}/.
	                    )
endforeach()
