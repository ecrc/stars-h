# Defines the source files of the tests. Each file generates 1 test
file(GLOB tests_files "*.c")

include_directories( ${PROJECT_SOURCE_DIR}/include)

foreach( test_src ${tests_files} )
	get_filename_component( test_exe ${test_src}  NAME_WE )
	add_executable( ${test_exe} ${test_src})
	target_link_libraries( ${test_exe} starsh ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES} )
	install( TARGETS ${test_exe} DESTINATION lib/testing )
endforeach()