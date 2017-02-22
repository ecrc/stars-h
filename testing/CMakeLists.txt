# Defines the source files of the tests. Each file generates 1 test
set(tests_files symm_positivity_after_compression.c)

include_directories( ${PROJECT_SOURCE_DIR}/include)


# Uses RUNPATH instead of RPATH
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

foreach( test_src ${tests_files} )
	get_filename_component( test_exe ${test_src}  NAME_WE )
	add_executable( ${test_exe} ${test_src} )
	target_link_libraries( ${test_exe} starsh ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES} )
	install( TARGETS ${test_exe} DESTINATION lib/testing )
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

# Add basic tests. To run them type: 'make test' in build directory
#add_test(NAME Array:ok COMMAND array )
#add_test(NAME Astronomy:ok COMMAND astronomy ./ 512 64 1e-3 astronomy.outfile )
#add_test(NAME Astronomy:usage COMMAND astronomy )
#set_tests_properties( Astronomy:usage PROPERTIES WILL_FAIL ON )
#add_test(NAME Spatial_sdd COMMAND spatial 100 1000 100 10 1e-12 1e-1 sdd)
#add_test(NAME Spatial_omp_sdd COMMAND spatial 100 1000 100 10 1e-12 1e-1 omp_sdd)
#add_test(NAME Spatial_starpu_sdd COMMAND spatial 100 1000 100 10 1e-12 1e-1 starpu_sdd)
#add_test(NAME Spatial_qp3 COMMAND spatial 100 1000 100 10 1e-12 1e-1 qp3)
#add_test(NAME Spatial_omp_qp3 COMMAND spatial 100 1000 100 10 1e-12 1e-1 omp_qp3)
#add_test(NAME Spatial_starpu_qp3 COMMAND spatial 100 1000 100 10 1e-12 1e-1 starpu_qp3)
#add_test(NAME Spatial_rsdd COMMAND spatial 100 1000 100 10 1e-12 1e-1 rsdd)
#add_test(NAME Spatial_omp_rsdd COMMAND spatial 100 1000 100 10 1e-12 1e-1 omp_rsdd)
#add_test(NAME Spatial_starpu_rsdd COMMAND spatial 100 1000 100 10 1e-12 1e-1 starpu_rsdd)
#add_test(NAME Spatial_rsdd2 COMMAND spatial 100 1000 100 10 1e-12 1e-1 rsdd2)
#add_test(NAME Spatial_omp_rsdd2 COMMAND spatial 100 1000 100 10 1e-12 1e-1 omp_rsdd2)
#add_test(NAME Spatial_starpu_rsdd2 COMMAND spatial 100 1000 100 10 1e-12 1e-1 starpu_rsdd2)
#add_test(NAME SpatialPerformance:ok COMMAND spatial_performance 8 8 64 32 1e-3 0.1 spatial.outfile )
#add_test(NAME Electrostatics:ok COMMAND electrostatics 8 8 64 32 1e-3 )

