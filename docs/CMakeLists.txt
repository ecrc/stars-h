# Collect Doxygen enabled sections
if(DOCS STREQUAL "FULL")
    set(DOXYGEN_ENABLED_SECTIONS OPENMP MPI STARPU GSL)
endif()

# Make enabled section appropriate for config file
string(REPLACE ";" " " DOXYGEN_ENABLED_SECTIONS "${DOXYGEN_ENABLED_SECTIONS}")

# Make DOXYGEN_INPUT appropriate for config file
string(REPLACE ";" "\"\\\n    \"" DOXYGEN_INPUT "${DOXYGEN_INPUT}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config")
add_custom_target(docs COMMAND
    "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/config")
