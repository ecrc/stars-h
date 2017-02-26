# Define the source files
add_subdirectory("control")
add_subdirectory("applications")
add_subdirectory("itersolvers")

# Set backends and check if there is support for them
add_subdirectory("backends")

# set the values of the variable in the parent scope
set(STARSH_SRC "${STARSH_SRC}" PARENT_SCOPE)
