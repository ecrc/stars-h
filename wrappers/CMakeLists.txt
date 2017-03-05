if(PLASMA)
    add_subdirectory("plasma")
    set(STARSH_SRC
        "${STARSH_SRC}" PARENT_SCOPE)
endif()
