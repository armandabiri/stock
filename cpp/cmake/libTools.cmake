
function(build_libraries)
    set(curdir ${CMAKE_CURRENT_SOURCE_DIR})
    file(GLOB folders LIST_DIRECTORIES true RELATIVE ${curdir} ${curdir}/*)

    foreach(folder ${folders})
        if(IS_DIRECTORY ${curdir}/${folder})
            string(TOUPPER ${folder} FOLDER_UPPERCASE)
            message(STATUS ">>>>>> Start to build the library [lib${FOLDER_UPPERCASE}.dll]")
            add_library(${FOLDER_UPPERCASE} SHARED "${folder}/${folder}.cpp")
            target_include_directories(${FOLDER_UPPERCASE} PUBLIC "${curdir}/${folder}/include")
        endif()
    endforeach()
endfunction()

function(build_library LIBRARY_NAME PROPERTIES1 PROPERTIES2 DEPENDENCIES)
    string(TOUPPER "${LIBRARY_NAME}" LIBRARY_UPPERCASE)
    set(FULL_LIBRARY_NAME "lib${LIBRARY_UPPERCASE}")
    message(STATUS ">>>>>> Start to build the library [${FULL_LIBRARY_NAME}] with source directory [${CMAKE_CURRENT_SOURCE_DIR}/include]")
    add_library(${FULL_LIBRARY_NAME} ${PROPERTIES1} "${LIBRARY_NAME}.cpp")
    target_include_directories(${FULL_LIBRARY_NAME} ${PROPERTIES2} "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${FULL_LIBRARY_NAME} ${DEPENDENCIES})

    add_library(adroco::${LIBRARY_NAME} ALIAS ${FULL_LIBRARY_NAME})
endfunction()

function(build_static_library_current DEPENDENCIES)
    get_filename_component(LIBRARY_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

    file(GLOB SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

    string(TOUPPER "${LIBRARY_NAME}" LIBRARY_UPPERCASE)
    set(FULL_LIBRARY_NAME "lib${LIBRARY_UPPERCASE}")
    message(STATUS ">>>>>> Start to build the STATIC library [${FULL_LIBRARY_NAME}] with source directory [${CMAKE_CURRENT_SOURCE_DIR}/include]")
    add_library(${FULL_LIBRARY_NAME} STATIC "${SRC}")
    target_include_directories(${FULL_LIBRARY_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${FULL_LIBRARY_NAME} ${DEPENDENCIES})

    add_library(adroco::${LIBRARY_NAME} ALIAS ${FULL_LIBRARY_NAME})
endfunction()

function(build_shared_library_current DEPENDENCIES VERSION)
    get_filename_component(LIBRARY_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

    string(REGEX MATCH "^([0-9]+).(.*)$" MATCHES ${VERSION})
    set(MAJOR_VERSION ${CMAKE_MATCH_1}) # Group 2: branch name
    set(MINOR_VERSION ${CMAKE_MATCH_2}) # Group 2: branch name

    file(GLOB SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

    file(GLOB HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}/*.hpp")

    string(TOUPPER "${LIBRARY_NAME}" LIBRARY_UPPERCASE)
    set(FULL_LIBRARY_NAME "lib${LIBRARY_UPPERCASE}")
    message(STATUS ">>>>>> Start to build the SHARED library [${FULL_LIBRARY_NAME}] with source directory [${CMAKE_CURRENT_SOURCE_DIR}/include]")
    add_library(${FULL_LIBRARY_NAME} SHARED "${SRC}")
    target_include_directories(${FULL_LIBRARY_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${FULL_LIBRARY_NAME} ${DEPENDENCIES})

    set_target_properties(${FULL_LIBRARY_NAME} PROPERTIES VERSION ${VERSION})

    SET_TARGET_PROPERTIES(
        ${FULL_LIBRARY_NAME} PROPERTIES
        VERSION ${VERSION}
        SOVERSION ${MAJOR_VERSION}
        PUBLIC_HEADER ${HEADERS}
    )

    INSTALL(
        TARGETS ${FULL_LIBRARY_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    add_library(adroco::${LIBRARY_NAME} ALIAS ${FULL_LIBRARY_NAME})
endfunction()

function(add_tests dependencies)
    get_filename_component(LIBRARY_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

    string(TOUPPER "${LIBRARY_NAME}" LIBRARY_UPPERCASE)
    set(FULL_LIBRARY_NAME "lib${LIBRARY_UPPERCASE}")

    file(GLOB SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")

    foreach(SRC ${SRC_LIST})
        get_filename_component(FILENAME ${SRC} NAME_WE)
        add_executable(${FILENAME} ${SRC})
        target_sources(${FILENAME} PRIVATE ${CMAKE_SOURCE_DIR}/sources/matrix.natvis)

        target_link_libraries(${FILENAME} PUBLIC ${FULL_LIBRARY_NAME} ${dependencies})
    endforeach()
endfunction()

function(add_gtests dependencies)
    get_filename_component(LIBRARY_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

    string(TOUPPER "${LIBRARY_NAME}" LIBRARY_UPPERCASE)
    set(FULL_LIBRARY_NAME "lib${LIBRARY_UPPERCASE}")

    file(GLOB SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/gtest/*.cpp")

    foreach(SRC ${SRC_LIST})
        get_filename_component(FILENAME ${SRC} NAME_WE)
        add_executable(${FILENAME} ${SRC})
        target_sources(${FILENAME} PRIVATE ${CMAKE_SOURCE_DIR}/sources/matrix.natvis)

        target_link_libraries(${FILENAME} PUBLIC ${FULL_LIBRARY_NAME} ${dependencies} GTest::GTest GTest::Main)
        set_target_properties(${FILENAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/gtests) # Set output directory
        add_test(NAME ${FILENAME} COMMAND ${FILENAME})
    endforeach()

    set(curdir ${CMAKE_CURRENT_SOURCE_DIR})
    file(GLOB libs LIST_DIRECTORIES true RELATIVE ${curdir} ${curdir}/*)
endfunction()

function(combine_static_libraries INPUT_LIBRARIES COMBINED_LIBRARY_NAME)
    string(TOUPPER "${COMBINED_LIBRARY_NAME}" COMBINED_LIBRARY_NAME_UPPERCASE)

    # Directory where the combined library will be created
    set(COMBINED_LIBRARY_DIR ${CMAKE_BINARY_DIR}/libs/${COMBINED_LIBRARY_NAME})

    # Ensure that input libraries are built before combining their object files
    add_dependencies(${INPUT_LIBRARIES})

    # Initialize a list to store the commands for combining libraries
    set(COMMAND_LIST "")

    # Loop through each input library to create a command for combining its object files
    foreach(INPUT_LIBRARY ${INPUT_LIBRARIES})
        list(APPEND COMMAND_LIST COMMAND ${CMAKE_AR} -rc ${COMBINED_LIBRARY_DIR}/${COMBINED_LIBRARY_NAME_UPPERCASE}.a $<TARGET_OBJECTS:${INPUT_LIBRARY}>)
    endforeach()

    # Combine object files from input libraries into a single library
    add_custom_command(
        OUTPUT ${COMBINED_LIBRARY_DIR}/${COMBINED_LIBRARY_NAME_UPPERCASE}.a
        ${COMMAND_LIST}
        DEPENDS ${INPUT_LIBRARIES}
    )

    # Inform the user about the combined library
    add_custom_target(${COMBINED_LIBRARY_NAME_UPPERCASE}_message
        DEPENDS ${COMBINED_LIBRARY_DIR}/${COMBINED_LIBRARY_NAME_UPPERCASE}.a
    )

    # Add a target to combine libraries and display the message
    add_custom_target(${COMBINED_LIBRARY_NAME_UPPERCASE} ALL DEPENDS ${COMBINED_LIBRARY_DIR}/${COMBINED_LIBRARY_NAME_UPPERCASE}.a ${COMBINED_LIBRARY_NAME_UPPERCASE}_message)

    # Copy only the .a files to COMBINED_LIBRARY_DIR/statics
    foreach(INPUT_LIBRARY ${INPUT_LIBRARIES})
        file(GLOB LIBRARY_A_FILES "${CMAKE_BINARY_DIR}/libs/${INPUT_LIBRARY}/*.a")
        message(STATUS "Copying ${LIBRARY_A_FILES} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/statics")
        file(COPY ${LIBRARY_A_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/statics)
    endforeach()
endfunction()
