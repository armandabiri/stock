cmake_minimum_required(VERSION 3.12)
message(STATUS "Fine all CMake files in the ${CMAKE_CURRENT_SOURCE_DIR}")
# Get a list of all CMake files in the 'cmake' directory
file(GLOB CMAKE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cmake")

# Iterate over each CMake file
foreach(CMAKE_FILE ${CMAKE_FILES})
    # Process each CMake file here
    message(STATUS "Found CMake file: ${CMAKE_FILE}")
    include(${CMAKE_FILE})
endforeach()

