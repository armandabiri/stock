cmake_minimum_required(VERSION 3.12)
project(MyApp)

# Set compiler-specific debug flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # For MSVC, use /Zi for debug information
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

add_subdirectory(cmake)

set(gtest_force_shared_crt on)
resolve_dependencies(requirements.yaml "C:/repos")

# include(FetchContent)
# FetchContent_Declare(googletest
# GIT_REPOSITORY https://github.com/google/googletest.git
# GIT_TAG main)
# FetchContent_MakeAvailable(googletest)
enable_testing()

message(" CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH} ")

include_directories("${gtest_SOURCE_DIR}/include")

find_package(Matplot++ REQUIRED)
find_package(Gtest REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

# find_package(OpenCV REQUIRED)

# find_package(yaml-cpp REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(libs)

# add_subdirectory(apps)

# add_subdirectory(tests)
