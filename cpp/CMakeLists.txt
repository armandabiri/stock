cmake_minimum_required(VERSION 3.12)

project(MyApp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

add_subdirectory(cmake)

resolve_dependencies(requirements.yaml "C:/repos")

message(" CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH} ")

find_package(Matplot++ REQUIRED)

# find_package(Gtest REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

# find_package(yaml-cpp REQUIRED)
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(libs)

# add_subdirectory(apps)

# add_subdirectory(tests)
