project(TestEigen)
find_package(Eigen3 3.3 REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
