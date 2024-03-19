cmake_minimum_required(VERSION 3.12)
project(TestApps)
include_directories(${OpenCV_INCLUDE_DIRS})
# Define the executable target
add_executable(${PROJECT_NAME} main.cpp)

# Link with the 'combined' library. Ensure 'combined' is defined in the project.
target_link_libraries(${PROJECT_NAME} LSTM)



