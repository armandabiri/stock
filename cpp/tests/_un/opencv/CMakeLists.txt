project(opencv_test)
message(STATUS "Configuring project ${PROJECT_NAME}...")
add_executable(${PROJECT_NAME} opencv_test.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
