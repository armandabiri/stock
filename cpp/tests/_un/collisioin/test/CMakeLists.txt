project(collision-test)

add_executable(${PROJECT_NAME} collision_test.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC GTest::GTest GTest::Main collision)

set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE "PRE_TEST")
gtest_discover_tests(${PROJECT_NAME})
