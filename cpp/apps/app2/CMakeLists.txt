project(AppLSTM)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} adroco::lstm)
