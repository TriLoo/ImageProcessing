cmake_minimum_required(VERSION 2.8)
project(GaussFilter)
message(STATUS "Project Name : GaussFilter")
message(STATUS "Project Dir  : ${PROJECT_SOURCE_DIR}")

set(SOURCE_FILES main.cpp)
add_executable(GaussFilter ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
target_link_libraries(GaussFilter ${OpenCV_LIBS})
