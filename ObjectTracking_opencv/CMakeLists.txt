cmake_minimum_required(VERSION 3.7)
project(ObjectTracking)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(ObjectTracking ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
include_directories(OpenCV_INCLUDE_DIR)
target_link_libraries(ObjectTracking ${OpenCV_LIBS})
