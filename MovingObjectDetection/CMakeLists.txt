cmake_minimum_required(VERSION 3.7)
project(MovingObjectDetects)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(MovingObjectDetects ${SOURCE_FILES})

target_link_libraries(MovingObjectDetects ${OpenCV_LIBS})
