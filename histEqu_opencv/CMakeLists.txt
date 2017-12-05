cmake_minimum_required(VERSION 3.2)
project(MVChapter3)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
message(STATUS ${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES histst.cpp)
add_executable(MVChapter3 ${SOURCE_FILES})

target_link_libraries(MVChapter3 ${OpenCV_LIBS})

