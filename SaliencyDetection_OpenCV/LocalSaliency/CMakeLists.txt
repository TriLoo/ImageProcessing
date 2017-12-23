cmake_minimum_required(VERSION 3.7)
project(LocalSaliencyGOL)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INC_DIRS})

set(SOURCE_FILES main.cpp GOL.h)
add_executable(LocalSaliencyGOL ${SOURCE_FILES})

target_link_libraries(LocalSaliencyGOL ${OpenCV_LIBS})
