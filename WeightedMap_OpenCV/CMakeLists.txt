cmake_minimum_required(VERSION 3.7)
project(WeightedMap)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INC_DIRS})

set(SOURCE_FILES main.cpp WeightedMap.cpp)
add_executable(WeightedMap ${SOURCE_FILES})

target_link_libraries(WeightedMap ${OpenCV_LIBS})

