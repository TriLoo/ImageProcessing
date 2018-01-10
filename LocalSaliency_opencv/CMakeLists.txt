cmake_minimum_required(VERSION 3.7)
project(LocalSaliency)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INC_DIRS})

set(SOURCE_FILES main.cpp LocalSaliency.cpp)
add_executable(LocalSaliency ${SOURCE_FILES})
target_link_libraries(LocalSaliency ${OpenCV_LIBS})

