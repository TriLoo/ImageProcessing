cmake_minimum_required(VERSION 3.2)
project(Transpose)

set(CMAKE_CXX_STANDARD 11)
include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cu)
cuda_add_executable(Transpose ${SOURCE_FILES})
target_link_libraries(Transpose ${OpenCV_LIBS})
