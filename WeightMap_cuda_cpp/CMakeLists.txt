cmake_minimum_required(VERSION 3.7)
project(WeightMap)
message(STATUS "Project Name : WeightMap")
message(STATUS "Project Dir  : ${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 11)

include(/usr/share/cmake-2.8/Modules/FindCUDA.cmake)
include_directories(./)

set(SOURCE_FILES main.cu weightmap.cu)
cuda_add_executable(WeightMap ${SOURCE_FILES})

link_directories(/usr/local/cuda/lib64)
link_libraries(/usr/local/cuda/lib64)
set(CUDA_LIBS /usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

target_link_libraries(WeightMap ${OpenCV_LIBS})
target_link_libraries(WeightMap ${CUDA_LIBS}/libcublas.so)
