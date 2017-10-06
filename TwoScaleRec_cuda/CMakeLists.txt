cmake_minimum_required(VERSION 3.7)
project(TwoScaleRec)

set(CMAKE_CXX_STANDARD 11)

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)
include_directories(./)

set(SOURCE_FILES main.cpp twoscalerec.cu)
cuda_add_executable(TwoScaleRec ${SOURCE_FILES})

link_libraries(/usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

target_link_libraries(TwoScaleRec ${OpenCV_LIBS})

