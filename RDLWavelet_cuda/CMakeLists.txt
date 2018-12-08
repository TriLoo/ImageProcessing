cmake_minimum_required(VERSION 3.2)
project(RDLWaveletCUDA)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 --keep")   # enable C++11 features in .cu file !!!

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp  RDLWavelet.cu headers.h)
cuda_add_executable(RDLWaveletCUDA ${SOURCE_FILES})

target_link_libraries(RDLWaveletCUDA ${OpenCV_LIBS})
target_link_libraries(RDLWaveletCUDA ${CUDA_LIBRARIES})
