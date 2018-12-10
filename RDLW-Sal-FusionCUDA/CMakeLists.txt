cmake_minimum_required(VERSION 3.2)
project(RDLW_Sal_FusionCUDA)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 --keep --gpu-architecture=sm_50")
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11 --keep")

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp FusionSystem.cu RDLWavelet.cu WeightedMap.cu GuidedFilter.cu headers.h)
cuda_add_executable(RDLW_Sal_FusionCUDA ${SOURCE_FILES})

target_link_libraries(RDLW_Sal_FusionCUDA ${OpenCV_LIBS})
target_link_libraries(RDLW_Sal_FusionCUDA ${CUDA_LIBRARIES})

