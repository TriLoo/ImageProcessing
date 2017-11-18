cmake_minimum_required(VERSION 3.2)
project(gffFusionFinal)

set(CMAKE_CXX_STANDARD 11)

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)   # No format library

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Boost})
#message(STATUS ${Boost_INCLUDE_DIRS})
#message(STATUS ${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp gff.cu BFilter.cu TwoScale.cu GFilter.cu WeightedMap.cu)
cuda_add_executable(gffFusionFinal ${SOURCE_FILES})

set(CUDA_LIBS /usr/local/cuda/lib64)

#message(STATUS ${CUDA_LIBRARIES})
message(STATUS ${Boost_LIBRARIES})

target_link_libraries(gffFusionFinal ${OpenCV_LIBS} ${CUDA_LIBRARIES})
target_link_libraries(gffFusionFinal ${CUDA_LIBS}/libcublas.so)
target_link_libraries(gffFusionFinal ${CUDA_LIBS}/libnppi.so)
target_link_libraries(gffFusionFinal ${CUDA_LIBS}/libcufft.so)
target_link_libraries(gffFusionFinal ${Boost_LIBRARIES})
