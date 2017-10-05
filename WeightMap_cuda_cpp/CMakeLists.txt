cmake_minimum_required(VERSION 3.2)
project(WeightedMap)
message(STATUS "Project Name : Weighted Map ")
message(STATUS "Project Dir  : ${PROJECT_SOURCE_DIR} ")

set(CMAKE_CXX_STANDARD 11)

include_directories(./)
include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)
set(SOURCE_FILES main.cpp WMap.cu)
cuda_add_executable(WeightedMap ${SOURCE_FILES})

link_libraries(/usr/local/cuda/lib64)
set(LIBS_DIR /usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

#message(STATUS "CUDA Library: ${CUDA_LIBRARIES}")
#message(STATUS "CUDA Library: ${CUDA_LIB}")

target_link_libraries(WeightedMap ${OpenCV_LIBS})
target_link_libraries(WeightedMap ${CUDA_LIBRARIES})
target_link_libraries(WeightedMap ${LIBS_DIR}/libcublas.so)
target_link_libraries(WeightedMap ${LIBS_DIR}/libcusolver.so)
target_link_libraries(WeightedMap ${LIBS_DIR}/libcufft.so)

link_directories(${LIBS_DIR})
