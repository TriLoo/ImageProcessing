cmake_minimum_required(VERSION 3.7)
project(TwoScale)
message(STATUS "Project Name: TwoScale")
message(STATUS "Project directory : ${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 11)

include_directories(./)
include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)

set(SOURCE_FILES main.cpp filter.cu twoscale.cu)
cuda_add_executable(TwoScale ${SOURCE_FILES})

link_libraries(/usr/local/cuda/lib64)
set(CUDA_LIBS /usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

target_link_libraries(TwoScale ${OpenCV_LIBS})
target_link_libraries(TwoScale ${CUDA_LIBRARIES})
target_link_libraries(TwoScale ${CUDA_LIBS}/libcublas.so)
target_link_libraries(TwoScale ${CUDA_LIBS}/libcusolver.so)
target_link_libraries(TwoScale ${CUDA_LIBS}/libcufft.so)

link_directories(${CUDA_LIBS})

