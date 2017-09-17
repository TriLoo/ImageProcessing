cmake_minimum_required(VERSION 3.7)
project(GaussFilter)
message(STATUS "Gauss Filter")
message(STATUS "Project Dir : ${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp gaussfilter.cu)

include_directories(./)
include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)
cuda_add_executable(GaussFilter ${SOURCE_FILES})

# add link libraries
link_libraries(/usr/local/cuda/lib64)

set(LIBS_DIR /usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

target_link_libraries(GaussFilter ${OpenCV_LIBS})
target_link_libraries(GaussFilter ${LIBS_DIR}/libcublas.so)
target_link_libraries(GaussFilter ${LIBS_DIR}/libcusolver.so)
target_link_libraries(GaussFilter ${LIBS_DIR}/libcufft.so)

link_directories(${LIBS_DIR})
