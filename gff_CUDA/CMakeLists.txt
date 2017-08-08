cmake_minimum_required(VERSION 3.7)
project(GffFusion)

message(STATUS  "Project: GffFusion")
message(STATUS  "Project Director: ${PROJECT_SOURCE_DIR}")

set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)

include(/usr/share/cmake-2.8/Modules/FindCUDA.cmake)
cuda_add_executable(GffFusion ${SOURCE_FILES})

# add link direcories
link_libraries(/usr/local/cuda/lib64)
set(LIBS_DIR /usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

message(STATUS "The OpenCV_LIBS is : ${OpenCV_LIBS}")
#message(STATUS "The CUDA_LIBS is : ${CUDA}")

target_link_libraries(GffFusion ${OpenCV_LIBS})
target_link_libraries(GffFusion ${LIBS_DIR}/libcublas.so)
target_link_libraries(GffFusion ${LIBS_DIR}/libcusolver.so)
link_directories("/usr/local/cuda/lib64")
