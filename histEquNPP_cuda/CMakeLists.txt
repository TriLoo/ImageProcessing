cmake_minimum_required(VERSION 3.7)
project(HistEquNPP)

set(CMAKE_CXX_STANDARD 11)

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)
include_directories(./)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(CUDA REQUIRED COMPONENTS cufft)
find_package(CUDA REQUIRED)
message(STATUS "CUDA Dris : ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp histEqu.cpp)
#cuda_add_executable(HistEquNPP ${SOURCE_FILES})
add_executable(HistEquNPP ${SOURCE_FILES})

set(CUDA_DIRS /usr/local/cuda/lib64)

target_link_libraries(HistEquNPP ${OpenCV_LIBS})
#message(STATUS "CUDA Libs : ${CUDA_LIBRARIES}")
target_link_libraries(HistEquNPP ${CUDA_LIBRARIES})
target_link_libraries(HistEquNPP ${CUDA_DIRS}/libnppc.so)
target_link_libraries(HistEquNPP ${CUDA_DIRS}/libnppi.so)
target_link_libraries(HistEquNPP ${CUDA_DIRS}/libcufft.so)

