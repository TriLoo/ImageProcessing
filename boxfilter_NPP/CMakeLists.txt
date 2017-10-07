cmake_minimum_required(VERSION 3.7)
project(boxFilterNPP)

set(CMAKE_CXX_STANDARD 11)

set(CUDA_NVCC_FLAGS -ptx)

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)
include_directories(./)

set(SOURCE_FILES main.cpp)
cuda_add_executable(boxFilterNPP ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(/usr/local/cuda/lib64)
link_directories(/usr/local/cuda/include)

#message(STATUS "CUDA_LIBS : ${CUDA_LIBRARIES}")
set(CUDA_LIBS /usr/local/cuda/lib64)

target_link_libraries(boxFilterNPP ${OpenCV_LIBS})
target_link_libraries(boxFilterNPP /usr/lib/libfreeimage.so)
target_link_libraries(boxFilterNPP ${CUDA_LIBRARIES})
target_link_libraries(boxFilterNPP ${CUDA_LIBS}/libnppi.so)
