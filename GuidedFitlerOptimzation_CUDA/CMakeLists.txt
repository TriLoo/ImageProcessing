cmake_minimum_required(VERSION 3.2)
project(GuidedFilterOptimize)

set(CMAKE_CXX_STANDARD 11)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("=std=c++11" COMPILER_SUPPORTS_CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp GuidedFilter.cu headers.h)
cuda_add_executable(GuidedFilterOptimize ${SOURCE_FILES})

#message(${CUDA_LIBRARIES})
link_directories(/usr/local/cuda/lib64/)
set(CUDA_DIR /usr/local/cuda/lib64)

target_link_libraries(GuidedFilterOptimize ${OpenCV_LIBS})
target_link_libraries(GuidedFilterOptimize ${CUDA_LIBRARIES})
target_link_libraries(GuidedFilterOptimize libnppc.so libnppi.so libcublas.so libcusolver.so)
