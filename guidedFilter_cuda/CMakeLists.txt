cmake_minimum_required(VERSION 3.7)
project(GuidedFilter)

set(CMAKE_CXX_STANDARD 11)

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)
include_directories(./)

set(SOURCE_FILES main.cpp BFilter.cu GFilter.cu guidedfilter.cpp)
cuda_add_executable(GuidedFilter ${SOURCE_FILES})

link_libraries(/usr/lcoal/cuda/lib64)
set(CUDA_LIBS /usr/local/cuda/lib64)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

target_link_libraries(GuidedFilter ${OpenCV_LIBS})
