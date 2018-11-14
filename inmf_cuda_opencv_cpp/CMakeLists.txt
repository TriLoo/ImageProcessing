cmake_minimum_required(VERSION 3.7)
project(imageINMF_revise)
message(STATUS "project : INMF")
message(STATUS "Project Directory : ${PROJECT_SOURCE_DIR}")

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 11)
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(SOURCE_FILES main.cu matrix.cu SVD.cpp matrix.h SVD.h)

#execute_process(COMMAND sh ../shell.sh
    #OUTPUT_VARIABLE FINDCUDA)
#message(STATUS "find cuda.cmake version is : ${FINDCUDA}")
#include(/usr/share/${FINDCUDA}/Modules/FindCUDA.cmake)
#include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)
#add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS ${CUDA_VERSION})    # 8.0
message(STATUS ${CUDA_TOOLKIT_ROOT_DIR})   # /usr/local/cuda

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_50)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50 -G -g)

set(SOURCE_FILES main.cpp matrix.cu SVD.cpp)
cuda_add_executable(imageINMF_revise ${SOURCE_FILES})

# add link directories
link_directories(/usr/local/cuda/lib64)
set(LIBS_DIR /usr/local/cuda/lib64)

# specif needed libraries
target_link_libraries(imageINMF_revise ${OpenCV_LIBS})
target_link_libraries(imageINMF_revise ${CUDA_LIBRARIES})
target_link_libraries(imageINMF_revise ${LIBS_DIR}/libcublas.so)
target_link_libraries(imageINMF_revise ${LIBS_DIR}/libcusolver.so)
