cmake_minimum_required(VERSION 3.2)
project(MSRCR)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS  "-std=c++11")


include_directories(./)
set(SOURCE_FILES main.cpp MSRCR.cu)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

include_directories(/usr/local/include/)
include_directories(/usr/local/cuda)
include_directories(${Boost_INCLUDE_DIR})

include(/usr/share/cmake-3.2/Modules/FindCUDA.cmake)

cuda_add_executable(MSRCR ${SOURCE_FILES})

#message(STATUS ${CUDA_LIBRARIES})

set(CUDA_LIBS "/usr/local/cuda/lib64")

link_directories(/usr/lib)
link_directories(/usr/local/lib)
link_directories(/usr/local/cuda/lib64)

target_link_libraries(MSRCR ${OpenCV_LIBS})
target_link_libraries(MSRCR ${Boost_LIBRARIES})
target_link_libraries(MSRCR ${CUDA_LIBRARIES})
target_link_libraries(MSRCR ${CUDA_LIBS}/libcublas.so)
target_link_libraries(MSRCR ${CUDA_LIBS}/libcufft.so)

