cmake_minimum_required(VERSION 3.7)
project(RDL_Wavelet)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INC_DIRS})

find_package(Boost REQUIRED COMPONENTS system thread)

set(SOURCE_FILES main.cpp RDL_Wavelet.cpp)
add_executable(RDL_Wavelet ${SOURCE_FILES})

#message(STATUS ${Boost_LIBRARIES})

target_link_libraries(RDL_Wavelet ${OpenCV_LIBS})
target_link_libraries(RDL_Wavelet ${Boost_LIBRARIES})
