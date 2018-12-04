cmake_minimum_required(VERSION 3.7)
project(RDLW_Sal_Fusion)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INC_DIRS})

set(SOURCE_FILES main.cpp RDL_Wavelet.cpp WeightedMap.cpp Fusion.cpp)
add_executable(RDLW_Sal_Fusion ${SOURCE_FILES})

target_link_libraries(RDLW_Sal_Fusion ${OpenCV_LIBS})
