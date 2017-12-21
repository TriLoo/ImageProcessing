cmake_minimum_required(VERSION 3.7)
project(RGF_Saliency)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)

find_package(OpenCV REQUIRED)
# message(STATUS ${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system thread)

set(SOURCE_FILES JBF.h RGF.h main.cpp )
#set(SOURCE_FILES JBF.h main.cpp )
add_executable(RGF_Saliency ${SOURCE_FILES})

target_link_libraries(RGF_Saliency ${OpenCV_LIBS})
target_link_libraries(RGF_Saliency ${Boost_LIBRARIES})

