cmake_minimum_required(VERSION 3.7)
project(LoadImages)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(LoadImages ${SOURCE_FILES})

target_link_libraries(LoadImages ${OpenCV_LIBS})
target_link_libraries(LoadImages ${Boost_LIBRARIES})
