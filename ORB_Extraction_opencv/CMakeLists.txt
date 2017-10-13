cmake_minimum_required(VERSION 3.7)
project(ORBFeatures)

set(CMAKE_BUILD_TYPE  DEBUG)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system)
#message(STATUS ${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(ORBFeatures ${SOURCE_FILES})

#message(STATUS ${Boost_LIBRARIES})
target_link_libraries(ORBFeatures ${OpenCV_LIBS} ${Boost_LIBRARIES})
