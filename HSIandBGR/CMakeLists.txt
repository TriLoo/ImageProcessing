cmake_minimum_required(VERSION 3.8)
project(HSIandBGR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

add_executable(HSIandBGR main.cpp)
#add_executable(HSIandBGR test.cpp)

target_link_libraries(HSIandBGR ${OpenCV_LIBS})
	
