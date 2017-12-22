cmake_minimum_required(VERSION 3.7)
project(SaliencyDetection)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INC_DIRS})
find_package(Boost REQUIRED COMPONENTS system thread date_time)

#set(SOURCE_FILES main.cpp globalSaliency.h FTSaliency.h)
set(SOURCE_FILES main.cpp FTSaliency.h)
add_executable(SaliencyDetection ${SOURCE_FILES})

message(STATUS ${Boost_LIBRARIES})
target_link_libraries(SaliencyDetection ${OpenCV_LIBS} ${Boost_LIBRARIES})
