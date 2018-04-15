cmake_minimum_required(VERSION 3.7)
project(MovingObjectDetects)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCE_FILES ViBe.cpp LKFlow.cpp main.cpp )
add_executable(MovingObjectDetects ${SOURCE_FILES})

target_link_libraries(MovingObjectDetects ${OpenCV_LIBS})
target_link_libraries(MovingObjectDetects ${Eigen3_LIBS})

