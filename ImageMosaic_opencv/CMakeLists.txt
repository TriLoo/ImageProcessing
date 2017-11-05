cmake_minimum_required(VERSION 3.7)
project(ImageMosaic)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp ImageMosaic.cpp)

find_package(OpenCV REQUIRED)
include_directories(/usr/local/include)

add_executable(ImageMosaic ${SOURCE_FILES})

target_link_libraries(ImageMosaic ${OpenCV_LIBS})

