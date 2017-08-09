cmake_minimum_required(VERSION 3.7)
project(NonPhotoRendering)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(NonPhotoRendering ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
target_link_libraries(NonPhotoRendering ${OpenCV_LIBS})