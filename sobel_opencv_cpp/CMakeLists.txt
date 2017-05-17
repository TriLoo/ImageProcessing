cmake_minimum_required(VERSION 2.8)
	project(sobel)
	message(STATUS "Project:sobel")
	message(STATUS "Project Directory: ${PROJECT_SOURCE_DIR}")

	add_definitions(-std=c++11)

	set(SOURCE_FILES main.cpp)
	add_executable(sobel ${SOURCE_FILES})
	find_package(OpenCV REQUIRED)
	target_link_libraries(sobel ${OpenCV_LIBS})
