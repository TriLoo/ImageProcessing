cmake_minimum_required(VERSION 2.8)
	project(GFF)
	message(STATUS "The Project directory: ${PROJECT_DIR}")

    set(CMAKE_BUILD_TYPE RELEASE)
	set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs")
	include_directories("/usr/local/include/opencv2/ximgproc")

	set( SOURCE_FILES main.cpp guidedfilter.cpp GffFusion.cpp)
	add_executable(gffFusion ${SOURCE_FILES})

	find_package(OpenCV REQUIRED)
	target_link_libraries(gffFusion ${OpenCV_LIBS})
