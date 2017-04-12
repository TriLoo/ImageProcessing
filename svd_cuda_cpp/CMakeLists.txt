cmake_minimum_required(VERSION 2.8)
	project(svd)
	message(STATUS "Project:svd")
	message(STATUS "Project Directory: ${PROJECT_SOURCE_DIR}")

	set(CMAKE_BUILE_TYPE DEBUG)
	#set(CMAKE_C_FLAGS_DEBUG "-g -Wall")
	set(SOURCE_FILES main.cu SVD.cpp SVD.h)

	include(/usr/share/cmake-2.8/Modules/FindCUDA.cmake)
	add_definitions(-std=c++11)
	cuda_add_executable(svd ${SOURCE_FILES})

	# add include directories
	include_directories(/usr/local/include ${PROJECT_SOURCE_DIR})
	include_directories(/usr/local/cuda/include)

	# add link directories
	link_directories(/usr/local/cuda/lib64)
	set(CUDA_LIBS cublas cusolver)

	find_package(OpenCV REQUIRED)
	find_package(CUDA REQUIRED)
	# specify needed libraries
	target_link_libraries(svd ${OpenCV_LIBS})
	#message(STATUS "${OpenCV_LIBS}")
	#target_link_libraries(svd cublas)
	target_link_libraries(svd /usr/local/cuda/lib64/libcublas.so)
	target_link_libraries(svd /usr/local/cuda/lib64/libcusolver.so)
	#target_link_libraries(svd ${CUDA_LIBS})
	#target_link_libraries(svd ${CUDA_LIBS})


