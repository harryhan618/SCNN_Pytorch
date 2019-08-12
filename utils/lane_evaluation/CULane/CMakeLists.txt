cmake_minimum_required(VERSION 2.8)
project (evaluate)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-DCPU_ONLY -fopenmp")

find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/include")

add_executable(evaluate 
	${PROJECT_SOURCE_DIR}/src/evaluate.cpp 
	${PROJECT_SOURCE_DIR}/src/counter.cpp 
	${PROJECT_SOURCE_DIR}/src/lane_compare.cpp 
	${PROJECT_SOURCE_DIR}/src/spline.cpp
)
target_link_libraries(evaluate ${OpenCV_LIBS})
