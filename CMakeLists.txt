cmake_minimum_required(VERSION 3.0.0)
project(Matrix VERSION 0.1.0)

# set(OpenCV_DIR D:\\Tools\\OpenCV\\opencv\\build)
FIND_PACKAGE(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(    ${PROJECT_SOURCE_DIR}    
           "/usr/include/opencv2" )

set(CMatrix
src/CMatrix.hpp
src/Cvector.hpp
src/Cexception.hpp
# src/testcase.cpp
src/demo.cpp
)

add_executable(main ${CMatrix})
target_link_libraries(main ${OpenCV_LIBS})
# add_library(CMatrix ${CMatrix})
