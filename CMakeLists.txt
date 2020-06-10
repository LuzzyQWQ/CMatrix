cmake_minimum_required(VERSION 3.0.0)
project(Matrix VERSION 0.1.0)


set(CMatrix
src/CMatrix.hpp
src/Cvector.hpp
src/Cexception.hpp
src/testcase.cpp
)

add_executable(main ${CMatrix})
# add_library(CMatrix ${CMatrix})
