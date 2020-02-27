cmake_minimum_required(VERSION 2.8)

set( Project_NAME dwt_mallat_omp)
set( LIB_NAME dwt_mallat_omp)
project(${Project_NAME})

find_package(OpenMP REQUIRED)

set(CMAKE_C_COMPILER /home/nvidia/GCC-8.2.0/bin/gcc)
set(CMAKE_CXX_COMPILER /home/nvidia/GCC-8.2.0/bin/g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#Specify Includes
include_directories(lib/inc)
#include_directories(/usr/lib/gcc/aarch64-linux-gnu/5/include/)

#add_definitions(-std=c++17)
add_definitions(-std=c++1z)

#Specify sources
file(GLOB CPP_SRC "lib/src/*.cpp")

#link cpp objects and cuda object to create target
add_executable(${LIB_NAME} ${CPP_SRC})
