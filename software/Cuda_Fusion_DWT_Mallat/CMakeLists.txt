cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(dwt_mallat_opt)

#Bring the sources
set(CPP_SRC 
lib/src/f_Lap_pyr.cpp
lib/src/f_wavelet.cpp
lib/src/wavefilt.cpp
)

find_package(CUDA REQUIRED)

set (opt O3)
if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
    set (opt g)
endif (CMAKE_BUILD_TYPE EQUAL "DEBUG")
  
#set compute capability of tx2
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11 -${opt} -gencode arch=compute_62,code=sm_62)
include_directories( lib/inc)
CUDA_INCLUDE_DIRECTORIES(lib/inc)
cuda_compile(CUDA_SRC lib/src/DWT_Kernels.cu lib/src/Lap_Pyr_Kernels.cu)

add_library(${PROJECT_NAME} ${CPP_SRC} ${CUDA_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE lib/inc)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})  
target_link_libraries(${PROJECT_NAME} PRIVATE -lnppicc -lnppial -lnppicom -lnppidei -lnppif -lnppig -lnppim -lnppisu)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
