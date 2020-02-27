cmake_minimum_required(VERSION 2.8)

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
  
#set compute capability 
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11 -${opt} -gencode arch=compute_${cc},code=sm_${cc})

add_definitions(-std=c++11)
include_directories(lib/inc)
include_directories(${CUDA_INCLUDE_DIRS})  

IF (WIN32)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
ELSE()
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)   
ENDIF()

cuda_compile(CUDA_SRC lib/src/DWT_Kernels.cu lib/src/Lap_Pyr_Kernels.cu)

add_library(${PROJECT_NAME} ${CPP_SRC} ${CUDA_SRC})
target_link_libraries(${PROJECT_NAME} -lnppicc -lnppial -lnppicom -lnppidei -lnppif -lnppig -lnppim -lnppisu)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)