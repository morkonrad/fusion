cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(fusion_stream)

find_package(CUDA REQUIRED)

if(WITH_OCV)
    add_definitions(-D_VISUAL_DBG_OPENCV_)
endif()

if(WITH_OCV)
    find_package(OpenCV REQUIRED)
endif()


#set debug symbols
set (opt O3)
if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
    set (opt g)
endif (CMAKE_BUILD_TYPE EQUAL "DEBUG")




#Specify sources
file(GLOB CPP_SRC "lib/src/*.cpp")

#set compute capability of tx2
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11 -${opt} -gencode arch=compute_62,code=sm_62)
include_directories(lib/inc)
CUDA_INCLUDE_DIRECTORIES(lib/inc)
cuda_compile(CUDA_SRC lib/src/fusion.cu lib/src/dwt_lift.cu lib/src/fusion_rule.cu)

if(WITH_OCV)
    list(APPEND OCV_LIBS "opencv_core" "opencv_imgproc" "opencv_imgcodecs" "opencv_highgui")
    add_executable(demo.exe lib/src/main.cpp ${CUDA_SRC})
    target_link_libraries(demo.exe PRIVATE ${OCV_LIBS} cudart cublas)
endif()

#link cpp objects and cuda object to create target
add_library(${PROJECT_NAME} ${CPP_SRC} ${CUDA_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
























