cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(SPIE18-Fusion)
add_subdirectory(Cuda_Fusion_DWT_Mallat)
add_subdirectory(Cuda_Fusion_DWT_Lift)  
add_subdirectory(OMP_Fusiom_DWT_Mallat)

